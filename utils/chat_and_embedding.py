"""
Helpers for Chat (Azure OpenAI) and Embeddings + Qdrant operations.

- LLMChat:
    Thin wrapper over AzureChatOpenAI to send prompts/messages, log token usage,
    and return the assistant's text content. Also includes a utility to robustly
    extract a JSON string from model output (handles ```json fences).

- LLMEmbed:
    Wraps AzureOpenAIEmbeddings + Qdrant client.
    Provides functions to upsert text+payload points, embed queries,
    and run ANN queries against a Qdrant collection.
"""
import json
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from utils.token_logger import count_tokens, log_tokens
from utils.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    EMBEDDING_DEPLOYMENT_NAME,
    CHAT_DEPLOYMENT_NAME,
    QDRANT_URL,
    QDRANT_API_KEY
)

# Embedding dimensionality for text-embedding-3-small
EMBEDDING_DIM = 1536


class LLMChat:
    """Lightweight chat client around AzureChatOpenAI with token logging."""

    def __init__(self, temp: float = 0):
        self.llm_chat = AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=CHAT_DEPLOYMENT_NAME,
            temperature=temp
        )

    def send_prompt(self, formatted_prompt: str) -> str:
        """
        Send a single string prompt (wrapped as a HumanMessage).
        Kept for backward compatibility.
        """
        messages = [HumanMessage(content=formatted_prompt)]
        return self.send_messages(messages)

    def send_messages(self, messages: list[BaseMessage]) -> str:
        """
        Send a list of messages (System/Human/AI) and return the model text.
        Also logs token usage for budgeting and cost estimation.
        """
        input_content = "\n".join([m.content for m in messages])
        input_tokens = count_tokens(input_content, model="gpt-4")
        log_tokens(input_tokens, category="chat")

        response = self.llm_chat.invoke(messages)

        output_text = response.content
        output_tokens = count_tokens(output_text, model="gpt-4")
        log_tokens(output_tokens, category="chat")

        return output_text

    def extract_json_string(self, text: str) -> str:
        """
        Best-effort JSON extractor from a model response:
        1) If there's a ```json fenced block, return its content.
        2) Otherwise, if the whole response parses as JSON, return it.
        3) Else raise a ValueError.
        """
        text = text.strip()

        # 1) Try fenced ```json blocks
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 2) Try whole text as raw JSON
        try:
            json.loads(text)  # validate only
            return text
        except json.JSONDecodeError:
            pass

        # 3) No valid JSON found
        raise ValueError("No valid JSON block or content found in LLM output.")


class LLMEmbed:
    """Embedding + Vector DB helper backed by Azure OpenAI and Qdrant."""

    def __init__(self):
        # Create embedding client
        self.llm_embed = AzureOpenAIEmbeddings(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=EMBEDDING_DEPLOYMENT_NAME,
        )
        # Create Qdrant client; bump timeout if large batches are used
        self.qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30  # seconds; increase if you expect longer upserts/queries
        )

    def embed_line_text_dict(self, collection_name: str, text_payload_pairs: list[tuple[str, dict]], embedding_dim: int = EMBEDDING_DIM) -> None:
        """
        Upsert a list of (text, payload) pairs into Qdrant as points.
        - `text` is embedded via Azure OpenAI.
        - `payload` MUST contain a unique integer/string `id` used as point ID.

        Creates the collection if it doesn't exist (COSINE distance).
        Logs total embedding tokens.
        """
        if not self.qdrant.collection_exists(collection_name):
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )
            print(f"Collection '{collection_name}' created")
        else:
            print(f"Collection '{collection_name}' already exists")

        points = []
        total_embedding_tokens = 0

        for text, payload in text_payload_pairs:
            vector = self.llm_embed.embed_query(text)

            # Pull point ID from the payload
            point_id = payload.get("id")
            if point_id is None:
                raise ValueError("Missing 'id' in payload")

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            n_tokens = count_tokens(text)
            total_embedding_tokens += n_tokens
            log_tokens(n_tokens, category="embedding")

        print(f"Total embedding tokens: {total_embedding_tokens}")
        self.qdrant.upsert(collection_name=collection_name, points=points)
        print(f"Uploaded {len(points)} items to Qdrant collection '{collection_name}'")

    def embed_query(self, query_str: str) -> list[float]:
        """
        Embed a search query string and log token usage. Returns the vector.
        """
        query_tokens = count_tokens(query_str)
        log_tokens(query_tokens, category="embedding")
        query_vector = self.llm_embed.embed_query(query_str)
        return query_vector

    def return_query_results(self, query_vector: list[float], collection_name: str, limit_res: int, with_payload: bool = True):
        """
        Query Qdrant ANN index for nearest vectors.
        """
        res = self.qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit_res,
            with_payload=with_payload
        )
        return res

    def print_results(self, res) -> None:
        """
        Pretty-print top matches from a Qdrant query response.
        """
        print("Top matching results:")
        for hit in res.points:
            print(f"{hit.payload.get('title')} (Score: {hit.score:.4f})")
