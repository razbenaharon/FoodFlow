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

EMBEDDING_DIM = 1536  # For text-embedding-3-small


class LLMChat:
    def __init__(self, temp=0):
        self.llm_chat = AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=CHAT_DEPLOYMENT_NAME,
            temperature=temp
        )

    def send_prompt(self, formatted_prompt):
        # Backward compatibility: wrap prompt as HumanMessage
        messages = [HumanMessage(content=formatted_prompt)]
        return self.send_messages(messages)

    def send_messages(self, messages: list[BaseMessage]):
        input_content = "\n".join([m.content for m in messages])
        input_tokens = count_tokens(input_content, model="gpt-4")
        log_tokens(input_tokens, category="chat")
        # print(f"Prompt token count: {input_tokens}")

        response = self.llm_chat.invoke(messages)

        output_text = response.content
        output_tokens = count_tokens(output_text, model="gpt-4")
        log_tokens(output_tokens, category="chat")
        # print(f"Completion token count: {output_tokens}")
        # print(f"Total tokens this run: {input_tokens + output_tokens}")

        return output_text

    def extract_json_string(self, text):
        text = text.strip()

        # 1. Try extracting from ```json block
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 2. Try using the whole text as raw JSON
        try:
            json.loads(text)  # just to validate it's valid
            return text
        except json.JSONDecodeError:
            pass

        # 3. Fallback: no valid JSON found
        raise ValueError("No valid JSON block or content found in LLM output.")

class LLMEmbed:
    def __init__(self):
        self.llm_embed = AzureOpenAIEmbeddings(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=EMBEDDING_DEPLOYMENT_NAME,
        )
        self.qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout= 30
        )

    def embed_line_text_dict(self, collection_name, text_payload_pairs, embedding_dim=EMBEDDING_DIM):
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

            # ✅ Pull point ID from the payload (created in main script)
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

    def embed_query(self, query_str):
        query_tokens = count_tokens(query_str)
        log_tokens(query_tokens, category="embedding")
        query_vector = self.llm_embed.embed_query(query_str)

        return query_vector

    def return_query_results(self, query_vector, collection_name, limit_res, with_payload=True):
        res = self.qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit_res,
            with_payload=with_payload
        )
        return res

    def print_results(self, res):
        # Print results
        print("Top matching results:")
        for hit in res.points:
            print(f"{hit.payload.get('title')} (Score: {hit.score:.4f})")
