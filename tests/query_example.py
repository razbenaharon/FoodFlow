import os
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest
from utils.token_logger import count_tokens, log_tokens
from utils.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    EMBEDDING_DEPLOYMENT_NAME,
    QDRANT_URL,
    QDRANT_API_KEY
)

from langchain_openai import AzureOpenAIEmbeddings

# ========== CONFIG ==========
COLLECTION_NAME = "recipes_mock"
# ============================

# Init embedding model
embedding_model = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=EMBEDDING_DEPLOYMENT_NAME,
)

# Init Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Input query
query_text = "I have leftover rice and chicken"

# Count and log tokens
n_tokens = count_tokens(query_text, model="text-embedding-3-small")
log_tokens(n_tokens, category="embedding")
print(f"🔢 Query tokens: {n_tokens}")

# Generate embedding
query_vector = embedding_model.embed_query(query_text)

# Perform search using query_points
results = client.query_points(
    collection_name=COLLECTION_NAME,
    limit=3,
    with_payload=True,
    query=query_vector
)

# Print results
print("🔍 Top matching recipes:")
for hit in results.points:
    print(f"🍽️ {hit.payload.get('title')} (Score: {hit.score:.4f})")
