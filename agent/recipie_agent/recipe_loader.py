import json
from agent.chat_and_embedding import LLMEmbed

COLLECTION_NAME = "recipes_mock"
RECIPE_FILE_PATH = "data/recipes_mock.json"

# ========== Load recipes file ==========
with open(RECIPE_FILE_PATH, "r") as f:
    recipes = json.load(f)

# ========== Create instance of embedding agent ==========
embedder = LLMEmbed()

# ========== Build list of (text, payload) tuples ==========
text_payload_pairs = []
for recipe in recipes:
    description = f"Recipe: {recipe['title']}. Ingredients: " + ", ".join([ing['item'] for ing in recipe['ingredients']])
    text_payload_pairs.append((description, recipe))

# ========== Embed and upload to Qdrant ==========
embedder.embed_line_text_dict(
    collection_name=COLLECTION_NAME,
    text_payload_pairs=text_payload_pairs
)

# import json
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct
# from langchain_openai import AzureOpenAIEmbeddings
# from utils.token_logger import count_tokens, log_tokens
# from utils.config import (
#     AZURE_OPENAI_API_KEY,
#     AZURE_OPENAI_API_VERSION,
#     AZURE_OPENAI_ENDPOINT,
#     EMBEDDING_DEPLOYMENT_NAME,
#     QDRANT_URL,
#     QDRANT_API_KEY
# )
#
# # ========== CONFIGURATION ==========
# COLLECTION_NAME = "recipes_mock"
# EMBEDDING_DIM = 1536  # For text-embedding-3-small
# RECIPE_FILE_PATH = "data/recipes_mock.json"
# # ===================================
#
#
# # Load recipes file
# with open(RECIPE_FILE_PATH, "r") as f:
#     recipes = json.load(f)
#
# # Create embedding model using Azure
# embedding_model = AzureOpenAIEmbeddings(
#     api_key=AZURE_OPENAI_API_KEY,
#     api_version=AZURE_OPENAI_API_VERSION,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     azure_deployment=EMBEDDING_DEPLOYMENT_NAME,
# )
#
# # Connect to Qdrant
# client = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY,
# )
#
# # Create collection if not exists
# if not client.collection_exists(COLLECTION_NAME):
#     client.create_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
#     )
#     print(f"🟢 Collection '{COLLECTION_NAME}' created")
# else:
#     print(f"🟡 Collection '{COLLECTION_NAME}' already exists")
#
# # Embed and upload recipes
# points = []
# total_embedding_tokens = 0
#
# for i, recipe in enumerate(recipes):
#     text = f"Recipe: {recipe['title']}. Ingredients: " + ", ".join([ing['item'] for ing in recipe['ingredients']])
#     vector = embedding_model.embed_query(text)
#     points.append(PointStruct(id=i, vector=vector, payload=recipe))
#
#     n_tokens = count_tokens(text)
#     total_embedding_tokens += n_tokens
#     log_tokens(n_tokens, category="embedding")
#
# print(f"🔢 Total embedding tokens: {total_embedding_tokens}")
#
# client.upsert(collection_name=COLLECTION_NAME, points=points)
#
# print(f"✅ Uploaded {len(points)} recipes to Qdrant collection '{COLLECTION_NAME}'")