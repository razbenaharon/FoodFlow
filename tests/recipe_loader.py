import json
from chat_and_embedding import LLMEmbed

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
