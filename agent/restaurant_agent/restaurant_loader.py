import json
from agent.chat_and_embedding import LLMEmbed
import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

NEARBY_RESTAURANT_FILE_PATH = os.path.join(DATA_DIR, "nearby_locations.json")

COLLECTION_NAME = "restaurants_mock"

# ========== Load restaurant data ==========
with open(NEARBY_RESTAURANT_FILE_PATH, "r", encoding="utf-8") as f:
    restaurant_data = json.load(f)
    restaurants = restaurant_data.get("restaurant_buyers", [])

# ========== Create instance of embedding agent ==========
embedder = LLMEmbed()

# ========== Build list of (text, payload) tuples ==========
text_payload_pairs = []
for r in restaurants:
    description = (
        f"{r['name']} is a {r['cuisine']} restaurant located {r['distance_km']} km away."
    )
    text_payload_pairs.append((description, r))

# ========== Embed and upload to Qdrant ==========
embedder.embed_line_text_dict(
    collection_name=COLLECTION_NAME,
    text_payload_pairs=text_payload_pairs
)