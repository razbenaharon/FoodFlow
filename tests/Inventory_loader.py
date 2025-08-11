import json
import os
from utils.chat_and_embedding import LLMEmbed
# TODO: currently not in use
COLLECTION_NAME = "current_inventory_mock"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CURRENT_INVENTORY_FILE = os.path.join(DATA_DIR, "full_inventory.json")

with open(CURRENT_INVENTORY_FILE) as f:
    inventory = json.load(f)

embedder = LLMEmbed()

# --- Build list of (text, payload) pairs ---
text_payload_pairs = []
for item in inventory:
    description = f"{item['item']} ({item['quantity_kg']}kg available)"
    text_payload_pairs.append((description, item))

# --- Embed and upload ---
embedder.embed_line_text_dict(
    collection_name=COLLECTION_NAME,
    text_payload_pairs=text_payload_pairs
)