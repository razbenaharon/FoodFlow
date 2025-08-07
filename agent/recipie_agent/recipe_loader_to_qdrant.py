import os
import json
import uuid
import pandas as pd
import time
from dotenv import load_dotenv
from chat_and_embedding import LLMEmbed

# --- Environment and configuration ---
load_dotenv()
embedder = LLMEmbed()
COLLECTION_NAME = "food_recipes"
STATE_PATH = "../uploaded.json"

# --- Load uploaded state from file ---
if os.path.exists(STATE_PATH):
    with open(STATE_PATH, 'r') as f:
        uploaded_keys = set(json.load(f))
else:
    uploaded_keys = set()

def save_state():
    with open(STATE_PATH, 'w') as f:
        json.dump(list(uploaded_keys), f)

# --- Upload a DataFrame to Qdrant ---
def upload_to_qdrant(data: pd.DataFrame, text_column: str, file_key: str):
    if text_column not in data.columns:
        raise ValueError(f"❌ Column '{text_column}' not found in data.")

    uploaded = 0
    batch = []

    total_rows = len(data)
    print(f"\n📂 Starting upload for: {file_key} ({total_rows} rows)")

    for idx, row in data.iterrows():
        if idx % 1000 == 0:
            print(f"[{file_key}] Processed {idx}/{total_rows} rows...")

        key = f"{file_key}::{idx}"
        if key in uploaded_keys:
            continue

        text = str(row.get(text_column, "")).strip()
        if not text or len(text) < 10:
            continue

        payload = row.to_dict()
        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, key).int >> 96  # 32-bit ID
        payload["id"] = point_id

        batch.append((text, payload))
        uploaded_keys.add(key)
        uploaded += 1

        if len(batch) >= 100:
            _upload_batch(batch, uploaded)
            batch = []

    if batch:
        _upload_batch(batch, uploaded)

    print(f"✅ Finished uploading file: {file_key}\n")

def _upload_batch(batch, uploaded):
    try:
        start = time.time()
        embedder.embed_line_text_dict(COLLECTION_NAME, batch)
        print(f"✅ Uploaded {len(batch)} points (batch), total new: {uploaded} "
              f"(took {time.time() - start:.2f}s)")
        total = embedder.qdrant.count(COLLECTION_NAME, exact=True).count
        print(f"📦 Qdrant now contains {total} total points")
        save_state()
    except Exception as e:
        print(f"❌ Upload failed: {e}")

# --- Upload from Excel or CSV file ---
def upload_excel(path: str, text_column: str):
    file_key = os.path.basename(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    upload_to_qdrant(df, text_column, file_key)

# --- Upload from JSON file ---
def upload_json(path: str, text_column: str):
    file_key = os.path.basename(path)
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    records = list(raw.values()) if isinstance(raw, dict) else raw
    df = pd.DataFrame(records)
    upload_to_qdrant(df, text_column, file_key)

# --- Base directory containing the files ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "recipes"))

# --- Run all uploads ---
upload_excel(os.path.join(BASE_DIR, "RAW_recipes.csv"), "description")
upload_excel(os.path.join(BASE_DIR, "13k-recipes.csv"), "Instructions")
upload_excel(os.path.join(BASE_DIR, "food_recipes.csv"), "description")
upload_json(os.path.join(BASE_DIR, "recipes_raw_nosource_fn.json"), "instructions")
upload_json(os.path.join(BASE_DIR, "recipes_raw_nosource_epi.json"), "instructions")
upload_json(os.path.join(BASE_DIR, "recipes_raw_nosource_ar.json"), "instructions")

print("✅ All files processed.")
