import os
import json
import uuid
import pandas as pd
import time
from dotenv import load_dotenv
from utils.chat_and_embedding import LLMEmbed

# This script is for one-time data loading. It is not part of the main agent execution loop.

# --- Environment and configuration ---
load_dotenv()  # Load environment variables from a .env file.
embedder = LLMEmbed()  # Initialize the embedding model and Qdrant client.
COLLECTION_NAME = "food_recipes"  # Name of the Qdrant collection to store embeddings.
STATE_PATH = "uploaded.json"  # File to keep track of uploaded data points to avoid duplicates.

# --- Load uploaded state from file ---
# Check if the state file exists to resume an interrupted upload process.
if os.path.exists(STATE_PATH):
    with open(STATE_PATH, 'r') as f:
        # Load the set of keys that have already been uploaded.
        uploaded_keys = set(json.load(f))
else:
    # If no state file exists, start with an empty set.
    uploaded_keys = set()

def save_state():
    """Saves the set of uploaded keys to a file."""
    with open(STATE_PATH, 'w') as f:
        json.dump(list(uploaded_keys), f)

# --- Upload a DataFrame to Qdrant ---
def upload_to_qdrant(data: pd.DataFrame, text_column: str, file_key: str):
    """
    Processes a DataFrame, generates embeddings, and uploads data to Qdrant.

    Args:
        data (pd.DataFrame): The DataFrame containing the recipe data.
        text_column (str): The name of the column to be used for text embedding.
        file_key (str): A unique identifier for the source file to track uploads.
    """
    if text_column not in data.columns:
        raise ValueError(f"❌ Column '{text_column}' not found in data.")

    uploaded = 0
    batch = []

    total_rows = len(data)
    print(f"\n📂 Starting upload for: {file_key} ({total_rows} rows)")

    for idx, row in data.iterrows():
        # Print progress every 1000 rows.
        if idx % 1000 == 0:
            print(f"[{file_key}] Processed {idx}/{total_rows} rows...")

        # Create a unique key for each row to check against the uploaded state.
        key = f"{file_key}::{idx}"
        if key in uploaded_keys:
            continue

        # Get the text for embedding, skipping empty or short entries.
        text = str(row.get(text_column, "")).strip()
        if not text or len(text) < 10:
            continue

        # Prepare the payload (the original row data).
        payload = row.to_dict()
        # Generate a unique 32-bit ID for the Qdrant point.
        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, key).int >> 96
        payload["id"] = point_id

        # Add the text and payload to the current batch.
        batch.append((text, payload))
        uploaded_keys.add(key)
        uploaded += 1

        # Upload the batch when it reaches a certain size.
        if len(batch) >= 100:
            _upload_batch(batch, uploaded)
            batch = []

    # Upload any remaining items in the last batch.
    if batch:
        _upload_batch(batch, uploaded)

    print(f"✅ Finished uploading file: {file_key}\n")

def _upload_batch(batch, uploaded):
    """Internal helper to perform the actual batch upload and update state."""
    try:
        start = time.time()
        # Call the embedding utility to embed and upload the batch.
        embedder.embed_line_text_dict(COLLECTION_NAME, batch)
        # Log the success and performance of the batch upload.
        print(f"✅ Uploaded {len(batch)} points (batch), total new: {uploaded} "
              f"(took {time.time() - start:.2f}s)")
        # Get and print the total count of points in the collection.
        total = embedder.qdrant.count(COLLECTION_NAME, exact=True).count
        print(f"📦 Qdrant now contains {total} total points")
        # Save the current state of uploaded keys.
        save_state()
    except Exception as e:
        # Log any errors that occur during the upload.
        print(f"❌ Upload failed: {e}")

# --- Upload from Excel or CSV file ---
def upload_excel(path: str, text_column: str):
    """Loads data from a .csv or .xlsx file and uploads it."""
    file_key = os.path.basename(path)
    # Use pandas to read the file based on its extension.
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    # Call the main upload function with the DataFrame.
    upload_to_qdrant(df, text_column, file_key)

# --- Upload from JSON file ---
def upload_json(path: str, text_column: str):
    """Loads data from a .json file and uploads it."""
    file_key = os.path.basename(path)
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # Handle both list and dictionary JSON formats.
    records = list(raw.values()) if isinstance(raw, dict) else raw
    # Convert the records to a DataFrame.
    df = pd.DataFrame(records)
    # Call the main upload function with the DataFrame.
    upload_to_qdrant(df, text_column, file_key)

# --- Base directory containing the files ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "recipes"))

# --- Run all uploads ---
# The main execution block that calls the upload functions for all specified files.
upload_excel(os.path.join(BASE_DIR, "RAW_recipes.csv"), "description")
upload_excel(os.path.join(BASE_DIR, "13k-recipes.csv"), "Instructions")
upload_excel(os.path.join(BASE_DIR, "food_recipes.csv"), "description")
upload_json(os.path.join(BASE_DIR, "recipes_raw_nosource_fn.json"), "instructions")
upload_json(os.path.join(BASE_DIR, "recipes_raw_nosource_epi.json"), "instructions")
upload_json(os.path.join(BASE_DIR, "recipes_raw_nosource_ar.json"), "instructions")

print("✅ All files processed.")