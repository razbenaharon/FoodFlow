from qdrant_client import QdrantClient
from pprint import pprint

client = QdrantClient("localhost", port=6333)

# --- 1. רשימת הקולקשנים ---
collections = client.get_collections()
collection_names = [col.name for col in collections.collections]
print("📦 Available collections:", collection_names)

# --- 2. בדיקה אם food_recipes קיים ---
if "food_recipes" not in collection_names:
    print("❌ Collection 'food_recipes' does not exist. Run your upload script first.")
else:
    print("✅ Collection 'food_recipes' exists.")

    # --- 3. שליפה של נקודה אחת ---
    result = client.scroll(
        collection_name="food_recipes",
        limit=1,
        with_payload=True,
        with_vectors=False  # לא צריך את הווקטור עצמו
    )

    if result and result[0]:
        print("\n🧾 Example payload from 'food_recipes':")
        pprint(result[0][0].payload)
    else:
        print("⚠️ Collection exists but contains no points.")
