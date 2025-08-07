import csv
import os

# --- Utility Functions ---
def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

NEARBY_RESTAURANT_FILE_PATH = os.path.join(DATA_DIR, "nearby_restaurants.csv")

# ========== Load and clean restaurant data from CSV ==========
restaurants = []
with open(NEARBY_RESTAURANT_FILE_PATH, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for idx, row in enumerate(reader):
        restaurant = {
            "id": f"rest_{idx}",
            "name": row["name"],
            "cuisine": row.get("types", "unknown"),
            "distance_km": safe_float(row.get("distance_from_ha_salon_km")),
            "address": row["address"],
            "lat": safe_float(row.get("lat")),
            "lng": safe_float(row.get("lng")),
            "rating": safe_float(row.get("rating")),
            "user_ratings_total": safe_int(row.get("user_ratings_total")),
            "name_english": row.get("name_english", row["name"])
        }
        restaurants.append(restaurant)

# ========== Generate descriptions (no embedding/upload) ==========
for r in restaurants:
    description = (
        f"{r['name_english']} is a {r['cuisine']} restaurant located {r['distance_km']} km away. "
        f"It has a rating of {r['rating']} based on {r['user_ratings_total']} reviews."
    )
    print(description)
