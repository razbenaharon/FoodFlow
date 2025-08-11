import csv
import os
import json
import random

def find_best_open_soup_kitchen():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    SOUP_KITCHEN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nearby_soup_kitchens.csv")
    OUTPUT_FILE = os.path.join(RESULTS_DIR, "best_soup_kitchen.json")

    kitchens = []
    with open(SOUP_KITCHEN_CSV, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                dist = float(row.get("distance_from_ha_salon_km", "999"))
                kitchens.append({
                    "name": row["name"],
                    "address": row["address"],
                    "distance_km": round(dist, 2),
                    "hours": row.get("opening_hours", "").strip()
                })
            except Exception:
                continue

    if not kitchens:
        print("❌ No soup kitchens found in data.")
        return

    top_5 = sorted(kitchens, key=lambda x: x["distance_km"])[:5]
    chosen = random.choice(top_5)

    print("\nBest matching soup kitchen:")
    print(f"Name: {chosen['name']}")
    print(f"Address: {chosen['address']}")
    print(f"Distance: {chosen['distance_km']} km")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    find_best_open_soup_kitchen()
