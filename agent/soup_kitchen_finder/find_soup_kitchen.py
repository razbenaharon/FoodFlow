import csv
import os
import json
import random


def find_best_open_soup_kitchen():
    """
    Select the best matching soup kitchen from a CSV of nearby options.
    - Reads soup kitchen info (name, address, distance, hours).
    - Picks the 5 closest by distance.
    - Randomly chooses one of the top 5.
    - Prints details and saves the chosen soup kitchen as JSON.
    """

    # --- Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    SOUP_KITCHEN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nearby_soup_kitchens.csv")
    OUTPUT_FILE = os.path.join(RESULTS_DIR, "best_soup_kitchen.json")

    # --- Load soup kitchen data from CSV ---
    kitchens = []
    with open(SOUP_KITCHEN_CSV, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Parse distance (default very large if missing)
                dist = float(row.get("distance_from_ha_salon_km", "999"))
                kitchens.append({
                    "name": row["name"],                          # soup kitchen name
                    "address": row["address"],                    # street address
                    "distance_km": round(dist, 2),                # distance in km (rounded to 2 decimals)
                    "hours": row.get("opening_hours", "").strip() # opening hours (optional)
                })
            except Exception:
                # Skip row if invalid data
                continue

    # --- Handle empty dataset ---
    if not kitchens:
        print("❌ No soup kitchens found in data.")
        return

    # --- Select candidate kitchens ---
    # Sort by distance and keep the 5 closest
    top_5 = sorted(kitchens, key=lambda x: x["distance_km"])[:5]

    # Randomly pick one from the top 5 closest
    chosen = random.choice(top_5)

    # --- Display result ---
    print("\nBest matching soup kitchen:")
    print(f"Name: {chosen['name']}")
    print(f"Address: {chosen['address']}")
    print(f"Distance: {chosen['distance_km']} km")

    # --- Save chosen soup kitchen to results file ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2, ensure_ascii=False)


# Allow running this file directly
if __name__ == "__main__":
    find_best_open_soup_kitchen()
