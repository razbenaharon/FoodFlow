import shutil
from datetime import datetime
from pathlib import Path
import json
import random

# === Define Paths ===
PROJECT_ROOT              = Path(__file__).resolve().parent.parent
DATA_DIR                  = PROJECT_ROOT / "data"
RESULTS_DIR               = PROJECT_ROOT / "results"
PROMPT_DIR                = PROJECT_ROOT / "agent" / "decision_agent" / "prompts"

FULL_INVENTORY_FILE       = DATA_DIR / "full_inventory.json"
CURRENT_INVENTORY_FILE    = DATA_DIR / "current_inventory.json"
EXPIRING_INGREDIENTS_FILE = DATA_DIR / "expiring_ingredients.json"
EXPIRED_HISTORY_FILE = DATA_DIR / "recent_expiring_ingredients.json"
NEARBY_LOCATIONS_FILE     = DATA_DIR / "nearby_restaurants.csv"
TOP_RECIPES_FILE          = RESULTS_DIR / "top_recipes.json"
TOP_RESTAURANTS_FILE      = RESULTS_DIR / "top_restaurants.json"
DECISION_PROMPT_FILE      = PROMPT_DIR / "ingredient_decision_prompt.txt"


def prepare_inventory():
    # 1. Load full inventory
    with open(FULL_INVENTORY_FILE, "r", encoding="utf-8") as f:
        full_inventory = json.load(f)

    # 2. Filter out salt and water from candidates for expiring items
    blacklist = {"salt", "water", "sea salt", "lemon"}
    expiring_candidates = [
        ing for ing in full_inventory
        if (ing.get("item") or ing.get("name")).lower() not in blacklist
    ]

    # 3. Pick 10 random expiring items
    expiring = random.sample(expiring_candidates, min(10, len(expiring_candidates)))

    # 4. Process expiring items: reduce quantity and add days_to_expire
    processed_expiring = []
    for ing in expiring:
        item_name = ing.get("item") or ing.get("name")
        factor = random.uniform(0.3, 0.8)
        new_qty = round(ing["quantity"] * factor, 2)
        processed_expiring.append({
            "name": item_name,
            "quantity": new_qty,
            "unit": ing.get("unit"),
            "days_to_expire": random.randint(1, 4)
        })

    # 5. Build current inventory (exclude expiring)
    expiring_names = {i["name"] for i in processed_expiring}
    current_inventory = []
    for ing in full_inventory:
        item_name = ing.get("item") or ing.get("name")
        if item_name not in expiring_names:
            current_inventory.append({
                "name": item_name,
                "quantity": ing["quantity"],
                "unit": ing.get("unit")
            })

    # 6. Save to files
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPIRING_INGREDIENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_expiring, f, ensure_ascii=False, indent=2)

    with open(CURRENT_INVENTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(current_inventory, f, ensure_ascii=False, indent=2)

    update_expired_history(processed_expiring)

    # 7. Print beautifully
    print("📦 Current Inventory (excluding expiring):")
    for ing in current_inventory:
        print(f"- {ing['name']}: {ing['quantity']} {ing['unit']}")

    print("\n⏳ Expiring Ingredients:")
    for ing in processed_expiring:
        print(f"- {ing['name']}: {ing['quantity']} {ing['unit']} (Expires in {ing['days_to_expire']} days)")

    return

def update_expired_history(new_expiring):
    # Load existing history (or start new)
    if EXPIRED_HISTORY_FILE.exists():
        with open(EXPIRED_HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            history = []
        else:
            history = json.loads(content)

    else:
        history = []

    # Add current batch
    history.append(new_expiring)

    # Save updated history
    with open(EXPIRED_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
