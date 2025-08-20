"""
Utility to prepare inventory for a simulation run.

- Reads the full inventory (`full_inventory.json`)
- Modes:
    1) "auto"  : Randomly selects up to 10 expiring items (excludes a blacklist),
                 reduces quantities and assigns 1–4 days_to_expire.
    2) "manual": Interactive selection of expiring items, setting quantity + days.
- Writes:
    * `expiring_ingredients.json` (reduced/entered quantities + days_to_expire)
    * `current_inventory.json`    (full inventory minus expiring items)
- Appends the expiring batch to `recent_expiring_ingredients.json`

Programmatic use (keeps current behavior by default):
    from prepare_inventory import prepare_inventory
    prepare_inventory()                  # auto (random) mode
    prepare_inventory(mode="manual")     # manual interactive mode

Running this file directly shows a banner and lets the user choose the mode.
"""

from pathlib import Path
import json
import random
from typing import List, Dict, Any, Tuple

# === Define Paths ===
PROJECT_ROOT              = Path(__file__).resolve().parent.parent
DATA_DIR                  = PROJECT_ROOT / "data"
RESULTS_DIR               = PROJECT_ROOT / "results"
PROMPT_DIR                = PROJECT_ROOT / "agent" / "decision_agent" / "prompts"

FULL_INVENTORY_FILE       = DATA_DIR / "full_inventory.json"
CURRENT_INVENTORY_FILE    = DATA_DIR / "current_inventory.json"
EXPIRING_INGREDIENTS_FILE = DATA_DIR / "expiring_ingredients.json"
EXPIRED_HISTORY_FILE      = DATA_DIR / "recent_expiring_ingredients.json"
NEARBY_LOCATIONS_FILE     = DATA_DIR / "nearby_restaurants.csv"
TOP_RECIPES_FILE          = RESULTS_DIR / "top_recipes.json"
TOP_RESTAURANTS_FILE      = RESULTS_DIR / "top_restaurants.json"
DECISION_PROMPT_FILE      = PROMPT_DIR / "ingredient_decision_prompt.txt"

# Items we avoid auto-marking as "expiring" in random mode
BLACKLIST_AUTO = {"salt", "water", "sea salt", "lemon"}


# ----------------------------- Helpers ----------------------------- #
def _load_full_inventory() -> List[Dict[str, Any]]:
    with open(FULL_INVENTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_expired_history(new_expiring: list) -> None:
    """Append the latest expiring batch to the rolling history JSON file."""
    if EXPIRED_HISTORY_FILE.exists():
        with open(EXPIRED_HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        history = json.loads(content) if content else []
    else:
        history = []

    history.append(new_expiring)
    _save_json(EXPIRED_HISTORY_FILE, history)


def _print_result(current_inventory: List[Dict[str, Any]],
                  processed_expiring: List[Dict[str, Any]]) -> None:
    print("📦 Current Inventory (excluding expiring):")
    for ing in current_inventory:
        print(f"- {ing['name']}: {ing['quantity']} {ing.get('unit')}")

    print("\n⏳ Expiring Ingredients:")
    for ing in processed_expiring:
        print(f"- {ing['name']}: {ing['quantity']} {ing.get('unit')} "
              f"(Expires in {ing['days_to_expire']} days)")


def _is_unlimited_entry(ing: Dict[str, Any]) -> bool:
    """True if the item is non-selectable (Water/Salt/unlimited)."""
    name = (ing.get("name") or ing.get("item") or "").strip().lower()
    qty = ing.get("quantity")
    if name in {"water", "salt"}:
        return True
    if isinstance(qty, str) and qty.strip().lower() == "unlimited":
        return True
    return False


def _name_qty_unit(ing: Dict[str, Any]) -> Tuple[str, Any, Any]:
    return (ing.get("name") or ing.get("item")), ing.get("quantity"), ing.get("unit")


# ----------------------------- Auto (random) mode ----------------------------- #
def _auto_prepare_inventory() -> None:
    """Create expiring and current inventory using random selection."""
    full_inventory = _load_full_inventory()

    # Filter out items we never want to mark as "expiring" automatically
    expiring_candidates = []
    for ing in full_inventory:
        name = (ing.get("item") or ing.get("name") or "").strip()
        if not name:
            continue
        if name.lower() in BLACKLIST_AUTO:
            continue
        expiring_candidates.append(ing)

    max_items = min(10, len(expiring_candidates))
    min_items = min(5, max_items)  # Avoids min > max

    num_to_expire = random.randint(min_items, max_items) if max_items > 0 else 0
    expiring = random.sample(expiring_candidates, num_to_expire) if num_to_expire > 0 else []

    processed_expiring = []
    for ing in expiring:
        item_name, qty, unit = _name_qty_unit(ing)
        # If quantity is numeric, multiply by factor; otherwise keep as-is (e.g., "unlimited")
        if isinstance(qty, (int, float)):
            factor = random.uniform(0.3, 0.8)  # keep 30–80% (simulate partial remaining)
            new_qty = round(qty * factor, 2)
        else:
            new_qty = qty

        processed_expiring.append({
            "name": item_name,
            "quantity": new_qty,
            "unit": unit,
            "days_to_expire": random.randint(1, 4)
        })

    # Build current inventory by excluding expiring item names
    expiring_names = {i["name"] for i in processed_expiring}
    current_inventory = []
    for ing in full_inventory:
        item_name, qty, unit = _name_qty_unit(ing)
        if item_name not in expiring_names:
            current_inventory.append({
                "name": item_name,
                "quantity": qty,
                "unit": unit
            })

    # Persist to disk
    _save_json(EXPIRING_INGREDIENTS_FILE, processed_expiring)
    _save_json(CURRENT_INVENTORY_FILE, current_inventory)
    update_expired_history(processed_expiring)

    _print_result(current_inventory, processed_expiring)


# ----------------------------- Manual (interactive) mode ----------------------------- #
def _manual_prepare_inventory() -> None:
    """
    Let the user manually choose expiring ingredients.
    - Shows a numbered list of full inventory
    - User selects indices (comma-separated)
    - For each chosen item, user sets quantity remaining and days to expire
    - Constraints:
        * Water/Salt/unlimited entries are NOT selectable.
        * days_to_expire must be 1..7
        * quantity must be > 0 and < original inventory amount
    """
    full_inventory = _load_full_inventory()

    # Determine non-selectable indices (Water/Salt/unlimited)
    non_selectable = set()
    print("\n📋 Full Inventory:")
    for idx, ing in enumerate(full_inventory, 1):
        name, qty, unit = _name_qty_unit(ing)
        is_blocked = _is_unlimited_entry(ing)
        if is_blocked:
            non_selectable.add(idx)

        suffix = " (not selectable)" if is_blocked else ""
        print(f"{idx}. {name} ({qty} {unit}){suffix}")

    # Read selections
    raw = input("\nEnter numbers of items to mark as expiring (comma separated, e.g. 1,4,8): ").strip()
    chosen_indices = []
    if raw:
        for token in raw.split(","):
            token = token.strip()
            if token.isdigit():
                i = int(token)
                if 1 <= i <= len(full_inventory) and i not in non_selectable:
                    chosen_indices.append(i)
                elif i in non_selectable:
                    print(f"Skipping {i}: This item is not selectable (Water/Salt/unlimited).")
    chosen_indices = sorted(set(chosen_indices))

    if not chosen_indices:
        print("No valid selections. Nothing to mark as expiring.")
        # Write consistent outputs (empty expiring -> current = full)
        processed_expiring = []
        current_inventory = [
            {"name": (ing.get("name") or ing.get("item")),
             "quantity": ing.get("quantity"),
             "unit": ing.get("unit")}
            for ing in full_inventory
        ]
        _save_json(EXPIRING_INGREDIENTS_FILE, processed_expiring)
        _save_json(CURRENT_INVENTORY_FILE, current_inventory)
        update_expired_history(processed_expiring)
        _print_result(current_inventory, processed_expiring)
        return

    # Collect details for each chosen item
    processed_expiring = []
    for idx in chosen_indices:
        ing = full_inventory[idx - 1]
        name, orig_qty, unit = _name_qty_unit(ing)

        # Guard again (shouldn't happen due to filtering)
        if _is_unlimited_entry(ing):
            print(f"Skipping '{name}': not selectable.")
            continue

        # Quantity (numeric, >0 and < original)
        if not isinstance(orig_qty, (int, float)):
            print(f"Skipping '{name}': original quantity is not numeric.")
            continue

        while True:
            q_in = input(f"Quantity remaining for '{name}' (same unit: {unit}, must be < {orig_qty}): ").strip()
            try:
                qty_val = float(q_in)
                if qty_val <= 0:
                    print("Quantity must be > 0.")
                    continue
                if qty_val >= float(orig_qty):
                    print(f"Quantity must be strictly less than original amount ({orig_qty}).")
                    continue
                break
            except ValueError:
                print("Please enter a valid number (e.g., 2 or 2.5).")

        # Days to expire: integer in [1, 7]
        while True:
            d_in = input(f"Days to expire for '{name}' (1-7): ").strip()
            if d_in.isdigit():
                days_val = int(d_in)
                if 1 <= days_val <= 7:
                    break
            print("Please enter an integer between 1 and 7.")

        processed_expiring.append({
            "name": name,
            "quantity": qty_val,
            "unit": unit,
            "days_to_expire": days_val
        })

    # Build current inventory = everything else
    expiring_names = {i["name"] for i in processed_expiring}
    current_inventory = [
        {"name": (ing.get("name") or ing.get("item")),
         "quantity": ing.get("quantity"),
         "unit": ing.get("unit")}
        for ing in full_inventory
        if (ing.get("name") or ing.get("item")) not in expiring_names
    ]

    # Persist
    _save_json(EXPIRING_INGREDIENTS_FILE, processed_expiring)
    _save_json(CURRENT_INVENTORY_FILE, current_inventory)
    update_expired_history(processed_expiring)

    _print_result(current_inventory, processed_expiring)


# ----------------------------- Public API ----------------------------- #
def prepare_inventory(mode: str = "auto") -> None:
    """
    Prepare inventory for a run.

    Parameters
    ----------
    mode : {"auto", "manual"}
        - "auto": Randomly selects expiring items (blacklist applied). DEFAULT.
        - "manual": Interactive selection via stdin.
    """
    if mode not in {"auto", "manual"}:
        raise ValueError("mode must be 'auto' or 'manual'")

    if mode == "auto":
        _auto_prepare_inventory()
    else:
        _manual_prepare_inventory()

