# agent/execution_agent/send_recipie_to_kitchen.py
"""
This module identifies, retrieves, and formats a selected recipe to be "sent to the kitchen."

It now uses:
- final_decision_output.json → to know which recipe title was selected.
- best_matching_recipes.json → to get the recipe details.

It outputs the formatted recipe message to a text file in results/messages/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

# --- Constants ---
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
RESULTS_DIR    = PROJECT_ROOT / "results"
MESSAGES_DIR   = RESULTS_DIR / "messages"
RECIPIES_DIR = RESULTS_DIR / "recipes"

DECISION_FILE  = RESULTS_DIR / "final_decision_output.json"
BEST_RECIPES_FILE = RECIPIES_DIR / "best_matching_recipes.json"

# --------------------------- helpers ---------------------------

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _selected_recipe_title(path: Path) -> str:
    """Extracts the selected recipe title from final_decision_output.json (COOK action)."""
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list.")
    for row in data:
        if isinstance(row, dict) and row.get("action") == "COOK":
            trg = row.get("target_recipes")
            if isinstance(trg, str) and trg.strip():
                return trg.strip()
            if isinstance(trg, list) and trg:
                return str(trg[0]).strip()
    raise RuntimeError("No COOK action with target_recipes found in final decisions.")

def _find_recipe_in_best(title: str, recipes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Finds a recipe by exact title inside best_matching_recipes.json."""
    for r in recipes:
        if r.get("title", "").strip().lower() == title.strip().lower():
            return r
    raise RuntimeError(f"Recipe '{title}' not found in {BEST_RECIPES_FILE}")

def _compose_kitchen_message(recipe: Dict[str, Any]) -> str:
    """Formats the kitchen message."""
    lines = []
    lines.append("===== KITCHEN DISPATCH =====")
    lines.append(f"Recipe: {recipe.get('title','(unknown)')}")
    lines.append("----------------------------")

    ingredients = recipe.get("matched_expiring", []) + recipe.get("inventory", []) + recipe.get("missing", [])
    if ingredients:
        lines.append("Ingredients:")
        for it in ingredients:
            lines.append(f" - {it}")
    else:
        lines.append("Ingredients: (not available)")

    lines.append("----------------------------")

    reason = recipe.get("reason")
    if reason:
        lines.append("Chef Notes / Reason:")
        lines.append(f" {reason}")

    lines.append("============================")
    return "\n".join(lines)

def _print_recipe(recipe: Dict[str, Any]) -> None:
    print("\n================= SEND TO KITCHEN =================")
    print(f"👨‍🍳  Recipe: {recipe.get('title','(unknown)')}")
    print("---------------------------------------------------")
    print("🧺  Ingredients:")
    for it in recipe.get("matched_expiring", []) + recipe.get("inventory", []) + recipe.get("missing", []):
        print(f"  • {it}")
    print("---------------------------------------------------")
    if recipe.get("reason"):
        print("🧾  Chef Notes / Reason:")
        print(f"  {recipe['reason']}")
    print("===================================================\n")

# ----------------------------- main -----------------------------

def main() -> None:
    if not DECISION_FILE.exists():
        raise FileNotFoundError(f"Missing decision file: {DECISION_FILE}")
    if not BEST_RECIPES_FILE.exists():
        raise FileNotFoundError(f"Missing best recipes file: {BEST_RECIPES_FILE}")

    # 1. Selected title from final_decision_output.json
    selected_title = _selected_recipe_title(DECISION_FILE)

    # 2. Load best_matching_recipes.json
    best_recipes = _load_json(BEST_RECIPES_FILE)

    # 3. Find the matching recipe
    recipe = _find_recipe_in_best(selected_title, best_recipes)

    # 4. Print to console
    _print_recipe(recipe)

    # 5. Save message to file
    MESSAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MESSAGES_DIR / "message_to_kitchen.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_compose_kitchen_message(recipe))

if __name__ == "__main__":
    main()
