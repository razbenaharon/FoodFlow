# agent/execution_agent/send_recipie_to_kitchen.py
"""
This module identifies, retrieves, and formats a selected recipe to be "sent to the kitchen."

It uses:
- final_decision_output.json → to know which recipe title was selected.
- best_matching_recipes.json → to get the recipe details.

If no recipe is available (e.g., no COOK action, missing files, or title not found),
the script will NOT raise; it prints a friendly notice and writes a placeholder
message to results/messages/message_to_kitchen.txt explaining the reason.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Constants ---
PROJECT_ROOT      = Path(__file__).resolve().parents[2]
RESULTS_DIR       = PROJECT_ROOT / "results"
MESSAGES_DIR      = RESULTS_DIR / "messages"
RECIPES_DIR       = RESULTS_DIR / "recipes"

DECISION_FILE     = RESULTS_DIR / "final_decision_output.json"
BEST_RECIPES_FILE = RECIPES_DIR / "best_matching_recipes.json"


# --------------------------- helpers ---------------------------

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _selected_recipe_title(path: Path) -> Optional[str]:
    """
    Extract the selected recipe title from final_decision_output.json (COOK action).

    Returns
    -------
    Optional[str]
        The selected title if found, else None. Never raises on normal "not found".
    """
    try:
        data = _load_json(path)
    except FileNotFoundError:
        return None
    except Exception:
        # Malformed JSON or other read error
        return None

    if not isinstance(data, list):
        return None

    for row in data:
        if isinstance(row, dict) and row.get("action") == "COOK":
            trg = row.get("target_recipes")
            if isinstance(trg, str) and trg.strip():
                return trg.strip()
            if isinstance(trg, list) and trg:
                return str(trg[0]).strip()
    return None


def _find_recipe_in_best(title: str, recipes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find a recipe by exact title inside best_matching_recipes.json (case-insensitive)."""
    title_norm = title.strip().lower()
    for r in recipes or []:
        if r.get("title", "").strip().lower() == title_norm:
            return r
    return None


def _compose_kitchen_message(recipe: Dict[str, Any]) -> str:
    """Formats the kitchen message for a found recipe."""
    lines = []
    lines.append("===== KITCHEN DISPATCH =====")
    lines.append(f"Recipe: {recipe.get('title','(unknown)')}")
    lines.append("----------------------------")

    ingredients = (
        recipe.get("matched_expiring", [])
        + recipe.get("inventory", [])
        + recipe.get("missing", [])
    )
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


def _compose_placeholder_message(reason: str) -> str:
    """Fallback message when there is no recipe to send."""
    return (
        "===== KITCHEN DISPATCH =====\n"
        "No recipe was selected to send to the kitchen.\n"
        f"Reason: {reason}\n"
        "============================\n"
    )


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


def _print_placeholder(reason: str) -> None:
    print("\n================= SEND TO KITCHEN =================")
    print("⚠️  No recipe to send to the kitchen.")
    print(f"🙇 Reason: {reason}")
    print("===================================================\n")


# ----------------------------- main -----------------------------

def main() -> None:
    # Ensure messages dir exists early
    MESSAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MESSAGES_DIR / "message_to_kitchen.txt"

    # 1) Get selected title (safe)
    selected_title = _selected_recipe_title(DECISION_FILE)
    if not selected_title:
        reason = (
            "No COOK action with a valid target recipe was found in final decisions "
            f"(expected file: {DECISION_FILE.name})."
        )
        _print_placeholder(reason)
        out_path.write_text(_compose_placeholder_message(reason), encoding="utf-8")
        return

    # 2) Load best recipes (safe)
    try:
        best_recipes = _load_json(BEST_RECIPES_FILE)
        if not isinstance(best_recipes, list):
            best_recipes = []
    except FileNotFoundError:
        reason = f"Missing best matches file: {BEST_RECIPES_FILE.name}."
        _print_placeholder(reason)
        out_path.write_text(_compose_placeholder_message(reason), encoding="utf-8")
        return
    except Exception:
        reason = f"Failed to read {BEST_RECIPES_FILE.name}."
        _print_placeholder(reason)
        out_path.write_text(_compose_placeholder_message(reason), encoding="utf-8")
        return

    # 3) Find the matching recipe by title
    recipe = _find_recipe_in_best(selected_title, best_recipes)
    if not recipe:
        reason = (
            f"Selected recipe '{selected_title}' was not found in "
            f"{BEST_RECIPES_FILE.name}."
        )
        _print_placeholder(reason)
        out_path.write_text(_compose_placeholder_message(reason), encoding="utf-8")
        return

    # 4) Print to console + save message
    _print_recipe(recipe)
    out_path.write_text(_compose_kitchen_message(recipe), encoding="utf-8")


if __name__ == "__main__":
    main()
