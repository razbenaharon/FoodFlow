# final_decision_agent.py
# Final Decision Agent - LLM-based decision making
# -----------------------------------------------------------------------------
# This agent uses an LLM to make the final decisions about ingredient actions.
# It loads the required data files and sends them to the LLM with the prompt
# template to get COOK, SELL, and DONATE decisions.

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

# Import the LLM chat utility (same as used in find_restaurant.py)
from utils.chat_and_embedding import LLMChat
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate
from langchain_core.messages import HumanMessage  # ✅ Use LangChain message type

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RECIPES_DIR = RESULTS_DIR / "recipes"

# Input files
BEST_RECIPES_FILE = RECIPES_DIR / "best_matching_recipes.json"
EXPIRING_FILE = PROJECT_ROOT / "data" / "expiring_ingredients.json"
TOP_RESTAURANTS_FILE = RESULTS_DIR / "top_restaurants.json"
DONATION_FILE = RESULTS_DIR / "best_soup_kitchen.json"
PROMPT_FILE = Path(__file__).parent / "prompts" / "ingredient_decision_prompt.txt"

# Output file
OUTPUT_FILE = RESULTS_DIR / "final_decision_output.json"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> Any:
    """Safely loads a JSON file with UTF-8 encoding."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def _load_prompt_template() -> str:
    """Load the prompt template from the prompts directory.
    Raises FileNotFoundError if the file is missing.
    """
    with open(PROMPT_FILE, encoding="utf-8") as f:
        return f.read()


def _format_expiring_ingredients(expiring_data: Any) -> str:
    """Format expiring ingredients data for the prompt."""
    if not expiring_data:
        return "No expiring ingredients found."

    if not isinstance(expiring_data, list):
        return str(expiring_data)

    formatted = []
    for item in expiring_data:
        if isinstance(item, dict):
            name = item.get("item") or item.get("name", "Unknown")
            quantity = item.get("quantity", "")
            unit = item.get("unit", "")
            days = item.get("days_to_expire") or item.get("days") or item.get("expires_in_days", "Unknown")
            formatted.append(f"- {name}: {quantity} {unit}, expires in {days} days")
        elif isinstance(item, str):
            formatted.append(f"- {item}")

    return "\n".join(formatted)


def _format_recipes(recipes_data: Any) -> str:
    """Format recipe data for the prompt."""
    if not recipes_data:
        return "No recipes found."

    if not isinstance(recipes_data, list):
        return str(recipes_data)

    formatted = []
    for i, recipe in enumerate(recipes_data[:5]):  # Limit to top 5 recipes
        if isinstance(recipe, dict):
            title = recipe.get("title", f"Recipe {i + 1}")
            rank = recipe.get("rank", "Unknown")
            matched = recipe.get("matched_expiring", [])
            instructions = recipe.get("instructions", "No instructions")

            formatted.append(f"Recipe {rank}: {title}")
            formatted.append(f"  Matched expiring ingredients: {', '.join(matched) if matched else 'None'}")
            formatted.append(f"  Instructions: {instructions[:200]}{'...' if len(instructions) > 200 else ''}")
            formatted.append("")

    return "\n".join(formatted)


def _format_restaurants(restaurants_data: Any) -> str:
    """Format restaurant data for the prompt."""
    if not restaurants_data:
        return "No restaurants found."

    if isinstance(restaurants_data, dict):
        # Handle single restaurant or dict with restaurants list
        if "name" in restaurants_data:
            name = restaurants_data.get("name", "Unknown Restaurant")
            matched = restaurants_data.get("matched_ingredients", [])
            return f"- {name}\n  Interested in: {', '.join(matched) if matched else 'Various ingredients'}"
        else:
            # Check for nested restaurant data
            for key in ["restaurants", "top", "results", "items", "matches"]:
                if key in restaurants_data and isinstance(restaurants_data[key], list):
                    restaurants_data = restaurants_data[key]
                    break

    if isinstance(restaurants_data, list):
        formatted = []
        for restaurant in restaurants_data[:3]:  # Limit to top 3
            if isinstance(restaurant, dict):
                name = restaurant.get("name", "Unknown Restaurant")
                matched = restaurant.get("matched_ingredients", [])
                formatted.append(f"- {name}")
                formatted.append(f"  Interested in: {', '.join(matched) if matched else 'Various ingredients'}")
        return "\n".join(formatted)

    return str(restaurants_data)


def _format_donation_center(donation_data: Any) -> str:
    """Format donation center data for the prompt."""
    if not donation_data:
        return "Local Soup Kitchen"

    if isinstance(donation_data, dict):
        return donation_data.get("name", "Local Soup Kitchen")

    return str(donation_data)


def _parse_llm_response(response: str) -> Tuple[List[str], List[str], List[str], str, str, str]:
    """
    Parse the LLM response to extract the three action lists and targets.
    Returns: (sell_items, cook_items, donate_items, restaurant_name, recipe_title, donation_center)
    """
    sell_items, cook_items, donate_items = [], [], []
    restaurant_name, recipe_title, donation_center = "", "", ""

    lines = response.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse selling line
        if line.lower().startswith('selling to'):
            # Extract restaurant name and items
            match = re.match(r'selling to ([^:]+):\s*\[(.*?)\]', line, re.IGNORECASE)
            if match:
                restaurant_name = match.group(1).strip()
                items_str = match.group(2).strip()
                if items_str:
                    sell_items = [item.strip() for item in items_str.split(',') if item.strip()]

        # Parse cooking line
        elif line.lower().startswith('cook recipe'):
            match = re.match(r'cook recipe ([^:]+):\s*\[(.*?)\]', line, re.IGNORECASE)
            if match:
                recipe_title = match.group(1).strip()
                items_str = match.group(2).strip()
                if items_str:
                    cook_items = [item.strip() for item in items_str.split(',') if item.strip()]

        # Parse donation line
        elif line.lower().startswith('donate to'):
            match = re.match(r'donate to ([^:]+):\s*\[(.*?)\]', line, re.IGNORECASE)
            if match:
                donation_center = match.group(1).strip()
                items_str = match.group(2).strip()
                if items_str:
                    donate_items = [item.strip() for item in items_str.split(',') if item.strip()]

    return sell_items, cook_items, donate_items, restaurant_name, recipe_title, donation_center


def _create_decision_output(sell_items: List[str], cook_items: List[str], donate_items: List[str],
                            restaurant_name: str, recipe_title: str, donation_center: str) -> List[Dict[str, Any]]:
    """Create the structured decision output."""
    decisions = []

    # Add COOK decisions
    for item in cook_items:
        decisions.append({
            "item": item,
            "action": "COOK",
            "reason": f"Used in selected recipe '{recipe_title}'",
            "target_recipes": [recipe_title],
        })

    # Add SELL decisions
    for item in sell_items:
        decisions.append({
            "item": item,
            "action": "SELL",
            "reason": f"Market demand and good shelf life for {restaurant_name}",
            "target_restaurants": [restaurant_name],
        })

    # Add DONATE decisions
    for item in donate_items:
        decisions.append({
            "item": item,
            "action": "DONATE",
            "reason": "Better suited for community benefit",
            "donation_center": donation_center,
        })

    return decisions


# ── Public API ───────────────────────────────────────────────────────────────
def decide_actions() -> List[Dict[str, Any]]:


    expiring_data = _load_json(EXPIRING_FILE)
    recipes_data = _load_json(BEST_RECIPES_FILE)
    restaurants_data = _load_json(TOP_RESTAURANTS_FILE)
    donation_data = _load_json(DONATION_FILE)

    if not expiring_data:
        raise FileNotFoundError(f"Missing or invalid {EXPIRING_FILE}")
    if not recipes_data:
        raise FileNotFoundError(f"Missing or invalid {BEST_RECIPES_FILE}")

    # --- 2. Load prompt template ---
    prompt_template = _load_prompt_template()

    # --- 3. Format data for the prompt ---

    formatted_expiring = _format_expiring_ingredients(expiring_data)
    formatted_recipes = _format_recipes(recipes_data)
    formatted_restaurants = _format_restaurants(restaurants_data)
    formatted_donation = _format_donation_center(donation_data)

    # --- 4. Create the complete prompt ---
    full_prompt = f"""{prompt_template}

Expiring ingredients:
{formatted_expiring}

Retrieved recipes:
{formatted_recipes}

Top restaurants:
{formatted_restaurants}

Donation centre:
{formatted_donation}

Please provide your decision in the exact format specified above."""

    # --- 5. Send to LLM ---
    print("Sending request to LLM...")

    try:
        chat = LLMChat()
        # ✅ Pass LangChain-native message object
        raw_response = chat.send_messages([HumanMessage(content=full_prompt)])
        print("Received response from LLM")
    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        raise

    # --- 6. Parse LLM response ---
    sell_items, cook_items, donate_items, restaurant_name, recipe_title, donation_center = _parse_llm_response(
        raw_response)

    # --- 7. Create structured output ---
    decisions = _create_decision_output(sell_items, cook_items, donate_items,
                                        restaurant_name, recipe_title, donation_center)

    # --- 8. Save to file ---
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)

    # --- 9. Display summary ---
    print("\n--- Final Decision Summary ---")
    print(f"COOK   ({len(cook_items)} items): ", ", ".join(cook_items) if cook_items else "none")
    if recipe_title:
        print(f"  ↳ Recipe: {recipe_title}")

    print(f"SELL   ({len(sell_items)} items): ", ", ".join(sell_items) if sell_items else "none")
    if restaurant_name:
        print(f"  ↳ Restaurant: {restaurant_name}")

    print(f"DONATE ({len(donate_items)} items): ", ", ".join(donate_items) if donate_items else "none")
    if donation_center:
        print(f"  ↳ Donation Center: {donation_center}")
    print("----------------------------")

    print(f"\nLLM Raw Response:\n{raw_response}")

    return decisions


def main():
    """Main execution function when script is run directly."""
    try:
        decide_actions()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
