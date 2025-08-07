"""
Final Decision Agent – Dish-Level Prompt
---------------------------------------

Builds a single chat prompt that includes:

• For each dish (dish1-dish3):
  - Expiring vs. inventory ingredient lists
  - Top retrieved recipe (title, score, *full* instructions)

• Full contents of top_restaurants.json (so the LLM can decide whom to sell to)

• One donation centre (best_soup_kitchen.json)

The LLM must answer with **EXACTLY** three lines:

  selling to <restaurant-name> – [comma-separated ingredients]
  cook recipe <recipe-title> – [comma-separated ingredients]
  donate to <donation-centre-name> – [comma-separated ingredients]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Any

from chat_and_embedding import LLMChat
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)

# ─────────────────────────── Paths ────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
RESULTS_DIR   = PROJECT_ROOT / "results"

RECIPES_DIR   = RESULTS_DIR / "recipes"
ANALYSIS_DIR  = RECIPES_DIR / "analyses"          # dish<i>_ingredient_analysis.json
RETRIEVED_DIR = RECIPES_DIR / "retrieved"         # retrieved_dish<i>.json

ANALYSIS_FILES  = sorted(ANALYSIS_DIR.glob("dish*_ingredient_analysis.json"))
RETRIEVED_FILES = sorted(RETRIEVED_DIR.glob("retrieved_dish*.json"))

TOP_RESTAURANTS_FILE = RESULTS_DIR / "top_restaurants.json"
DONATION_FILE        = RESULTS_DIR / "best_soup_kitchen.json"

# ────────────────────────── Helpers ───────────────────────────────
def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────── Prompt pieces ────────────────────────────
SYSTEM_PROMPT = """\
You are a smart decision agent for a restaurant.

Task:
Decide what to do with **each EXPIRING ingredient** given:
 • Three dishes’ ingredient breakdowns (expiring vs. inventory).
 • One matched recipe for each dish.
 • A list of restaurants you could sell to.
 • One soup-kitchen donation centre.

Return **exactly three lines** (no extra text):

  selling to <restaurant-name> – [comma-separated ingredients]
  cook recipe <recipe-title> – [comma-separated ingredients]
  donate to <donation-centre-name> – [comma-separated ingredients]

Rules:
 • An ingredient may appear in only ONE line.
 • Choose “cook” only if the ingredient appears in the matched recipe.
 • Choose “sell” if another restaurant would likely buy it.
 • Otherwise choose “donate”.
"""

# ------------------------------------------------------------------#
#  Dish blocks (include FULL instructions)                          #
# ------------------------------------------------------------------#
def _dish_blocks() -> List[str]:
    blocks: List[str] = []

    for analysis_path, recipe_path in zip(ANALYSIS_FILES, RETRIEVED_FILES):
        dish     = analysis_path.stem.replace("_ingredient_analysis", "")
        analysis = _load_json(analysis_path)
        recipe   = _load_json(recipe_path)[0]  # first (and only) hit

        lines = [f"🧂 Ingredients for **{dish}**:"]
        lines.append("  • Expiring:")
        for ing in analysis.get("expiring", []):
            name = ing["name"] if isinstance(ing, dict) else ing
            lines.append(f"    - {name}")
        lines.append("  • Inventory:")
        for ing in analysis.get("inventory", []):
            name = ing["name"] if isinstance(ing, dict) else ing
            lines.append(f"    - {name}")

        title         = recipe["payload"].get("title", "")
        score         = recipe.get("score", 0.0)
        instructions  = recipe["payload"].get("instructions", "").strip()
        indented_instr = "      " + instructions.replace("\n", "\n      ")

        lines.append("\n🍲 Retrieved recipe:")
        lines.append(f"    - {title} (score {score:.2f})")
        lines.append("      Instructions:\n" + indented_instr)

        blocks.append("\n".join(lines))

    return blocks

# ------------------------------------------------------------------#
#  Restaurants block                                                #
# ------------------------------------------------------------------#
def _restaurants_block() -> str:
    top_restos = _load_json(TOP_RESTAURANTS_FILE)
    json_text  = json.dumps(top_restos, indent=2, ensure_ascii=False)
    return "🏪 Top restaurants you could sell to (file contents):\n" + json_text

# ------------------------------------------------------------------#
#  Donation block                                                   #
# ------------------------------------------------------------------#
def _donation_block() -> str:
    d = _load_json(DONATION_FILE)
    return (
        "❤️ Donation centre candidate:\n"
        f"  - {d['name']} ({d.get('distance_km', d.get('distance', 'n/a'))} km), "
        f"{d.get('address', '')}"
    )

# ------------------------------------------------------------------#
#  Build full prompt                                                #
# ------------------------------------------------------------------#
def build_prompt_messages():
    body = (
        "\n\n====================\n\n".join(_dish_blocks())
        + "\n\n====================\n\n"
        + _restaurants_block()
        + "\n\n====================\n\n"
        + _donation_block()
    )

    system_msg = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=[],
            template_format="jinja2"
        )
    )
    human_msg  = HumanMessagePromptTemplate.from_template("{payload}")
    chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
    return chat_prompt.format_messages(payload=body)

# ------------------------------------------------------------------#
#  Driver – always runs full decision (no dry-run)                  #
# ------------------------------------------------------------------#
def decide_actions():
    messages   = build_prompt_messages()
    chat_model = LLMChat(temp=0.3)

    raw_response = chat_model.send_messages(messages)

    try:
        cleaned = chat_model.extract_json_string(raw_response)
        decision_result = json.loads(cleaned)
    except Exception:
        decision_result = raw_response.strip().splitlines()

    print("\n🤖 Final decision result:")
    if isinstance(decision_result, list):
        for line in decision_result:
            print("-", line)
    else:
        print(decision_result)

    output_path = RESULTS_DIR / "final_decision_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(decision_result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved result to: {output_path}")


