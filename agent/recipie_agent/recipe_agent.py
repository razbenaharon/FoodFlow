import os
import re
import json
import time
from chat_and_embedding import LLMChat, LLMEmbed
from agent.recipie_agent.json_handler import load_json
from agent.recipie_agent.rag_retriever import retrieve_similar_recipes
from agent.recipie_agent.recipe_query_builder import build_query_string


def run_find_recipes():
    # ------------------------------------------------------------------ #
    #  Base paths                                                        #
    # ------------------------------------------------------------------ #
    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    # ------------------------------------------------------------------ #
    #  Directory structure inside results/recipes                        #
    # ------------------------------------------------------------------ #
    RECIPES_DIR   = os.path.join(RESULTS_DIR, "recipes")   # results/recipes
    DISH_DIR      = os.path.join(RECIPES_DIR, "dishes")    # results/recipes/dishes
    RETRIEVED_DIR = os.path.join(RECIPES_DIR, "retrieved") # results/recipes/retrieved
    ANALYSIS_DIR  = os.path.join(RECIPES_DIR, "analyses")  # results/recipes/analyses

    for path in (RECIPES_DIR, DISH_DIR, RETRIEVED_DIR, ANALYSIS_DIR):
        os.makedirs(path, exist_ok=True)

    EXPIRING_INGREDIENTS_FILE = os.path.join(DATA_DIR, "expiring_ingredients.json")
    CURRENT_INVENTORY_FILE    = os.path.join(DATA_DIR, "full_inventory.json")
    QUERY_LOG_FILE            = os.path.join(RECIPES_DIR, "recipe_builder.txt")

    # ------------------------------------------------------------------ #
    #  Load expiring & inventory                                         #
    # ------------------------------------------------------------------ #
    expiring, _ = load_json(EXPIRING_INGREDIENTS_FILE, "expiring ingredients")
    current,  _ = load_json(CURRENT_INVENTORY_FILE,    "current inventory")

    # ------------------------------------------------------------------ #
    #  Build query & save                                                #
    # ------------------------------------------------------------------ #
    full_prompt = build_query_string(expiring, current)
    with open(QUERY_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(full_prompt)
    print(f"📝 Saved query to: {QUERY_LOG_FILE}")

    # ------------------------------------------------------------------ #
    #  Get three dishes from GPT                                         #
    # ------------------------------------------------------------------ #
    chat_model  = LLMChat(temp=0.7)
    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        response = chat_model.send_prompt(full_prompt)

        # split on: 1. **Dish Title**
        parts  = re.split(r"\n?\d\.\s+\*\*(.+?)\*\*", response.strip())
        dishes = [
            f"{parts[i].strip()}\n{parts[i + 1].strip() if i + 1 < len(parts) else ''}"
            for i in range(1, len(parts), 2)
        ]

        if len(dishes) == 3:
            break
        print(f"❌ Attempt {attempt + 1}: Only {len(dishes)} dishes found. Retrying…\n")
        time.sleep(2)
    else:
        raise ValueError(
            f"❌ Failed to get 3 dishes after {MAX_RETRIES} attempts.\n\nLast response:\n{response}"
        )

    print(f"\n🍽️ Parsed {len(dishes)} distinct dishes from GPT output.")

    # ------------------------------------------------------------------ #
    #  Helper to clean a dish for RAG                                    #
    # ------------------------------------------------------------------ #
    def clean_dish_for_rag_format(raw_dish: str) -> str:
        lines = raw_dish.strip().splitlines()
        if not lines:
            return ""
        title   = lines[0].strip()
        content = "\n".join(lines[1:])
        content = re.sub(r"\*\*.*?\*\*", "", content)  # bold
        content = re.sub(r"[-•*]\s*", "", content)     # bullets
        content = re.sub(r"---+", "", content)         # hr
        content_lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        return "\n".join([title] + content_lines)

    # ------------------------------------------------------------------ #
    #  Save dish<i>.txt                                                  #
    # ------------------------------------------------------------------ #
    for idx, dish in enumerate(dishes, 1):
        cleaned   = clean_dish_for_rag_format(dish)
        file_path = os.path.join(DISH_DIR, f"dish{idx}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"✅ Saved cleaned RAG-formatted dish {idx} to {file_path}")

    # ------------------------------------------------------------------ #
    #  Retrieve similar recipes (1 per dish)                             #
    # ------------------------------------------------------------------ #
    embed_model = LLMEmbed()

    for i, dish in enumerate(dishes, 1):
        print(f"\n🔍 Retrieving recipe similar to dish {i}…")
        save_path = os.path.join(RETRIEVED_DIR, f"retrieved_dish{i}.json")
        retrieve_similar_recipes(
            query_str=dish,
            embedder=embed_model,
            collection_name="food_recipes",
            result_limit=1,
            save_path=save_path
        )

    # ------------------------------------------------------------------ #
    #  Ingredient classification per dish                                #
    # ------------------------------------------------------------------ #
    CLASSIFICATION_PROMPT_FILE = os.path.join(
        PROJECT_ROOT, "agent", "recipie_agent", "prompts",
        "ingredient_classification_prompt.txt"
    )
    with open(CLASSIFICATION_PROMPT_FILE, "r", encoding="utf-8") as f:
        classification_prompt_template = f.read()

    expiring_names  = [d.get("item") or d.get("name") for d in expiring  if isinstance(d, dict)]
    inventory_names = [d.get("item") or d.get("name") for d in current   if isinstance(d, dict)]

    def extract_json_block(text):
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in response.")
        return match.group(0)

    for idx, dish in enumerate(dishes, 1):
        print(f"\n🔎 Classifying ingredients for dish {idx}…")

        filled_prompt = (
            classification_prompt_template
            .replace("<DISH_TEXT>", dish)
            .replace("<EXPIRING_LIST>",  ", ".join(expiring_names))
            .replace("<INVENTORY_LIST>", ", ".join(inventory_names))
        )

        response = chat_model.send_prompt(filled_prompt)

        try:
            parsed = json.loads(extract_json_block(response))
        except Exception as e:
            print(f"⚠️ Failed to parse JSON for dish {idx}, error: {e}\nFull response:\n{response}")
            continue

        analysis_path = os.path.join(ANALYSIS_DIR, f"dish{idx}_ingredient_analysis.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved ingredient analysis to {analysis_path}")

    print("\n🎉 run_find_recipes completed.")
