from agent.chat_and_embedding import LLMChat
import json
import os
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate


def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_ingredient_level_prompt_with_template(ingredients, recipes, restaurants, donation_centers, decision_prompt_file_path):
    # For each ingredient, gather info string
    ingredient_blocks = []
    for ing in ingredients:
        name = ing["item"]
        qty = ing["quantity"]
        unit = ing["unit"]
        expires = ing["expires_in_days"]

        relevant_recipes = [
            {
                "title": r["title"],
                "score": r["absolute_score"],
                "rank": r["relative_rank"],
                "reason": r.get("reason", "No reason provided")
            }
            for r in recipes
            if name in r.get("used_expiring_ingredients", [])
        ]

        recipe_str = "\n".join([
            f"- {r['title']} (score: {r['score']}, rank: {r['rank']})\n  💡 {r['reason']}"
            for r in relevant_recipes
        ]) or "None"

        matching_restaurants = [
            f"- {res['name']} (score: {res['absolute_score']}, distance: {res['original_data']['distance_km']} km)"
            for res in restaurants
            if name in res.get("matched_ingredients", [])
        ] or ["None"]

        donation_str = "\n".join([
            f"- {d['name']} (Distance: {d['distance_km']} km)"
            for d in sorted(donation_centers, key=lambda d: d['distance_km'])
        ])

        block = (
            f"🫒 Ingredient: {name}\n"
            f"Quantity: {qty} {unit}\n"
            f"Expires in: {expires} days\n\n"
            f"🍳 Appears in Recipes:\n{recipe_str}\n\n"
            f"🏪 Restaurants interested in this ingredient:\n" + "\n".join(matching_restaurants) + "\n\n"
            f"❤️ Nearby Donation Centers:\n{donation_str}"
        )

        ingredient_blocks.append(block)

    joined_blocks = "\n---\n".join(ingredient_blocks)

    # Load instructions from external prompt file
    system_prompt = load_system_prompt(decision_prompt_file_path)

    system_template = PromptTemplate(
        template=system_prompt,
        input_variables=[],  # אין משתנים למילוי בתוך הטמפלט
        template_format="jinja2"
    )

    system_message = SystemMessagePromptTemplate(prompt=system_template)


    prompt_template = ChatPromptTemplate.from_messages([
        system_message,
        HumanMessagePromptTemplate.from_template("Ingredients to evaluate:\n{ingredient_blocks}")
    ])

    # Format messages with input
    messages = prompt_template.format_messages(ingredient_blocks=joined_blocks)

    return messages


def decide_action_per_ingredient(ingredients, recipes, restaurants, donation_centers, decision_prompt_file_path):
    chat = LLMChat(temp=0)
    messages = build_ingredient_level_prompt_with_template(ingredients, recipes, restaurants, donation_centers, decision_prompt_file_path)
    print("\n📤 Prompt sent to LLM:\n")
    for m in messages:
        print(f"[{m.type.upper()}]\n{m.content}\n")
    response = chat.send_messages(messages)
    print("\n🤖 LLM Response:\n")
    print(response)
    return response


# --- Example usage with dummy data ---
if __name__ == "__main__":
    # --- Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    PROMPT_FILE = os.path.join(PROMPT_DIR, "ingredient_decision_prompt.txt")
    TOP_RECIPIES_FILE = os.path.join(RESULTS_DIR, "top_recipies.json")
    TOP_RESTAURANTS_FILE = os.path.join(RESULTS_DIR, "top_restaurants.json")
    LOCATIONS_FILE = os.path.join(DATA_DIR, "nearby_locations.json")
    INGREDIENTS_FILE = os.path.join(DATA_DIR, "expiring_ingredients.json")

    with open(TOP_RECIPIES_FILE, "r") as f:
        recipes = json.load(f)
    with open(TOP_RESTAURANTS_FILE, "r") as f:
        restaurants = json.load(f)
    with open(LOCATIONS_FILE, "r") as f:
        locations = json.load(f)
        donation_centers = locations['donation_targets']
    with open(INGREDIENTS_FILE, "r") as f:
        ingredients = json.load(f)

    response = decide_action_per_ingredient(ingredients, recipes, restaurants, donation_centers, PROMPT_FILE)

# TODO: add more data to examine the reasoning
# TODO: currently, the decision is local - examine each product by it's own -- need to add global decision part
