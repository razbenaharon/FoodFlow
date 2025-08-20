import json
import csv
import os
from utils.chat_and_embedding import LLMChat
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
import random


def run_find_restaurant():
    # --- Utility ---
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # --- Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    RESTAURANT_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "restaurant_agent_prompt.txt")
    TOP_RESTAURANTS_FILE = os.path.join(RESULTS_DIR, "top_restaurants.json")
    EXPIRING_INGREDIENTS_FILE = os.path.join(DATA_DIR, "expiring_ingredients.json")
    RESTAURANTS_CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nearby_restaurants.csv")

    # --- Load system prompt ---
    def load_system_prompt(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # --- Init LLM ---
    chat_model = LLMChat(temp=0)

    # --- Load data ---
    with open(EXPIRING_INGREDIENTS_FILE) as f:
        expiring = json.load(f)

    # Load restaurants from CSV
    restaurants = []
    with open(RESTAURANTS_CSV_FILE, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            restaurants.append({
                "name": row["name"],
                "cuisine": row.get("types", "unknown"),
                "distance_km": safe_float(row.get("distance_from_ha_salon_km"))
            })

    # Limit token load
    # Step 1: Randomly sample up to 50 restaurants
    sampled_restaurants = random.sample(restaurants, min(50, len(restaurants)))

    # Step 2: Sort the sampled list by distance, take the closest 20
    restaurants = sorted(sampled_restaurants, key=lambda r: r["distance_km"])[:20]

    # --- Format input strings ---
    ingredient_json_str = json.dumps(expiring, indent=2)
    restaurant_list = "\n".join([
        f"- {r['name']}: {r['cuisine']} (distance: {r['distance_km']} km)"
        for r in restaurants
    ])

    # --- Build messages ---
    system_prompt_text = load_system_prompt(RESTAURANT_PROMPT_FILE)
    system_template = PromptTemplate(template=system_prompt_text, input_variables=[], template_format="jinja2")
    system_message = SystemMessagePromptTemplate(prompt=system_template)

    human_message = HumanMessagePromptTemplate.from_template(
        "Here is the list of ingredients (in JSON format):\n{ingredients}\n\nHere are nearby restaurants:\n{restaurants}"
    )

    prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])

    messages = prompt_template.format_messages(
        ingredients=ingredient_json_str,
        restaurants=restaurant_list
    )

    # --- Send to LLM ---
    raw_response = chat_model.send_messages(messages)

    # --- Parse LLM output ---
    try:
        cleaned_response = chat_model.extract_json_string(raw_response)
        restaurant_scores = json.loads(cleaned_response)
    except Exception as e:
        print(f"⚠️ Failed to parse GPT output: {str(e)}")
        restaurant_scores = []

    # --- Display results ---
    print("\nBest matching restaurants:")
    for idx, r in enumerate(restaurant_scores, 1):
        print(f"{idx}. {r['name']}")
        print(f"   Matched ingredients: {', '.join(r.get('matched_ingredients', []))}")
        print(f"   Reason: {r['reason']}")

    # --- Save results ---
    with open(TOP_RESTAURANTS_FILE, "w", encoding="utf-8") as f:
        json.dump(restaurant_scores, f, indent=2, ensure_ascii=False)

