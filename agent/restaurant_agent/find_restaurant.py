import json
import os
from agent.chat_and_embedding import LLMChat
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
RESTAURANT_PROMPT_FILE = os.path.join(PROMPT_DIR, "prompts/restaurant_scoring_prompt.txt")
TOP_RESTAURANTS_FILE = os.path.join(RESULTS_DIR, "top_restaurants.json")
EXPIRING_INGREDIENTS_FILE = os.path.join(DATA_DIR, "expiring_ingredients.json")
RESTAURANTS_FILE = os.path.join(DATA_DIR, "nearby_locations.json")


# --- Load prompt from file ---
def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --- Init LLM ---
chat_model = LLMChat(temp=0)

# Load data
with open(EXPIRING_INGREDIENTS_FILE) as f:
    expiring = json.load(f)

with open(RESTAURANTS_FILE) as f:
    locations = json.load(f)
    restaurants = locations["restaurant_buyers"]

# Format input strings
ingredient_json_str = json.dumps(expiring, indent=2)
restaurant_list = "\n".join([
    f"- {r['name']}: {r['cuisine']} (distance: {r['distance_km']} km)"
    for r in restaurants
])

# Build messages
system_prompt_text = load_system_prompt(RESTAURANT_PROMPT_FILE)
system_template = PromptTemplate(template=system_prompt_text, input_variables=[], template_format="jinja2")
system_message = SystemMessagePromptTemplate(prompt=system_template)

human_message = HumanMessagePromptTemplate.from_template(
    "Here is the list of ingredients (in JSON format):\n{ingredients}\n\nHere are nearby restaurants:\n{restaurants}"
)

prompt_template = ChatPromptTemplate.from_messages([
    system_message,
    human_message
])

messages = prompt_template.format_messages(
    ingredients=ingredient_json_str,
    restaurants=restaurant_list
)

# Send to GPT
raw_response = chat_model.send_messages(messages)

# Parse GPT response
try:
    print("\U0001f4be Raw response from LLM:\n", raw_response)
    cleaned_response = chat_model.extract_json_string(raw_response)
    restaurant_scores = json.loads(cleaned_response)
except Exception as e:
    print(f"\u26a0\ufe0f Failed to parse GPT output: {str(e)}")
    restaurant_scores = []

# Display results
restaurant_scores.sort(key=lambda x: x.get("relative_rank", 99))
print("\n\U0001f3ea Suggested restaurants to offer ingredients (ranked):")
for r in restaurant_scores:
    print(f"{r['relative_rank']}. {r['name']} (Score: {r['absolute_score']:.1f})")
    print(f"    \U0001f9c2 Matched ingredients: {', '.join(r.get('matched_ingredients', []))}")
    print(f"    \ud83d\udca1 Reason: {r['reason']}")

# Save results
restaurants_for_saving = []
for scored in restaurant_scores:
    name = scored["name"]
    original = next((r for r in restaurants if r["name"] == name), {})
    restaurant_entry = {
        "name": name,
        "absolute_score": round(scored.get("absolute_score", 0), 2),
        "relative_rank": scored.get("relative_rank"),
        "matched_ingredients": scored.get("matched_ingredients", []),
        "reason": scored.get("reason", ""),
        "original_data": original
    }
    restaurants_for_saving.append(restaurant_entry)

with open(TOP_RESTAURANTS_FILE, "w", encoding="utf-8") as f:
    json.dump(restaurants_for_saving, f, indent=2, ensure_ascii=False)

print(f"\n\U0001f4be Saved ranked restaurants to: {TOP_RESTAURANTS_FILE}")


# TODO: use embedding for restaurants in order to save tokens
#  (if we have a lot of restaurants we might want to filter the relevant first)
