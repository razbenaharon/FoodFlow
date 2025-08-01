import json
import os
from agent.chat_and_embedding import LLMChat, LLMEmbed
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
RECIPE_PROMPT_FILE = os.path.join(PROMPT_DIR, "recipe_scoring_prompt.txt")
TOP_RECIPIES_FILE = os.path.join(RESULTS_DIR, "top_recipies.json")
EXPIRING_INGREDIENTS_FILE = os.path.join(DATA_DIR, "expiring_ingredients.json")
CURRENT_INVENTORY_FILE = os.path.join(DATA_DIR, "current_inventory.json")


# --- Load prompt from file ---
def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --- Load data ---
with open(EXPIRING_INGREDIENTS_FILE) as f:
    expiring = json.load(f)

with open(CURRENT_INVENTORY_FILE) as f:
    current = json.load(f)

# --- Init models ---
chat_model = LLMChat(temp=0)
embed_model = LLMEmbed()

# --- Ingredient sets ---
expiring_items = {item['item'] for item in expiring}
available_items = {item['item'] for item in current}

# --- Embed query for semantic search ---
query_str = (
    "I want to cook something using ingredients that are about to expire. "
    "Here is what I have:\n" +
    "\n".join([
        f"- {item['item']} ({item['quantity']}{item['unit']}, expires in {item['expires_in_days']} days)"
        for item in expiring
    ])
)
query_vector = embed_model.embed_query(query_str)

# --- Search for recipe candidates ---
recipe_results = embed_model.return_query_results(
    query_vector=query_vector,
    collection_name='recipes_mock',
    limit_res=10
)


# --- Prepare recipes summary for LLM ---
def summarize_recipe_ingredients(recipe_hit, available_items, expiring_items):
    ingredients_raw = recipe_hit.payload.get("ingredients", [])
    summarized = []
    for ing in ingredients_raw:
        name = ing['item']
        summarized.append({
            "item": name,
            "in_inventory": name in available_items,
            "is_expiring": name in expiring_items
        })
    return summarized


all_recipes_summary = []
for hit in recipe_results.points:
    recipe_title = hit.payload.get("title")
    ingredients = summarize_recipe_ingredients(hit, available_items, expiring_items)
    all_recipes_summary.append({
        "title": recipe_title,
        "semantic_score": round(hit.score, 4),
        "ingredients": ingredients
    })

# --- Build prompt with SystemMessage and HumanMessage ---
system_prompt_text = load_system_prompt(RECIPE_PROMPT_FILE)
system_template = PromptTemplate(template=system_prompt_text, input_variables=[], template_format="jinja2")
system_message = SystemMessagePromptTemplate(prompt=system_template)

user_input = json.dumps({"recipes": all_recipes_summary}, indent=2)
human_message = HumanMessagePromptTemplate.from_template("Here is the input:\n{input}")

prompt_template = ChatPromptTemplate.from_messages([
    system_message,
    human_message
])

messages = prompt_template.format_messages(input=user_input)

# --- Send to LLM ---
raw_response = chat_model.send_messages(messages)

# --- Parse response and match to results ---
try:
    print("\U0001f4be Raw response from LLM:\n", raw_response)
    cleaned_response = chat_model.extract_json_string(raw_response)
    scored_recipes = json.loads(cleaned_response)

    recipes_with_scores = []
    for hit in recipe_results.points:
        title = hit.payload.get("title")
        match = next((r for r in scored_recipes if r['title'] == title), None)
        if match:
            recipes_with_scores.append({
                "title": title,
                "absolute_score": match.get("absolute_score", 0.0),
                "relative_rank": match.get("relative_rank", None),
                "reason": match.get("reason", ""),
                "used_expiring_ingredients": match.get("used_expiring_ingredients", []),
                "payload": hit.payload
            })
except Exception as e:
    print(f"\u26a0\ufe0f Failed to parse LLM output: {str(e)}")
    recipes_with_scores = []

# --- Print final output ---
print("\n\U0001f372 Final suggested recipes (with absolute & relative scores):")
for recipe in sorted(recipes_with_scores, key=lambda x: x['relative_rank']):
    print(f"\U0001f37d️ {recipe['title']}")
    print(f"    \U0001f9ee Absolute score: {recipe['absolute_score']:.1f}")
    print(f"    \U0001f3c5 Relative rank: {recipe['relative_rank']}")
    print(f"    \U0001f9c2 Uses: {', '.join(recipe.get('used_expiring_ingredients', []))}")
    print(f"    \ud83d\udca1 Reason: {recipe['reason']}")

# --- Save output ---
recipes_for_saving = []
for r in recipes_with_scores:
    recipe_entry = {
        "title": r["title"],
        "absolute_score": round(r["absolute_score"], 2),
        "relative_rank": r["relative_rank"],
        "reason": r["reason"],
        "used_expiring_ingredients": r.get("used_expiring_ingredients", []),
        "payload": r["payload"]
    }
    recipes_for_saving.append(recipe_entry)

with open(TOP_RECIPIES_FILE, "w", encoding="utf-8") as f:
    json.dump(recipes_for_saving, f, indent=2, ensure_ascii=False)

print(f"\n\U0001f4be Saved ranked recipes to: {TOP_RECIPIES_FILE}")
