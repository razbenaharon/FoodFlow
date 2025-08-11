import json
import os
from utils.chat_and_embedding import LLMChat, LLMEmbed
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

RESTAURANT_PROMPT_FILE = os.path.join(PROMPT_DIR, "restaurant_agent_prompt.txt")
TOP_RESTAURANTS_FILE = os.path.join(RESULTS_DIR, "top_restaurants.json")
EXPIRING_INGREDIENTS_FILE = os.path.join(DATA_DIR, "expiring_ingredients.json")


# --- Load system prompt ---
def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --- Load data ---
with open(EXPIRING_INGREDIENTS_FILE) as f:
    expiring = json.load(f)

# --- Init models ---
chat_model = LLMChat(temp=0)
embed_model = LLMEmbed()

# --- Prepare query for embedding ---
query_str = "I want to offer these ingredients to nearby restaurants:\n" + "\n".join([
    f"- {item['item']} ({item['quantity']}{item['unit']}, expires in {item['expires_in_days']} days)"
    for item in expiring
])
query_vector = embed_model.embed_query(query_str)

# --- Search Qdrant for relevant restaurants ---
restaurant_results = embed_model.return_query_results(
    query_vector=query_vector,
    collection_name='restaurants_mock',
    limit_res=10
)

# --- Summarize restaurant results ---
all_restaurant_summary = []
expiring_items = {item["item"] for item in expiring}

for hit in restaurant_results.points:
    restaurant_info = hit.payload
    name = restaurant_info.get("name")
    cuisine = restaurant_info.get("cuisine", "")
    distance = restaurant_info.get("distance_km", "?")
    summary = {
        "name": name,
        "cuisine": cuisine,
        "distance_km": distance,
        "expiring_ingredients": list(expiring_items)
    }
    all_restaurant_summary.append(summary)

# --- Build prompt for LLM ---
system_prompt_text = load_system_prompt(RESTAURANT_PROMPT_FILE)
system_template = PromptTemplate(template=system_prompt_text, input_variables=[], template_format="jinja2")
system_message = SystemMessagePromptTemplate(prompt=system_template)

user_input = json.dumps({"restaurants": all_restaurant_summary}, indent=2, ensure_ascii=False)
human_message = HumanMessagePromptTemplate.from_template("Here is the input:\n{input}")

prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])
messages = prompt_template.format_messages(input=user_input)

# --- Get LLM response ---
raw_response = chat_model.send_messages(messages)

# --- Parse and match LLM output ---
try:
    print("📥 Raw response from LLM:\n", raw_response)
    cleaned_response = chat_model.extract_json_string(raw_response)
    scored_restaurants = json.loads(cleaned_response)

    final_restaurants = []
    for hit in restaurant_results.points:
        name = hit.payload.get("name")
        match = next((r for r in scored_restaurants if r["name"] == name), None)
        if match:
            final_restaurants.append({
                "name": name,
                "absolute_score": match.get("absolute_score", 0.0),
                "relative_rank": match.get("relative_rank", None),
                "matched_ingredients": match.get("matched_ingredients", []),
                "reason": match.get("reason", ""),
                "payload": hit.payload
            })
except Exception as e:
    print(f"⚠️ Failed to parse LLM output: {str(e)}")
    final_restaurants = []

# --- Display results ---
print("\n🏪 Final suggested restaurants:")
for r in sorted(final_restaurants, key=lambda x: x['relative_rank']):
    print(f"{r['relative_rank']}. {r['name']} (Score: {r['absolute_score']:.1f})")
    print(f"   🧂 Matched: {', '.join(r['matched_ingredients'])}")
    print(f"   💡 Reason: {r['reason']}")

# --- Save to file ---
restaurants_for_saving = []
for r in final_restaurants:
    restaurant_entry = {
        "name": r["name"],
        "absolute_score": round(r["absolute_score"], 2),
        "relative_rank": r["relative_rank"],
        "matched_ingredients": r["matched_ingredients"],
        "reason": r["reason"],
        "original_data": r["payload"]
    }
    restaurants_for_saving.append(restaurant_entry)

with open(TOP_RESTAURANTS_FILE, "w", encoding="utf-8") as f:
    json.dump(restaurants_for_saving, f, indent=2, ensure_ascii=False)

print(f"\n💾 Saved ranked restaurants to: {TOP_RESTAURANTS_FILE}")
