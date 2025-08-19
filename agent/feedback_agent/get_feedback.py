import os
import json
import shutil
from datetime import datetime
from utils.chat_and_embedding import LLMChat
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate

LEN_FOR_FEEDBACK = 10
def run_feedback_agent():
    # --- Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    HISTORY_FILE = os.path.join(DATA_DIR, "recent_expiring_ingredients.json")
    FEEDBACK_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_agent_prompt.txt")
    FEEDBACK_RESULT_FILE = os.path.join(RESULTS_DIR, "feedback_suggestions.json")

    # --- Load expired ingredients history ---
    if not os.path.exists(HISTORY_FILE):
        print("❌ No expired ingredient history found.")
        return

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    if len(history) < LEN_FOR_FEEDBACK:
        print(f"ℹ️ Only {len(history)} runs found. Need {LEN_FOR_FEEDBACK} to activate feedback agent.")
        return

    # --- Flatten ingredient data ---
    flat = []
    for run in history:
        for ing in run:
            flat.append(ing)
    ingredient_json_str = json.dumps(flat, indent=2, ensure_ascii=False)

    # --- Load system prompt ---
    with open(FEEDBACK_PROMPT_FILE, "r", encoding="utf-8") as f:
        system_prompt_text = f.read()

    # --- Build prompt ---
    system_template = PromptTemplate(template=system_prompt_text, input_variables=["n_runs"], template_format="jinja2")
    system_message = SystemMessagePromptTemplate(prompt=system_template)

    human_message = HumanMessagePromptTemplate.from_template(
        "Here is the history of expiring ingredients from the past {n_runs} runs (in JSON format):\n{ingredients}"
    )

    prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])

    messages = prompt_template.format_messages(
        ingredients=ingredient_json_str,
        n_runs=LEN_FOR_FEEDBACK)

    # --- Init LLM ---
    chat_model = LLMChat(temp=0)

    # --- Send to LLM ---
    raw_response = chat_model.send_messages(messages)

    try:
        cleaned_response = chat_model.extract_json_string(raw_response)
        feedback = json.loads(cleaned_response)
    except Exception as e:
        print(f"⚠️ Failed to parse GPT output: {str(e)}")
        feedback = []

    # --- Display results ---
    print("\n📣 Feedback Suggestions:")
    for item in feedback:
        print(f"- {item['item']}: reduce by ~{item['suggested_reduction']} {item.get('unit', '')}")
        print(f"  Reason: {item['reason']}")

    # --- Save results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(FEEDBACK_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2, ensure_ascii=False)

    # --- Backup and reset history ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(DATA_DIR, f"recent_expiring_ingredients_backup_{timestamp}.json")
    shutil.copy(HISTORY_FILE, backup_path)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)
