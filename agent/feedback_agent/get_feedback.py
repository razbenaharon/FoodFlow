import os
import json
import shutil
from datetime import datetime
from utils.chat_and_embedding import LLMChat
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate

# The number of past runs (data points) required to trigger the feedback agent.
LEN_FOR_FEEDBACK = 10

def run_feedback_agent():
    # --- Paths ---
    # Determine the project root directory by moving up three levels from the current file's location.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Define the data and results directories.
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    # Define the file paths for the input history, prompt template, and output results.
    HISTORY_FILE = os.path.join(DATA_DIR, "recent_expiring_ingredients.json")
    FEEDBACK_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_agent_prompt.txt")
    FEEDBACK_RESULT_FILE = os.path.join(RESULTS_DIR, "feedback_suggestions.json")

    # --- Load expired ingredients history ---
    # Check if the history file exists. If not, print a message and exit.
    if not os.path.exists(HISTORY_FILE):
        print("❌ No expired ingredient history found.")
        return

    # Read the historical data of expiring ingredients from the JSON file.
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    # Check if there is enough historical data to run the feedback agent.
    if len(history) < LEN_FOR_FEEDBACK:
        print(f"ℹ️ Only {len(history)} runs found. Need {LEN_FOR_FEEDBACK} to activate feedback agent.")
        return

    # --- Flatten ingredient data ---
    # Create a single, flat list of all ingredient entries from the historical runs.
    flat = []
    for run in history:
        for ing in run:
            flat.append(ing)
    # Convert the flattened list back into a JSON string for use in the prompt.
    ingredient_json_str = json.dumps(flat, indent=2, ensure_ascii=False)

    # --- Load system prompt ---
    # Read the system prompt template from a text file.
    with open(FEEDBACK_PROMPT_FILE, "r", encoding="utf-8") as f:
        system_prompt_text = f.read()

    # --- Build prompt ---
    # Create a SystemMessagePromptTemplate using the loaded prompt text and a variable for the number of runs.
    system_template = PromptTemplate(template=system_prompt_text, input_variables=["n_runs"], template_format="jinja2")
    system_message = SystemMessagePromptTemplate(prompt=system_template)

    # Create a HumanMessagePromptTemplate with a placeholder for the ingredient history.
    human_message = HumanMessagePromptTemplate.from_template(
        "Here is the history of expiring ingredients from the past {n_runs} runs (in JSON format):\n{ingredients}"
    )

    # Combine the system and human messages into a single chat prompt template.
    prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])

    # Format the prompt with the flattened ingredient data and the number of runs.
    messages = prompt_template.format_messages(
        ingredients=ingredient_json_str,
        n_runs=LEN_FOR_FEEDBACK)

    # --- Init LLM ---
    # Initialize the LLM chat model with a temperature of 0 for deterministic output.
    chat_model = LLMChat(temp=0)

    # --- Send to LLM ---
    # Send the formatted messages to the LLM and get the raw response.
    raw_response = chat_model.send_messages(messages)

    try:
        # Attempt to extract and parse the JSON string from the LLM's raw response.
        cleaned_response = chat_model.extract_json_string(raw_response)
        feedback = json.loads(cleaned_response)
    except Exception as e:
        # If parsing fails, print a warning and set feedback to an empty list.
        print(f"⚠️ Failed to parse GPT output: {str(e)}")
        feedback = []

    # --- Display results ---
    # Print the feedback suggestions to the console in a user-friendly format.
    print("\n📣 Feedback Suggestions:")
    for item in feedback:
        print(f"- {item['item']}: reduce by ~{item['suggested_reduction']} {item.get('unit', '')}")
        print(f"  Reason: {item['reason']}")

    # --- Save results ---
    # Ensure the results directory exists, then save the parsed feedback to a JSON file.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(FEEDBACK_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2, ensure_ascii=False)

    # --- Backup and reset history ---
    # Generate a timestamp for the backup file.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(DATA_DIR, f"recent_expiring_ingredients_backup_{timestamp}.json")
    # Copy the current history file to a backup file.
    shutil.copy(HISTORY_FILE, backup_path)
    # Overwrite the history file with an empty list to reset it for future runs.
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)

# Entry point of the script.
if __name__ == "__main__":
    run_feedback_agent()