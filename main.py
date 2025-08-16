from pathlib import Path
from agent.decision_agent.final_decision_agent import decide_actions
from agent.restaurant_agent.find_restaurant import run_find_restaurant
from agent.soup_kitchen_finder.find_soup_kitchen import find_best_open_soup_kitchen
from agent.recipie_agent.recipe_agent import run_find_recipes
from utils.prepare_inventory import prepare_inventory
from agent.execution_agent.send_recipie_to_kitchen import main as send_recipe_to_kitchen
from agent.execution_agent.send_message import send_message
from utils.token_logger import print_estimated_cost_header


# === Define Paths ===
PROJECT_ROOT              = Path(__file__).resolve().parent
DATA_DIR                  = PROJECT_ROOT / "data"
RESULTS_DIR               = PROJECT_ROOT / "results"
PROMPT_DIR                = PROJECT_ROOT / "agent" / "decision_agent" / "prompts"

FULL_INVENTORY_FILE       = DATA_DIR / "full_inventory.json"
CURRENT_INVENTORY_FILE    = DATA_DIR / "current_inventory.json"
EXPIRING_INGREDIENTS_FILE = DATA_DIR / "expiring_ingredients.json"
NEARBY_LOCATIONS_FILE     = DATA_DIR / "nearby_restaurants.csv"
TOP_RECIPES_FILE          = RESULTS_DIR / "top_recipes.json"
TOP_RESTAURANTS_FILE      = RESULTS_DIR / "top_restaurants.json"
DECISION_PROMPT_FILE      = PROMPT_DIR / "ingredient_decision_prompt.txt"



def main():
    print_estimated_cost_header()

    print("\n===========================================================================\n")

    print("👋 Welcome to FoodFlow!")
    print("You are now simulating operations for 'HaSalon' restaurant in Tel Aviv, owned by chef Eyal Shani.")
    print("Our goal is to make smart decisions about what to cook, sell, or donate.\n")

    print("\n===========================================================================\n")



    prepare_inventory()
    input("\nPress Enter to continue...")
    print("\n===========================================================================\n")


    print("🥣 Running soup_kitchen recommendation agent...")
    find_best_open_soup_kitchen()
    input("\nPress Enter to continue...")
    print("\n===========================================================================\n")



    print("🧑‍🍳 Running recipe recommendation agent...")
    run_find_recipes()
    input("\nPress Enter to continue...")
    print("\n===========================================================================\n")


    print("🍽️ Running restaurant recommendation agent...")
    run_find_restaurant()
    input("\nPress Enter to continue...")
    print("\n===========================================================================\n")


    print("🧠 Running decision agent...")
    decide_actions()
    input("\nPress Enter to continue...")
    print("\n===========================================================================\n")

    print("🧑‍🍳 Sending selected recipe to kitchen...")
    send_recipe_to_kitchen()
    input("\nPress Enter to continue...")
    print("\n===========================================================================\n")


    print("✉️ Preparing outbound messages...")
    send_message()
    input("\nPress Enter to continue...")
    print("\n===========================================================================\n")



if __name__ == "__main__":
    main()

# TODO: phone number - write the same number for all prints (mock number)
