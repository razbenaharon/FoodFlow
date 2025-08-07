import json
import random
from pathlib import Path

from agent.decision_agent.final_decision_agent import decide_actions
from agent.restaurant_agent.find_restaurant import run_find_restaurant
from agent.soup_kitchen_finder.find_soup_kitchen import find_best_open_soup_kitchen
from agent.recipie_agent.recipe_agent import run_find_recipes
from prepare_inventory import prepare_inventory
from agent.decision_agent import final_decision_agent


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
    print("👋 Welcome to FoodFlow!")
    print("You are now simulating operations for 'HaSalon' restaurant in Tel Aviv, owned by chef Eyal Shani.")
    print("Our goal is to make smart decisions about what to cook, sell, or donate.\n")

    # Ensure results directory exists (data dir is created in prepare_inventory)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


    prepare_inventory()
    input("\nPress Enter to continue...")


    print("🍽️ Running recipe recommendation agent...")
    run_find_recipes()
    input("\nPress Enter to continue...")


    print("🥣 Running soup_kitchen recommendation agent...")
    find_best_open_soup_kitchen()
    input("\nPress Enter to continue...")


    print("🍽️ Running restaurant recommendation agent...")
    run_find_restaurant()
    input("\nPress Enter to continue...")

    print("🧠 Running decision agent...")
    decide_actions()
    input("\nPress Enter to continue...")




if __name__ == "__main__":
    main()
