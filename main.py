from pathlib import Path
from agent.decision_agent.final_decision_agent import decide_actions
from agent.restaurant_agent.find_restaurant import run_find_restaurant
from agent.soup_kitchen_finder.find_soup_kitchen import find_best_open_soup_kitchen
from agent.recipe_agent.recipe_agent import run_find_recipes
from utils.prepare_inventory import prepare_inventory
from agent.execution_agent.send_recipie_to_kitchen import main as send_recipe_to_kitchen
from agent.execution_agent.send_message import send_message
from utils.token_logger import print_estimated_cost_header
from agent.feedback_agent.get_feedback import run_feedback_agent

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
EXPIRED_HISTORY_FILE      = DATA_DIR / "recent_expiring_ingredients.json"


def main():
    print_estimated_cost_header()

    print("\n===========================================================================\n")
    print("👋 Welcome to FoodFlow!")
    print("You are now simulating operations for 'HaSalon' restaurant in Tel Aviv, owned by chef Eyal Shani.")
    print("Our goal is to make smart, sustainable decisions about what to cook, sell, or donate "
          "among expiring ingredients.\n")

    print("For each expiring ingredient, the agent will determine the most suitable action:")
    print("- 🍽️ Cook: A message will be sent to the chef with today’s special based on this ingredient.")
    print("- 🍴 Sell: A message will be sent to a nearby restaurant most likely to purchase it.")
    print("- ❤️ Donate: A pickup will be scheduled with a nearby donation center.")

    print("\n=========================================="
          "=================================\n")
    input("Press Enter to see the magic happen...")

    print("\n===========================================================================\n")

    prepare_inventory()

    print("\n===========================================================================\n")

    print("🥣 choosing soup kitchen for today...")
    find_best_open_soup_kitchen()
    # input("\nPress Enter to continue...")
    print("\n===========================================================================\n")


    print("🧑‍🍳 Running recipe recommendation agent...")
    run_find_recipes()
    # input("\nPress Enter to continue...")
    print("\n===========================================================================\n")

    print("🍽️ Running restaurant recommendation agent...")
    run_find_restaurant()
    # input("\nPress Enter to continue...")
    print("\n===========================================================================\n")

    print("🧠 Running decision agent...")
    decide_actions()
    # input("\nPress Enter to continue...")
    print("\n===========================================================================\n")

    print("🧑‍🍳 Sending selected recipe to kitchen...")
    send_recipe_to_kitchen()
    # input("\nPress Enter to continue...")
    print("\n===========================================================================\n")

    print("✉️ Preparing outbound messages...")
    send_message()
    # input("\nPress Enter to continue...")
    print("\n===========================================================================\n")

    print("📊 Checking if feedback agent should run -- asking for feedback only each 10 iterations of the agent...")
    run_feedback_agent()

    print("\n✅ Simulation complete.")


if __name__ == "__main__":
    main()

