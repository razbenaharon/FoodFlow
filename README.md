# FoodFlow – Developer Guide & Runbook

This repository simulates day‑to‑day operations for a restaurant (HaSalon, Tel Aviv), helping decide **what to cook, sell, or donate** based on inventory and expiring items. It stitches together several agents:
- **Inventory prep** → generates `current_inventory.json` and `expiring_ingredients.json`
- **Soup kitchen finder** → finds the best open donation center
- **Recipe agent (RAG)** → retrieves relevant recipes
- **Restaurant recommender** → suggests nearby restaurants to sell surplus to
- **Decision agent** → decides per ingredient (COOK / SELL / DONATE) and selects a dish
- **Execution agents** → send recipe to kitchen + outbound messages
- **Feedback agent** → optionally collects downstream feedback
- **Token logger** → tracks token usage and prints an estimated cost banner

> This guide explains **how to run**, **what each file does**, and **where data lives**.

---

## Quickstart

1) **Install dependencies**
```bash
pip install -r requirements.txt
```

2) **Set environment variables** (for Azure OpenAI). Either export them or use a `.env` file:
```
API_KEY=...
```

3) **Run the pipeline**
```bash
python main.py
```
You’ll see step‑by‑step progress with pauses between phases (press Enter to continue).

---

## File Map

```
.
├─ main.py                      # Orchestrates the whole simulation
├─ chat_and_embedding.py        # Azure OpenAI chat + embeddings helpers, Qdrant client
├─ config.py                    # Centralized configuration (API keys, endpoints)
├─ prepare_inventory.py         # Builds current & expiring inventory, logs history
├─ token_logger.py              # Token usage recorder + cost estimator banner
├─ requirements.txt             # Python deps
└─ data/                        # (created/used at runtime) JSON inventory, history
    ├─ full_inventory.json
    ├─ current_inventory.json
    ├─ expiring_ingredients.json
    └─ recent_expiring_ingredients.json
```

> Other agents (restaurant finder, soup kitchen, recipes, decision, execution, feedback) are imported from your project’s `agent/` and `utils/` packages. Ensure your PYTHONPATH / package structure matches those imports.

---

## Runtime Artifacts

- `data/current_inventory.json` – full inventory **minus** the randomly sampled expiring items
- `data/expiring_ingredients.json` – 10 random expiring items with reduced quantities and `days_to_expire`
- `data/recent_expiring_ingredients.json` – rolling history list of recent runs (append‑only)
- `results/top_recipes.json`, `results/top_restaurants.json` – produced by recipe / restaurant agents
- `tokens_count/chat_tokens.txt`, `tokens_count/embedding_tokens.txt` – accumulated token logs

---

## Configuration

Edit `config.py` (or, preferably, rely on environment variables) to set:
- **Azure OpenAI**: `API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_ENDPOINT`, `CHAT_DEPLOYMENT_NAME`, `EMBEDDING_DEPLOYMENT_NAME`
- **Qdrant**: `QDRANT_URL`, `QDRANT_API_KEY`

> ⚠ **Security note**: Avoid committing real keys to version control. Use environment variables or a `.env` file excluded by `.gitignore`.

---

## How the Pipeline Flows

1. `prepare_inventory()`
   - Loads `full_inventory.json`
   - Randomly selects up to 10 expiring items (excluding blacklist: salt, water, sea salt, lemon)
   - Decreases their quantities and assigns `days_to_expire ∈ [1..4]`
   - Writes both `current_inventory.json` and `expiring_ingredients.json`
   - Appends the expiring batch to `recent_expiring_ingredients.json`

2. `find_best_open_soup_kitchen()`
   - Chooses donation destination for the donation path

3. `run_find_recipes()`
   - Retrieves top candidate recipes using embeddings / RAG

4. `run_find_restaurant()`
   - Finds nearby restaurants most likely to buy the surplus

5. `decide_actions()`
   - Consumes all above outputs and produces actionable decisions (COOK / SELL / DONATE)

6. `send_recipe_to_kitchen()` and `send_message()`
   - Dispatches the final recipe + messages (kitchen, soup kitchen, restaurant)

7. `run_feedback_agent()`
   - Optional feedback harvesting (based on your agent’s logic)

Throughout the run, `token_logger.py` records token usage and `main.py` prints a **cost banner** up front.

---

## Troubleshooting

- **Imports / paths**: If your project uses `utils.*` or `agent.*` packages, ensure your folder structure matches (or adjust imports).
- **Long‑running Qdrant ops**: Add client `timeout=` in `chat_and_embedding.py` (the sample already uses `timeout=30`).
- **Token logs empty**: Logs accumulate under `tokens_count/`. They’re created at first write.
- **Costs look off**: The banner uses a simple split (50/50 input/output). Tweak `estimate_cost()` if your pattern differs.

---

## Contributing Notes

- Prefer **English comments** and docstrings.
- Keep functions small and single‑purpose.
- When adding an agent, expose a single `run_*` function and print concise progress updates.
- Avoid interactive pauses in production (replace `input()` with a CLI flag or config).

Happy shipping 🚀
