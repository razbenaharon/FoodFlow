from typing import Dict, List, Any


def _name_qty_unit(item: Dict[str, Any]) -> str:
    """
    Formats a dictionary item with name, quantity, and unit into a concise string.

    Handles cases where quantity or unit might be missing.
    """
    name = item.get("item") or item.get("name") or ""
    qty = item.get("quantity")
    unit = item.get("unit")
    if qty is None and not unit:
        return f"- {name}".strip()
    if qty is None:
        return f"- {name}: {unit}".strip()
    if unit:
        return f"- {name}: {qty} {unit}".strip()
    return f"- {name}: {qty}".strip()


def build_query_string(expiring: List[Dict[str, Any]], inventory: List[Dict[str, Any]]) -> str:
    """
    Builds a deterministic prompt for an LLM to generate recipe suggestions.

    The prompt is designed to elicit exactly three dishes, with specific formatting
    including a bold title, steps, and a one-sentence reason, based on the provided
    expiring and inventory ingredients.
    """
    exp_lines = "\n".join(_name_qty_unit(x) for x in expiring if isinstance(x, dict))
    inv_lines = "\n".join(_name_qty_unit(x) for x in inventory if isinstance(x, dict))

    return f"""You are a creative chef and menu developer.

    You are working for **HaSalon**, a high-end restaurant in Tel Aviv known for Israeli Mediterranean cuisine,
    live fire cooking, bold produce-driven dishes, and the signature style of chef Eyal Shani.

    Use the provided ingredients to invent EXACTLY three dishes that HaSalon could serve tonight.
    Prefer dishes that leverage expiring items; you may assume common pantry items (oil, salt, pepper) are available.

    Rules:
    - Return EXACTLY three items, numbered 1–3.
    - First line of each item is the dish TITLE in bold like: **Title**
    - Then 3–8 short lines of how to make it (no long paragraphs).
    - Finish each item with a single line starting with: Reason: <one concise sentence>

    Ingredients that are about to expire:
    {exp_lines}

    Available inventory:
    {inv_lines}

    Return format example:
    1. **Dish Title**
       step one
       step two
       step three
       Reason: one short sentence
    2. **Second Dish**
       ...
       Reason: one short sentence
    3. **Third Dish**
       ...
       Reason: one short sentence
    """.strip()
