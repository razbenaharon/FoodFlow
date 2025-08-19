import json, os
from typing import Any, Dict, List, Optional
from agent.recipe_agent.utilities.parsing import no_en_dashes
from agent.recipe_agent.utilities.console import suppress_stdout_stderr
from utils.chat_and_embedding import LLMChat


def load_top_restaurant(results_dir: str) -> Optional[Dict[str, Any]]:
    """
    Loads and identifies the top-ranked restaurant from a JSON file.

    The function handles various JSON structures by searching for common keys
    like 'restaurants', 'top', or 'results'. It then iterates through the
    found candidates to find the one with the highest score, rank, or rating.
    """
    path = os.path.join(results_dir, "top_restaurants.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    candidates: List[Dict[str, Any]] = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        for key in ["restaurants", "top", "results", "items", "matches"]:
            if isinstance(data.get(key), list):
                candidates = data[key]
                break
        if not candidates:
            candidates = [data]

    best, best_score = None, None
    for r in candidates:
        if not isinstance(r, dict):
            continue
        sc = None
        for k in ["score", "rank", "rating", "confidence"]:
            v = r.get(k)
            try:
                sc = float(v)
                break
            except Exception:
                continue
        if best is None or (sc is not None and (best_score is None or sc > best_score)):
            best, best_score = r, sc
    return best


def restaurant_capsule(rest: Dict[str, Any]) -> str:
    """
    Creates a concise summary string (a "capsule") from a restaurant dictionary.

    It extracts relevant details like name, cuisine, city, tags, and specialties,
    then formats them into a single, semicolon-separated sentence.
    """
    if not isinstance(rest, dict):
        return ""
    name = rest.get("name") or rest.get("title") or rest.get("restaurant") or ""
    cuisine = rest.get("cuisine") or rest.get("style") or rest.get("category") or ""
    city = rest.get("city") or rest.get("location") or ""
    tags = rest.get("tags") or rest.get("keywords") or rest.get("attributes") or []
    specialties = rest.get("specialties") or rest.get("signature_dishes") or rest.get("popular_dishes") or []
    vibe = rest.get("vibe") or rest.get("concept") or ""

    def _flat(x):
        if isinstance(x, list):
            return ", ".join([str(t) for t in x if t])
        return str(x) if x else ""

    parts: List[str] = []
    if cuisine: parts.append(str(cuisine))
    if vibe:    parts.append(str(vibe))
    if specialties: parts.append(f"specialties: {_flat(specialties)}")
    if tags:        parts.append(f"tags: {_flat(tags)}")
    if city:        parts.append(str(city))

    capsule = "; ".join([p for p in parts if p])
    if name:
        capsule = f"{name}; " + capsule if capsule else name
    return no_en_dashes(capsule)


def tailor_reason_for_restaurant(
        dish_title: str,
        expiring_list: List[str],
        inventory_list: List[str],
        missing_list: List[str],
        restaurant_name: str,
        restaurant_capsule_text: str
) -> str:
    """
    Uses a large language model (LLM) to generate a concise, tailored reason
    explaining why a dish is suitable for a particular restaurant.

    The function formats a prompt with all the relevant context (dish, ingredients,
    restaurant info) and sends it to the LLM for a single-sentence response.
    """
    chat = LLMChat(temp=0.2)
    exp_str = ", ".join(expiring_list) if expiring_list else "none"
    inv_str = ", ".join(inventory_list) if inventory_list else "none"
    miss_str = ", ".join(missing_list) if missing_list else "none"
    prompt = (
        "You write one concise sentence that explains why a dish suits a specific restaurant.\n"
        f"Restaurant: {restaurant_name}\n"
        f"Context: {restaurant_capsule_text}\n"
        f"Dish: {dish_title}\n"
        f"Matched expiring ingredients: {exp_str}\n"
        f"Inventory ingredients: {inv_str}\n"
        f"Missing ingredients: {miss_str}\n"
        "Rules: reference one or two matched expiring items if present, tie to cuisine or concept, "
        "if missing list is not none you may mention they are simple to source, keep it under 22 words, "
        "no hyphens or en dashes, use commas.\n"
        "Answer with one sentence only."
    )
    with suppress_stdout_stderr():
        out = chat.send_prompt(prompt).strip().splitlines()[0]
    return no_en_dashes(out)