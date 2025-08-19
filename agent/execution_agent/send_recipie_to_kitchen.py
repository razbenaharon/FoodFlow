# agent/execution_agent/send_recipie_to_kitchen.py
"""
This module identifies, retrieves, and formats a selected recipe to be "sent to the kitchen."

It orchestrates the final step of the recipe-selection process by:
- Reading a decision file to determine the chosen recipe's title.
- Searching through a directory of retrieved recipe data for the best match.
- Normalizing the matched recipe's data (title, ingredients, and instructions).
- Formatting the recipe details into a human-readable text message.

The module outputs the final formatted recipe message to a text file in the `results/messages/` directory.
"""
from __future__ import annotations

import json
import ast
import re
import unicodedata
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Constants for Directory and File Paths ---


# Traverses up two levels to get the project root directory.
PROJECT_ROOT   = Path(__file__).resolve().parents[2]  # repo root if file in execution_agent/

# Directory to store all output results.
RESULTS_DIR    = PROJECT_ROOT / "results"

# Subdirectory within results for recipe-related files.
RECIPES_DIR    = RESULTS_DIR / "recipes"

# Subdirectory where raw retrieved recipe files (JSONs) are stored.
RETRIEVED_DIR  = RECIPES_DIR / "retrieved"

# Directory where the final formatted message for the "kitchen" is saved.
MESSAGES_DIR   = RESULTS_DIR / "messages"             # where we save the kitchen message

# The final decision file that specifies which recipe to cook.
DECISION_FILE  = RESULTS_DIR / "final_decision_output.json"

# --------------------------- helpers ---------------------------

def _strip_accents(s: str) -> str:
    """Removes accents from a string by normalizing to NFKD form."""
    # Example: "é" becomes "e"
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")

def _norm(s: str) -> str:
    """Normalizes a string by stripping accents, leading/trailing whitespace, and lowering case."""
    return re.sub(r"\s+", " ", (_strip_accents(s).strip().lower()))

def _simplify(s: str) -> str:
    """
    Simplifies a string for matching by normalizing it and removing all non-alphanumeric characters.
    """
    # lower, strip accents, remove punctuation except spaces
    s = _norm(s)
    # Keep only letters, numbers, and spaces
    return re.sub(r"[^a-z0-9 ]+", "", s)

def _tokenize(s: str) -> List[str]:
    """
    Tokenizes a simplified string into a list of meaningful words, excluding common stopwords.
    """
    # Define a set of common, low-information words to exclude.
    stop = {"with", "and", "the", "a", "an", "of", "in", "to", "on", "for"}
    # Simplify the string, split it into words, and filter out stopwords.
    toks = [t for t in _simplify(s).split() if t and t not in stop]
    return toks

def _token_overlap_score(a: str, b: str) -> float:
    """
    Calculates the similarity between two strings based on the Sørensen–Dice coefficient
    applied to their token sets.
    """
    # Convert strings into sets of tokens for efficient comparison.
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    # If either string has no meaningful tokens, similarity is 0.
    if not ta or not tb:
        return 0.0
    # Calculate the size of the intersection of the two token sets.
    inter = len(ta & tb)
    # Apply the Sørensen–Dice formula: 2 * |A ∩ B| / (|A| + |B|)
    return (2.0 * inter) / (len(ta) + len(tb))

def _load_json(path: Path) -> Any:
    """Loads and parses a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _selected_recipe_title_from_decisions(path: Path) -> str:
    """
    Extracts the selected recipe title from the final decision JSON file.

    The decision file contains a list of actions for various ingredients. This function
    scans for "COOK" actions and extracts the 'target_recipes' value, which is
    expected to be the chosen recipe's title. It handles cases where the same
    title appears multiple times and raises errors if no COOK action is found
    or if the file format is incorrect.
    """
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list.")

    titles = []
    # Iterate through each decision entry to find COOK actions.
    for row in data:
        if isinstance(row, dict) and row.get("action") == "COOK":
            trg = row.get("target_recipes") or []
            # The recipe title is expected to be the first element in the 'target_recipes' list.
            if isinstance(trg, list) and trg and isinstance(trg[0], str):
                titles.append(trg[0].strip())

    if not titles:
        raise RuntimeError("No COOK actions found in final decisions — cannot determine selected recipe.")

    # De-duplicate the found titles while preserving order.
    # By design, all COOK actions should point to the same recipe.
    out, seen = [], set()
    for t in titles:
        k = _norm(t) # Use normalized form for de-duplication
        if k and k not in seen:
            seen.add(k)
            out.append(t)

    # If multiple different recipes are found, warn the user and proceed with the first one.
    if len(out) > 1:
        print(f"⚠️ Multiple COOK target recipes found, using the first: {out[0]}")

    return out[0]

def _safe_parse_ingredient_list(value: Any) -> List[str]:
    """
    Safely parses an ingredient list that could be a list, a stringified list,
    or a simple string with delimiters.

    Handles various shapes:
      - A standard list of strings: ['item1', 'item2']
      - A string that looks like a list: "['item1', 'item2']"
      - A string with comma or semicolon delimiters: "item1, item2; item3"
    """
    if isinstance(value, list):
        # If it's already a list, just ensure all elements are clean strings.
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        s = value.strip()
        # Case 1: Try to parse as a stringified Python list (e.g., "['a', 'b']").
        if s.startswith("[") and s.endswith("]"):
            try:
                # ast.literal_eval is a safe way to evaluate a string containing a Python literal.
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                # If parsing fails, fall through to the next method.
                pass
        # Case 2: Fallback for simple delimited strings (e.g., "a, b; c").
        parts = re.split(r"[;,]\s*", s)
        return [p for p in (x.strip() for x in parts) if p]

    # If the input is neither a list nor a string, return an empty list.
    return []

def _unify_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a recipe dictionary to a consistent format.

    Handles various possible key names for title, ingredients, and instructions
    (e.g., "title" vs "Title", "ingredients" vs "Cleaned_Ingredients") and
    ensures the ingredients are a list of strings.
    """
    # Normalize title: Look for "title", "Title", or "Name", and default to empty string.
    title = payload.get("title") or payload.get("Title") or payload.get("Name") or ""
    title = str(title).strip()

    # Normalize ingredients: Look for various keys and use the safe parser.
    ing = (
        payload.get("ingredients")
        or payload.get("Ingredients")
        or payload.get("Cleaned_Ingredients")
        or []
    )
    ingredients = _safe_parse_ingredient_list(ing)

    # Normalize instructions: Look for "instructions" or "Instructions".
    instr = payload.get("instructions") or payload.get("Instructions") or ""
    instructions = str(instr).strip()

    return {"title": title, "ingredients": ingredients, "instructions": instructions}

def _load_all_retrieved() -> List[Tuple[float, Dict[str, Any], Path]]:
    """
    Reads all `retrieved_dish*.json` files from the RETRIEVED_DIR, normalizes their
    content, and returns a list of tuples containing the score, the normalized
    recipe data, and the source file path.
    """
    if not RETRIEVED_DIR.exists():
        return []

    items: List[Tuple[float, Dict[str, Any], Path]] = []
    # Iterate over all JSON files matching the pattern "retrieved_dish*.json".
    for fp in sorted(RETRIEVED_DIR.glob("retrieved_dish*.json")):
        try:
            data = _load_json(fp)
        except Exception:
            # Skip files that fail to load or parse.
            continue

        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                # Extract score and payload, providing defaults.
                score = float(entry.get("score", 0.0))
                payload = entry.get("payload") or {}
                if isinstance(payload, dict):
                    # Unify the payload to a standard format.
                    norm = _unify_payload(payload)
                    # Only include recipes that have a title.
                    if norm.get("title"):
                        items.append((score, norm, fp))
    return items

def _find_recipe_in_retrieved(
    selected_title: str,
    candidates: List[Tuple[float, Dict[str, Any], Path]],
    fuzzy_threshold: float = 0.15
) -> Optional[Tuple[float, Dict[str, Any], Path]]:
    """
    Finds the best matching recipe from a list of candidates based on the selected title.

    It uses a tiered matching strategy:
    1.  Exact Match: Looks for a case-insensitive, normalized title match. The candidate
        with the highest retrieval score wins.
    2.  Relaxed Contains Match: Checks if one simplified title string contains the other.
        Again, the highest score wins.
    3.  Fuzzy Token Match: If no match is found, it calculates a token-based
        similarity (Sørensen–Dice) score. The candidate with the highest similarity
        is chosen, provided it meets the `fuzzy_threshold`.
    """
    # --- 1. Exact match ---
    sel_n = _norm(selected_title)
    exact = [t for t in candidates if _norm(t[1].get("title", "")) == sel_n]
    if exact:
        # If multiple exact matches exist, return the one with the highest original score.
        return sorted(exact, key=lambda x: x[0], reverse=True)[0]

    # --- 2. Relaxed "contains" match ---
    sel_s = _simplify(selected_title)
    relaxed = []
    for tup in candidates:
        cand_s = _simplify(tup[1].get("title", ""))
        # Check for substring relationship in both directions.
        if sel_s in cand_s or cand_s in sel_s:
            relaxed.append(tup)
    if relaxed:
        # Return the best-scored relaxed match.
        return sorted(relaxed, key=lambda x: x[0], reverse=True)[0]

    # --- 3. Fuzzy match by token overlap ---
    scored: List[Tuple[float, Tuple[float, Dict[str, Any], Path]]] = []
    for tup in candidates:
        cand_title = tup[1].get("title", "")
        # Calculate the similarity score between the selected title and the candidate title.
        sim = _token_overlap_score(selected_title, cand_title)
        scored.append((sim, tup))
    # Sort candidates by their new similarity score in descending order.
    scored.sort(key=lambda z: z[0], reverse=True)

    # Return the top result if its similarity is above the threshold.
    if scored and scored[0][0] >= fuzzy_threshold:
        return scored[0][1]

    # --- Not Found ---
    # If no suitable match is found, print top suggestions to help with debugging.
    top = [f"{z[1][1].get('title','')} (sim={z[0]:.2f})" for z in scored[:5]]
    print("🔍 No direct match. Top similar retrieved titles:")
    for t in top:
        print("   •", t)
    return None

def _compose_kitchen_message(title: str, ingredients: List[str], instructions: str) -> str:
    """Builds the final, formatted text message for the kitchen."""
    lines = []
    lines.append("===== KITCHEN DISPATCH =====")
    lines.append(f"Recipe: {title}")
    lines.append("----------------------------")

    if ingredients:
        lines.append("Ingredients:")
        for it in ingredients:
            lines.append(f" - {it}")
    else:
        lines.append("Ingredients: (not available)")

    lines.append("----------------------------")

    if instructions:
        lines.append("Instructions:")
        # Split instructions into lines and format them neatly.
        for ln in instructions.splitlines():
            ln = ln.strip()
            if ln:
                lines.append(f" {ln}")
    else:
        lines.append("Instructions: (not available)")

    lines.append("============================")
    return "\n".join(lines)

def _print_recipe(selected_title: str, payload: Dict[str, Any]) -> None:
    """Prints the final recipe details to the console in a user-friendly format."""
    title = payload.get("title") or selected_title
    ingredients = payload.get("ingredients") or []
    instructions = payload.get("instructions") or ""

    print("\n================= SEND TO KITCHEN =================")
    print(f"👨‍🍳  Recipe: {title}")
    print("---------------------------------------------------")
    if ingredients:
        print("🧺  Ingredients:")
        for item in ingredients:
            print(f"  • {item}")
    else:
        print("🧺  Ingredients: (not available)")
    print("---------------------------------------------------")
    if instructions:
        print("🧾  Instructions:")
        for line in instructions.splitlines():
            ln = line.strip()
            # Print non-empty lines of instructions.
            if ln:
                print(f"  {ln}")
    else:
        print("🧾  Instructions: (not available)")
    print("===================================================\n")


# ----------------------------- main -----------------------------

def main() -> None:
    """
    Main execution function. Orchestrates the process of identifying, finding,
    and formatting the selected recipe to be "sent to the kitchen".
    """
    # 1. Get the selected recipe title from the final decision output JSON.
    if not DECISION_FILE.exists():
        raise FileNotFoundError(f"Missing decisions file: {DECISION_FILE}")
    selected_title = _selected_recipe_title_from_decisions(DECISION_FILE)

    # 2. Load all retrieved dish JSON files into a list of candidate recipes.
    candidates = _load_all_retrieved()
    if not candidates:
        raise FileNotFoundError(f"No retrieved dish files found in: {RETRIEVED_DIR}")

    # 3. Find the best matching recipe from the candidates using the selected title.
    hit = _find_recipe_in_retrieved(selected_title, candidates)
    if not hit:
        raise RuntimeError(f"Could not find retrieved dish matching: {selected_title}")

    # Unpack the found recipe data.
    score, norm_payload, source_path = hit

    # 4. Print the formatted recipe to the console for visibility.
    _print_recipe(selected_title, norm_payload)

    # 5. Save the formatted recipe to a text file for external systems (the "kitchen").
    MESSAGES_DIR.mkdir(parents=True, exist_ok=True)
    kitchen_msg = _compose_kitchen_message(
        title=norm_payload.get("title") or selected_title,
        ingredients=norm_payload.get("ingredients") or [],
        instructions=norm_payload.get("instructions") or ""
    )
    # Write to a file with a fixed, predictable name.
    out_path = MESSAGES_DIR / "message_to_kitchen.txt"   # ← fixed filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(kitchen_msg)

if __name__ == "__main__":
    # Standard entry point for executing the script directly.
    main()