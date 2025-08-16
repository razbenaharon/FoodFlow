# agent/execution_agent/send_recipie_to_kitchen.py
from __future__ import annotations

import json
import ast
import re
import unicodedata
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT   = Path(__file__).resolve().parents[2]  # repo root if file in execution_agent/
RESULTS_DIR    = PROJECT_ROOT / "results"
RECIPES_DIR    = RESULTS_DIR / "recipes"
RETRIEVED_DIR  = RECIPES_DIR / "retrieved"
MESSAGES_DIR   = RESULTS_DIR / "messages"             # where we save the kitchen message

DECISION_FILE  = RESULTS_DIR / "final_decision_output.json"

# --------------------------- helpers ---------------------------

def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (_strip_accents(s).strip().lower()))

def _simplify(s: str) -> str:
    # lower, strip accents, remove punctuation except spaces
    s = _norm(s)
    return re.sub(r"[^a-z0-9 ]+", "", s)

def _tokenize(s: str) -> List[str]:
    # drop tiny stopwords; keep meaningful tokens
    stop = {"with", "and", "the", "a", "an", "of", "in", "to", "on", "for"}
    toks = [t for t in _simplify(s).split() if t and t not in stop]
    return toks

def _token_overlap_score(a: str, b: str) -> float:
    # Sørensen–Dice coefficient on token sets
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return (2.0 * inter) / (len(ta) + len(tb))

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _selected_recipe_title_from_decisions(path: Path) -> str:
    """
    Decisions contain many per-ingredient actions; the selected recipe name
    is on COOK items under 'target_recipes': ['Recipe Title'].
    We collect and dedupe; expect exactly one unique title (by design).
    """
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list.")
    titles = []
    for row in data:
        if isinstance(row, dict) and row.get("action") == "COOK":
            trg = row.get("target_recipes") or []
            if isinstance(trg, list) and trg and isinstance(trg[0], str):
                titles.append(trg[0].strip())
    if not titles:
        raise RuntimeError("No COOK actions found in final decisions — cannot determine selected recipe.")
    # de-dup but keep order
    out, seen = [], set()
    for t in titles:
        k = _norm(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    if len(out) > 1:
        print(f"⚠️ Multiple COOK target recipes found, using the first: {out[0]}")
    return out[0]

def _safe_parse_ingredient_list(value: Any) -> List[str]:
    """
    Handles various shapes:
      - list[str]
      - stringified list: "['a','b']"
      - free text string: "a; b; c"
    """
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        # try to parse "['a', 'b']"
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        # fall back: split by commas or semicolons
        parts = re.split(r"[;,]\s*", s)
        return [p for p in (x.strip() for x in parts) if p]
    return []

def _unify_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize fields across different shapes:
      title / Title
      ingredients / Ingredients / Cleaned_Ingredients (string or list)
      instructions / Instructions
    """
    # title
    title = payload.get("title") or payload.get("Title") or payload.get("Name") or ""
    title = str(title).strip()

    # ingredients
    ing = (
        payload.get("ingredients")
        or payload.get("Ingredients")
        or payload.get("Cleaned_Ingredients")
        or []
    )
    ingredients = _safe_parse_ingredient_list(ing)

    # instructions
    instr = payload.get("instructions") or payload.get("Instructions") or ""
    instructions = str(instr).strip()

    return {"title": title, "ingredients": ingredients, "instructions": instructions}

def _load_all_retrieved() -> List[Tuple[float, Dict[str, Any], Path]]:
    """
    Read all retrieved_dish*.json files and return list of (score, normalized_payload, path).
    Each file is expected to contain a list of objects with 'score' and 'payload'.
    """
    if not RETRIEVED_DIR.exists():
        return []
    items: List[Tuple[float, Dict[str, Any], Path]] = []
    for fp in sorted(RETRIEVED_DIR.glob("retrieved_dish*.json")):
        try:
            data = _load_json(fp)
        except Exception:
            continue
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                score = float(entry.get("score", 0.0))
                payload = entry.get("payload") or {}
                if isinstance(payload, dict):
                    norm = _unify_payload(payload)
                    if norm.get("title"):
                        items.append((score, norm, fp))
    return items

def _find_recipe_in_retrieved(
    selected_title: str,
    candidates: List[Tuple[float, Dict[str, Any], Path]],
    fuzzy_threshold: float = 0.15
) -> Optional[Tuple[float, Dict[str, Any], Path]]:
    """
    1) Exact (case-insensitive) title match; highest score wins.
    2) Relaxed contains-match after punctuation stripping.
    3) Fuzzy token-overlap (Dice); pick best >= threshold.
    """
    sel_n = _norm(selected_title)
    exact = [t for t in candidates if _norm(t[1].get("title", "")) == sel_n]
    if exact:
        return sorted(exact, key=lambda x: x[0], reverse=True)[0]

    # relaxed contains
    sel_s = _simplify(selected_title)
    relaxed = []
    for tup in candidates:
        cand_s = _simplify(tup[1].get("title", ""))
        if sel_s in cand_s or cand_s in sel_s:
            relaxed.append(tup)
    if relaxed:
        return sorted(relaxed, key=lambda x: x[0], reverse=True)[0]

    # fuzzy by token overlap
    scored: List[Tuple[float, Tuple[float, Dict[str, Any], Path]]] = []
    for tup in candidates:
        cand_title = tup[1].get("title", "")
        sim = _token_overlap_score(selected_title, cand_title)
        scored.append((sim, tup))
    scored.sort(key=lambda z: z[0], reverse=True)

    if scored and scored[0][0] >= fuzzy_threshold:
        return scored[0][1]

    # not found
    # print top suggestions to help debugging
    top = [f"{z[1][1].get('title','')} (sim={z[0]:.2f})" for z in scored[:5]]
    print("🔍 No direct match. Top similar retrieved titles:")
    for t in top:
        print("   •", t)
    return None

def _compose_kitchen_message(title: str, ingredients: List[str], instructions: str) -> str:
    """Build the text that will be saved to results/messages/message_to_kitchen.txt."""
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
        for ln in instructions.splitlines():
            ln = ln.strip()
            if ln:
                lines.append(f" {ln}")
    else:
        lines.append("Instructions: (not available)")
    lines.append("============================")
    return "\n".join(lines)

def _print_recipe(selected_title: str, payload: Dict[str, Any]) -> None:
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
            if ln:
                print(f"  {ln}")
    else:
        print("🧾  Instructions: (not available)")
    print("===================================================\n")


# ----------------------------- main -----------------------------

def main() -> None:
    # 1) get the selected recipe title from decision output
    if not DECISION_FILE.exists():
        raise FileNotFoundError(f"Missing decisions file: {DECISION_FILE}")
    selected_title = _selected_recipe_title_from_decisions(DECISION_FILE)

    # 2) load all retrieved dish files
    candidates = _load_all_retrieved()
    if not candidates:
        raise FileNotFoundError(f"No retrieved dish files found in: {RETRIEVED_DIR}")

    # 3) find the matching retrieved recipe
    hit = _find_recipe_in_retrieved(selected_title, candidates)
    if not hit:
        raise RuntimeError(f"Could not find retrieved dish matching: {selected_title}")

    score, norm_payload, source_path = hit

    # 4) print it
    _print_recipe(selected_title, norm_payload)

    # 5) save a stable-named message file for the kitchen
    MESSAGES_DIR.mkdir(parents=True, exist_ok=True)
    kitchen_msg = _compose_kitchen_message(
        title=norm_payload.get("title") or selected_title,
        ingredients=norm_payload.get("ingredients") or [],
        instructions=norm_payload.get("instructions") or ""
    )
    out_path = MESSAGES_DIR / "message_to_kitchen.txt"   # ← fixed filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(kitchen_msg)

if __name__ == "__main__":
    main()

