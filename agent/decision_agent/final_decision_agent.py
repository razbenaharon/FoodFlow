# Final Decision Agent (single target recipe, human-ish randomness)
# -----------------------------------------------------------------
# Picks exactly ONE recipe using results/recipes/best_matching_recipes.json.
# Non deterministic:
#   • Selects the one recipe by sampling with a softmax over coverage and rank (temperature controlled).
#   • Leftover expiring ingredients are split between SELL and DONATE probabilistically.
#
# Guarantees:
#   • All ingredients in the chosen recipe's 'matched_expiring' that are truly expiring => COOK (target_recipes = [that one recipe])
#   • Each expiring ingredient has exactly one action (COOK or SELL or DONATE)
#   • Console prints cook, sell, donate lists exactly as requested.

from __future__ import annotations

import json
import math
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

# ── Paths
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
RESULTS_DIR    = PROJECT_ROOT / "results"
RECIPES_DIR    = RESULTS_DIR / "recipes"

BEST_RECIPES_FILE    = RECIPES_DIR / "best_matching_recipes.json"
EXPIRING_FILE        = PROJECT_ROOT / "data" / "expiring_ingredients.json"   # required to define the universe
TOP_RESTAURANTS_FILE = RESULTS_DIR / "top_restaurants.json"                  # optional
DONATION_FILE        = RESULTS_DIR / "best_soup_kitchen.json"                # optional

OUTPUT_FILE = RESULTS_DIR / "final_decision_output.json"

# ── Behavior knobs (tweak to taste or via env)
HUMAN_TEMPERATURE  = float(os.getenv("FOODFLOW_TEMPERATURE", "0.9"))  # higher => more random (0.3 greedy, 1.2 wilder)
SEED_ENV           = os.getenv("FOODFLOW_SEED")  # set to a number to make runs reproducible
LEFTOVER_SELL_PROB = float(os.getenv("FOODFLOW_SELL_PROB", "0.7"))    # with a SELL target available

# ── Random instance (optionally seeded)
_rng = random.Random()
if SEED_ENV is not None:
    try:
        _rng.seed(int(SEED_ENV))
    except Exception:
        _rng.seed(SEED_ENV)  # string seeds also ok

# ── Helpers
def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', s or "").encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', s.lower()).strip()

def _canonicalize_list(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for v in values:
        if not isinstance(v, str):
            continue
        key = _norm(v)
        if key and key not in seen:
            seen.add(key)
            out.append(v.strip())
    return out

def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _expiring_universe(best: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, str]]:
    """
    Build the expiring ingredient list and a norm->canonical map.
    FIX: only read from expiring_ingredients.json. Do NOT merge in recipe matched_expiring.
    """
    universe: List[str] = []
    if EXPIRING_FILE.exists():
        raw = _load_json(EXPIRING_FILE)
        for row in raw:
            if isinstance(row, dict):
                nm = row.get("item") or row.get("name")
            else:
                nm = str(row)
            if isinstance(nm, str) and nm.strip():
                universe.append(nm.strip())

    universe = _canonicalize_list(universe)
    canon_by_norm = {_norm(x): x for x in universe}
    return universe, canon_by_norm

# ── Scoring and softmax sampling of a SINGLE recipe
def _softmax_sample(weights: List[float], temperature: float, rng: random.Random) -> int:
    """Sample index using softmax over weights and temperature."""
    if temperature <= 0:
        return max(range(len(weights)), key=lambda i: weights[i])
    scaled = [w / temperature for w in weights]
    m = max(scaled) if scaled else 0.0
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps) or 1.0
    probs = [x / s for x in exps]
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1

def _select_one_recipe_humanish(best: List[Dict[str, Any]], target_norms: Set[str],
                                rng: random.Random, temperature: float) -> Optional[Dict[str, Any]]:
    """
    Score each recipe by:
      score = w_cov * coverage + w_rank * rank_norm + small_noise
    then sample one via softmax( score / T ).
    """
    if not best:
        return None

    coverages: List[int] = []
    ranks: List[int] = []
    for rec in best:
        cov = {_norm(x) for x in (rec.get("matched_expiring") or []) if isinstance(x, str)}
        coverages.append(len(target_norms & cov))
        ranks.append(int(rec.get("rank", 9999)))

    max_cov = max(coverages) if coverages else 1
    max_rank = max(ranks) if ranks else 1
    max_cov = max(1, max_cov)
    max_rank = max(1, max_rank)

    weights: List[float] = []
    for cov, rk in zip(coverages, ranks):
        cov_norm = cov / max_cov
        rank_norm = 1.0 - (rk - 1) / (max_rank - 1) if max_rank > 1 else 1.0
        w_cov = 0.75
        w_rank = 0.25
        base = w_cov * cov_norm + w_rank * rank_norm
        noise = rng.uniform(-0.05, 0.05)
        weights.append(base + noise)

    idx = _softmax_sample(weights, temperature=temperature, rng=rng)
    return best[idx] if 0 <= idx < len(best) else None

def _pick_sell_and_donate_targets() -> Tuple[Optional[str], Optional[str]]:
    sell_target = None
    if TOP_RESTAURANTS_FILE.exists():
        try:
            top = _load_json(TOP_RESTAURANTS_FILE)
            if isinstance(top, list) and top:
                sell_target = (top[0].get("name") or top[0].get("title") or "").strip() or None
            elif isinstance(top, dict):
                for key in ["restaurants", "top", "results", "items", "matches"]:
                    arr = top.get(key)
                    if isinstance(arr, list) and arr:
                        sell_target = (arr[0].get("name") or arr[0].get("title") or "").strip() or None
                        break
                if not sell_target:
                    sell_target = (top.get("name") or top.get("title") or "").strip() or None
        except Exception:
            sell_target = None

    donate_target = None
    if DONATION_FILE.exists():
        try:
            d = _load_json(DONATION_FILE)
            donate_target = (d.get("name") or "").strip() or None
        except Exception:
            donate_target = None

    return sell_target, donate_target

def _build_decisions_single_recipe_humanish(
    expiring_universe: List[str],
    canon_by_norm: Dict[str, str],
    selected: Optional[Dict[str, Any]],
    sell_target: Optional[str],
    donate_target: Optional[str],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Per ingredient decisions with a SINGLE selected recipe.
    - Ingredients present in selected['matched_expiring'] AND in expiring_universe => COOK
    - Others in expiring_universe => SELL or DONATE, chosen probabilistically
    """
    selected_title = (selected or {}).get("title", "").strip() if selected else ""
    selected_norms = {_norm(x) for x in ((selected or {}).get("matched_expiring") or []) if isinstance(x, str)}
    expiring_norms = {_norm(x) for x in expiring_universe}
    selected_norms &= expiring_norms  # safety: only cook items that are actually expiring

    decisions: List[Dict[str, Any]] = []
    for item in expiring_universe:
        k = _norm(item)
        if selected and k in selected_norms:
            decisions.append({
                "item": canon_by_norm.get(k, item),
                "action": "COOK",
                "reason": f"Used in the selected recipe: {selected_title}.",
                "target_recipes": [selected_title],
            })
        else:
            # human-ish leftover choice for expiring items only
            if sell_target and rng.random() < LEFTOVER_SELL_PROB:
                decisions.append({
                    "item": canon_by_norm.get(k, item),
                    "action": "SELL",
                    "reason": f"Not used by the selected recipe; likely demand at {sell_target}.",
                    "target_restaurants": [sell_target],
                })
            else:
                decisions.append({
                    "item": canon_by_norm.get(k, item),
                    "action": "DONATE",
                    "reason": "Not used by the selected recipe; donation reduces waste and aids community.",
                    "donation_center": donate_target or "Local soup kitchen",
                })
    return decisions

# ── Public API
def decide_actions() -> List[Dict[str, Any]]:
    """
    Entry point used by main.py:
      from agent.decision_agent.final_decision_agent import decide_actions
      decide_actions()

    Returns: the list of per ingredient decisions.
    """
    if not BEST_RECIPES_FILE.exists():
        raise FileNotFoundError(
            f"Missing {BEST_RECIPES_FILE}. Run the recipe agent first to create it."
        )

    best = _load_json(BEST_RECIPES_FILE)
    if not isinstance(best, list) or not best:
        raise ValueError("best_matching_recipes.json must be a non empty list of recipes.")

    # Sort by rank asc for a stable base; sampling adds the human-ish variability
    best_sorted = sorted(best, key=lambda r: int(r.get("rank", 9999)))

    expiring_list, canon_by_norm = _expiring_universe(best_sorted)
    target_norms = {_norm(x) for x in expiring_list}

    # Choose exactly ONE recipe via softmax sampling
    selected = _select_one_recipe_humanish(best_sorted, target_norms, rng=_rng, temperature=HUMAN_TEMPERATURE)
    sell_target, donate_target = _pick_sell_and_donate_targets()
    decisions = _build_decisions_single_recipe_humanish(
        expiring_universe=expiring_list,
        canon_by_norm=canon_by_norm,
        selected=selected,
        sell_target=sell_target,
        donate_target=donate_target,
        rng=_rng,
    )

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)

    # Console summary
    cook_list   = [d["item"] for d in decisions if d.get("action") == "COOK"]
    sell_list   = [d["item"] for d in decisions if d.get("action") == "SELL"]
    donate_list = [d["item"] for d in decisions if d.get("action") == "DONATE"]

    print("")  # spacer
    print("cook-list ingirdients:",   ", ".join(cook_list)   if cook_list   else "none")
    print("sell-list ingirdients:",   ", ".join(sell_list)   if sell_list   else "none")
    print("donate-list ingirdients:", ", ".join(donate_list) if donate_list else "none")
    # TODO: add the reason for each ingredient - why the agent chose to sell/cook/donate and where to
    return decisions

# Keep script runnable directly
def main():
    decide_actions()

if __name__ == "__main__":
    main()

