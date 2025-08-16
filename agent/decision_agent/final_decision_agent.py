# final_decision_agent.py
# Final Decision Agent (single target recipe, human-ish randomness) — with rich reasons
# -----------------------------------------------------------------------------
# Picks exactly ONE recipe using results/recipes/best_matching_recipes.json.
# Assigns every expiring ingredient to COOK or SELL or DONATE.
# Adds human-readable reasons per group that explain *why*, not just "not used".

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

BEST_RECIPES_FILE       = RECIPES_DIR / "best_matching_recipes.json"
EXPIRING_FILE           = PROJECT_ROOT / "data" / "expiring_ingredients.json"
TOP_RESTAURANTS_FILE    = RESULTS_DIR / "top_restaurants.json"
DONATION_FILE           = RESULTS_DIR / "best_soup_kitchen.json"
REPLY_TABLE_FILE        = RESULTS_DIR / "learning" / "top_restaurants_by_reply.json"

OUTPUT_FILE = RESULTS_DIR / "final_decision_output.json"

# ── Behavior knobs (tweak to taste or via env)
HUMAN_TEMPERATURE  = float(os.getenv("FOODFLOW_TEMPERATURE", "0.9"))  # 0.3=greedy, 1.2=wilder
SEED_ENV           = os.getenv("FOODFLOW_SEED")  # set to a number for reproducibility
LEFTOVER_SELL_PROB = float(os.getenv("FOODFLOW_SELL_PROB", "0.7"))    # with a SELL target

# ── Random instance (optionally seeded)
_rng = random.Random()
if SEED_ENV is not None:
    try:
        _rng.seed(int(SEED_ENV))
    except Exception:
        _rng.seed(SEED_ENV)

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

def _safe_get_days(meta_row: Dict[str, Any]) -> Optional[float]:
    d = meta_row.get("days_to_expire")
    try:
        return float(d) if d is not None else None
    except Exception:
        return None

def _expiring_universe() -> Tuple[List[str], Dict[str, str]]:
    """Universe of expiring items and norm->canonical map from expiring_ingredients.json only."""
    universe: List[str] = []
    if EXPIRING_FILE.exists():
        raw = _load_json(EXPIRING_FILE)
        for row in raw:
            nm = (row.get("item") if isinstance(row, dict) else None) or (row.get("name") if isinstance(row, dict) else None)
            if not nm and isinstance(row, str):
                nm = row
            if isinstance(nm, str) and nm.strip():
                universe.append(nm.strip())
    universe = _canonicalize_list(universe)
    canon_by_norm = {_norm(x): x for x in universe}
    return universe, canon_by_norm

def _expiring_meta_map() -> Dict[str, Dict[str, Any]]:
    """norm(name) -> {days_to_expire, quantity, unit} for reasoning."""
    meta: Dict[str, Dict[str, Any]] = {}
    if not EXPIRING_FILE.exists(): return meta
    for row in _load_json(EXPIRING_FILE):
        if not isinstance(row, dict): continue
        nm = (row.get("item") or row.get("name") or "").strip()
        if not nm: continue
        meta[_norm(nm)] = {
            "days_to_expire": row.get("days_to_expire") or row.get("days") or row.get("expires_in_days"),
            "quantity": row.get("quantity"),
            "unit": row.get("unit"),
        }
    return meta

# --- Light knowledge to craft better reasons ---
COOK_PRIORITY = {
    # spoils fast or benefits from cooking now
    "mushrooms", "duck breast", "shrimp", "sea bass", "eggplant", "zucchini", "cauliflower", "white wine",
}
SELL_MARKETABLE = {
    # common cross-restaurant staples
    "artichoke", "lemon zest", "rosemary", "parsley", "mint", "cream", "butter",
    "crushed tomato", "white wine", "zucchini", "cauliflower", "lamb chops",
}
DONATE_FRIENDLY = {
    # good for community kitchens when near expiry (assuming safe handling)
    "yogurt", "labneh", "bread", "beets", "leeks", "cabbage",
}

def _softmax_sample(weights: List[float], temperature: float, rng: random.Random) -> int:
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

def _select_one_recipe(best: List[Dict[str, Any]], expiring_norms: Set[str],
                       rng: random.Random, temperature: float) -> Optional[Dict[str, Any]]:
    """Score = 0.75*coverage + 0.25*rank_norm + noise; sample via softmax(T)."""
    if not best: return None
    coverages, ranks = [], []
    for rec in best:
        cov = {_norm(x) for x in (rec.get("matched_expiring") or []) if isinstance(x, str)}
        coverages.append(len(expiring_norms & cov))
        ranks.append(int(rec.get("rank", 9999)))
    max_cov = max(coverages) if coverages else 1
    max_rank = max(ranks) if ranks else 1
    max_cov = max(1, max_cov); max_rank = max(1, max_rank)

    weights = []
    for cov, rk in zip(coverages, ranks):
        cov_norm = cov / max_cov
        rank_norm = 1.0 - (rk - 1) / (max_rank - 1) if max_rank > 1 else 1.0
        base = 0.75 * cov_norm + 0.25 * rank_norm
        weights.append(base + rng.uniform(-0.05, 0.05))
    idx = _softmax_sample(weights, temperature=temperature, rng=rng)
    return best[idx] if 0 <= idx < len(best) else None

def _pick_targets_and_reply_prob() -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[float]]:
    """
    Returns:
      (sell_top_entry, sell_target_name, reply_prob_if_known)
      sell_top_entry is the first object from top_restaurants.json (may hold matched_ingredients, reason, expected_reply_prob)
    """
    sell_entry = None
    sell_name, reply_prob = None, None

    if TOP_RESTAURANTS_FILE.exists():
        try:
            top = _load_json(TOP_RESTAURANTS_FILE)
            if isinstance(top, list) and top:
                sell_entry = top[0]
            elif isinstance(top, dict):
                for key in ["restaurants", "top", "results", "items", "matches"]:
                    arr = top.get(key)
                    if isinstance(arr, list) and arr:
                        sell_entry = arr[0]; break
                if not sell_entry:
                    sell_entry = top
        except Exception:
            sell_entry = None

    if isinstance(sell_entry, dict):
        sell_name = (sell_entry.get("name") or sell_entry.get("title") or "").strip() or None
        # use embedded expected_reply_prob if present
        rp = sell_entry.get("expected_reply_prob")
        try:
            reply_prob = float(rp) if rp is not None else None
        except Exception:
            reply_prob = None

    # overwrite with learned table if available (source of truth)
    if sell_name and REPLY_TABLE_FILE.exists():
        try:
            table = _load_json(REPLY_TABLE_FILE)
            by = {(_norm(r.get("name",""))): r for r in table if isinstance(r, dict)}
            row = by.get(_norm(sell_name))
            if row and "expected_reply_prob" in row:
                reply_prob = float(row["expected_reply_prob"])
        except Exception:
            pass

    return sell_entry, sell_name, reply_prob

def _build_decisions_single_recipe(
    expiring_universe: List[str],
    canon_by_norm: Dict[str, str],
    selected: Optional[Dict[str, Any]],
    sell_target: Optional[str],
    donate_target: Optional[str],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Per-ingredient decisions with a SINGLE selected recipe."""
    selected_title = (selected or {}).get("title", "").strip() if selected else ""
    selected_norms = {_norm(x) for x in ((selected or {}).get("matched_expiring") or []) if isinstance(x, str)}
    expiring_norms = {_norm(x) for x in expiring_universe}
    selected_norms &= expiring_norms  # only cook things truly expiring

    decisions: List[Dict[str, Any]] = []
    for item in expiring_universe:
        k = _norm(item)
        if selected and k in selected_norms:
            decisions.append({
                "item": canon_by_norm.get(k, item),
                "action": "COOK",
                "reason": f"Used in selected recipe '{selected_title}'.",
                "target_recipes": [selected_title],
            })
        else:
            # human-ish leftover choice for expiring items only
            if sell_target and rng.random() < LEFTOVER_SELL_PROB:
                decisions.append({
                    "item": canon_by_norm.get(k, item),
                    "action": "SELL",
                    "reason": f"Better matched to {sell_target}'s menu and demand window.",
                    "target_restaurants": [sell_target],
                })
            else:
                decisions.append({
                    "item": canon_by_norm.get(k, item),
                    "action": "DONATE",
                    "reason": "Short shelf-life / lower marketability today; donation reduces waste.",
                    "donation_center": donate_target or "Local soup kitchen",
                })
    return decisions

# —— Group reason builders (richer, multi-signal) ————————————————

def _split_by_days(items: List[str], meta: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    """Return (<=1 day, 2–3 days, >3 days) lists by canonical item name."""
    soon, mid, long = [], [], []
    for it in items:
        d = _safe_get_days(meta.get(_norm(it), {}))
        if d is None:
            mid.append(it)  # treat unknown as mid to avoid extreme choices
        elif d <= 1:
            soon.append(it)
        elif d <= 3:
            mid.append(it)
        else:
            long.append(it)
    return soon, mid, long

def _pick_examples(items: List[str], n: int = 3) -> List[str]:
    return items[:n]

def _intersect_with_sell_matches(sell_items: List[str], sell_entry: Optional[Dict[str, Any]]) -> List[str]:
    """Return items that the sell target explicitly matched (if the file contains 'matched_ingredients')."""
    if not sell_entry: return []
    matched = set(_norm(x) for x in (sell_entry.get("matched_ingredients") or []) if isinstance(x, str))
    out = []
    for it in sell_items:
        if _norm(it) in matched:
            out.append(it)
    return out

def _group_reason_cook(selected: Optional[Dict[str, Any]], cook_items: List[str], meta: Dict[str, Dict[str, Any]]) -> str:
    if not selected:
        return "No recipe selected; nothing to cook."
    title = (selected.get("title") or "").strip()
    rank  = selected.get("rank")
    soon, mid, long = _split_by_days(cook_items, meta)
    high_risk_hits = [it for it in cook_items if _norm(it) in COOK_PRIORITY]
    used_preview = ", ".join(_pick_examples(cook_items, 3)) if cook_items else "none"
    bits = [f"picked '{title}'"]
    if isinstance(rank, int):
        bits.append(f"(rank #{rank})")
    if cook_items:
        bits.append(f"to convert {len(cook_items)} expiring items into menu value ({used_preview})")
    if soon:
        bits.append(f"including {len(soon)} due ≤1 day")
    if high_risk_hits:
        bits.append(f"; prioritizes quick-spoil items (e.g., {', '.join(_pick_examples(high_risk_hits, 2))})")
    return " ".join(bits) + "."

def _group_reason_sell(
    sell_items: List[str],
    sell_target: Optional[str],
    sell_entry: Optional[Dict[str, Any]],
    reply_prob: Optional[float],
    meta: Dict[str, Dict[str, Any]],
) -> str:
    if not sell_items:
        return "No items selected to sell."
    soon, mid, long = _split_by_days(sell_items, meta)
    menu_overlap = _intersect_with_sell_matches(sell_items, sell_entry)
    marketables = [it for it in sell_items if _norm(it) in SELL_MARKETABLE]

    tgt = sell_target or "the top nearby restaurant"
    bits = [f"sending {len(sell_items)} items to {tgt}"]

    # why sell vs cook/donate:
    if long or mid:
        bits.append(f"because {len(mid)+len(long)} item(s) have ≥2 days left")
    if menu_overlap:
        bits.append(f"and their menu matches items like {', '.join(_pick_examples(menu_overlap, 3))}")
    elif marketables:
        bits.append(f"and they commonly use staples like {', '.join(_pick_examples(marketables, 3))}")
    if reply_prob is not None:
        bits.append(f"(expected reply ≈ {reply_prob:.2f})")
    if soon and not mid and not long:
        bits.append("— only a few are close to expiry, still saleable today")

    return " ".join(bits) + "."

def _group_reason_donate(
    donate_items: List[str],
    donate_target: Optional[str],
    meta: Dict[str, Dict[str, Any]],
) -> str:
    if not donate_items:
        return "No items selected to donate."
    soon, mid, long = _split_by_days(donate_items, meta)
    donor_pref = [it for it in donate_items if _norm(it) in DONATE_FRIENDLY]

    tgt = donate_target or "a local soup kitchen"
    bits = [f"donating {len(donate_items)} item(s) to {tgt}"]

    # why donate vs sell:
    if soon:
        bits.append(f"because {len(soon)} expire ≤1 day")
    if donor_pref:
        bits.append(f"and items are serviceable for bulk/meal-prep (e.g., {', '.join(_pick_examples(donor_pref, 2))})")
    if mid and not soon:
        bits.append("to reduce overstock where market demand is uncertain today")

    return " ".join(bits) + "."

# ── Public API
def decide_actions() -> List[Dict[str, Any]]:
    """
    Entry point used by main.py:
      from agent.decision_agent.final_decision_agent import decide_actions
      decide_actions()
    """
    if not BEST_RECIPES_FILE.exists():
        raise FileNotFoundError(f"Missing {BEST_RECIPES_FILE}. Run the recipe agent first.")

    best = _load_json(BEST_RECIPES_FILE)
    if not isinstance(best, list) or not best:
        raise ValueError("best_matching_recipes.json must be a non-empty list of recipes.")

    best_sorted = sorted(best, key=lambda r: int(r.get("rank", 9999)))
    expiring_list, canon_by_norm = _expiring_universe()
    expiring_norms = {_norm(x) for x in expiring_list}

    # Choose exactly ONE recipe via softmax sampling
    selected = _select_one_recipe(best_sorted, expiring_norms, rng=_rng, temperature=HUMAN_TEMPERATURE)

    # Targets & reply-prob for SELL
    sell_entry, sell_target, reply_prob = _pick_targets_and_reply_prob()
    donate_target = None
    if DONATION_FILE.exists():
        try:
            d = _load_json(DONATION_FILE)
            donate_target = (d.get("name") or "").strip() or None
        except Exception:
            donate_target = None

    # Per-ingredient decisions
    decisions = _build_decisions_single_recipe(
        expiring_universe=expiring_list,
        canon_by_norm=canon_by_norm,
        selected=selected,
        sell_target=sell_target,
        donate_target=donate_target,
        rng=_rng,
    )

    # Save per-ingredient decisions
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)

    # Console summary with rich reasons
    cook_list   = [d["item"] for d in decisions if d.get("action") == "COOK"]
    sell_list   = [d["item"] for d in decisions if d.get("action") == "SELL"]
    donate_list = [d["item"] for d in decisions if d.get("action") == "DONATE"]

    meta_map = _expiring_meta_map()

    print("")  # spacer
    print("cook-list ingirdients:",   ", ".join(cook_list)   if cook_list   else "none")
    print("reason:", _group_reason_cook(selected, cook_list, meta_map))

    print("sell-list ingirdients:",   ", ".join(sell_list)   if sell_list   else "none")
    print("reason:", _group_reason_sell(sell_list, sell_target, sell_entry, reply_prob, meta_map))

    print("donate-list ingirdients:", ", ".join(donate_list) if donate_list else "none")
    print("reason:", _group_reason_donate(donate_list, donate_target, meta_map))

    return decisions

# Keep script runnable directly
def main():
    decide_actions()

if __name__ == "__main__":
    main()
