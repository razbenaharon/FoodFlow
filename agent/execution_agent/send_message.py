# agent/execution_agent/send_message.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.chat_and_embedding import LLMChat

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
RESULTS_DIR    = PROJECT_ROOT / "results"
RECIPES_DIR    = RESULTS_DIR / "recipes"
DATA_DIR       = PROJECT_ROOT / "data"
MESSAGES_DIR   = RESULTS_DIR / "messages"

DECISIONS_FILE = RESULTS_DIR / "final_decision_output.json"
EXPIRING_FILE  = DATA_DIR / "expiring_ingredients.json"

DEFAULT_RESTAURANT_NAME = "HaSalon"
DEFAULT_CITY = "Tel Aviv"


# ---------------- helpers ----------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_expiring_map() -> Dict[str, Dict[str, Any]]:
    """name -> {quantity, unit, days_to_expire}"""
    if not EXPIRING_FILE.exists():
        return {}
    raw = _load_json(EXPIRING_FILE)
    out: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, dict):
            continue
        name = (row.get("item") or row.get("name") or "").strip()
        if not name:
            continue
        out[_norm(name)] = {
            "name": name,
            "quantity": row.get("quantity"),
            "unit": row.get("unit"),
            "days_to_expire": row.get("days_to_expire"),
        }
    return out

def _group_from_decisions(
    restaurant_hint: Optional[str] = None,
    soup_kitchen_hint: Optional[str] = None,
) -> Tuple[str, List[str], str, List[str]]:
    """
    Returns:
      (restaurant_name, sell_items[], soup_kitchen_name, donate_items[])
    If hints provided, filter to those; otherwise pick the first targets found.
    """
    if not DECISIONS_FILE.exists():
        raise FileNotFoundError(f"Missing decisions file: {DECISIONS_FILE}")
    decisions = _load_json(DECISIONS_FILE)
    if not isinstance(decisions, list):
        raise ValueError("final_decision_output.json must be a list.")

    sell_by_target: Dict[str, List[str]] = {}
    donate_by_target: Dict[str, List[str]] = {}

    for row in decisions:
        if not isinstance(row, dict):
            continue
        item = (row.get("item") or "").strip()
        action = (row.get("action") or "").strip().upper()
        if action == "SELL":
            trg = ""
            trg_list = row.get("target_restaurants") or []
            if isinstance(trg_list, list) and trg_list:
                trg = str(trg_list[0]).strip()
            if not trg:
                trg = "Nearby partner"
            sell_by_target.setdefault(trg, []).append(item)
        elif action == "DONATE":
            trg = str(row.get("donation_center") or "").strip() or "Local soup kitchen"
            donate_by_target.setdefault(trg, []).append(item)

    # choose targets
    restaurant_name = restaurant_hint or (next(iter(sell_by_target)) if sell_by_target else "")
    soup_kitchen_name = soup_kitchen_hint or (next(iter(donate_by_target)) if donate_by_target else "")

    sell_items = sell_by_target.get(restaurant_name, [])
    donate_items = donate_by_target.get(soup_kitchen_name, [])

    return restaurant_name, sell_items, soup_kitchen_name, donate_items

def _prep_item_lines(items: List[str], exp_map: Dict[str, Dict[str, Any]]) -> List[str]:
    """Create compact item lines with qty/unit/expiry for the prompt."""
    lines = []
    for it in items:
        meta = exp_map.get(_norm(it), {})
        q = meta.get("quantity")
        u = meta.get("unit")
        d = meta.get("days_to_expire")
        qty = f"{q:.2f}" if isinstance(q, (int, float)) else (str(q) if q is not None else "")
        unit = u or ""
        exp  = f"{d} days" if isinstance(d, (int, float)) else ("soon" if d is not None else "n/a")
        if qty and unit:
            lines.append(f"- {it}: {qty} {unit}, expires in {exp}")
        elif qty:
            lines.append(f"- {it}: {qty}, expires in {exp}")
        else:
            lines.append(f"- {it}, expires in {exp}")
    return lines

def _gpt_message_to_restaurant(
    restaurant: str,
    sell_lines: List[str],
    brand: str = DEFAULT_RESTAURANT_NAME,
    city: str = DEFAULT_CITY
) -> str:
    if not sell_lines:
        return ""
    prompt = f"""You write a concise, professional WhatsApp-style message in English from {brand} in {city} to a partner restaurant named "{restaurant}".
We want to SELL the following surplus ingredients today (quantities and expiry below).
Please:
- Start with a warm one-line intro that we're {brand} and have high-quality surplus available today.
- Bullet the items exactly as given THEN add a per-item suggested price and line subtotal in ILS (₪), based on freshness (bigger discount if fewer days to expire).
- After bullets, include: a bundled offer price, pickup window today, and a contact phone number placeholder.
- Keep it under 1200 characters. Avoid markdown code fences.

Surplus list:
{os.linesep.join(sell_lines)}

Output: message text only."""
    chat = LLMChat(temp=0.5)
    return chat.send_prompt(prompt).strip()

def _gpt_message_to_soup_kitchen(
    kitchen: str,
    donate_lines: List[str],
    brand: str = DEFAULT_RESTAURANT_NAME,
    city: str = DEFAULT_CITY
) -> str:
    if not donate_lines:
        return ""
    prompt = f"""Write a kind, short WhatsApp-style message in English from {brand} in {city} to "{kitchen}".
We want to DONATE the following fresh ingredients today (quantities and expiry below).
Please:
- One-line intro that we're {brand} and we have same-day donation.
- Bullet the items exactly as given (no prices).
- Add an estimated pickup window today and a contact phone number placeholder.
- Keep it under 800 characters. Avoid markdown code fences.

Donation list:
{os.linesep.join(donate_lines)}

Output: message text only."""
    chat = LLMChat(temp=0.5)
    return chat.send_prompt(prompt).strip()


# ---------------- public API ----------------
def send_message(restaurant: Optional[str] = None, soup_kitchen: Optional[str] = None) -> Dict[str, str]:
    """
    Build and save messages for SELL and DONATE flows using GPT.
    If 'restaurant' or 'soup_kitchen' provided, targets are forced to those names.
    Returns dict with keys 'restaurant_message_path' and/or 'soup_kitchen_message_path'.
    """
    exp_map = _load_expiring_map()
    r_name, sell_items, k_name, donate_items = _group_from_decisions(restaurant, soup_kitchen)

    sell_lines = _prep_item_lines(sell_items, exp_map)
    donate_lines = _prep_item_lines(donate_items, exp_map)

    MESSAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_paths: Dict[str, str] = {}

    if r_name and sell_lines:
        msg_r = _gpt_message_to_restaurant(r_name, sell_lines)
        r_path = MESSAGES_DIR / "message_to_restaurant.txt"     # ← fixed filename
        with open(r_path, "w", encoding="utf-8") as f:
            f.write(msg_r)
        print("\n====== Message to restaurant ======")
        print(msg_r)
        out_paths["restaurant_message_path"] = str(r_path)

    if k_name and donate_lines:
        msg_k = _gpt_message_to_soup_kitchen(k_name, donate_lines)
        k_path = MESSAGES_DIR / "message_to_soup_kitchen.txt"   # ← fixed filename
        with open(k_path, "w", encoding="utf-8") as f:
            f.write(msg_k)
        print("\n====== Message to soup kitchen ======")
        print(msg_k)
        out_paths["soup_kitchen_message_path"] = str(k_path)

    if not out_paths:
        print("No messages generated (no SELL/DONATE items or no targets).")
    return out_paths


# --------------- CLI ---------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Send sales/donation messages via GPT.")
    p.add_argument("--restaurant", type=str, default=None, help="Target restaurant name (optional)")
    p.add_argument("--soup_kitchen", type=str, default=None, help="Target soup kitchen name (optional)")
    args = p.parse_args()
    send_message(args.restaurant, args.soup_kitchen)
