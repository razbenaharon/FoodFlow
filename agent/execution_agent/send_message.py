# agent/execution_agent/send_message.py
"""
This module generates outbound WhatsApp-style messages for:
- SELLING surplus ingredients to a restaurant
- DONATING surplus ingredients to a soup kitchen

It reads:
  • expiring_ingredients.json (for quantities/expiry)
  • final_decision_output.json (decisions from decision agent)

Then uses GPT (via LLMChat) to draft natural-language messages.
Each message is saved under results/messages/.
"""

from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from utils.chat_and_embedding import LLMChat   # wrapper around Azure GPT

# --- Paths ---
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
RESULTS_DIR    = PROJECT_ROOT / "results"
RECIPES_DIR    = RESULTS_DIR / "recipes"
DATA_DIR       = PROJECT_ROOT / "data"
MESSAGES_DIR   = RESULTS_DIR / "messages"

DECISIONS_FILE = RESULTS_DIR / "final_decision_output.json"
EXPIRING_FILE  = DATA_DIR / "expiring_ingredients.json"

DEFAULT_RESTAURANT_NAME = "HaSalon"
DEFAULT_CITY = "Tel Aviv"

# Mock contact number; real number can be passed via env var
CONTACT_PHONE = os.getenv("FOODFLOW_CONTACT_PHONE", "052-1234567")

# ---------------- helpers ----------------
def _norm(s: str) -> str:
    """Normalize a string for dictionary lookups (lowercased, no extra spaces)."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_expiring_map() -> Dict[str, Dict[str, Any]]:
    """
    Load expiring ingredients and build a lookup:
    normalized_name -> {quantity, unit, days_to_expire}
    """
    if not EXPIRING_FILE.exists():
        return {}
    raw = _load_json(EXPIRING_FILE)
    out: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, dict): continue
        name = (row.get("item") or row.get("name") or "").strip()
        if not name: continue
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
    Read final_decision_output.json and group items by SELL/restaurant and DONATE/kitchen.
    Returns: (restaurant_name, sell_items[], soup_kitchen_name, donate_items[])
    """
    if not DECISIONS_FILE.exists():
        raise FileNotFoundError(f"Missing decisions file: {DECISIONS_FILE}")
    decisions = _load_json(DECISIONS_FILE)
    if not isinstance(decisions, list):
        raise ValueError("final_decision_output.json must be a list.")

    sell_by_target, donate_by_target = {}, {}
    for row in decisions:
        if not isinstance(row, dict): continue
        item = (row.get("item") or "").strip()
        action = (row.get("action") or "").strip().upper()
        if action == "SELL":
            trg_list = row.get("target_restaurants") or []
            trg = str(trg_list[0]).strip() if isinstance(trg_list, list) and trg_list else "Nearby partner"
            sell_by_target.setdefault(trg, []).append(item)
        elif action == "DONATE":
            trg = str(row.get("donation_center") or "").strip() or "Local soup kitchen"
            donate_by_target.setdefault(trg, []).append(item)

    # Choose targets (first found, unless hints provided)
    restaurant_name = restaurant_hint or (next(iter(sell_by_target)) if sell_by_target else "")
    soup_kitchen_name = soup_kitchen_hint or (next(iter(donate_by_target)) if donate_by_target else "")
    return restaurant_name, sell_by_target.get(restaurant_name, []), soup_kitchen_name, donate_by_target.get(soup_kitchen_name, [])

def _prep_item_lines(items: List[str], exp_map: Dict[str, Dict[str, Any]]) -> List[str]:
    """Format expiring items with qty/unit/expiry into human-readable bullet lines."""
    lines = []
    for it in items:
        meta = exp_map.get(_norm(it), {})
        q, u, d = meta.get("quantity"), meta.get("unit"), meta.get("days_to_expire")
        qty = f"{q:.2f}" if isinstance(q, (int, float)) else (str(q) if q is not None else "")
        exp  = f"{d} days" if isinstance(d, (int, float)) else ("soon" if d is not None else "n/a")
        if qty and u: lines.append(f"- {it}: {qty} {u}, expires in {exp}")
        elif qty:     lines.append(f"- {it}: {qty}, expires in {exp}")
        else:         lines.append(f"- {it}, expires in {exp}")
    return lines

def _ensure_contact_phone(text: str) -> str:
    """
    Make sure CONTACT_PHONE is present in the generated message.
    Replace placeholders like [phone number] if necessary.
    """
    if CONTACT_PHONE in text:
        return text
    patterns = [r"\[\s*phone\s*number\s*\]", r"\[\s*insert\s*phone\s*number\s*\]",
                r"\[\s*phone\s*\]", r"\(\s*phone\s*\)", r"\{\s*phone\s*\}",
                r"\b(?:XXX-XXX-XXXX|000-000-0000)\b"]
    new_text, replaced = text, False
    for pat in patterns:
        if re.search(pat, new_text, flags=re.IGNORECASE):
            new_text = re.sub(pat, CONTACT_PHONE, new_text, flags=re.IGNORECASE); replaced = True
    if not replaced:
        new_text += f"\nContact: {CONTACT_PHONE}"
    return new_text

# --- GPT prompts for restaurant and soup kitchen messages ---
def _gpt_message_to_restaurant(restaurant: str, sell_lines: List[str], brand: str = DEFAULT_RESTAURANT_NAME, city: str = DEFAULT_CITY) -> str:
    """Generate GPT message offering surplus to a restaurant."""
    if not sell_lines: return ""
    prompt = f"""You write a concise, professional WhatsApp-style message ... (prompt omitted for brevity)"""
    msg = LLMChat(temp=0.5).send_prompt(prompt).strip()
    return _ensure_contact_phone(msg)

def _gpt_message_to_soup_kitchen(kitchen: str, donate_lines: List[str], brand: str = DEFAULT_RESTAURANT_NAME, city: str = DEFAULT_CITY) -> str:
    """Generate GPT message offering donation to a soup kitchen."""
    if not donate_lines: return ""
    prompt = f"""Write a kind, short WhatsApp-style message ... (prompt omitted for brevity)"""
    msg = LLMChat(temp=0.5).send_prompt(prompt).strip()
    return _ensure_contact_phone(msg)

# ---------------- public API ----------------
def send_message(restaurant: Optional[str] = None, soup_kitchen: Optional[str] = None) -> Dict[str, str]:
    """
    Build and save messages for SELL and DONATE flows.
    Saves to results/messages/message_to_restaurant.txt and message_to_soup_kitchen.txt.
    Returns dict of output paths.
    """
    exp_map = _load_expiring_map()
    r_name, sell_items, k_name, donate_items = _group_from_decisions(restaurant, soup_kitchen)

    sell_lines, donate_lines = _prep_item_lines(sell_items, exp_map), _prep_item_lines(donate_items, exp_map)
    MESSAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = {}

    if r_name and sell_lines:
        msg_r = _gpt_message_to_restaurant(r_name, sell_lines)
        r_path = MESSAGES_DIR / "message_to_restaurant.txt"
        r_path.write_text(msg_r, encoding="utf-8")
        print("\n====== Message to restaurant ======\n", msg_r); out_paths["restaurant_message_path"] = str(r_path)

    if k_name and donate_lines:
        msg_k = _gpt_message_to_soup_kitchen(k_name, donate_lines)
        k_path = MESSAGES_DIR / "message_to_soup_kitchen.txt"
        k_path.write_text(msg_k, encoding="utf-8")
        print("\n====== Message to soup kitchen ======\n", msg_k); out_paths["soup_kitchen_message_path"] = str(k_path)

    if not out_paths: print("No messages generated.")
    return out_paths

# --------------- CLI ---------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Send sales/donation messages via GPT.")
    p.add_argument("--restaurant", type=str, help="Target restaurant name (optional)")
    p.add_argument("--soup_kitchen", type=str, help="Target soup kitchen name (optional)")
    args = p.parse_args()
    send_message(args.restaurant, args.soup_kitchen)
