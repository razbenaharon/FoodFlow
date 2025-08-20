import re
from typing import Any, Dict, List, Optional, Tuple


def _dedupe_keep_order(items: List[str]) -> List[str]:
    """Removes duplicate strings from a list while preserving their original order."""
    seen, out = set(), []
    for x in items:
        if not isinstance(x, str):
            continue
        key = x.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(x.strip())
    return out


def _collect_lists_by_keys(obj: Any, keys_exact=None, key_substrings=None) -> List[str]:
    """
    Recursively walks through a nested dictionary or list to find lists of strings
    associated with specific keys.

    It searches for keys that are an exact match or contain a specific substring.
    """
    keys_exact = [k.lower() for k in (keys_exact or [])]
    key_substrings = [s.lower() for s in (key_substrings or [])]
    out: List[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                hit = (kl in keys_exact) or any(sub in kl for sub in key_substrings)
                if hit and isinstance(v, list):
                    out.extend([x for x in v if isinstance(x, str)])
                walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)

    walk(obj)
    return _dedupe_keep_order(out)


def extract_matched_sets(analysis: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Extracts and categorizes ingredients from an analysis dictionary into three lists:
    expiring, inventory, and missing ingredients.

    It uses a predefined set of key names and substrings to find the relevant lists
    within the input analysis.
    """
    analysis = analysis or {}
    expiring = _collect_lists_by_keys(
        analysis,
        keys_exact=["expiring_matched", "expiring_used", "matched_expiring", "expiring"],
        key_substrings=["expiring", "about to expire"],
    )
    inventory = _collect_lists_by_keys(
        analysis,
        keys_exact=["inventory_matched", "inventory_used", "matched_inventory", "available_used", "inventory"],
        key_substrings=["inventory", "available", "pantry", "stock"],
    )
    missing = _collect_lists_by_keys(
        analysis,
        keys_exact=["missing", "missing_ingredients", "needed"],
        key_substrings=["missing", "need", "shopping", "shortfall", "not available", "lacking"],
    )
    return expiring, inventory, missing


def safe_join(lst: List[str]) -> str:
    """Joins a list of strings into a comma-separated string, handling empty lists gracefully."""
    return ", ".join(lst) if lst else "none"


def extract_json_block(text: str) -> str:
    """
    Extracts the first JSON object string from a given text block.

    Raises a ValueError if no JSON object is found.
    """
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in response.")