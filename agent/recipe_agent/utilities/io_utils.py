import json
from typing import Any, Dict, List, Tuple


def load_json(path: str, label: str, key_fields: Tuple[str, str] = ("item", "name")) -> Tuple[
    List[Dict[str, Any]], set]:
    """
    Safely loads a JSON file from the given path and extracts a set of unique items.

    Args:
        path (str): The file path to the JSON file.
        label (str): A descriptive label for logging purposes.
        key_fields (Tuple[str, str]): A tuple of keys to try for extracting item names.

    Returns:
        A tuple containing the raw JSON data (as a list) and a set of unique item names.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        items = {
            item.get(key_fields[0], item.get(key_fields[1]))
            for item in raw if isinstance(item, dict)
        }
        print(f"✅ Loaded {len(items)} items from {label}")
        return raw, items
    except Exception as e:
        print(f"❌ Failed to load {label} from {path}: {e}")
        return [], set()