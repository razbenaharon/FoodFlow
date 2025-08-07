import json

def load_json(path, label, key_fields=("item", "name")):
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
