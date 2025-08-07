import os
import json
import pandas as pd

BASE_DIR = r"C:\Users\User\Desktop\AI AGENT\recapies_dataset"



csv_files = [
    "recipes_data.csv",
    "recipes_w_search_terms.csv",
]


def print_sample_json(file_name):
    print("\n" + "=" * 80)
    print(f"📂 JSON File: {file_name}")
    print("=" * 80)

    path = os.path.join(BASE_DIR, file_name)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            for i, key in enumerate(list(data.keys())[:3]):
                print(f"\n🔑 Key: {key}")
                print(json.dumps(data[key], indent=2, ensure_ascii=False))
        elif isinstance(data, list):
            for i, item in enumerate(data[:3]):
                print(f"\n🔢 Item {i+1}:")
                print(json.dumps(item, indent=2, ensure_ascii=False))
        else:
            print(f"⚠️ Unknown JSON format: {type(data)}")

    except Exception as e:
        print(f"❌ Failed to read {file_name}: {e}")


def print_sample_csv(file_name):
    print("\n" + "=" * 80)
    print(f"📂 CSV File: {file_name}")
    print("=" * 80)

    path = os.path.join(BASE_DIR, file_name)
    try:
        df = pd.read_csv(path)
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"❌ Failed to read {file_name}: {e}")




# Run for all CSV files
for file_name in csv_files:
    print_sample_csv(file_name)
