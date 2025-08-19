import requests
import pandas as pd
import time
from geopy.distance import geodesic

API_KEY = ""
SEARCH_RADIUS = 5000
BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
HA_SALON_COORDS = (32.0861, 34.7825)

KEYWORDS = ["בית תמחוי"]

locations = [
    "32.0730,34.8003", "32.0853,34.7818", "32.0635,34.7722", "32.0554,34.7522",
    "32.0742,34.7920", "32.0925,34.8034", "32.1000,34.8240", "32.0690,34.8244",
    "32.0816,34.8486", "32.0765,34.8241", "32.0697,34.7947", "32.0231,34.7482",
    "32.0260,34.8100", "32.0755,34.8254", "32.1850,34.8260"
]

all_results = []

for keyword in KEYWORDS:
    for loc in locations:
        print(f"\n🔍 Searching '{keyword}' near {loc}...")
        next_page_token = None

        while True:
            params = {
                "location": loc,
                "radius": SEARCH_RADIUS,
                "keyword": keyword,
                "key": API_KEY
            }

            if next_page_token:
                params["pagetoken"] = next_page_token
                time.sleep(2)

            response = requests.get(BASE_URL, params=params)
            data = response.json()

            if "results" not in data:
                print("⚠️ No API results for this query.")
                break

            for place in data.get("results", []):
                place_id = place.get("place_id")

                # Fetch place details for opening_hours
                details_params = {
                    "place_id": place_id,
                    "fields": "opening_hours",
                    "key": API_KEY
                }
                time.sleep(0.1)
                details_resp = requests.get(DETAILS_URL, params=details_params)
                details = details_resp.json()
                opening_hours = None

                if details.get("result", {}).get("opening_hours", {}).get("weekday_text"):
                    opening_hours = "\n".join(details["result"]["opening_hours"]["weekday_text"])

                result = {
                    "name": place.get("name"),
                    "address": place.get("vicinity"),
                    "lat": place["geometry"]["location"]["lat"],
                    "lng": place["geometry"]["location"]["lng"],
                    "rating": place.get("rating"),
                    "user_ratings_total": place.get("user_ratings_total"),
                    "types": ", ".join(place.get("types", [])),
                    "keyword_matched": keyword,
                    "opening_hours": opening_hours
                }
                all_results.append(result)

            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break

# Build DataFrame
df = pd.DataFrame(all_results)
df.drop_duplicates(subset=["name", "address"], inplace=True)

if df.empty:
    print("\n⚠️ No food donation places found. Try different keywords or locations.")
else:
    df["distance_from_ha_salon_km"] = df.apply(
        lambda row: geodesic((row["lat"], row["lng"]), HA_SALON_COORDS).km, axis=1
    )

    # Print a sample
    print("\n✅ Sample results found:")
    print(df[["name", "address", "distance_from_ha_salon_km", "opening_hours"]].head())

    # Save to CSV
    df.to_csv("nearby_soup_kitchens.csv", index=False, encoding="utf-8-sig")
    print(f"\n📁 Saved {len(df)} food donation places with hours to 'nearby_soup_kitchens.csv'")
