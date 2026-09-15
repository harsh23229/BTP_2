import requests
import pandas as pd
from pathlib import Path

BASE_URL = "https://cosylab.iiitd.edu.in/carbon-food-print-2-api/api/getAllIngredients"
OUTPUT = Path(__file__).resolve().parent / "SustainableFoodDB_CarbonFootprints.csv"

session = requests.Session()
session.headers.update({
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://cosylab.iiitd.edu.in/SustainableFoodDB/",
    "User-Agent": "Mozilla/5.0"
})

print("Fetching SustainableFoodDB...")

response = session.get(BASE_URL, params={"page": 1}, timeout=30)
response.raise_for_status()

data = response.json()
total_pages = data["total_pages"]
total_count = data["total_count"]
all_data = data.get("Ingredient_Data", [])

print(f"Expected records: {total_count}")
print(f"Total pages: {total_pages}")
print(f"Page 1/{total_pages}: {len(all_data)} records")

for page in range(2, total_pages + 1):
    response = session.get(BASE_URL, params={"page": page}, timeout=30)
    response.raise_for_status()

    rows = response.json().get("Ingredient_Data", [])
    all_data.extend(rows)

    print(f"Page {page}/{total_pages} | Downloaded: {len(all_data)}")

df = pd.DataFrame(all_data)

if "_id" in df.columns:
    before = len(df)
    df = df.drop_duplicates("_id")
    print(f"Duplicate records removed: {before - len(df)}")

print("\nData audit")
print(f"Records downloaded: {len(df)}")
print(f"API expected: {total_count}")
print(f"Count matches: {'YES' if len(df) == total_count else 'NO'}")

if "Ingredient" in df.columns:
    print(f"Unique RecipeDB ingredients: {df['Ingredient'].nunique()}")

if "SueatableIngredient" in df.columns:
    print(f"Unique SuEatable ingredients: {df['SueatableIngredient'].nunique()}")

if "CarbonFootprint" in df.columns:
    print(f"Missing carbon footprints: {df['CarbonFootprint'].isna().sum()}")

if "Similarity" in df.columns:
    print(f"Missing similarity values: {df['Similarity'].isna().sum()}")

df.to_csv(OUTPUT, index=False)

print("\nDOWNLOAD COMPLETE")
print(f"Records saved: {len(df)}")
print(f"File: {OUTPUT}")