import requests
import pandas as pd
import time
from pathlib import Path

BASE_URL = (
    "https://cosylab.iiitd.edu.in/"
    "carbon-food-print-2-api/api/getAllIngredients"
)

OUTPUT = Path("SustainableFoodDB_CarbonFootprints.csv")

session = requests.Session()

session.headers.update({
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://cosylab.iiitd.edu.in/SustainableFoodDB/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    ),
})

# ---------------------------------------------------------
# STEP 1: Get first page and discover total pages
# ---------------------------------------------------------

print("Fetching first page...")

response = session.get(
    BASE_URL,
    params={"page": 1},
    timeout=30
)

response.raise_for_status()

data = response.json()

total_pages = data["total_pages"]
total_count = data["total_count"]

print(f"Total records : {total_count}")
print(f"Total pages   : {total_pages}")

all_data = data["Ingredient_Data"]

print(f"Page 1/{total_pages}: {len(data['Ingredient_Data'])} records")


# ---------------------------------------------------------
# STEP 2: Download remaining pages
# ---------------------------------------------------------

for page in range(2, total_pages + 1):

    try:
        response = session.get(
            BASE_URL,
            params={"page": page},
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        rows = data.get("Ingredient_Data", [])

        all_data.extend(rows)

        print(
            f"Page {page}/{total_pages}: "
            f"{len(rows)} records | "
            f"Total downloaded: {len(all_data)}"
        )

        time.sleep(0.1)

    except Exception as e:
        print(f"\nERROR on page {page}: {e}")
        raise


# ---------------------------------------------------------
# STEP 3: Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(all_data)

print("\nDownloaded columns:")
print(df.columns.tolist())

print(f"\nRecords before deduplication: {len(df)}")


# ---------------------------------------------------------
# STEP 4: Remove duplicate IDs
# ---------------------------------------------------------

if "_id" in df.columns:
    before = len(df)

    df = df.drop_duplicates(subset="_id")

    duplicates_removed = before - len(df)

    print(f"Duplicate records removed: {duplicates_removed}")


# ---------------------------------------------------------
# STEP 5: Save CSV
# ---------------------------------------------------------

df.to_csv(OUTPUT, index=False)

print("\n==========================================")
print("DOWNLOAD COMPLETE")
print("==========================================")

print(f"Records saved : {len(df)}")
print(f"Expected      : {total_count}")
print(f"File          : {OUTPUT.absolute()}")


# ---------------------------------------------------------
# STEP 6: Basic data audit
# ---------------------------------------------------------

print("\nMissing values:")
print(df.isna().sum())

if "CarbonFootprint" in df.columns:

    print("\nCarbon footprint statistics:")
    print(df["CarbonFootprint"].describe())

    print("\nMissing carbon footprints:")
    print(df["CarbonFootprint"].isna().sum())

if "Similarity" in df.columns:

    print("\nSimilarity statistics:")
    print(df["Similarity"].describe())