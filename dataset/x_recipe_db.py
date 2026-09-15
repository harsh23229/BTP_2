import re, json, time
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, pandas as pd
from bs4 import BeautifulSoup

BASE = "https://cosylab.iiitd.edu.in"
SEARCH_URL = f"{BASE}/recipedb/search_recipe"
OUT = Path(__file__).resolve().parent / "RecipeDB_full"
OUT.mkdir(parents=True, exist_ok=True)

WORKERS = 8
BATCH_SIZE = 500
RETRIES = 3

def text(x):
    return x.get_text(" ", strip=True) if x else ""

def search(page):
    r = requests.post(
        SEARCH_URL,
        data={
            "page": str(page),
            "autocomplete_region": "",
            "autocomplete_cuisine": "",
            "autocomplete_recipe": ""
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60
    )
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    out = {}

    for a in soup.select('a[href*="/recipedb/search_recipeInfo/"]'):
        m = re.search(r"/search_recipeInfo/(\d+)", a.get("href", ""))
        if m:
            rid = m.group(1)
            out[rid] = {
                "recipe_id": rid,
                "search_title": text(a),
                "url": urljoin(BASE, a["href"])
            }

    return list(out.values())

def table_data(table):
    rows = []

    for tr in table.select("tr"):
        cells = [text(x) for x in tr.select("th,td")]
        if cells:
            rows.append(cells)

    return rows

def detail(r):
    last_error = None

    for attempt in range(1, RETRIES + 1):
        try:
            s = requests.Session()
            s.headers.update({"User-Agent": "Mozilla/5.0"})

            x = s.get(r["url"], timeout=60)
            x.raise_for_status()

            soup = BeautifulSoup(x.text, "html.parser")
            rid = r["recipe_id"]

            title = text(soup.select_one("h3")) or r["search_title"]

            data = {
                "recipe_id": rid,
                "recipe_title": title,
                "url": r["url"],
                "image": None,
                "overview": "",
                "nutrition": [],
                "ingredients": [],
                "processes": [],
                "utensils": [],
                "steps": [],
                "links": []
            }

            img = soup.select_one("img")
            if img:
                data["image"] = urljoin(BASE, img.get("src", ""))

            info = soup.select_one("#recipe_info")
            if info:
                data["overview"] = text(info)

            for table in soup.find_all("table"):
                rows = table_data(table)

                if not rows:
                    continue

                header = [x.lower() for x in rows[0]]

                if header[:2] == ["nutrient", "quantity"]:
                    for row in rows[1:]:
                        if len(row) >= 2:
                            data["nutrition"].append({
                                "nutrient": row[0],
                                "quantity": row[1]
                            })

                elif "ingredient name" in header:
                    for row in rows[1:]:
                        row += [""] * max(0, 8 - len(row))

                        data["ingredients"].append({
                            "ingredient_name": row[0],
                            "quantity": row[1],
                            "unit": row[2],
                            "state": row[3],
                            "energy_kcal": row[4],
                            "carbohydrates": row[5],
                            "protein_g": row[6],
                            "fat_g": row[7]
                        })

            sections = {
                "earlyStageP": "early",
                "midStageP": "middle",
                "endStageP": "late",
                "utensils": "utensil",
                "steps": "step"
            }

            for sid, kind in sections.items():
                node = soup.select_one(f"#{sid}")

                if not node:
                    continue

                if kind == "utensil":
                    for a in node.select("a"):
                        v = text(a)
                        if v:
                            data["utensils"].append(v)

                elif kind == "step":
                    for li in node.select("li"):
                        v = text(li)
                        if v:
                            data["steps"].append(v)

                    if not data["steps"]:
                        v = text(node)
                        if v:
                            data["steps"].append(v)

                else:
                    for a in node.select("a"):
                        v = text(a)
                        if v:
                            data["processes"].append({
                                "stage": kind,
                                "process": v
                            })

                    if not data["processes"]:
                        v = text(node)
                        if v:
                            data["processes"].append({
                                "stage": kind,
                                "process": v
                            })

            for a in soup.select("a[href]"):
                href = urljoin(BASE, a["href"])
                label = text(a)

                if label or href:
                    data["links"].append({
                        "text": label,
                        "url": href
                    })

            return data

        except Exception as e:
            last_error = e

            if attempt < RETRIES:
                time.sleep(1)

    raise last_error

# ---------------------------------------------------------
# 1. Discover all recipes
# ---------------------------------------------------------

print("Discovering RecipeDB recipes...\n")

all_recipes = {}
page = 1

while True:
    print(f"Searching page {page}...")

    try:
        found = search(page)
    except Exception as e:
        print(f"ERROR searching page {page}: {e}")
        print("Stopping discovery. Rerun the script to continue.")
        break

    new = 0

    for r in found:
        if r["recipe_id"] not in all_recipes:
            all_recipes[r["recipe_id"]] = r
            new += 1

    print(f"  found={len(found)}, new={new}, total={len(all_recipes)}")

    if not found or new == 0:
        break

    page += 1

print(f"\nTotal recipes discovered: {len(all_recipes)}")

# ---------------------------------------------------------
# 2. Resume from existing JSONL
# ---------------------------------------------------------

JSONL = OUT / "recipe_pages.jsonl"
FAILED = OUT / "failed_recipe_ids.txt"

processed = set()

if JSONL.exists():
    print("\nChecking existing progress...")

    with open(JSONL, "r", encoding="utf-8") as jf:
        for line in jf:
            try:
                d = json.loads(line)
                rid = str(d.get("recipe_id", ""))

                if rid:
                    processed.add(rid)
            except json.JSONDecodeError:
                continue

print(f"Already downloaded: {len(processed)}")

remaining = [
    r for r in all_recipes.values()
    if r["recipe_id"] not in processed
]

print(f"Remaining: {len(remaining)}")

if not remaining:
    print("\nAll recipes already downloaded.")
else:
    # -----------------------------------------------------
    # 3. Download in batches
    # -----------------------------------------------------

    mode = "a" if JSONL.exists() else "w"

    with open(JSONL, mode, encoding="utf-8") as jf:

        for batch_start in range(0, len(remaining), BATCH_SIZE):

            batch = remaining[batch_start:batch_start + BATCH_SIZE]

            batch_no = batch_start // BATCH_SIZE + 1
            total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

            print(
                f"\nBatch {batch_no}/{total_batches} "
                f"| Recipes {batch_start + 1}-{batch_start + len(batch)}"
            )

            success = 0
            failed = []

            with ThreadPoolExecutor(max_workers=WORKERS) as executor:

                futures = {
                    executor.submit(detail, r): r
                    for r in batch
                }

                for future in as_completed(futures):

                    r = futures[future]

                    try:
                        d = future.result()

                        jf.write(
                            json.dumps(d, ensure_ascii=False) + "\n"
                        )
                        jf.flush()

                        success += 1
                        processed.add(r["recipe_id"])

                    except Exception as e:
                        failed.append(r["recipe_id"])
                        print(
                            f"ERROR {r['recipe_id']}: {e}"
                        )

            print(
                f"Batch complete | "
                f"Success: {success} | "
                f"Failed: {len(failed)} | "
                f"Total saved: {len(processed)}"
            )

            if failed:
                with open(FAILED, "a", encoding="utf-8") as f:
                    for rid in failed:
                        f.write(f"{rid}\n")

# ---------------------------------------------------------
# 4. Rebuild CSV files from JSONL
# ---------------------------------------------------------

print("\nBuilding CSV files from recipe_pages.jsonl...")

recipes = []
nutrition = []
ingredients = []
processes = []
utensils = []
steps = []
links = []

with open(JSONL, "r", encoding="utf-8") as jf:

    for line in jf:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        rid = d["recipe_id"]
        title = d["recipe_title"]

        recipes.append({
            "recipe_id": rid,
            "recipe_title": title,
            "url": d["url"],
            "image": d["image"],
            "overview": d["overview"]
        })

        for n in d["nutrition"]:
            nutrition.append({
                "recipe_id": rid,
                "recipe_title": title,
                **n
            })

        for ing in d["ingredients"]:
            ingredients.append({
                "recipe_id": rid,
                "recipe_title": title,
                **ing
            })

        for p in d["processes"]:
            processes.append({
                "recipe_id": rid,
                "recipe_title": title,
                **p
            })

        for u in d["utensils"]:
            utensils.append({
                "recipe_id": rid,
                "recipe_title": title,
                "utensil": u
            })

        for st in d["steps"]:
            steps.append({
                "recipe_id": rid,
                "recipe_title": title,
                "step": st
            })

        for l in d["links"]:
            links.append({
                "recipe_id": rid,
                "recipe_title": title,
                **l
            })

pd.DataFrame(recipes).drop_duplicates(
    "recipe_id"
).to_csv(
    OUT / "recipes.csv",
    index=False
)

pd.DataFrame(nutrition).to_csv(
    OUT / "recipe_nutrition.csv",
    index=False
)

pd.DataFrame(ingredients).to_csv(
    OUT / "recipe_ingredients.csv",
    index=False
)

pd.DataFrame(processes).to_csv(
    OUT / "recipe_processes.csv",
    index=False
)

pd.DataFrame(utensils).to_csv(
    OUT / "recipe_utensils.csv",
    index=False
)

pd.DataFrame(steps).to_csv(
    OUT / "recipe_steps.csv",
    index=False
)

pd.DataFrame(links).to_csv(
    OUT / "recipe_links.csv",
    index=False
)

print("\nDONE")
print(f"Output: {OUT.resolve()}")
print(f"Recipes: {len(recipes)}")
print(f"Nutrition rows: {len(nutrition)}")
print(f"Ingredient rows: {len(ingredients)}")
print(f"Process rows: {len(processes)}")
print(f"Utensil rows: {len(utensils)}")
print(f"Step rows: {len(steps)}")
print(f"Link rows: {len(links)}")

if FAILED.exists():
    print(f"Failed IDs: {FAILED}")