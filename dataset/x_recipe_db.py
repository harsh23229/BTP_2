import re, json, time
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, pandas as pd
from bs4 import BeautifulSoup

BASE = "https://cosylab.iiitd.edu.in"
SEARCH_URL = f"{BASE}/recipedb/search_recipe"
OUT = Path(__file__).resolve().parent / "raw" / "RecipeDB_full"
OUT.mkdir(parents=True, exist_ok=True)

SEARCH_BATCH = 500
WORKERS = 8
RETRIES = 3
START_PAGE = 501

JSONL = OUT / "recipe_pages.jsonl"
FAILED = OUT / "failed_recipe_ids.txt"
DISCOVERY = OUT / "recipe_discovery.jsonl"
CHECKPOINT = OUT / "search_checkpoint.txt"

def text(x):
    return x.get_text(" ", strip=True) if x else ""

def search(page):
    r = requests.post(SEARCH_URL, data={"page": str(page), "autocomplete_region": "", "autocomplete_cuisine": "", "autocomplete_recipe": ""}, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out = {}
    for a in soup.select('a[href*="/recipedb/search_recipeInfo/"]'):
        m = re.search(r"/search_recipeInfo/(\d+)", a.get("href", ""))
        if m:
            rid = m.group(1)
            out[rid] = {"recipe_id": rid, "search_title": text(a), "url": urljoin(BASE, a["href"])}
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
            data = {"recipe_id": rid, "recipe_title": title, "url": r["url"], "image": None, "overview": "", "nutrition": [], "ingredients": [], "processes": [], "utensils": [], "steps": [], "links": []}

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
                            data["nutrition"].append({"nutrient": row[0], "quantity": row[1]})
                elif "ingredient name" in header:
                    for row in rows[1:]:
                        row += [""] * max(0, 8 - len(row))
                        data["ingredients"].append({"ingredient_name": row[0], "quantity": row[1], "unit": row[2], "state": row[3], "energy_kcal": row[4], "carbohydrates": row[5], "protein_g": row[6], "fat_g": row[7]})

            sections = {"earlyStageP": "early", "midStageP": "middle", "endStageP": "late", "utensils": "utensil", "steps": "step"}

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
                            data["processes"].append({"stage": kind, "process": v})
                    if not data["processes"]:
                        v = text(node)
                        if v:
                            data["processes"].append({"stage": kind, "process": v})

            for a in soup.select("a[href]"):
                href = urljoin(BASE, a["href"])
                label = text(a)
                if label or href:
                    data["links"].append({"text": label, "url": href})

            return data
        except Exception as e:
            last_error = e
            if attempt < RETRIES:
                time.sleep(1)
    raise last_error

# Load previous discovery
all_recipes = {}
if DISCOVERY.exists():
    print("Loading previous discovery...")
    with open(DISCOVERY, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                all_recipes[r["recipe_id"]] = r
            except json.JSONDecodeError:
                continue
    print(f"Previously discovered: {len(all_recipes)}")

# Load downloaded recipes
processed = set()
if JSONL.exists():
    print("Checking downloaded recipes...")
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                rid = str(d.get("recipe_id", ""))
                if rid:
                    processed.add(rid)
            except json.JSONDecodeError:
                continue
print(f"Already downloaded: {len(processed)}")

# Load search checkpoint
if CHECKPOINT.exists():
    try:
        page = int(CHECKPOINT.read_text().strip())
    except ValueError:
        page = START_PAGE
else:
    page = START_PAGE
    CHECKPOINT.write_text(str(page))

print(f"Starting search from page {page}")

# Search and download in blocks
while True:
    block_end = page + SEARCH_BATCH - 1
    print("\n" + "=" * 60)
    print(f"SEARCHING PAGES {page} - {block_end}")
    print("=" * 60)

    batch_recipes = []
    search_complete = True

    for p in range(page, block_end + 1):
        print(f"Searching page {p}...", end="\r", flush=True)
        try:
            found = search(p)
        except Exception as e:
            print(f"\nERROR searching page {p}: {e}")
            print("Search block incomplete. Restart to retry this block.")
            search_complete = False
            break

        if not found:
            print(f"\nNo recipes found on page {p}.")
            search_complete = False
            end_reached = True
            break

        new = 0
        for r in found:
            if r["recipe_id"] not in all_recipes:
                all_recipes[r["recipe_id"]] = r
                with open(DISCOVERY, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                new += 1
            if r["recipe_id"] not in processed:
                batch_recipes.append(r)

        print(f"Page {p} | found={len(found)} | new={new} | total discovered={len(all_recipes)}")
    else:
        end_reached = False

    if end_reached:
        break

    if not search_complete:
        break

    unique_batch = {}
    for r in batch_recipes:
        unique_batch[r["recipe_id"]] = r
    batch_recipes = list(unique_batch.values())

    print(f"\nSearch block complete.")
    print(f"Recipes discovered so far: {len(all_recipes)}")
    print(f"Recipes to download: {len(batch_recipes)}")

    success = 0
    failed = []

    if batch_recipes:
        print(f"\nDownloading {len(batch_recipes)} recipes with {WORKERS} workers...")

        with open(JSONL, "a", encoding="utf-8") as jf:
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = {executor.submit(detail, r): r for r in batch_recipes}

                for future in as_completed(futures):
                    r = futures[future]
                    try:
                        d = future.result()
                        jf.write(json.dumps(d, ensure_ascii=False) + "\n")
                        jf.flush()
                        processed.add(r["recipe_id"])
                        success += 1
                        print(f"Downloaded: {success}/{len(batch_recipes)} | Total saved: {len(processed)}", end="\r", flush=True)
                    except Exception as e:
                        failed.append(r["recipe_id"])
                        print(f"\nERROR {r['recipe_id']}: {e}")

    print(f"\nDownload block complete.")
    print(f"Success: {success}")
    print(f"Failed: {len(failed)}")
    print(f"Total saved: {len(processed)}")

    if failed:
        with open(FAILED, "a", encoding="utf-8") as f:
            for rid in failed:
                f.write(f"{rid}\n")
        print("Checkpoint NOT advanced because some recipes failed.")
        print("Restart the script to retry the missing recipes.")
        break

    next_page = block_end + 1
    CHECKPOINT.write_text(str(next_page))
    print(f"Checkpoint saved: next search page = {next_page}")
    page = next_page

# Build CSV files
print("\n" + "=" * 60)
print("BUILDING CSV FILES")
print("=" * 60)

recipes, nutrition, ingredients = [], [], []
processes, utensils, steps, links = [], [], [], []

if JSONL.exists():
    with open(JSONL, "r", encoding="utf-8") as jf:
        for line in jf:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            rid = d["recipe_id"]
            title = d["recipe_title"]

            recipes.append({"recipe_id": rid, "recipe_title": title, "url": d["url"], "image": d["image"], "overview": d["overview"]})

            for n in d["nutrition"]:
                nutrition.append({"recipe_id": rid, "recipe_title": title, **n})

            for ing in d["ingredients"]:
                ingredients.append({"recipe_id": rid, "recipe_title": title, **ing})

            for p in d["processes"]:
                processes.append({"recipe_id": rid, "recipe_title": title, **p})

            for u in d["utensils"]:
                utensils.append({"recipe_id": rid, "recipe_title": title, "utensil": u})

            for st in d["steps"]:
                steps.append({"recipe_id": rid, "recipe_title": title, "step": st})

            for l in d["links"]:
                links.append({"recipe_id": rid, "recipe_title": title, **l})

pd.DataFrame(recipes).drop_duplicates("recipe_id").to_csv(OUT / "recipes.csv", index=False)
pd.DataFrame(nutrition).to_csv(OUT / "recipe_nutrition.csv", index=False)
pd.DataFrame(ingredients).to_csv(OUT / "recipe_ingredients.csv", index=False)
pd.DataFrame(processes).to_csv(OUT / "recipe_processes.csv", index=False)
pd.DataFrame(utensils).to_csv(OUT / "recipe_utensils.csv", index=False)
pd.DataFrame(steps).to_csv(OUT / "recipe_steps.csv", index=False)
pd.DataFrame(links).to_csv(OUT / "recipe_links.csv", index=False)

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