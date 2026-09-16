import json, string
from pathlib import Path
import requests
import pandas as pd

BASE = "https://cosylab.iiitd.edu.in/flavordb"
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
RAW_JSON = RAW / "FlavorDB_raw_entity_json"
RAW_JSON.mkdir(parents=True, exist_ok=True)

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0"})

entities = {}

for letter in string.ascii_lowercase:
    try:
        r = s.get(f"{BASE}/entities", params={"entity": letter}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, str):
            data = json.loads(data)
        print(f"{letter}: {len(data)}")
        for e in data:
            entities[e["entity_id"]] = e
    except Exception as ex:
        print(f"Failed {letter}: {ex}")

print(f"Unique entities: {len(entities)}")

entity_rows, molecules, relations = [], {}, []

for n, (eid, _) in enumerate(entities.items(), 1):
    path = RAW_JSON / f"{eid}.json"
    try:
        if path.exists():
            data = json.loads(path.read_text())
        else:
            r = s.get(f"{BASE}/entities_json", params={"id": eid}, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, str):
                data = json.loads(data)
            path.write_text(json.dumps(data, ensure_ascii=False))

        entity_rows.append({
            k: data.get(k, "") for k in [
                "entity_id", "category", "category_readable",
                "entity_alias", "entity_alias_basket",
                "entity_alias_readable", "entity_alias_synonyms",
                "natural_source_name", "natural_source_url"
            ]
        })

        for m in data.get("molecules", []):
            mid = m.get("pubchem_id")
            if mid:
                molecules[mid] = m
                relations.append({"entity_id": eid, "pubchem_id": mid})

    except Exception as ex:
        print(f"Failed entity {eid}: {ex}")

    if n % 25 == 0:
        print(f"{n}/{len(entities)} processed")

pd.DataFrame(entity_rows).drop_duplicates("entity_id").to_csv(
    RAW / "FlavorDB_entities.csv", index=False
)
pd.DataFrame(molecules.values()).to_csv(
    RAW / "FlavorDB_molecules.csv", index=False
)
pd.DataFrame(relations).drop_duplicates().to_csv(
    RAW / "FlavorDB_entity_molecules.csv", index=False
)

print("\nDONE")
print(f"Entities: {len(entity_rows)}")
print(f"Molecules: {len(molecules)}")
print(f"Links: {len(set((x['entity_id'], x['pubchem_id']) for x in relations))}")