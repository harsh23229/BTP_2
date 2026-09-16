import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "dataset" / "processed"
OUT = BASE_DIR / "EDA" / "results" / "flavor"
OUT.mkdir(parents=True, exist_ok=True)

entities = pd.read_csv(DATA / "FlavorDB_entities.csv")
molecules = pd.read_csv(DATA / "FlavorDB_molecules.csv")
links = pd.read_csv(DATA / "FlavorDB_entity_molecules.csv")

print("=" * 60)
print("FLAVORDB EDA")
print("=" * 60)

print("\nEntities")
print(f"Rows: {len(entities)}")
print(f"Unique entity IDs: {entities['entity_id'].nunique()}")
print(f"Duplicate rows: {entities.duplicated().sum()}")
print(f"Duplicate entity IDs: {entities['entity_id'].duplicated().sum()}")

print("\nEntity categories:")
print(entities["category_readable"].value_counts())

print("\nEntity-Molecule links")
print(f"Rows: {len(links)}")
print(f"Unique entity-molecule pairs: {links.drop_duplicates(['entity_id', 'pubchem_id']).shape[0]}")
print(f"Duplicate pairs: {links.duplicated(['entity_id', 'pubchem_id']).sum()}")

print("\nMolecules")
print(f"Rows: {len(molecules)}")
print(f"Unique PubChem IDs: {molecules['pubchem_id'].nunique()}")
print(f"Duplicate PubChem IDs: {molecules['pubchem_id'].duplicated().sum()}")

molecules_per_entity = links.groupby("entity_id")["pubchem_id"].nunique()

print("\nMolecules per entity:")
print(molecules_per_entity.describe())

numeric = [
    "molecular_weight",
    "xlogp",
    "heavy_atom_count"
]

print("\nMolecular descriptor statistics:")
print(molecules[numeric].describe().T)

print("\nMissing values:")
print(molecules[numeric].isna().sum())

print("\nSkewness:")
print(molecules[numeric].skew())

print("\nMolecular descriptor percentiles:")
print(molecules[numeric].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))

for col in numeric:
    plt.figure(figsize=(8, 5))
    plt.hist(molecules[col].dropna(), bins=50)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(OUT / f"{col}_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.boxplot(molecules[col].dropna(), vert=False)
    plt.xlabel(col)
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(OUT / f"{col}_boxplot.png", dpi=150)
    plt.close()

molecules_per_entity.describe().to_csv(
    OUT / "molecules_per_entity_statistics.csv"
)

molecules[numeric].describe().T.assign(
    skewness=molecules[numeric].skew()
).to_csv(
    OUT / "molecular_statistics.csv"
)

print(f"\nResults saved to: {OUT}")