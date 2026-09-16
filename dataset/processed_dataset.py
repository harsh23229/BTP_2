import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def prepare_carbon_dataset():
    input_file = RAW_DIR / "SustainableFoodDB_CarbonFootprints.csv"
    output_file = PROCESSED_DIR / "SustainableFoodDB_CarbonFootprints.csv"

    df = pd.read_csv(input_file)

    columns = [
        "Ingredient",
        "SueatableIngredient",
        "Similarity",
        "CarbonFootprint",
        "Count"
    ]

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[columns].copy()
    df["Ingredient"] = df["Ingredient"].astype("string").str.strip()
    df["SueatableIngredient"] = df["SueatableIngredient"].astype("string").str.strip()

    for col in ["Similarity", "CarbonFootprint", "Count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all").reset_index(drop=True)
    df.to_csv(output_file, index=False)

    print(f"Carbon dataset: {df.shape}")
    print(f"Saved to: {output_file}")


def prepare_flavor_entities():
    input_file = RAW_DIR / "FlavorDB_entities.csv"
    output_file = PROCESSED_DIR / "FlavorDB_entities.csv"

    df = pd.read_csv(input_file)

    columns = [
        "entity_id",
        "category",
        "category_readable",
        "entity_alias",
        "entity_alias_readable",
        "entity_alias_synonyms"
    ]

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[columns].copy()
    df = df.drop_duplicates("entity_id").reset_index(drop=True)
    df.to_csv(output_file, index=False)

    print(f"FlavorDB entities: {df.shape}")
    print(f"Saved to: {output_file}")


def prepare_flavor_molecules():
    input_file = RAW_DIR / "FlavorDB_molecules.csv"
    output_file = PROCESSED_DIR / "FlavorDB_molecules.csv"

    df = pd.read_csv(input_file)

    columns = [
        "pubchem_id",
        "iupac_name",
        "common_name",
        "smile",
        "molecular_weight",
        "hbd_count",
        "hba_count",
        "num_rotatablebonds",
        "complexity",
        "topological_polor_surfacearea",
        "monoisotopic_mass",
        "exact_mass",
        "xlogp",
        "charge",
        "heavy_atom_count",
        "atom_stereo_count",
        "defined_atom_stereocenter_count",
        "undefined_atom_stereocenter_count",
        "bond_stereo_count",
        "defined_bond_stereocenter_count",
        "undefined_bond_stereocenter_count",
        "isotope_atom_count",
        "covalently_bonded_unit_count"
    ]

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[columns].copy()
    df = df.drop_duplicates("pubchem_id").reset_index(drop=True)

    numeric = [col for col in columns if col not in [
        "pubchem_id", "iupac_name", "common_name", "smile"
    ]]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(output_file, index=False)

    print(f"FlavorDB molecules: {df.shape}")
    print(f"Saved to: {output_file}")


def prepare_flavor_links():
    input_file = RAW_DIR / "FlavorDB_entity_molecules.csv"
    output_file = PROCESSED_DIR / "FlavorDB_entity_molecules.csv"

    df = pd.read_csv(input_file)

    columns = ["entity_id", "pubchem_id"]

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[columns].drop_duplicates().reset_index(drop=True)
    df.to_csv(output_file, index=False)

    print(f"FlavorDB links: {df.shape}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    prepare_carbon_dataset()
    prepare_flavor_entities()
    prepare_flavor_molecules()
    prepare_flavor_links()