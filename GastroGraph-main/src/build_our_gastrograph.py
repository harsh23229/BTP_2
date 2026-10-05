# ================================================================
# BUILD OUR GASTROGRAPH
#
# RecipeDB
#   ├── Ingredient nodes
#   └── CO_OCCURS edges
#
# FlavorDB
#   ├── Molecule nodes
#   └── CONTAINS / ingr-dcomp edges
#
# Output:
#   GastroGraph-custom/data/
#
# IMPORTANT:
# This script only builds and validates OUR raw graph.
# It does not modify or run the original GastroGraph builder.
# ================================================================

from pathlib import Path
from itertools import combinations
from collections import Counter
import hashlib
import re

import pandas as pd
import networkx as nx


# ================================================================
# 1. PATHS
# ================================================================

# This script lives at:
#
# BTP_2/
#   GastroGraph-main/
#       src/
#           build_our_gastrograph.py
#
# Therefore:

SCRIPT_DIR = Path(__file__).resolve().parent
GASTROGRAPH_ROOT = SCRIPT_DIR.parent
BTP_ROOT = GASTROGRAPH_ROOT.parent

DATASET = BTP_ROOT / "dataset"

RECIPE_DB = DATASET / "RecipeDB_full"
PROCESSED = DATASET / "processed"

CUSTOM_ROOT = BTP_ROOT / "GastroGraph-custom"
CUSTOM_DATA = CUSTOM_ROOT / "data"
CUSTOM_MODELS = CUSTOM_ROOT / "models"

CUSTOM_DATA.mkdir(parents=True, exist_ok=True)
CUSTOM_MODELS.mkdir(parents=True, exist_ok=True)


# ================================================================
# 2. INPUT FILES
# ================================================================

RECIPE_INGREDIENTS_PATH = (
    RECIPE_DB / "recipe_ingredients.csv"
)

FLAVOR_ENTITIES_PATH = (
    PROCESSED / "FlavorDB_entities.csv"
)

FLAVOR_LINKS_PATH = (
    PROCESSED / "FlavorDB_entity_molecules.csv"
)

FLAVOR_MOLECULES_PATH = (
    PROCESSED / "FlavorDB_molecules.csv"
)

# This is produced by the FINAL eda_05_mapping notebook.
FLAVOR_MAPPING_PATH = (
    BTP_ROOT
    / "EDA"
    / "results"
    / "mapping"
    / "flavor_recipe_ingredient_mapping.csv"
)


# ================================================================
# 3. OUTPUT FILES
# ================================================================

NODES_PATH = (
    CUSTOM_DATA / "nodes_our_recipeflavor.csv"
)

EDGES_PATH = (
    CUSTOM_DATA / "edges_our_recipeflavor.csv"
)

STATS_PATH = (
    CUSTOM_DATA / "graph_statistics.txt"
)


# ================================================================
# 4. HELPERS
# ================================================================

def clean_name(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)

    return value


def make_node_id(prefix, value):
    """
    Deterministic graph node ID.

    Examples:
        ing_beef_xxxxxxxx
        mol_6275_xxxxxxxx
    """

    value = str(value)

    safe = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        value.lower()
    ).strip("_")

    digest = hashlib.md5(
        value.encode("utf-8")
    ).hexdigest()[:8]

    return f"{prefix}_{safe}_{digest}"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# 5. CHECK INPUT FILES
# ================================================================

section("CHECKING INPUT FILES")

required_files = [
    RECIPE_INGREDIENTS_PATH,
    FLAVOR_ENTITIES_PATH,
    FLAVOR_LINKS_PATH,
    FLAVOR_MOLECULES_PATH,
    FLAVOR_MAPPING_PATH,
]

for path in required_files:

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    print("OK:", path)


# ================================================================
# 6. LOAD DATA
# ================================================================

section("LOADING DATA")

recipe_ingredients = pd.read_csv(
    RECIPE_INGREDIENTS_PATH
)

flavor_entities = pd.read_csv(
    FLAVOR_ENTITIES_PATH
)

flavor_links = pd.read_csv(
    FLAVOR_LINKS_PATH
)

flavor_molecules = pd.read_csv(
    FLAVOR_MOLECULES_PATH
)

flavor_mapping = pd.read_csv(
    FLAVOR_MAPPING_PATH
)

print(
    "RecipeDB ingredient rows :",
    len(recipe_ingredients)
)

print(
    "FlavorDB entities        :",
    len(flavor_entities)
)

print(
    "FlavorDB molecule links  :",
    len(flavor_links)
)

print(
    "FlavorDB molecules       :",
    len(flavor_molecules)
)

print(
    "Controlled Flavor mappings:",
    len(flavor_mapping)
)


# ================================================================
# 7. CLEAN RECIPEDB
# ================================================================

section("PREPARING RECIPEDB")

recipe_ingredients["ingredient_name"] = (
    recipe_ingredients["ingredient_name"]
    .fillna("")
    .astype(str)
    .map(clean_name)
)

recipe_ingredients = recipe_ingredients[
    recipe_ingredients["ingredient_name"] != ""
].copy()

recipe_ingredients = recipe_ingredients[
    recipe_ingredients["recipe_id"].notna()
].copy()

recipe_ingredients["recipe_id"] = (
    recipe_ingredients["recipe_id"]
    .astype(str)
)

print(
    "Clean ingredient occurrences:",
    len(recipe_ingredients)
)

print(
    "Unique recipes:",
    recipe_ingredients["recipe_id"].nunique()
)

print(
    "Unique ingredient types:",
    recipe_ingredients["ingredient_name"].nunique()
)


# ================================================================
# 8. BUILD INGREDIENT NODES
# ================================================================

section("BUILDING INGREDIENT NODES")

ingredient_names = sorted(
    recipe_ingredients["ingredient_name"]
    .unique()
)

ingredient_frequency = (
    recipe_ingredients["ingredient_name"]
    .value_counts()
)

ingredient_nodes = []

for ingredient in ingredient_names:

    occurrence_count = int(
        ingredient_frequency.get(
            ingredient,
            0
        )
    )

    ingredient_nodes.append({
        "node_id": make_node_id(
            "ing",
            ingredient
        ),
        "name": ingredient,
        "node_type": "ingredient",

        # GastroGraph expects this field.
        # We use the same basic idea:
        # frequently occurring ingredients are hubs.
        "is_hub": occurrence_count >= 100,

        "occurrence_count": occurrence_count,
    })

ingredient_nodes_df = pd.DataFrame(
    ingredient_nodes
)

print(
    "Ingredient nodes:",
    len(ingredient_nodes_df)
)


# ================================================================
# 9. INGREDIENT -> NODE LOOKUP
# ================================================================

ingredient_to_node = dict(
    zip(
        ingredient_nodes_df["name"],
        ingredient_nodes_df["node_id"]
    )
)


# ================================================================
# 10. BUILD RECIPE-LEVEL INGREDIENT SETS
# ================================================================

section("BUILDING RECIPE INGREDIENT SETS")

recipe_ingredient_sets = (
    recipe_ingredients
    .groupby("recipe_id")["ingredient_name"]
    .apply(lambda x: sorted(set(x)))
)

print(
    "Recipes represented:",
    len(recipe_ingredient_sets)
)


# ================================================================
# 11. BUILD CO-OCCURRENCE COUNTS
# ================================================================

section("BUILDING CO_OCCURS EDGES")

cooccurrence_counter = Counter()

for ingredients in recipe_ingredient_sets:

    if len(ingredients) < 2:
        continue

    for ingredient_a, ingredient_b in combinations(
        ingredients,
        2
    ):

        cooccurrence_counter[
            (ingredient_a, ingredient_b)
        ] += 1

print(
    "Unique ingredient pairs:",
    len(cooccurrence_counter)
)


# ================================================================
# 12. CONSTRUCT CO_OCCURS EDGES
# ================================================================

cooccurrence_edges = []

for (
    ingredient_a,
    ingredient_b
), count in cooccurrence_counter.items():

    node_a = ingredient_to_node.get(
        ingredient_a
    )

    node_b = ingredient_to_node.get(
        ingredient_b
    )

    if node_a is None or node_b is None:
        continue

    cooccurrence_edges.append({
        "id_1": node_a,
        "id_2": node_b,

        # IMPORTANT:
        # This is the schema used by original GastroGraph.
        "edge_type": "ingr-ingr",

        # Number of recipes containing both.
        "score": float(count),
    })

cooccurrence_edges_df = pd.DataFrame(
    cooccurrence_edges
)

print(
    "CO_OCCURS edges:",
    len(cooccurrence_edges_df)
)


# ================================================================
# 13. BUILD FLAVORDB ENTITY LOOKUP
# ================================================================

section("BUILDING FLAVORDB ENTITY LOOKUP")

flavor_entities["entity_alias_key"] = (
    flavor_entities["entity_alias"]
    .map(clean_name)
)

flavor_entities = flavor_entities[
    flavor_entities["entity_alias_key"] != ""
].copy()

entity_name_to_ids = {}

for _, row in flavor_entities.iterrows():

    name = row["entity_alias_key"]

    entity_id = row["entity_id"]

    entity_name_to_ids.setdefault(
        name,
        []
    ).append(entity_id)

print(
    "Unique FlavorDB entity names:",
    len(entity_name_to_ids)
)


# ================================================================
# 14. RESOLVE OUR CONTROLLED MAPPINGS
# ================================================================

section("RESOLVING RECIPEDB -> FLAVORDB")

flavor_mapping["recipe_ingredient"] = (
    flavor_mapping["recipe_ingredient"]
    .map(clean_name)
)

flavor_mapping["mapped_entity"] = (
    flavor_mapping["mapped_entity"]
    .map(clean_name)
)

recipe_entity_rows = []

for _, row in flavor_mapping.iterrows():

    recipe_ingredient = (
        row["recipe_ingredient"]
    )

    mapped_entity = (
        row["mapped_entity"]
    )

    entity_ids = entity_name_to_ids.get(
        mapped_entity,
        []
    )

    for entity_id in entity_ids:

        recipe_entity_rows.append({
            "recipe_ingredient": recipe_ingredient,
            "mapped_entity": mapped_entity,
            "entity_id": str(entity_id),
            "mapping_type": row["mapping_type"],
        })

recipe_entity_df = pd.DataFrame(
    recipe_entity_rows
)

print(
    "Recipe ingredient -> entity mappings:",
    len(recipe_entity_df)
)

print(
    "Recipe ingredients with FlavorDB entities:",
    recipe_entity_df[
        "recipe_ingredient"
    ].nunique()
)


# ================================================================
# 15. BUILD INGREDIENT -> MOLECULE LINKS
# ================================================================

section("BUILDING INGREDIENT -> MOLECULE LINKS")

flavor_links["entity_id"] = (
    flavor_links["entity_id"]
    .astype(str)
)

flavor_links["pubchem_id"] = (
    flavor_links["pubchem_id"]
    .astype(str)
)

recipe_molecules = (
    recipe_entity_df
    .merge(
        flavor_links[
            [
                "entity_id",
                "pubchem_id"
            ]
        ].drop_duplicates(),
        on="entity_id",
        how="inner"
    )
)

recipe_molecules = (
    recipe_molecules[
        [
            "recipe_ingredient",
            "mapped_entity",
            "entity_id",
            "pubchem_id",
            "mapping_type",
        ]
    ]
    .drop_duplicates()
)

print(
    "Ingredient-molecule links:",
    len(recipe_molecules)
)

print(
    "Ingredients with molecules:",
    recipe_molecules[
        "recipe_ingredient"
    ].nunique()
)

print(
    "Unique molecules:",
    recipe_molecules[
        "pubchem_id"
    ].nunique()
)


# ================================================================
# 16. BUILD MOLECULE NODES
# ================================================================

section("BUILDING MOLECULE NODES")

molecule_ids = sorted(
    recipe_molecules[
        "pubchem_id"
    ].unique()
)

molecule_metadata = (
    flavor_molecules.copy()
)

molecule_metadata["pubchem_id"] = (
    molecule_metadata["pubchem_id"]
    .astype(str)
)

molecule_metadata = (
    molecule_metadata
    .drop_duplicates("pubchem_id")
    .set_index("pubchem_id")
)

molecule_nodes = []

for pubchem_id in molecule_ids:

    common_name = ""

    if pubchem_id in molecule_metadata.index:

        value = molecule_metadata.loc[
            pubchem_id,
            "common_name"
        ]

        if pd.notna(value):
            common_name = str(value)

    molecule_nodes.append({
        "node_id": make_node_id(
            "mol",
            pubchem_id
        ),

        "name": (
            common_name
            if common_name
            else f"pubchem_{pubchem_id}"
        ),

        # Original GastroGraph uses
        # node_type = compound.
        "node_type": "compound",

        "is_hub": False,

        "pubchem_id": pubchem_id,
    })

molecule_nodes_df = pd.DataFrame(
    molecule_nodes
)

print(
    "Molecule nodes:",
    len(molecule_nodes_df)
)


# ================================================================
# 17. MOLECULE -> NODE LOOKUP
# ================================================================

molecule_to_node = dict(
    zip(
        molecule_nodes_df["pubchem_id"],
        molecule_nodes_df["node_id"]
    )
)


# ================================================================
# 18. BUILD CONTAINS / CHEMISTRY EDGES
# ================================================================

section("BUILDING CONTAINS EDGES")

ingredient_molecule_edges = []

for _, row in recipe_molecules.iterrows():

    ingredient = (
        row["recipe_ingredient"]
    )

    pubchem_id = (
        row["pubchem_id"]
    )

    ingredient_node = ingredient_to_node.get(
        ingredient
    )

    molecule_node = molecule_to_node.get(
        pubchem_id
    )

    if (
        ingredient_node is None
        or molecule_node is None
    ):
        continue

    ingredient_molecule_edges.append({
        "id_1": ingredient_node,
        "id_2": molecule_node,

        # IMPORTANT:
        # Original GastroGraph accepts this
        # and converts it to CONTAINS.
        "edge_type": "ingr-dcomp",

        # Original build_graph.py subsequently
        # replaces chemistry-edge weights with IDF.
        "score": 1.0,
    })

ingredient_molecule_edges_df = (
    pd.DataFrame(
        ingredient_molecule_edges
    )
    .drop_duplicates(
        subset=[
            "id_1",
            "id_2",
            "edge_type"
        ]
    )
)

print(
    "CONTAINS edges:",
    len(ingredient_molecule_edges_df)
)


# ================================================================
# 19. COMBINE NODES
# ================================================================

section("COMBINING GRAPH")

nodes_df = pd.concat(
    [
        ingredient_nodes_df[
            [
                "node_id",
                "name",
                "node_type",
                "is_hub",
            ]
        ],

        molecule_nodes_df[
            [
                "node_id",
                "name",
                "node_type",
                "is_hub",
            ]
        ],
    ],
    ignore_index=True
)


# ================================================================
# 20. COMBINE EDGES
# ================================================================

edges_df = pd.concat(
    [
        cooccurrence_edges_df,
        ingredient_molecule_edges_df,
    ],
    ignore_index=True
)


# ================================================================
# 21. BASIC GRAPH STATISTICS
# ================================================================

print(
    "\nNodes:"
)

print(
    nodes_df["node_type"]
    .value_counts()
)

print(
    "\nEdges:"
)

print(
    edges_df["edge_type"]
    .value_counts()
)

print(
    "\nTOTAL NODES:",
    len(nodes_df)
)

print(
    "TOTAL EDGES:",
    len(edges_df)
)


# ================================================================
# 22. STRUCTURAL VALIDATION
# ================================================================

section("STRUCTURAL VALIDATION")

node_ids = set(
    nodes_df["node_id"]
)

missing_sources = (
    ~edges_df["id_1"].isin(node_ids)
).sum()

missing_targets = (
    ~edges_df["id_2"].isin(node_ids)
).sum()

self_loops = (
    edges_df["id_1"]
    == edges_df["id_2"]
).sum()

duplicate_edges = (
    edges_df
    .duplicated(
        subset=[
            "id_1",
            "id_2",
            "edge_type"
        ]
    )
    .sum()
)

print(
    "Missing source nodes:",
    missing_sources
)

print(
    "Missing target nodes:",
    missing_targets
)

print(
    "Self loops:",
    self_loops
)

print(
    "Duplicate edges:",
    duplicate_edges
)

assert missing_sources == 0
assert missing_targets == 0
assert self_loops == 0
assert duplicate_edges == 0


# ================================================================
# 23. BUILD NETWORKX GRAPH FOR VALIDATION
# ================================================================

section("GRAPH CONNECTIVITY")

G = nx.Graph()

for _, row in nodes_df.iterrows():

    G.add_node(
        row["node_id"],
        name=row["name"],
        node_type=row["node_type"],
    )

for _, row in edges_df.iterrows():

    G.add_edge(
        row["id_1"],
        row["id_2"],
        edge_type=row["edge_type"],
        weight=row["score"],
    )

print(
    "NetworkX nodes:",
    G.number_of_nodes()
)

print(
    "NetworkX edges:",
    G.number_of_edges()
)

components = list(
    nx.connected_components(G)
)

components.sort(
    key=len,
    reverse=True
)

print(
    "Connected components:",
    len(components)
)

print(
    "Largest component:",
    len(components[0])
    if components
    else 0
)

print(
    "Smallest component:",
    len(components[-1])
    if components
    else 0
)


# ================================================================
# 24. DEGREE STATISTICS
# ================================================================

section("DEGREE STATISTICS")

degree_series = pd.Series(
    dict(G.degree()),
    name="degree"
)

print(
    degree_series.describe().round(2)
)

ingredient_node_ids = set(
    ingredient_nodes_df["node_id"]
)

ingredient_degrees = (
    degree_series[
        degree_series.index.isin(
            ingredient_node_ids
        )
    ]
)

isolated_ingredients = (
    ingredient_degrees == 0
).sum()

print(
    "\nIsolated ingredient nodes:",
    isolated_ingredients
)


# ================================================================
# 25. MOLECULAR COVERAGE CHECK
# ================================================================

section("MOLECULAR COVERAGE")

ingredients_with_molecules = set(
    recipe_molecules[
        "recipe_ingredient"
    ]
)

all_ingredients = set(
    ingredient_names
)

ingredients_without_molecules = (
    all_ingredients
    - ingredients_with_molecules
)

print(
    "Total ingredient nodes:",
    len(all_ingredients)
)

print(
    "Ingredients with molecules:",
    len(ingredients_with_molecules)
)

print(
    "Ingredients without molecules:",
    len(ingredients_without_molecules)
)

print(
    "Molecule coverage:",
    round(
        100
        * len(ingredients_with_molecules)
        / len(all_ingredients),
        2
    ),
    "%"
)


# ================================================================
# 26. MANUAL SANITY CHECKS
# ================================================================

section("MANUAL SANITY CHECKS")

test_ingredients = [
    "beef",
    "onion",
    "garlic",
    "tomato",
    "black pepper",
    "olive oil",
]

print("\nIngredient node checks:")

for ingredient in test_ingredients:

    node = ingredient_to_node.get(
        ingredient
    )

    if node is None:

        print(
            f"  {ingredient:<20} NOT FOUND"
        )

        continue

    degree = G.degree(node)

    molecule_neighbors = [
        G.nodes[n]["name"]
        for n in G.neighbors(node)
        if G.nodes[n]["node_type"]
        == "compound"
    ]

    print(
        f"  {ingredient:<20}"
        f" node={node}"
        f" degree={degree}"
        f" molecules={len(molecule_neighbors)}"
    )


# ================================================================
# 27. MANUAL CO-OCCURRENCE CHECKS
# ================================================================

print("\nCo-occurrence checks:")

test_pairs = [
    ("beef", "onion"),
    ("onion", "garlic"),
    ("tomato", "onion"),
    ("olive oil", "onion"),
]

for ingredient_a, ingredient_b in test_pairs:

    node_a = ingredient_to_node.get(
        ingredient_a
    )

    node_b = ingredient_to_node.get(
        ingredient_b
    )

    if node_a is None or node_b is None:

        print(
            f"  {ingredient_a} <-> {ingredient_b}"
            " : ingredient not found"
        )

        continue

    edge_data = G.get_edge_data(
        node_a,
        node_b
    )

    if edge_data is None:

        print(
            f"  {ingredient_a} <-> {ingredient_b}"
            " : NO CO_OCCURRENCE"
        )

    else:

        print(
            f"  {ingredient_a} <-> {ingredient_b}"
            f" : {edge_data['weight']:.0f} recipes"
        )


# ================================================================
# 28. SAVE RAW GRAPH
# ================================================================

section("SAVING GRAPH")

nodes_df.to_csv(
    NODES_PATH,
    index=False
)

edges_df.to_csv(
    EDGES_PATH,
    index=False
)

print(
    "Nodes saved to:"
)

print(
    NODES_PATH
)

print(
    "\nEdges saved to:"
)

print(
    EDGES_PATH
)


# ================================================================
# 29. SAVE GRAPH STATISTICS
# ================================================================

ingredient_count = int(
    (
        nodes_df["node_type"]
        == "ingredient"
    ).sum()
)

molecule_count = int(
    (
        nodes_df["node_type"]
        == "compound"
    ).sum()
)

cooccur_count = int(
    (
        edges_df["edge_type"]
        == "ingr-ingr"
    ).sum()
)

contains_count = int(
    (
        edges_df["edge_type"]
        == "ingr-dcomp"
    ).sum()
)

stats_text = f"""
OUR GASTROGRAPH
================

INPUT DATA
----------

RecipeDB ingredient occurrences : {len(recipe_ingredients)}
RecipeDB recipes                : {recipe_ingredients["recipe_id"].nunique()}
RecipeDB ingredient types       : {len(all_ingredients)}

FlavorDB controlled mappings     : {len(flavor_mapping)}
Ingredients with molecules      : {len(ingredients_with_molecules)}
Unique molecules                : {len(molecule_ids)}

GRAPH
-----

Ingredient nodes : {ingredient_count}
Molecule nodes   : {molecule_count}
Total nodes      : {len(nodes_df)}

CO_OCCURS edges  : {cooccur_count}
CONTAINS edges   : {contains_count}
Total edges      : {len(edges_df)}

STRUCTURAL CHECKS
-----------------

Missing source nodes : {missing_sources}
Missing target nodes : {missing_targets}
Self loops           : {self_loops}
Duplicate edges      : {duplicate_edges}

CONNECTIVITY
------------

Connected components : {len(components)}
Largest component    : {len(components[0]) if components else 0}
Smallest component   : {len(components[-1]) if components else 0}

INGREDIENT COVERAGE
-------------------

Ingredients with molecules    : {len(ingredients_with_molecules)}
Ingredients without molecules : {len(ingredients_without_molecules)}
Molecule coverage             : {100 * len(ingredients_with_molecules) / len(all_ingredients):.2f}%

ISOLATED INGREDIENTS
--------------------

Isolated ingredients : {isolated_ingredients}
"""

with open(
    STATS_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        stats_text.strip()
        + "\n"
    )

print(
    "\nStatistics saved to:"
)

print(
    STATS_PATH
)


# ================================================================
# 30. FINAL
# ================================================================

section("BUILD COMPLETE")

print(
    "OUR CUSTOM GASTROGRAPH HAS BEEN BUILT."
)

print(
    "\nOriginal GastroGraph files were NOT modified."
)

print(
    "\nNext step:"
)

print(
    "Inspect the graph statistics and sanity checks"
)

print(
    "BEFORE running GastroGraph's graph-processing/training pipeline."
)