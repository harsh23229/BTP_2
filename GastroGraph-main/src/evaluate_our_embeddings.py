"""
evaluate_our_embeddings.py

Sanity-check the learned GastroGraph ingredient embeddings.

For selected ingredients:
    ingredient
        ↓
    64-D GraphSAGE embedding
        ↓
    cosine similarity
        ↓
    nearest ingredient neighbors

This is an embedding sanity check, NOT the substitution algorithm.
"""

import os
import pickle
import numpy as np
import networkx as nx
import pandas as pd


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CUSTOM_ROOT = os.path.join(
    PROJECT_ROOT,
    "..",
    "GastroGraph-custom"
)

MODEL_DIR = os.path.join(
    CUSTOM_ROOT,
    "models"
)

GRAPH_FILE = os.path.join(
    MODEL_DIR,
    "gastro_graph.gpickle"
)

MAPPING_FILE = os.path.join(
    MODEL_DIR,
    "node2idx.pkl"
)

EMBEDDINGS_FILE = os.path.join(
    MODEL_DIR,
    "node_embeddings.npy"
)

OUTPUT_DIR = os.path.join(
    CUSTOM_ROOT,
    "data"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "embedding_nearest_neighbors.csv"
)

TOP_K = 10


IMPORTANT_INGREDIENTS = [
    "beef",
    "chicken",
    "pork",
    "lamb",
    "fish",
    "shrimp",
    "prawn",
    "lobster",
    "mushroom",
    "onion",
    "garlic",
    "tomato",
    "potato",
    "carrot",
    "olive oil",
    "butter",
    "milk",
    "egg",
    "black pepper",
    "salt",
    "lemon",
    "rice",
    "soybean",
]


# ================================================================
# 1. LOAD GRAPH
# ================================================================

print("=" * 70)
print("OUR GASTROGRAPH — EMBEDDING SANITY CHECK")
print("=" * 70)

print("\n" + "=" * 70)
print("1. LOADING GRAPH")
print("=" * 70)

with open(GRAPH_FILE, "rb") as f:
    G = pickle.load(f)

print(f"Nodes : {G.number_of_nodes():,}")
print(f"Edges : {G.number_of_edges():,}")


# ================================================================
# 2. LOAD NODE MAPPING
# ================================================================

print("\n" + "=" * 70)
print("2. LOADING NODE MAPPING")
print("=" * 70)

with open(MAPPING_FILE, "rb") as f:
    mapping_data = pickle.load(f)

node2idx = mapping_data["node2idx"]
nodes = mapping_data["nodes"]

idx2node = {
    idx: node
    for node, idx in node2idx.items()
}

print(f"Mapping nodes : {len(node2idx):,}")


# ================================================================
# 3. LOAD EMBEDDINGS
# ================================================================

print("\n" + "=" * 70)
print("3. LOADING EMBEDDINGS")
print("=" * 70)

embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embedding shape : {embeddings.shape}")

if embeddings.shape[0] != len(nodes):
    raise ValueError(
        "Embedding row count does not match number of graph nodes."
    )

print("Embedding count matches graph nodes.")


# ================================================================
# 4. IDENTIFY INGREDIENT NODES
# ================================================================

print("\n" + "=" * 70)
print("4. IDENTIFYING INGREDIENT NODES")
print("=" * 70)

ingredient_nodes = []

for node in G.nodes():

    node_data = G.nodes[node]

    if node_data.get("node_type") == "ingredient":
        ingredient_nodes.append(node)

print(f"Ingredient nodes : {len(ingredient_nodes):,}")

ingredient_indices = []

for node in ingredient_nodes:

    if node not in node2idx:
        continue

    ingredient_indices.append(
        node2idx[node]
    )

ingredient_indices = np.array(
    ingredient_indices,
    dtype=np.int64
)

print(
    f"Ingredient embedding rows : "
    f"{len(ingredient_indices):,}"
)


# ================================================================
# 5. BUILD INGREDIENT NAME LOOKUP
# ================================================================

print("\n" + "=" * 70)
print("5. BUILDING INGREDIENT LOOKUP")
print("=" * 70)

ingredient_lookup = {}

for node in ingredient_nodes:

    node_data = G.nodes[node]

    name = node_data.get("name")

    if name is None:
        name = str(node)

    name = str(name).strip().lower()

    ingredient_lookup[name] = node

print(
    f"Ingredient names available : "
    f"{len(ingredient_lookup):,}"
)


# ================================================================
# 6. NORMALIZE EMBEDDINGS
# ================================================================

print("\n" + "=" * 70)
print("6. NORMALIZING EMBEDDINGS")
print("=" * 70)

ingredient_embeddings = embeddings[
    ingredient_indices
].astype(np.float32)

norms = np.linalg.norm(
    ingredient_embeddings,
    axis=1,
    keepdims=True
)

norms[norms == 0] = 1.0

ingredient_embeddings_normalized = (
    ingredient_embeddings / norms
)

print(
    "Ingredient embeddings normalized "
    "for cosine similarity."
)


# ================================================================
# 7. FIND NEAREST NEIGHBORS
# ================================================================

print("\n" + "=" * 70)
print("7. NEAREST INGREDIENT NEIGHBORS")
print("=" * 70)

results = []

for ingredient in IMPORTANT_INGREDIENTS:

    ingredient = ingredient.lower().strip()

    if ingredient not in ingredient_lookup:

        print(
            f"\n{ingredient:20s} | NOT FOUND"
        )

        continue

    node = ingredient_lookup[ingredient]

    query_idx = node2idx[node]

    query_embedding = embeddings[
        query_idx
    ].astype(np.float32)

    query_norm = np.linalg.norm(
        query_embedding
    )

    if query_norm == 0:
        print(
            f"\n{ingredient:20s} | ZERO EMBEDDING"
        )
        continue

    query_embedding = (
        query_embedding / query_norm
    )

    similarities = np.dot(
        ingredient_embeddings_normalized,
        query_embedding
    )

    ranking = np.argsort(
        similarities
    )[::-1]

    neighbors_found = 0

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"{ingredient.upper()}"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        f"{'Rank':<6}"
        f"{'Ingredient':<25}"
        f"{'Similarity':>12}"
    )

    print("-" * 45)

    for rank, local_idx in enumerate(
        ranking,
        start=1
    ):

        neighbor_node = ingredient_nodes[
            local_idx
        ]

        neighbor_data = G.nodes[
            neighbor_node
        ]

        neighbor_name = neighbor_data.get(
            "name"
        )

        if neighbor_name is None:
            neighbor_name = str(
                neighbor_node
            )

        neighbor_name = str(
            neighbor_name
        ).strip()

        # Don't return the ingredient itself.
        if neighbor_name.lower() == ingredient:
            continue

        similarity = float(
            similarities[local_idx]
        )

        print(
            f"{neighbors_found + 1:<6}"
            f"{neighbor_name:<25}"
            f"{similarity:>12.4f}"
        )

        results.append(
            {
                "query_ingredient": ingredient,
                "neighbor_ingredient": neighbor_name,
                "rank": neighbors_found + 1,
                "cosine_similarity": similarity,
            }
        )

        neighbors_found += 1

        if neighbors_found >= TOP_K:
            break


# ================================================================
# 8. SAVE RESULTS
# ================================================================

print("\n" + "=" * 70)
print("8. SAVING RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Saved to:\n{OUTPUT_FILE}"
)

print(
    f"\nTotal neighbor relationships saved: "
    f"{len(results_df):,}"
)


# ================================================================
# 9. BASIC SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("9. SUMMARY")
print("=" * 70)

if not results_df.empty:

    print(
        f"Queries evaluated : "
        f"{results_df['query_ingredient'].nunique()}"
    )

    print(
        f"Neighbors per query : "
        f"{TOP_K}"
    )

    print(
        f"Mean cosine similarity : "
        f"{results_df['cosine_similarity'].mean():.4f}"
    )

    print(
        f"Median cosine similarity : "
        f"{results_df['cosine_similarity'].median():.4f}"
    )

print("\n" + "=" * 70)
print("EMBEDDING SANITY CHECK COMPLETE")
print("=" * 70)

print(
    "\nImportant:"
    "\nThese nearest neighbors are NOT substitution recommendations."
    "\nThey only show what the trained graph considers structurally similar."
)