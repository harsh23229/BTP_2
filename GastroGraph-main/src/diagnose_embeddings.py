import os
import pickle
import numpy as np
import networkx as nx


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "GastroGraph-custom")
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "node_embeddings.npy")
NODE2IDX_PATH = os.path.join(MODEL_DIR, "node2idx.pkl")
GRAPH_PATH = os.path.join(MODEL_DIR, "gastro_graph.gpickle")

DIAGNOSTICS_PATH = os.path.join(DATA_DIR, "embedding_diagnostics.txt")
RANDOM_PAIRS_PATH = os.path.join(DATA_DIR, "embedding_random_pair_cosines.csv")
NN_STATS_PATH = os.path.join(DATA_DIR, "embedding_nearest_neighbor_statistics.csv")

os.makedirs(DATA_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def cosine_matrix_stats(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    return normalized


section("OUR GASTROGRAPH — CORRECTED EMBEDDING DIAGNOSTICS")

print("""
Purpose:
Determine whether the learned INGREDIENT embeddings are diverse
and distinguishable enough to be useful.

This version identifies ingredient nodes using the node_type attribute
stored in the GastroGraph rather than node-name heuristics.
""")


# ============================================================
# 1. LOAD EMBEDDINGS
# ============================================================

section("1. LOADING EMBEDDINGS")

embeddings = np.load(EMBEDDINGS_PATH)

print(f"Embedding matrix shape : {embeddings.shape}")
print(f"Number of nodes        : {len(embeddings):,}")
print(f"Embedding dimension    : {embeddings.shape[1]}")


# ============================================================
# 2. LOAD NODE MAPPING
# ============================================================

section("2. LOADING NODE MAPPING")

with open(NODE2IDX_PATH, "rb") as f:
    node2idx = pickle.load(f)

print(f"Mapped nodes : {len(node2idx):,}")

if len(node2idx) != len(embeddings):
    raise RuntimeError(
        "Embedding rows do not match node mapping."
    )

print("[PASS] Embedding rows match node mapping.")


# ============================================================
# 3. LOAD GRAPH AND IDENTIFY INGREDIENT NODES
# ============================================================

section("3. IDENTIFYING INGREDIENT NODES FROM GRAPH")

graph = nx.read_gpickle(GRAPH_PATH)

print(f"Graph nodes : {graph.number_of_nodes():,}")
print(f"Graph edges : {graph.number_of_edges():,}")

ingredient_nodes = []

node_type_counts = {}

for node, attrs in graph.nodes(data=True):

    node_type = attrs.get("node_type", None)

    node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

    if node_type == "ingredient":
        ingredient_nodes.append(node)


print("\nNode types found:")

for node_type, count in sorted(
    node_type_counts.items(),
    key=lambda x: str(x[0])
):
    print(f"  {node_type}: {count:,}")


print(f"\nIngredient nodes : {len(ingredient_nodes):,}")


if len(ingredient_nodes) != 5693:
    print(
        "\n[WARNING] Expected 5,693 ingredient nodes "
        "based on graph validation."
    )


# ============================================================
# 4. EXTRACT INGREDIENT EMBEDDINGS
# ============================================================

section("4. EXTRACTING INGREDIENT EMBEDDINGS")

ingredient_indices = []

for node in ingredient_nodes:

    if node not in node2idx:
        raise RuntimeError(
            f"Ingredient node missing from node2idx: {node}"
        )

    ingredient_indices.append(node2idx[node])


ingredient_indices = np.array(
    ingredient_indices,
    dtype=np.int64
)

ingredient_embeddings = embeddings[ingredient_indices]

print(
    f"Ingredient embedding matrix : "
    f"{ingredient_embeddings.shape}"
)


# ============================================================
# 5. EMBEDDING NORMS
# ============================================================

section("5. EMBEDDING NORMS")

norms = np.linalg.norm(
    ingredient_embeddings,
    axis=1
)

print(f"Mean norm   : {norms.mean():.6f}")
print(f"Median norm : {np.median(norms):.6f}")
print(f"Minimum norm: {norms.min():.6f}")
print(f"Maximum norm: {norms.max():.6f}")


# ============================================================
# 6. DIMENSION STATISTICS
# ============================================================

section("6. EMBEDDING DIMENSION STATISTICS")

dimension_means = ingredient_embeddings.mean(axis=0)
dimension_stds = ingredient_embeddings.std(axis=0)
dimension_vars = ingredient_embeddings.var(axis=0)

print(
    f"Overall mean of dimension means : "
    f"{dimension_means.mean():.8f}"
)

print(
    f"Overall std of dimension means  : "
    f"{dimension_means.std():.8f}"
)

print(
    f"Mean dimension std              : "
    f"{dimension_stds.mean():.8f}"
)

print(
    f"Minimum dimension std           : "
    f"{dimension_stds.min():.8f}"
)

print(
    f"Maximum dimension std           : "
    f"{dimension_stds.max():.8f}"
)

print(
    f"Mean dimension variance         : "
    f"{dimension_vars.mean():.8f}"
)

print(
    f"Minimum dimension variance      : "
    f"{dimension_vars.min():.8f}"
)

print(
    f"Maximum dimension variance      : "
    f"{dimension_vars.max():.8f}"
)


# ============================================================
# 7. EFFECTIVE DIMENSIONALITY
# ============================================================

section("7. EFFECTIVE DIMENSIONALITY / RANK")

X = ingredient_embeddings - ingredient_embeddings.mean(axis=0)

_, singular_values, _ = np.linalg.svd(
    X,
    full_matrices=False
)

singular_values_sq = singular_values ** 2

explained = (
    singular_values_sq /
    singular_values_sq.sum()
)

cumulative = np.cumsum(explained)

numerical_rank = np.linalg.matrix_rank(X)

effective_rank = (
    singular_values_sq.sum() ** 2 /
    (singular_values_sq ** 2).sum()
)

print(f"Embedding dimension      : {X.shape[1]}")
print(f"Numerical rank            : {numerical_rank}")
print(f"Effective rank            : {effective_rank:.4f}")

for threshold in [0.50, 0.80, 0.90, 0.95, 0.99]:

    dims = np.searchsorted(
        cumulative,
        threshold
    ) + 1

    print(
        f"Dimensions for "
        f"{int(threshold * 100)}% variance : {dims}"
    )


# ============================================================
# 8. RANDOM PAIR COSINE SIMILARITY
# ============================================================

section("8. RANDOM PAIR COSINE SIMILARITY")

rng = np.random.default_rng(42)

normalized = cosine_matrix_stats(
    ingredient_embeddings
)

n = len(normalized)

num_pairs = 100_000

i = rng.integers(
    0,
    n,
    size=num_pairs
)

j = rng.integers(
    0,
    n,
    size=num_pairs
)

# Avoid self-pairs
same = i == j

while np.any(same):

    j[same] = rng.integers(
        0,
        n,
        size=same.sum()
    )

    same = i == j


cosines = np.sum(
    normalized[i] * normalized[j],
    axis=1
)

print(f"Random pairs analyzed : {len(cosines):,}")

print(
    f"Mean cosine similarity   : "
    f"{cosines.mean():.6f}"
)

print(
    f"Median cosine similarity : "
    f"{np.median(cosines):.6f}"
)

print(
    f"Minimum cosine similarity: "
    f"{cosines.min():.6f}"
)

print(
    f"Maximum cosine similarity: "
    f"{cosines.max():.6f}"
)

print(
    f"Std cosine similarity    : "
    f"{cosines.std():.6f}"
)

for threshold in [0.90, 0.95, 0.99]:

    count = np.sum(cosines >= threshold)

    print(
        f"Cosine >= {threshold:.2f}: "
        f"{count:,} "
        f"({count / len(cosines) * 100:.2f}%)"
    )


np.savetxt(
    RANDOM_PAIRS_PATH,
    cosines,
    delimiter=",",
    header="cosine_similarity",
    comments=""
)

print(
    "\nRandom-pair cosine results saved to:"
)
print(RANDOM_PAIRS_PATH)


# ============================================================
# 9. COSINE DISTRIBUTION
# ============================================================

section("9. COSINE SIMILARITY DISTRIBUTION")

bins = [
    -1.0,
    0.0,
    0.25,
    0.50,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
    1.001
]

hist, _ = np.histogram(
    cosines,
    bins=bins
)

labels = [
    "[-1.00, 0.00)",
    "[0.00, 0.25)",
    "[0.25, 0.50)",
    "[0.50, 0.70)",
    "[0.70, 0.80)",
    "[0.80, 0.90)",
    "[0.90, 0.95)",
    "[0.95, 0.99)",
    "[0.99, 1.00)"
]

for label, count in zip(labels, hist):

    print(
        f"{label} : "
        f"{count:6d} "
        f"({count / len(cosines) * 100:5.2f}%)"
    )


# ============================================================
# 10. TOP-10 NEAREST NEIGHBORS
# ============================================================

section("10. NEAREST-NEIGHBOR SIMILARITY")

# 5693 x 5693 matrix is manageable
similarity_matrix = (
    normalized @ normalized.T
)

np.fill_diagonal(
    similarity_matrix,
    -np.inf
)

top_k = 10

top10 = np.partition(
    similarity_matrix,
    -top_k,
    axis=1
)[:, -top_k:]

top10 = np.sort(
    top10,
    axis=1
)[:, ::-1]

top1 = top10[:, 0]

top10_mean = top10.mean(axis=1)

print(
    f"Ingredients analyzed : "
    f"{len(ingredient_embeddings):,}"
)

print(
    f"Top-{top_k} neighbors per ingredient."
)

print(
    f"Mean top-1 similarity : "
    f"{top1.mean():.6f}"
)

print(
    f"Median top-1 similarity : "
    f"{np.median(top1):.6f}"
)

print(
    f"Mean top-{top_k} similarity : "
    f"{top10_mean.mean():.6f}"
)

print(
    f"Median top-{top_k} similarity : "
    f"{np.median(top10_mean):.6f}"
)

print(
    f"Minimum top-{top_k} similarity : "
    f"{top10_mean.min():.6f}"
)

print(
    f"Maximum top-{top_k} similarity : "
    f"{top10_mean.max():.6f}"
)


with open(NN_STATS_PATH, "w", encoding="utf-8") as f:

    f.write(
        "ingredient,top1_similarity,"
        "top5_mean_similarity,"
        "top10_mean_similarity\n"
    )

    for idx, node in enumerate(ingredient_nodes):

        f.write(
            f"{node},"
            f"{top1[idx]},"
            f"{top10[idx, :5].mean()},"
            f"{top10_mean[idx]}\n"
        )


print(
    "\nNearest-neighbor statistics saved to:"
)

print(NN_STATS_PATH)


# ============================================================
# 11. DIAGNOSTIC ASSESSMENT
# ============================================================

section("11. EMBEDDING DISTINGUISHABILITY")

high_similarity_fraction = (
    np.sum(cosines >= 0.99) /
    len(cosines)
)

print(
    f"Random-pair cosine std : "
    f"{cosines.std():.6f}"
)

print(
    f"Random pairs >= 0.90 : "
    f"{np.sum(cosines >= 0.90) / len(cosines) * 100:.2f}%"
)

print(
    f"Random pairs >= 0.95 : "
    f"{np.sum(cosines >= 0.95) / len(cosines) * 100:.2f}%"
)

print(
    f"Random pairs >= 0.99 : "
    f"{high_similarity_fraction * 100:.2f}%"
)

print(
    f"Mean top-1 similarity : "
    f"{top1.mean():.6f}"
)

print(
    f"Median top-1 similarity : "
    f"{np.median(top1):.6f}"
)


if effective_rank < 8 and top1.mean() > 0.90:

    assessment = (
        "WARNING — embeddings show strong concentration/collapse. "
        "Nearest-neighbor similarity is very high."
    )

elif high_similarity_fraction > 0.05:

    assessment = (
        "WARNING — many random ingredient pairs have extremely "
        "high cosine similarity."
    )

elif top1.mean() > 0.95:

    assessment = (
        "CAUTION — nearest neighbors are highly similar; "
        "inspect semantic quality before using embeddings."
    )

else:

    assessment = (
        "REASONABLE — embeddings show measurable variation "
        "across ingredient pairs."
    )


print("\nDIAGNOSTIC ASSESSMENT:")
print(assessment)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

section("12. FINAL SANITY SUMMARY")

print(
    f"[INFO] Ingredient embeddings : "
    f"{len(ingredient_embeddings):,}"
)

print(
    f"[INFO] Embedding dimension : "
    f"{ingredient_embeddings.shape[1]}"
)

print(
    f"[INFO] Numerical rank : "
    f"{numerical_rank}"
)

print(
    f"[INFO] Effective rank : "
    f"{effective_rank:.4f}"
)

print(
    f"[INFO] Random-pair mean cosine : "
    f"{cosines.mean():.6f}"
)

print(
    f"[INFO] Random-pair std cosine : "
    f"{cosines.std():.6f}"
)

print(
    f"[INFO] Random pairs >= 0.99 : "
    f"{high_similarity_fraction * 100:.2f}%"
)

print(
    f"[INFO] Mean top-1 cosine : "
    f"{top1.mean():.6f}"
)

print("""
[IMPORTANT]

This diagnostic tests geometric diversity of the INGREDIENT
embedding space.

It does NOT establish that embeddings represent chemical flavor
similarity or culinary substitution suitability.

The embedding should NOT be used directly as a substitution score
until the diagnostic and semantic nearest neighbors have been
inspected.
""")


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    DIAGNOSTICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("OUR GASTROGRAPH — EMBEDDING DIAGNOSTICS\n\n")

    f.write(
        f"Ingredient embeddings: "
        f"{len(ingredient_embeddings)}\n"
    )

    f.write(
        f"Embedding dimension: "
        f"{ingredient_embeddings.shape[1]}\n"
    )

    f.write(
        f"Numerical rank: "
        f"{numerical_rank}\n"
    )

    f.write(
        f"Effective rank: "
        f"{effective_rank:.4f}\n"
    )

    f.write(
        f"Random pair mean cosine: "
        f"{cosines.mean():.6f}\n"
    )

    f.write(
        f"Random pair std cosine: "
        f"{cosines.std():.6f}\n"
    )

    f.write(
        f"Random pairs >= 0.99: "
        f"{high_similarity_fraction * 100:.2f}%\n"
    )

    f.write(
        f"Mean top-1 cosine: "
        f"{top1.mean():.6f}\n"
    )

    f.write(
        f"Median top-1 cosine: "
        f"{np.median(top1):.6f}\n"
    )

    f.write(
        f"\nAssessment:\n{assessment}\n"
    )


print("\nDIAGNOSTIC REPORT SAVED")
print(DIAGNOSTICS_PATH)

print("\n" + "=" * 70)
print("CORRECTED EMBEDDING DIAGNOSTICS COMPLETE")
print("=" * 70)