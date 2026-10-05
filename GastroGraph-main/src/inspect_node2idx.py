import os
import pickle
import numpy as np
import networkx as nx


BASE = r"C:\Users\Parth\semester7\BTP_2\BTP_2\GastroGraph-custom"
MODEL_DIR = os.path.join(BASE, "models")

GRAPH_PATH = os.path.join(MODEL_DIR, "gastro_graph.gpickle")
NODE2IDX_PATH = os.path.join(MODEL_DIR, "node2idx.pkl")
EMBEDDING_PATH = os.path.join(MODEL_DIR, "node_embeddings.npy")

OUTPUT_PATH = os.path.join(BASE, "data", "embedding_diagnostics.txt")


def cosine_similarity_matrix(a, b):
    return np.matmul(a, b.T)


print("=" * 70)
print("GASTROGRAPH EMBEDDING DIAGNOSTICS")
print("=" * 70)

# ------------------------------------------------------------
# Load files
# ------------------------------------------------------------

print("\nLoading graph...")

with open(GRAPH_PATH, "rb") as f:
    graph = pickle.load(f)

print("Graph loaded.")
print("Nodes:", graph.number_of_nodes())
print("Edges:", graph.number_of_edges())


print("\nLoading node2idx...")

with open(NODE2IDX_PATH, "rb") as f:
    node2idx = pickle.load(f)

nodes = node2idx["nodes"]

print("Node list length:", len(nodes))


print("\nLoading embeddings...")

embeddings = np.load(EMBEDDING_PATH)

print("Embedding shape:", embeddings.shape)


# ------------------------------------------------------------
# Basic consistency checks
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BASIC CONSISTENCY")
print("=" * 70)

assert len(nodes) == embeddings.shape[0], (
    f"Node count {len(nodes)} != embedding rows {embeddings.shape[0]}"
)

print("[PASS] node2idx node list matches embedding rows.")

assert len(nodes) == graph.number_of_nodes(), (
    f"Node list {len(nodes)} != graph nodes {graph.number_of_nodes()}"
)

print("[PASS] node list matches graph node count.")


graph_nodes = set(graph.nodes())

missing_from_graph = [n for n in nodes if n not in graph_nodes]

if missing_from_graph:
    print(
        f"[FAIL] {len(missing_from_graph)} nodes from node2idx "
        f"are missing from graph."
    )
else:
    print("[PASS] Every node2idx node exists in graph.")


# ------------------------------------------------------------
# Determine ingredient nodes from graph attributes
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NODE TYPES")
print("=" * 70)

node_type_counts = {}

for node, attrs in graph.nodes(data=True):
    node_type = attrs.get("node_type")

    if node_type not in node_type_counts:
        node_type_counts[node_type] = 0

    node_type_counts[node_type] += 1

for node_type, count in node_type_counts.items():
    print(f"{node_type}: {count}")


ingredient_nodes = [
    node
    for node, attrs in graph.nodes(data=True)
    if attrs.get("node_type") == "ingredient"
]

compound_nodes = [
    node
    for node, attrs in graph.nodes(data=True)
    if attrs.get("node_type") == "compound"
]

print("\nIngredient nodes:", len(ingredient_nodes))
print("Compound nodes:", len(compound_nodes))


# ------------------------------------------------------------
# Map graph nodes -> embedding rows
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NODE -> EMBEDDING ROW MAPPING")
print("=" * 70)

node_to_row = {
    node: i
    for i, node in enumerate(nodes)
}

missing_ingredients = [
    node
    for node in ingredient_nodes
    if node not in node_to_row
]

missing_compounds = [
    node
    for node in compound_nodes
    if node not in node_to_row
]

print("Ingredients missing from embedding mapping:",
      len(missing_ingredients))

print("Compounds missing from embedding mapping:",
      len(missing_compounds))

if missing_ingredients:
    print("[FAIL] Some ingredient nodes have no embedding row.")
    print("Examples:", missing_ingredients[:10])
    raise SystemExit(1)

if missing_compounds:
    print("[FAIL] Some compound nodes have no embedding row.")
    print("Examples:", missing_compounds[:10])
    raise SystemExit(1)

print("[PASS] Every ingredient has an embedding row.")
print("[PASS] Every compound has an embedding row.")


ingredient_indices = np.array(
    [node_to_row[node] for node in ingredient_nodes],
    dtype=np.int64
)

compound_indices = np.array(
    [node_to_row[node] for node in compound_nodes],
    dtype=np.int64
)

ingredient_embeddings = embeddings[ingredient_indices]
compound_embeddings = embeddings[compound_indices]

print("\nIngredient embedding shape:",
      ingredient_embeddings.shape)

print("Compound embedding shape:",
      compound_embeddings.shape)


# ------------------------------------------------------------
# Check ordering
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EMBEDDING ORDER SANITY CHECK")
print("=" * 70)

print("\nFirst 10 nodes in embedding order:")

for i in range(min(10, len(nodes))):
    node = nodes[i]
    node_type = graph.nodes[node].get("node_type")
    print(f"{i:5d} | {node_type:10s} | {node}")

print("\nLast 10 nodes in embedding order:")

start = max(0, len(nodes) - 10)

for i in range(start, len(nodes)):
    node = nodes[i]
    node_type = graph.nodes[node].get("node_type")
    print(f"{i:5d} | {node_type:10s} | {node}")


# ------------------------------------------------------------
# Norms
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INGREDIENT EMBEDDING NORMS")
print("=" * 70)

ingredient_norms = np.linalg.norm(
    ingredient_embeddings,
    axis=1
)

print("Mean   :", ingredient_norms.mean())
print("Median :", np.median(ingredient_norms))
print("Min    :", ingredient_norms.min())
print("Max    :", ingredient_norms.max())

if np.allclose(ingredient_norms, 1.0, atol=1e-5):
    print("[PASS] Ingredient embeddings are L2-normalized.")
else:
    print("[INFO] Ingredient embeddings are not exactly normalized.")


# ------------------------------------------------------------
# Dimension statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIMENSION STATISTICS")
print("=" * 70)

dim_means = ingredient_embeddings.mean(axis=0)
dim_stds = ingredient_embeddings.std(axis=0)
dim_vars = ingredient_embeddings.var(axis=0)

print("Overall mean of dimension means:",
      dim_means.mean())

print("Overall std of dimension means:",
      dim_means.std())

print("Mean dimension std:",
      dim_stds.mean())

print("Min dimension std:",
      dim_stds.min())

print("Max dimension std:",
      dim_stds.max())

print("Mean dimension variance:",
      dim_vars.mean())

print("Min dimension variance:",
      dim_vars.min())

print("Max dimension variance:",
      dim_vars.max())


# ------------------------------------------------------------
# SVD / effective rank
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SVD / EFFECTIVE RANK")
print("=" * 70)

X = ingredient_embeddings.astype(np.float64)

X_centered = X - X.mean(axis=0, keepdims=True)

_, singular_values, _ = np.linalg.svd(
    X_centered,
    full_matrices=False
)

variance = singular_values ** 2
variance_ratio = variance / variance.sum()
cumulative_variance = np.cumsum(variance_ratio)

numerical_rank = np.sum(
    singular_values > singular_values.max() * 1e-6
)

effective_rank = np.exp(
    -np.sum(
        variance_ratio
        * np.log(variance_ratio + 1e-12)
    )
)

print("Numerical rank:", numerical_rank)
print("Effective rank:", effective_rank)

for target in [0.50, 0.80, 0.90, 0.95, 0.99]:

    dims_needed = np.searchsorted(
        cumulative_variance,
        target
    ) + 1

    print(
        f"{int(target * 100)}% variance: "
        f"{dims_needed} dimensions"
    )


# ------------------------------------------------------------
# Random pair cosine similarities
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RANDOM INGREDIENT PAIR COSINES")
print("=" * 70)

rng = np.random.default_rng(42)

n_ingredients = len(ingredient_embeddings)
n_pairs = 100_000

i1 = rng.integers(
    0,
    n_ingredients,
    size=n_pairs
)

i2 = rng.integers(
    0,
    n_ingredients,
    size=n_pairs
)

same = i1 == i2

while np.any(same):
    i2[same] = rng.integers(
        0,
        n_ingredients,
        size=np.sum(same)
    )
    same = i1 == i2

a = ingredient_embeddings[i1]
b = ingredient_embeddings[i2]

a_norm = np.linalg.norm(a, axis=1)
b_norm = np.linalg.norm(b, axis=1)

cosines = np.sum(a * b, axis=1) / (
    a_norm * b_norm + 1e-12
)

print("Pairs:", len(cosines))

print("Mean   :", cosines.mean())
print("Median :", np.median(cosines))
print("Std    :", cosines.std())
print("Min    :", cosines.min())
print("Max    :", cosines.max())

for threshold in [0.80, 0.90, 0.95, 0.99]:

    percentage = (
        np.mean(cosines >= threshold) * 100
    )

    print(
        f">= {threshold:.2f}: "
        f"{percentage:.2f}%"
    )


# ------------------------------------------------------------
# Nearest-neighbor analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INGREDIENT NEAREST NEIGHBORS")
print("=" * 70)

# Embeddings are already normalized by GraphSAGE,
# but normalize again for safety.

E = ingredient_embeddings.astype(np.float32)

norms = np.linalg.norm(E, axis=1, keepdims=True)

E = E / np.maximum(norms, 1e-12)

n = len(E)

top1_scores = []
top10_scores = []

batch_size = 500

print(f"Computing cosine similarities in batches of {batch_size}...")

for start in range(0, n, batch_size):

    end = min(start + batch_size, n)

    batch = E[start:end]

    similarities = np.matmul(batch, E.T)

    # Remove self similarity.
    for local_i in range(end - start):
        global_i = start + local_i
        similarities[local_i, global_i] = -np.inf

    top10 = np.partition(
        similarities,
        -10,
        axis=1
    )[:, -10:]

    top1 = top10[:, -1]

    top1_scores.extend(top1.tolist())
    top10_scores.extend(top10.flatten().tolist())


top1_scores = np.array(top1_scores)

top10_scores = np.array(top10_scores)

print("\nTop-1 nearest-neighbor cosine:")
print("Mean   :", top1_scores.mean())
print("Median :", np.median(top1_scores))
print("Min    :", top1_scores.min())
print("Max    :", top1_scores.max())

print("\nTop-10 neighbor cosine:")
print("Mean   :", top10_scores.mean())
print("Median :", np.median(top10_scores))
print("Min    :", top10_scores.min())
print("Max    :", top10_scores.max())


# ------------------------------------------------------------
# Assessment
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ASSESSMENT")
print("=" * 70)

warnings = []

if effective_rank < 8:
    warnings.append(
        f"Very low effective rank ({effective_rank:.2f}/64)."
    )

if effective_rank < 12:
    warnings.append(
        "Embedding variance is concentrated in a small number "
        "of dimensions."
    )

if np.mean(cosines >= 0.99) > 0.05:
    warnings.append(
        "More than 5% of random ingredient pairs have cosine >= 0.99."
    )

if top1_scores.mean() > 0.95:
    warnings.append(
        "Mean nearest-neighbor cosine is extremely high."
    )

elif top1_scores.mean() > 0.90:
    warnings.append(
        "Mean nearest-neighbor cosine is high."
    )


if warnings:

    print("WARNING: Potential embedding collapse detected.")

    for warning in warnings:
        print("-", warning)

    print(
        "\nInterpretation: the embedding space may be "
        "too concentrated for reliable ingredient similarity."
    )

else:

    print(
        "Embedding space does not show strong evidence "
        "of catastrophic collapse."
    )


# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    f.write("GASTROGRAPH EMBEDDING DIAGNOSTICS\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Graph nodes: {graph.number_of_nodes()}\n")
    f.write(f"Graph edges: {graph.number_of_edges()}\n")

    f.write(f"Ingredient nodes: {len(ingredient_nodes)}\n")
    f.write(f"Compound nodes: {len(compound_nodes)}\n")

    f.write(
        f"Embedding shape: {embeddings.shape}\n"
    )

    f.write(
        f"Effective rank: {effective_rank}\n"
    )

    f.write(
        f"Numerical rank: {numerical_rank}\n"
    )

    f.write(
        f"Random pair mean cosine: {cosines.mean()}\n"
    )

    f.write(
        f"Random pair median cosine: {np.median(cosines)}\n"
    )

    f.write(
        f"Random pair >=0.99: "
        f"{np.mean(cosines >= 0.99) * 100:.2f}%\n"
    )

    f.write(
        f"Nearest-neighbor mean cosine: "
        f"{top1_scores.mean()}\n"
    )

    f.write(
        f"Nearest-neighbor median cosine: "
        f"{np.median(top1_scores)}\n"
    )

    f.write("\nASSESSMENT\n")
    f.write("-" * 70 + "\n")

    if warnings:

        f.write(
            "WARNING: Potential embedding collapse detected.\n"
        )

        for warning in warnings:
            f.write(f"- {warning}\n")

    else:

        f.write(
            "No strong evidence of catastrophic collapse.\n"
        )


print("\nReport saved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)