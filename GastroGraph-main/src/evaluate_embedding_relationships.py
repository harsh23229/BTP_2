import os
import pickle
import numpy as np


BASE = r"C:\Users\Parth\semester7\BTP_2\BTP_2\GastroGraph-custom"
MODEL_DIR = os.path.join(BASE, "models")
DATA_DIR = os.path.join(BASE, "data")

GRAPH_PATH = os.path.join(
    MODEL_DIR,
    "gastro_graph.gpickle"
)

NODE2IDX_PATH = os.path.join(
    MODEL_DIR,
    "node2idx.pkl"
)

EMBEDDING_PATH = os.path.join(
    MODEL_DIR,
    "node_embeddings.npy"
)

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "embedding_relationship_evaluation.txt"
)


# ============================================================
# Expected related pairs
# ============================================================

RELATED_PAIRS = [
    ("beef", "beef stock"),
    ("chicken", "chicken breast"),
    ("tomato", "tomato paste"),
    ("olive oil", "extra virgin olive oil"),
    ("black pepper", "pepper"),
    ("onion", "yellow onion"),
]


# ============================================================
# Expected unrelated pairs
# ============================================================

UNRELATED_PAIRS = [
    ("beef", "bay leaf"),
    ("beef", "wine vinegar"),
    ("chicken", "dill seed"),
    ("lobster", "tofu"),
    ("fish", "broccoli"),
]


# ============================================================
# Helpers
# ============================================================

def normalize_name(name):
    return " ".join(
        name.lower().strip().split()
    )


def find_ingredient_node(graph, ingredient_name):
    """
    Find an ingredient graph node whose normalized
    display/name attribute matches the requested ingredient.

    We inspect common possible attributes rather than
    assuming one exact attribute name.
    """

    target = normalize_name(ingredient_name)

    candidates = []

    for node, attrs in graph.nodes(data=True):

        if attrs.get("node_type") != "ingredient":
            continue

        values = []

        for key in [
            "ingredient_name",
            "name",
            "label",
            "entity_alias",
            "display_name",
        ]:

            value = attrs.get(key)

            if isinstance(value, str):
                values.append(value)

        # Also inspect the node ID itself.
        if isinstance(node, str):
            values.append(node)

        for value in values:

            normalized = normalize_name(value)

            if normalized == target:
                return node

            candidates.append(
                (normalized, node)
            )

    return None


def build_ingredient_name_index(graph):
    """
    Build a lookup table from all textual ingredient
    representations to graph node IDs.
    """

    lookup = {}

    for node, attrs in graph.nodes(data=True):

        if attrs.get("node_type") != "ingredient":
            continue

        values = []

        for key in [
            "ingredient_name",
            "name",
            "label",
            "entity_alias",
            "display_name",
        ]:

            value = attrs.get(key)

            if isinstance(value, str):
                values.append(value)

        if isinstance(node, str):
            values.append(node)

        for value in values:

            normalized = normalize_name(value)

            if normalized not in lookup:
                lookup[normalized] = node

    return lookup


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    return float(np.dot(a, b))


# ============================================================
# Load files
# ============================================================

print("=" * 70)
print("GASTROGRAPH EMBEDDING RELATIONSHIP EVALUATION")
print("=" * 70)

print("\nLoading graph...")

with open(GRAPH_PATH, "rb") as f:
    graph = pickle.load(f)

print(
    f"Graph: {graph.number_of_nodes()} nodes, "
    f"{graph.number_of_edges()} edges"
)


print("\nLoading node mapping...")

with open(NODE2IDX_PATH, "rb") as f:
    node2idx = pickle.load(f)

nodes = node2idx["nodes"]

print("Node list:", len(nodes))


print("\nLoading embeddings...")

embeddings = np.load(EMBEDDING_PATH)

print("Embedding matrix:", embeddings.shape)


# ============================================================
# Build node -> row mapping
# ============================================================

node_to_row = {
    node: i
    for i, node in enumerate(nodes)
}


# ============================================================
# Extract ingredient nodes
# ============================================================

ingredient_nodes = [
    node
    for node, attrs in graph.nodes(data=True)
    if attrs.get("node_type") == "ingredient"
]

print(
    "\nIngredient nodes:",
    len(ingredient_nodes)
)


ingredient_indices = np.array(
    [
        node_to_row[node]
        for node in ingredient_nodes
    ],
    dtype=np.int64
)

ingredient_embeddings = embeddings[
    ingredient_indices
]


# ============================================================
# Normalize embeddings once
# ============================================================

ingredient_embeddings = (
    ingredient_embeddings
    /
    np.maximum(
        np.linalg.norm(
            ingredient_embeddings,
            axis=1,
            keepdims=True
        ),
        1e-12
    )
)


# ============================================================
# Build lookup
# ============================================================

print("\nBuilding ingredient name lookup...")

name_to_node = build_ingredient_name_index(graph)

print(
    "Indexed ingredient names:",
    len(name_to_node)
)


# ============================================================
# Create compact node -> ingredient embedding index
# ============================================================

ingredient_node_to_index = {
    node: i
    for i, node in enumerate(ingredient_nodes)
}


# ============================================================
# Evaluate pair
# ============================================================

def evaluate_pair(a_name, b_name):

    a_key = normalize_name(a_name)
    b_key = normalize_name(b_name)

    a_node = name_to_node.get(a_key)
    b_node = name_to_node.get(b_key)

    if a_node is None:
        return {
            "a": a_name,
            "b": b_name,
            "status": "MISSING_A",
        }

    if b_node is None:
        return {
            "a": a_name,
            "b": b_name,
            "status": "MISSING_B",
        }

    a_idx = ingredient_node_to_index.get(a_node)
    b_idx = ingredient_node_to_index.get(b_node)

    if a_idx is None:
        return {
            "a": a_name,
            "b": b_name,
            "status": "A_NOT_INGREDIENT",
        }

    if b_idx is None:
        return {
            "a": a_name,
            "b": b_name,
            "status": "B_NOT_INGREDIENT",
        }

    a_embedding = ingredient_embeddings[a_idx]
    b_embedding = ingredient_embeddings[b_idx]

    score = cosine(
        a_embedding,
        b_embedding
    )

    # --------------------------------------------------------
    # Rank B among every ingredient for A
    # --------------------------------------------------------

    similarities = np.dot(
        ingredient_embeddings,
        a_embedding
    )

    # Exclude A itself.
    similarities[a_idx] = -np.inf

    rank_order = np.argsort(
        -similarities
    )

    rank_positions = np.where(
        rank_order == b_idx
    )[0]

    rank = (
        int(rank_positions[0]) + 1
        if len(rank_positions) > 0
        else None
    )

    # Percentile:
    # 1.0 = extremely high
    # 0.5 = middle
    # 0.0 = very low
    percentile = (
        1.0 - (
            (rank - 1)
            /
            (len(ingredient_nodes) - 1)
        )
        if rank is not None
        else None
    )

    # --------------------------------------------------------
    # Top 10 neighbors
    # --------------------------------------------------------

    top10_indices = rank_order[:10]

    top10 = []

    for idx in top10_indices:

        node = ingredient_nodes[idx]

        attrs = graph.nodes[node]

        display_name = (
            attrs.get("ingredient_name")
            or attrs.get("name")
            or attrs.get("label")
            or node
        )

        top10.append(
            (
                display_name,
                float(similarities[idx])
            )
        )

    return {
        "a": a_name,
        "b": b_name,
        "status": "OK",
        "a_node": a_node,
        "b_node": b_node,
        "cosine": score,
        "rank": rank,
        "percentile": percentile,
        "top10": top10,
    }


# ============================================================
# Print result
# ============================================================

def print_result(result):

    print("\n" + "-" * 70)

    print(
        f"{result['a']}  <->  {result['b']}"
    )

    if result["status"] != "OK":

        print(
            "STATUS:",
            result["status"]
        )

        return

    print(
        f"Cosine similarity : "
        f"{result['cosine']:.6f}"
    )

    print(
        f"Rank              : "
        f"{result['rank']} / {len(ingredient_nodes)}"
    )

    print(
        f"Rank percentile   : "
        f"{result['percentile'] * 100:.2f}%"
    )

    print("\nTop 10 neighbors of", result["a"])

    for i, (name, score) in enumerate(
        result["top10"],
        start=1
    ):

        print(
            f"{i:2d}. "
            f"{name:<40} "
            f"{score:.6f}"
        )


# ============================================================
# Evaluate related pairs
# ============================================================

print("\n")
print("=" * 70)
print("EXPECTED RELATED PAIRS")
print("=" * 70)

related_results = []

for a, b in RELATED_PAIRS:

    result = evaluate_pair(a, b)

    related_results.append(result)

    print_result(result)


# ============================================================
# Evaluate unrelated pairs
# ============================================================

print("\n")
print("=" * 70)
print("EXPECTED UNRELATED PAIRS")
print("=" * 70)

unrelated_results = []

for a, b in UNRELATED_PAIRS:

    result = evaluate_pair(a, b)

    unrelated_results.append(result)

    print_result(result)


# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)


def valid_results(results):

    return [
        r
        for r in results
        if r["status"] == "OK"
    ]


valid_related = valid_results(
    related_results
)

valid_unrelated = valid_results(
    unrelated_results
)


if valid_related:

    related_cosines = np.array(
        [
            r["cosine"]
            for r in valid_related
        ]
    )

    related_ranks = np.array(
        [
            r["rank"]
            for r in valid_related
        ]
    )

    print("\nRELATED")

    print(
        "Pairs evaluated:",
        len(valid_related)
    )

    print(
        "Mean cosine:",
        related_cosines.mean()
    )

    print(
        "Median cosine:",
        np.median(related_cosines)
    )

    print(
        "Mean rank:",
        related_ranks.mean()
    )

    print(
        "Median rank:",
        np.median(related_ranks)
    )


if valid_unrelated:

    unrelated_cosines = np.array(
        [
            r["cosine"]
            for r in valid_unrelated
        ]
    )

    unrelated_ranks = np.array(
        [
            r["rank"]
            for r in valid_unrelated
        ]
    )

    print("\nUNRELATED")

    print(
        "Pairs evaluated:",
        len(valid_unrelated)
    )

    print(
        "Mean cosine:",
        unrelated_cosines.mean()
    )

    print(
        "Median cosine:",
        np.median(unrelated_cosines)
    )

    print(
        "Mean rank:",
        unrelated_ranks.mean()
    )

    print(
        "Median rank:",
        np.median(unrelated_ranks)
    )


# ============================================================
# Separation
# ============================================================

if valid_related and valid_unrelated:

    related_mean = np.mean(
        [
            r["cosine"]
            for r in valid_related
        ]
    )

    unrelated_mean = np.mean(
        [
            r["cosine"]
            for r in valid_unrelated
        ]
    )

    cosine_gap = (
        related_mean
        -
        unrelated_mean
    )

    related_rank_mean = np.mean(
        [
            r["rank"]
            for r in valid_related
        ]
    )

    unrelated_rank_mean = np.mean(
        [
            r["rank"]
            for r in valid_unrelated
        ]
    )

    print("\n")
    print("=" * 70)
    print("SEPARATION")
    print("=" * 70)

    print(
        f"Related mean cosine   : "
        f"{related_mean:.6f}"
    )

    print(
        f"Unrelated mean cosine : "
        f"{unrelated_mean:.6f}"
    )

    print(
        f"Cosine gap            : "
        f"{cosine_gap:.6f}"
    )

    print(
        f"Related mean rank     : "
        f"{related_rank_mean:.1f}"
    )

    print(
        f"Unrelated mean rank   : "
        f"{unrelated_rank_mean:.1f}"
    )

    print("\nInterpretation:")

    if cosine_gap > 0.03:
        print(
            "[GOOD] Related pairs have noticeably higher "
            "cosine similarity."
        )

    elif cosine_gap > 0.01:
        print(
            "[WEAK] Related pairs are somewhat separated, "
            "but the difference is small."
        )

    else:
        print(
            "[POOR] Related and unrelated pairs have "
            "very similar cosine scores."
        )

    if related_rank_mean < unrelated_rank_mean:

        print(
            "[GOOD] Related pairs tend to rank higher "
            "than unrelated pairs."
        )

    else:

        print(
            "[POOR] Related pairs do not rank higher "
            "than unrelated pairs."
        )


# ============================================================
# Save report
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GASTROGRAPH EMBEDDING RELATIONSHIP EVALUATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "EXPECTED RELATED PAIRS\n"
    )

    f.write("=" * 70 + "\n")

    for result in related_results:

        f.write(
            f"\n{result['a']} <-> {result['b']}\n"
        )

        f.write(
            f"Status: {result['status']}\n"
        )

        if result["status"] == "OK":

            f.write(
                f"Cosine: {result['cosine']:.6f}\n"
            )

            f.write(
                f"Rank: {result['rank']} / "
                f"{len(ingredient_nodes)}\n"
            )

            f.write(
                f"Percentile: "
                f"{result['percentile'] * 100:.2f}%\n"
            )


    f.write(
        "\n\nEXPECTED UNRELATED PAIRS\n"
    )

    f.write("=" * 70 + "\n")

    for result in unrelated_results:

        f.write(
            f"\n{result['a']} <-> {result['b']}\n"
        )

        f.write(
            f"Status: {result['status']}\n"
        )

        if result["status"] == "OK":

            f.write(
                f"Cosine: {result['cosine']:.6f}\n"
            )

            f.write(
                f"Rank: {result['rank']} / "
                f"{len(ingredient_nodes)}\n"
            )

            f.write(
                f"Percentile: "
                f"{result['percentile'] * 100:.2f}%\n"
            )


    if valid_related and valid_unrelated:

        f.write(
            "\n\nSEPARATION\n"
        )

        f.write("=" * 70 + "\n")

        f.write(
            f"Related mean cosine: "
            f"{related_mean:.6f}\n"
        )

        f.write(
            f"Unrelated mean cosine: "
            f"{unrelated_mean:.6f}\n"
        )

        f.write(
            f"Cosine gap: "
            f"{cosine_gap:.6f}\n"
        )

        f.write(
            f"Related mean rank: "
            f"{related_rank_mean:.1f}\n"
        )

        f.write(
            f"Unrelated mean rank: "
            f"{unrelated_rank_mean:.1f}\n"
        )


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print("\nReport saved to:")
print(OUTPUT_PATH)