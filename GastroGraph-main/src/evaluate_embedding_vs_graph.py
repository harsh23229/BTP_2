import os
import pickle
import numpy as np
import networkx as nx
from itertools import combinations


# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\Parth\semester7\BTP_2\BTP_2\GastroGraph-custom"

GRAPH_PATH = os.path.join(BASE, "models", "gastro_graph.gpickle")
NODE2IDX_PATH = os.path.join(BASE, "models", "node2idx.pkl")
EMBEDDINGS_PATH = os.path.join(BASE, "models", "node_embeddings.npy")

OUTPUT_PATH = os.path.join(
    BASE,
    "data",
    "embedding_vs_graph_evaluation.txt"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
NUM_RANDOM_PAIRS = 10000


# ============================================================
# LOAD GRAPH
# ============================================================

print("=" * 70)
print("GASTROGRAPH: EMBEDDING vs GRAPH RELATIONSHIP EVALUATION")
print("=" * 70)

print("\nLoading graph...")

with open(GRAPH_PATH, "rb") as f:
    graph = pickle.load(f)

print(f"Graph nodes: {graph.number_of_nodes()}")
print(f"Graph edges: {graph.number_of_edges()}")


# ============================================================
# LOAD NODE MAPPING
# ============================================================

print("\nLoading node mapping...")

with open(NODE2IDX_PATH, "rb") as f:
    node2idx = pickle.load(f)

nodes = node2idx["nodes"]

print(f"Node list: {len(nodes)}")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(EMBEDDINGS_PATH)

print(f"Embedding matrix: {embeddings.shape}")


# ============================================================
# VALIDATE ALIGNMENT
# ============================================================

if len(nodes) != len(embeddings):
    raise ValueError(
        f"Node/embedding mismatch: "
        f"{len(nodes)} nodes vs {len(embeddings)} embeddings"
    )

if len(nodes) != graph.number_of_nodes():
    raise ValueError(
        f"Node/graph mismatch: "
        f"{len(nodes)} mapping entries vs "
        f"{graph.number_of_nodes()} graph nodes"
    )

node_to_row = {
    node: i
    for i, node in enumerate(nodes)
}


# ============================================================
# IDENTIFY INGREDIENT NODES
# ============================================================

ingredient_nodes = [
    node
    for node in graph.nodes
    if graph.nodes[node].get("node_type") == "ingredient"
]

print(f"\nIngredient nodes: {len(ingredient_nodes)}")


# ============================================================
# INGREDIENT EMBEDDINGS
# ============================================================

ingredient_rows = [
    node_to_row[node]
    for node in ingredient_nodes
]

ingredient_embeddings = embeddings[ingredient_rows].astype(np.float32)

# Normalize explicitly for cosine similarity.
norms = np.linalg.norm(
    ingredient_embeddings,
    axis=1,
    keepdims=True
)

ingredient_embeddings = (
    ingredient_embeddings /
    np.maximum(norms, 1e-12)
)

ingredient_index = {
    node: i
    for i, node in enumerate(ingredient_nodes)
}


# ============================================================
# BUILD INGREDIENT NAME LOOKUP
# ============================================================

print("\nBuilding ingredient name lookup...")

name_to_node = {}


def add_name(name, node):
    if name is None:
        return

    name = str(name).strip().lower()

    if not name:
        return

    if name not in name_to_node:
        name_to_node[name] = node


for node in ingredient_nodes:

    attrs = graph.nodes[node]

    possible_names = [
        attrs.get("ingredient_name"),
        attrs.get("name"),
        attrs.get("label"),
        attrs.get("entity_alias"),
        attrs.get("display_name"),
    ]

    for name in possible_names:
        add_name(name, node)

    # Also index the node ID itself.
    add_name(node, node)


print(f"Indexed ingredient names: {len(name_to_node)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_node(name):
    name = name.strip().lower()

    if name in name_to_node:
        return name_to_node[name]

    # Try substring matching as fallback.
    matches = [
        node
        for indexed_name, node in name_to_node.items()
        if name == indexed_name
    ]

    if matches:
        return matches[0]

    return None


def cosine_similarity(node_a, node_b):

    ia = ingredient_index[node_a]
    ib = ingredient_index[node_b]

    return float(
        np.dot(
            ingredient_embeddings[ia],
            ingredient_embeddings[ib]
        )
    )


def shared_molecule_count(node_a, node_b):
    """
    Count common compound neighbors between two ingredients.
    """

    neighbors_a = set(graph.neighbors(node_a))
    neighbors_b = set(graph.neighbors(node_b))

    compounds_a = {
        n for n in neighbors_a
        if graph.nodes[n].get("node_type") == "compound"
    }

    compounds_b = {
        n for n in neighbors_b
        if graph.nodes[n].get("node_type") == "compound"
    }

    return len(compounds_a & compounds_b)


def cooccurrence_count(node_a, node_b):
    """
    Number of CO_OCCURS edges between the two ingredients.

    In this graph, ingredient-ingredient edges represent
    recipe co-occurrence relationships.
    """

    if not graph.has_edge(node_a, node_b):
        return 0

    edge_data = graph.get_edge_data(node_a, node_b)

    if edge_data is None:
        return 0

    # Graph should normally have one edge per ingredient pair.
    # Handle both ordinary and multi-edge style data safely.

    if isinstance(edge_data, dict):

        if "edge_type" in edge_data:
            if edge_data["edge_type"] == "CO_OCCURS":
                return edge_data.get("weight", 1)

        if "weight" in edge_data:
            return edge_data["weight"]

        # MultiGraph-like structure.
        values = []

        for value in edge_data.values():

            if isinstance(value, dict):
                if value.get("edge_type") == "CO_OCCURS":
                    values.append(value.get("weight", 1))

        if values:
            return sum(values)

    return 1


def recipe_degree(node):
    """
    Number of ingredient neighbors connected by CO_OCCURS.
    """

    count = 0

    for neighbor in graph.neighbors(node):

        if graph.nodes[neighbor].get("node_type") != "ingredient":
            continue

        edge_data = graph.get_edge_data(node, neighbor)

        if edge_data is None:
            continue

        if isinstance(edge_data, dict):

            if edge_data.get("edge_type") == "CO_OCCURS":
                count += 1

            elif "edge_type" not in edge_data:
                count += 1

    return count


# ============================================================
# EXPLICIT TEST PAIRS
# ============================================================

RELATED_PAIRS = [
    ("beef", "beef stock"),
    ("chicken", "chicken breast"),
    ("tomato", "tomato paste"),
    ("olive oil", "extra virgin olive oil"),
    ("black pepper", "pepper"),
    ("onion", "yellow onion"),
]

UNRELATED_PAIRS = [
    ("beef", "bay leaf"),
    ("beef", "wine vinegar"),
    ("chicken", "dill seed"),
    ("lobster", "tofu"),
    ("fish", "broccoli"),
]


# ============================================================
# EVALUATE PAIR
# ============================================================

def evaluate_pair(name_a, name_b):

    node_a = find_node(name_a)
    node_b = find_node(name_b)

    if node_a is None or node_b is None:
        return None

    cosine = cosine_similarity(node_a, node_b)

    molecules = shared_molecule_count(node_a, node_b)

    cooccur = cooccurrence_count(node_a, node_b)

    degree_a = recipe_degree(node_a)
    degree_b = recipe_degree(node_b)

    return {
        "a": name_a,
        "b": name_b,
        "node_a": node_a,
        "node_b": node_b,
        "cosine": cosine,
        "shared_molecules": molecules,
        "cooccurrence": cooccur,
        "degree_a": degree_a,
        "degree_b": degree_b,
    }


# ============================================================
# OUTPUT STORAGE
# ============================================================

report_lines = []


def report(text=""):
    print(text)
    report_lines.append(text)


# ============================================================
# EXPLICIT PAIR ANALYSIS
# ============================================================

report("\n" + "=" * 70)
report("EXPLICIT PAIR ANALYSIS")
report("=" * 70)


all_results = []


def run_pair_group(title, pairs):

    report("\n" + "-" * 70)
    report(title)
    report("-" * 70)

    results = []

    for a, b in pairs:

        result = evaluate_pair(a, b)

        if result is None:

            report(f"\n{a} <-> {b}")
            report("  [MISSING] One or both ingredients not found.")
            continue

        results.append(result)
        all_results.append(result)

        report(f"\n{a} <-> {b}")
        report(f"  GraphSAGE cosine       : {result['cosine']:.6f}")
        report(f"  Shared molecules       : {result['shared_molecules']}")
        report(f"  CO_OCCURS weight       : {result['cooccurrence']}")
        report(f"  Recipe-neighbor degree : "
               f"{result['degree_a']} / {result['degree_b']}")

    return results


related_results = run_pair_group(
    "EXPECTED RELATED PAIRS",
    RELATED_PAIRS
)

unrelated_results = run_pair_group(
    "EXPECTED UNRELATED PAIRS",
    UNRELATED_PAIRS
)


# ============================================================
# SUMMARY OF EXPLICIT PAIRS
# ============================================================

report("\n" + "=" * 70)
report("EXPLICIT PAIR SUMMARY")
report("=" * 70)


if related_results:

    report("\nRELATED")

    report(
        f"Mean cosine: "
        f"{np.mean([r['cosine'] for r in related_results]):.6f}"
    )

    report(
        f"Mean shared molecules: "
        f"{np.mean([r['shared_molecules'] for r in related_results]):.2f}"
    )

    report(
        f"Mean CO_OCCURS weight: "
        f"{np.mean([r['cooccurrence'] for r in related_results]):.2f}"
    )


if unrelated_results:

    report("\nUNRELATED")

    report(
        f"Mean cosine: "
        f"{np.mean([r['cosine'] for r in unrelated_results]):.6f}"
    )

    report(
        f"Mean shared molecules: "
        f"{np.mean([r['shared_molecules'] for r in unrelated_results]):.2f}"
    )

    report(
        f"Mean CO_OCCURS weight: "
        f"{np.mean([r['cooccurrence'] for r in unrelated_results]):.2f}"
    )


# ============================================================
# RANDOM PAIR CORRELATION ANALYSIS
# ============================================================

report("\n" + "=" * 70)
report("RANDOM PAIR ANALYSIS")
report("=" * 70)

rng = np.random.default_rng(RANDOM_SEED)

num_ingredients = len(ingredient_nodes)

random_pairs = set()

while len(random_pairs) < NUM_RANDOM_PAIRS:

    i = int(rng.integers(0, num_ingredients))
    j = int(rng.integers(0, num_ingredients))

    if i == j:
        continue

    if i > j:
        i, j = j, i

    random_pairs.add((i, j))


cosines = []
cooccurrences = []
molecule_overlaps = []


for i, j in random_pairs:

    node_a = ingredient_nodes[i]
    node_b = ingredient_nodes[j]

    cosine = float(
        np.dot(
            ingredient_embeddings[i],
            ingredient_embeddings[j]
        )
    )

    cooccur = cooccurrence_count(node_a, node_b)

    molecules = shared_molecule_count(node_a, node_b)

    cosines.append(cosine)
    cooccurrences.append(cooccur)
    molecule_overlaps.append(molecules)


cosines = np.array(cosines, dtype=np.float64)
cooccurrences = np.array(cooccurrences, dtype=np.float64)
molecule_overlaps = np.array(
    molecule_overlaps,
    dtype=np.float64
)


# ============================================================
# CORRELATION HELPER
# ============================================================

def safe_correlation(x, y):

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


cosine_cooccur_corr = safe_correlation(
    cosines,
    cooccurrences
)

cosine_molecule_corr = safe_correlation(
    cosines,
    molecule_overlaps
)

cooccur_molecule_corr = safe_correlation(
    cooccurrences,
    molecule_overlaps
)


report(f"\nRandom pairs: {len(random_pairs)}")

report(
    f"Mean GraphSAGE cosine: "
    f"{np.mean(cosines):.6f}"
)

report(
    f"Median GraphSAGE cosine: "
    f"{np.median(cosines):.6f}"
)

report(
    f"Mean CO_OCCURS weight: "
    f"{np.mean(cooccurrences):.6f}"
)

report(
    f"Pairs with CO_OCCURS > 0: "
    f"{np.sum(cooccurrences > 0)} "
    f"({100 * np.mean(cooccurrences > 0):.2f}%)"
)

report(
    f"Pairs with shared molecules > 0: "
    f"{np.sum(molecule_overlaps > 0)} "
    f"({100 * np.mean(molecule_overlaps > 0):.2f}%)"
)

report("\nCORRELATIONS")

report(
    f"GraphSAGE cosine <-> CO_OCCURS      : "
    f"{cosine_cooccur_corr:.6f}"
)

report(
    f"GraphSAGE cosine <-> molecule overlap: "
    f"{cosine_molecule_corr:.6f}"
)

report(
    f"CO_OCCURS <-> molecule overlap       : "
    f"{cooccur_molecule_corr:.6f}"
)


# ============================================================
# CO-OCCURRENCE GROUP COMPARISON
# ============================================================

report("\n" + "=" * 70)
report("GRAPH SAGE COSINE BY CO-OCCURRENCE")
report("=" * 70)


zero_mask = cooccurrences == 0
positive_mask = cooccurrences > 0


if np.any(zero_mask):

    report(
        f"\nNo CO_OCCURS:"
        f"\n  Count: {np.sum(zero_mask)}"
        f"\n  Mean cosine: {np.mean(cosines[zero_mask]):.6f}"
        f"\n  Median cosine: {np.median(cosines[zero_mask]):.6f}"
    )


if np.any(positive_mask):

    report(
        f"\nCO_OCCURS > 0:"
        f"\n  Count: {np.sum(positive_mask)}"
        f"\n  Mean cosine: {np.mean(cosines[positive_mask]):.6f}"
        f"\n  Median cosine: {np.median(cosines[positive_mask]):.6f}"
    )


# ============================================================
# MOLECULAR OVERLAP GROUP COMPARISON
# ============================================================

report("\n" + "=" * 70)
report("GRAPH SAGE COSINE BY MOLECULAR OVERLAP")
report("=" * 70)


zero_mol_mask = molecule_overlaps == 0
positive_mol_mask = molecule_overlaps > 0


if np.any(zero_mol_mask):

    report(
        f"\nNo shared molecules:"
        f"\n  Count: {np.sum(zero_mol_mask)}"
        f"\n  Mean cosine: {np.mean(cosines[zero_mol_mask]):.6f}"
        f"\n  Median cosine: {np.median(cosines[zero_mol_mask]):.6f}"
    )


if np.any(positive_mol_mask):

    report(
        f"\nShared molecules > 0:"
        f"\n  Count: {np.sum(positive_mol_mask)}"
        f"\n  Mean cosine: {np.mean(cosines[positive_mol_mask]):.6f}"
        f"\n  Median cosine: {np.median(cosines[positive_mol_mask]):.6f}"
    )


# ============================================================
# INTERPRETATION
# ============================================================

report("\n" + "=" * 70)
report("INTERPRETATION")
report("=" * 70)


if np.isnan(cosine_cooccur_corr):

    report(
        "\nCould not calculate GraphSAGE/co-occurrence correlation."
    )

elif abs(cosine_cooccur_corr) >= 0.50:

    report(
        "\n[STRONG]"
        "\nGraphSAGE cosine has a strong relationship with "
        "recipe co-occurrence."
    )

elif abs(cosine_cooccur_corr) >= 0.25:

    report(
        "\n[MODERATE]"
        "\nGraphSAGE cosine has a moderate relationship with "
        "recipe co-occurrence."
    )

else:

    report(
        "\n[WEAK]"
        "\nGraphSAGE cosine has a weak relationship with "
        "recipe co-occurrence."
    )


if np.isnan(cosine_molecule_corr):

    report(
        "\nCould not calculate GraphSAGE/molecular correlation."
    )

elif abs(cosine_molecule_corr) >= 0.50:

    report(
        "\n[STRONG]"
        "\nGraphSAGE cosine has a strong relationship with "
        "molecular overlap."
    )

elif abs(cosine_molecule_corr) >= 0.25:

    report(
        "\n[MODERATE]"
        "\nGraphSAGE cosine has a moderate relationship with "
        "molecular overlap."
    )

else:

    report(
        "\n[WEAK]"
        "\nGraphSAGE cosine has a weak relationship with "
        "molecular overlap."
    )


if (
    np.any(zero_mask)
    and np.any(positive_mask)
):

    difference = (
        np.mean(cosines[positive_mask])
        -
        np.mean(cosines[zero_mask])
    )

    report(
        f"\nCosine difference:"
        f"\n  CO_OCCURS > 0 minus CO_OCCURS = 0"
        f"\n  {difference:.6f}"
    )

    if difference > 0.01:

        report(
            "\nGraphSAGE clearly gives higher similarity "
            "to ingredients that co-occur."
        )

    elif difference > 0:

        report(
            "\nGraphSAGE gives slightly higher similarity "
            "to ingredients that co-occur."
        )

    else:

        report(
            "\nGraphSAGE does not give higher similarity "
            "to ingredients that co-occur."
        )


report("\nIMPORTANT:")
report(
    "CO_OCCURS measures culinary compatibility/context, "
    "not substitution suitability."
)

report(
    "Shared molecules measure molecular overlap, "
    "not automatically sensory flavor similarity."
)

report(
    "GraphSAGE cosine should therefore not be interpreted "
    "as a substitution score without validation."
)


# ============================================================
# SAVE REPORT
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

report("\n" + "=" * 70)
report("DONE")
report("=" * 70)

report(f"\nReport saved to:\n{OUTPUT_PATH}")