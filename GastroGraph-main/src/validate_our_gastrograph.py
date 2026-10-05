# ================================================================
# VALIDATE OUR CUSTOM GASTROGRAPH
#
# Checks:
#   1. Connected components
#   2. Node types
#   3. Ingredient / molecule counts
#   4. Important ingredient connectivity
#   5. CO_OCCURS / CONTAINS connections
#   6. Degree distribution
#   7. Isolated nodes
#   8. Learned embedding dimensions
#   9. Embedding norms
#  10. Important ingredient embeddings
#
# Input:
#   GastroGraph-custom/models/
#       gastro_graph.gpickle
#       node2idx.pkl
#       node_embeddings.npy
#
# Output:
#   GastroGraph-custom/data/validation_statistics.txt
# ================================================================

import os
import pickle

import numpy as np
import networkx as nx


# ================================================================
# PATHS
# ================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

GASTROGRAPH_ROOT = os.path.dirname(
    SCRIPT_DIR
)

BTP_ROOT = os.path.dirname(
    GASTROGRAPH_ROOT
)

CUSTOM_ROOT = os.path.join(
    BTP_ROOT,
    "GastroGraph-custom"
)

DATA_DIR = os.path.join(
    CUSTOM_ROOT,
    "data"
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

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "validation_statistics.txt"
)


# ================================================================
# IMPORTANT INGREDIENTS TO CHECK
# ================================================================

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
# HELPERS
# ================================================================

def print_section(title, output_lines):

    line = "=" * 70

    print("\n" + line)
    print(title)
    print(line)

    output_lines.append("")
    output_lines.append(line)
    output_lines.append(title)
    output_lines.append(line)


def get_node_name(G, node):

    data = G.nodes[node]

    name = data.get("name")

    if name is None:
        name = str(node)

    return str(name).strip().lower()


def find_ingredient_nodes(G):

    ingredient_nodes = {}

    for node, data in G.nodes(data=True):

        node_type = str(
            data.get("node_type", "")
        ).lower()

        if node_type != "ingredient":
            continue

        name = str(
            data.get("name", node)
        ).strip().lower()

        ingredient_nodes[name] = node

    return ingredient_nodes


# ================================================================
# MAIN VALIDATION
# ================================================================

def main():

    output_lines = []

    print("=" * 70)
    print("OUR GASTROGRAPH — VALIDATION")
    print("=" * 70)

    output_lines.append(
        "OUR GASTROGRAPH — VALIDATION"
    )

    # ------------------------------------------------------------
    # Check files
    # ------------------------------------------------------------

    required_files = [
        GRAPH_FILE,
        MAPPING_FILE,
        EMBEDDINGS_FILE,
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    # ------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------

    print_section(
        "1. LOADING GRAPH",
        output_lines
    )

    with open(
        GRAPH_FILE,
        "rb"
    ) as f:

        G = pickle.load(f)

    print(
        f"Graph type: {type(G).__name__}"
    )

    print(
        f"Nodes: {G.number_of_nodes():,}"
    )

    print(
        f"Edges: {G.number_of_edges():,}"
    )

    output_lines.append(
        f"Nodes: {G.number_of_nodes():,}"
    )

    output_lines.append(
        f"Edges: {G.number_of_edges():,}"
    )


    # ============================================================
    # 2. NODE TYPES
    # ============================================================

    print_section(
        "2. NODE TYPES",
        output_lines
    )

    node_type_counts = {}

    for node, data in G.nodes(data=True):

        node_type = str(
            data.get(
                "node_type",
                "UNKNOWN"
            )
        )

        node_type_counts[node_type] = (
            node_type_counts.get(
                node_type,
                0
            ) + 1
        )

    for node_type, count in sorted(
        node_type_counts.items(),
        key=lambda x: -x[1]
    ):

        print(
            f"{node_type:15s}: {count:,}"
        )

        output_lines.append(
            f"{node_type:15s}: {count:,}"
        )


    # ============================================================
    # 3. CONNECTED COMPONENTS
    # ============================================================

    print_section(
        "3. CONNECTED COMPONENTS",
        output_lines
    )

    components = sorted(
        nx.connected_components(G),
        key=len,
        reverse=True
    )

    print(
        f"Number of connected components: "
        f"{len(components):,}"
    )

    output_lines.append(
        f"Number of connected components: "
        f"{len(components):,}"
    )

    print(
        "\nLargest components:"
    )

    output_lines.append(
        "\nLargest components:"
    )

    for i, component in enumerate(
        components[:10],
        start=1
    ):

        component_nodes = list(
            component
        )

        type_counts = {}

        for node in component_nodes:

            node_type = str(
                G.nodes[node].get(
                    "node_type",
                    "UNKNOWN"
                )
            )

            type_counts[node_type] = (
                type_counts.get(
                    node_type,
                    0
                ) + 1
            )

        type_text = ", ".join(
            f"{k}={v:,}"
            for k, v in type_counts.items()
        )

        line = (
            f"Component {i}: "
            f"{len(component_nodes):,} nodes "
            f"({type_text})"
        )

        print(line)
        output_lines.append(line)


    # ============================================================
    # 4. INGREDIENT / MOLECULE COUNTS
    # ============================================================

    print_section(
        "4. INGREDIENT / MOLECULE COUNTS",
        output_lines
    )

    ingredient_nodes = [
        node
        for node, data in G.nodes(data=True)
        if str(
            data.get(
                "node_type",
                ""
            )
        ).lower() == "ingredient"
    ]

    molecule_nodes = [
        node
        for node, data in G.nodes(data=True)
        if str(
            data.get(
                "node_type",
                ""
            )
        ).lower() in [
            "compound",
            "molecule"
        ]
    ]

    print(
        f"Ingredient nodes : {len(ingredient_nodes):,}"
    )

    print(
        f"Molecule nodes   : {len(molecule_nodes):,}"
    )

    output_lines.append(
        f"Ingredient nodes : {len(ingredient_nodes):,}"
    )

    output_lines.append(
        f"Molecule nodes   : {len(molecule_nodes):,}"
    )


    # ============================================================
    # 5. EDGE TYPES
    # ============================================================

    print_section(
        "5. EDGE TYPES",
        output_lines
    )

    edge_type_counts = {}

    for u, v, data in G.edges(
        data=True
    ):

        edge_type = str(
            data.get(
                "edge_type",
                "UNKNOWN"
            )
        )

        edge_type_counts[edge_type] = (
            edge_type_counts.get(
                edge_type,
                0
            ) + 1
        )

    for edge_type, count in sorted(
        edge_type_counts.items(),
        key=lambda x: -x[1]
    ):

        print(
            f"{edge_type:15s}: {count:,}"
        )

        output_lines.append(
            f"{edge_type:15s}: {count:,}"
        )


    # ============================================================
    # 6. ISOLATED NODES
    # ============================================================

    print_section(
        "6. ISOLATED NODES",
        output_lines
    )

    isolated_nodes = list(
        nx.isolates(G)
    )

    print(
        f"Isolated nodes: "
        f"{len(isolated_nodes):,}"
    )

    output_lines.append(
        f"Isolated nodes: "
        f"{len(isolated_nodes):,}"
    )

    if isolated_nodes:

        print("\nFirst isolated nodes:")

        for node in isolated_nodes[:20]:

            name = get_node_name(
                G,
                node
            )

            node_type = G.nodes[node].get(
                "node_type",
                "UNKNOWN"
            )

            line = (
                f"  {name} "
                f"({node_type})"
            )

            print(line)
            output_lines.append(line)


    # ============================================================
    # 7. DEGREE DISTRIBUTION
    # ============================================================

    print_section(
        "7. DEGREE DISTRIBUTION",
        output_lines
    )

    degrees = np.array(
        [
            degree
            for _, degree in G.degree()
        ],
        dtype=float
    )

    print(
        f"Mean degree   : "
        f"{degrees.mean():.4f}"
    )

    print(
        f"Median degree : "
        f"{np.median(degrees):.4f}"
    )

    print(
        f"Minimum degree: "
        f"{degrees.min():.0f}"
    )

    print(
        f"Maximum degree: "
        f"{degrees.max():.0f}"
    )

    print(
        f"90th percentile: "
        f"{np.percentile(degrees, 90):.2f}"
    )

    print(
        f"99th percentile: "
        f"{np.percentile(degrees, 99):.2f}"
    )

    output_lines.extend([
        f"Mean degree   : {degrees.mean():.4f}",
        f"Median degree : {np.median(degrees):.4f}",
        f"Minimum degree: {degrees.min():.0f}",
        f"Maximum degree: {degrees.max():.0f}",
        f"90th percentile: {np.percentile(degrees, 90):.2f}",
        f"99th percentile: {np.percentile(degrees, 99):.2f}",
    ])


    # ============================================================
    # 8. IMPORTANT INGREDIENTS
    # ============================================================

    print_section(
        "8. IMPORTANT INGREDIENT CONNECTIVITY",
        output_lines
    )

    ingredient_lookup = find_ingredient_nodes(
        G
    )

    print(
        f"Ingredient names available: "
        f"{len(ingredient_lookup):,}"
    )

    output_lines.append(
        f"Ingredient names available: "
        f"{len(ingredient_lookup):,}"
    )

    print(
        "\nIngredient checks:"
    )

    output_lines.append(
        "\nIngredient checks:"
    )

    # ------------------------------------------------------------
    # Identify the largest component
    # ------------------------------------------------------------

    largest_component = set(
        components[0]
    )

    for ingredient in IMPORTANT_INGREDIENTS:

        node = ingredient_lookup.get(
            ingredient
        )

        if node is None:

            line = (
                f"{ingredient:20s} "
                f"NOT FOUND"
            )

            print(line)
            output_lines.append(line)

            continue


        # --------------------------------------------------------
        # Basic graph information
        # --------------------------------------------------------

        degree = G.degree(
            node
        )

        in_main_component = (
            node in largest_component
        )


        # --------------------------------------------------------
        # Edge type information
        # --------------------------------------------------------

        co_occurs_count = 0
        molecule_count = 0

        for _, _, data in G.edges(
            node,
            data=True
        ):

            edge_type = str(
                data.get(
                    "edge_type",
                    ""
                )
            ).lower()

            if edge_type == "ingr-ingr":

                co_occurs_count += 1

            elif edge_type == "ingr-dcomp":

                molecule_count += 1


        line = (
            f"{ingredient:20s} "
            f"| degree={degree:4d} "
            f"| CO_OCCURS={co_occurs_count:4d} "
            f"| molecules={molecule_count:4d} "
            f"| main_component={in_main_component}"
        )

        print(line)
        output_lines.append(line)


    # ============================================================
    # 9. LOAD EMBEDDINGS
    # ============================================================

    print_section(
        "9. LEARNED EMBEDDINGS",
        output_lines
    )

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    output_lines.append(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )


    # ------------------------------------------------------------
    # Load node mapping
    # ------------------------------------------------------------

    with open(
        MAPPING_FILE,
        "rb"
    ) as f:

        mapping_data = pickle.load(f)

    node2idx = mapping_data[
        "node2idx"
    ]

    mapped_nodes = mapping_data[
        "nodes"
    ]


    # ------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------

    print(
        f"Graph nodes       : "
        f"{G.number_of_nodes():,}"
    )

    print(
        f"Mapping nodes     : "
        f"{len(node2idx):,}"
    )

    print(
        f"Embedding rows    : "
        f"{embeddings.shape[0]:,}"
    )

    output_lines.extend([
        f"Graph nodes       : {G.number_of_nodes():,}",
        f"Mapping nodes     : {len(node2idx):,}",
        f"Embedding rows    : {embeddings.shape[0]:,}",
    ])


    if (
        len(node2idx)
        != G.number_of_nodes()
    ):

        print(
            "WARNING: mapping/node count mismatch!"
        )

    if (
        embeddings.shape[0]
        != len(node2idx)
    ):

        print(
            "WARNING: embedding/mapping count mismatch!"
        )


    # ============================================================
    # 10. EMBEDDING NORMS
    # ============================================================

    print_section(
        "10. EMBEDDING NORMALIZATION",
        output_lines
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    print(
        f"Mean norm   : {norms.mean():.6f}"
    )

    print(
        f"Median norm : {np.median(norms):.6f}"
    )

    print(
        f"Min norm    : {norms.min():.6f}"
    )

    print(
        f"Max norm    : {norms.max():.6f}"
    )

    output_lines.extend([
        f"Mean norm   : {norms.mean():.6f}",
        f"Median norm : {np.median(norms):.6f}",
        f"Min norm    : {norms.min():.6f}",
        f"Max norm    : {norms.max():.6f}",
    ])


    # ============================================================
    # 11. IMPORTANT INGREDIENT EMBEDDINGS
    # ============================================================

    print_section(
        "11. IMPORTANT INGREDIENT EMBEDDINGS",
        output_lines
    )

    print(
        "Showing embedding statistics for "
        "important ingredients:"
    )

    output_lines.append(
        "Showing embedding statistics for "
        "important ingredients:"
    )

    for ingredient in IMPORTANT_INGREDIENTS:

        node = ingredient_lookup.get(
            ingredient
        )

        if node is None:
            continue

        if node not in node2idx:
            continue

        idx = node2idx[
            node
        ]

        embedding = embeddings[
            idx
        ]

        norm = np.linalg.norm(
            embedding
        )

        first_values = ", ".join(
            f"{x:.4f}"
            for x in embedding[:5]
        )

        line = (
            f"{ingredient:20s} "
            f"| index={idx:5d} "
            f"| norm={norm:.6f} "
            f"| first5=[{first_values}]"
        )

        print(line)
        output_lines.append(line)


    # ============================================================
    # 12. FINAL SANITY SUMMARY
    # ============================================================

    print_section(
        "12. FINAL SANITY SUMMARY",
        output_lines
    )

    checks = []

    checks.append(
        (
            "Graph contains ingredient nodes",
            len(ingredient_nodes) > 0
        )
    )

    checks.append(
        (
            "Graph contains molecule nodes",
            len(molecule_nodes) > 0
        )
    )

    checks.append(
        (
            "Graph contains CO_OCCURS edges",
            edge_type_counts.get(
                "ingr-ingr",
                0
            ) > 0
        )
    )

    checks.append(
        (
            "Graph contains molecule edges",
            edge_type_counts.get(
                "ingr-dcomp",
                0
            ) > 0
        )
    )

    checks.append(
        (
            "Embeddings have 64 dimensions",
            embeddings.ndim == 2
            and embeddings.shape[1] == 64
        )
    )

    checks.append(
        (
            "Embedding count matches graph",
            embeddings.shape[0]
            == G.number_of_nodes()
        )
    )

    checks.append(
        (
            "No NaN embeddings",
            not np.isnan(
                embeddings
            ).any()
        )
    )

    checks.append(
        (
            "No infinite embeddings",
            not np.isinf(
                embeddings
            ).any()
        )
    )


    for description, passed in checks:

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        line = (
            f"[{status}] {description}"
        )

        print(line)
        output_lines.append(line)


    # ============================================================
    # SAVE REPORT
    # ============================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                output_lines
            )
        )

    print(
        "\nValidation report saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n" + "=" * 70)
    print(
        "GRAPH VALIDATION COMPLETE"
    )
    print("=" * 70)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    main()