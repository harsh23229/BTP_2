# ================================================================
# TRAIN OUR GASTROGRAPH
#
# Uses:
#   GastroGraph-custom/data/
#       nodes_our_recipeflavor.csv
#       edges_our_recipeflavor.csv
#
# Uses the ORIGINAL GastroGraph GraphSAGE implementation:
#   GastroGraph-main/src/model.py
#
# Outputs:
#   GastroGraph-custom/models/
#       gastro_graph.gpickle
#       node2idx.pkl
#       graphsage_model.pth
#       node_embeddings.npy
# ================================================================

import os
import sys
import pickle

import numpy as np
import networkx as nx
import torch
import torch.optim as optim


# ================================================================
# IMPORT ORIGINAL GASTROGRAPH MODEL
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

# Make sure Python can import model.py
# from the same src/ directory.
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import GraphSAGE


# ================================================================
# PATHS
# ================================================================

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

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


NODES_FILE = os.path.join(
    DATA_DIR,
    "nodes_our_recipeflavor.csv"
)

EDGES_FILE = os.path.join(
    DATA_DIR,
    "edges_our_recipeflavor.csv"
)

GRAPH_FILE = os.path.join(
    MODEL_DIR,
    "gastro_graph.gpickle"
)

MAPPING_FILE = os.path.join(
    MODEL_DIR,
    "node2idx.pkl"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "graphsage_model.pth"
)

EMBEDDINGS_FILE = os.path.join(
    MODEL_DIR,
    "node_embeddings.npy"
)


# ================================================================
# HYPERPARAMETERS
# ================================================================

EPOCHS = 100
LR = 0.01

HIDDEN_DIM = 128
OUTPUT_DIM = 64

SEED = 42
BATCH_SIZE = 2048


# ================================================================
# LOAD OUR GRAPH
# ================================================================

def load_our_graph():

    print("=" * 70)
    print("LOADING OUR CUSTOM GASTROGRAPH")
    print("=" * 70)

    if not os.path.exists(NODES_FILE):
        raise FileNotFoundError(
            f"Nodes file not found:\n{NODES_FILE}"
        )

    if not os.path.exists(EDGES_FILE):
        raise FileNotFoundError(
            f"Edges file not found:\n{EDGES_FILE}"
        )

    import pandas as pd

    nodes_df = pd.read_csv(
        NODES_FILE
    )

    edges_df = pd.read_csv(
        EDGES_FILE
    )

    print(
        f"Nodes CSV loaded: {len(nodes_df):,}"
    )

    print(
        f"Edges CSV loaded: {len(edges_df):,}"
    )

    print("\nNode types:")

    print(
        nodes_df["node_type"]
        .value_counts()
    )

    print("\nEdge types:")

    print(
        edges_df["edge_type"]
        .value_counts()
    )


    # ------------------------------------------------------------
    # Build NetworkX graph
    # ------------------------------------------------------------

    G = nx.Graph()

    for _, row in nodes_df.iterrows():

        G.add_node(
            row["node_id"],
            name=row["name"],
            node_type=row["node_type"],
            is_hub=bool(row["is_hub"]),
        )


    for _, row in edges_df.iterrows():

        G.add_edge(
            row["id_1"],
            row["id_2"],
            edge_type=row["edge_type"],
            weight=float(row["score"]),
        )


    print("\nNetworkX graph:")

    print(
        f"Nodes : {G.number_of_nodes():,}"
    )

    print(
        f"Edges : {G.number_of_edges():,}"
    )

    print(
        f"Connected components : "
        f"{nx.number_connected_components(G):,}"
    )


    # ------------------------------------------------------------
    # Save graph
    # ------------------------------------------------------------

    with open(
        GRAPH_FILE,
        "wb"
    ) as f:

        pickle.dump(
            G,
            f
        )

    print(
        "\nGraph saved to:"
    )

    print(
        GRAPH_FILE
    )

    return G


# ================================================================
# PREPARE ADJACENCY MATRIX
# ================================================================

def prepare_data(G):

    print("\n" + "=" * 70)
    print("PREPARING GRAPH FOR GRAPHSAGE")
    print("=" * 70)

    nodes = list(
        G.nodes()
    )

    node2idx = {
        node: i
        for i, node in enumerate(nodes)
    }

    # ------------------------------------------------------------
    # Save mapping
    # ------------------------------------------------------------

    with open(
        MAPPING_FILE,
        "wb"
    ) as f:

        pickle.dump(
            {
                "node2idx": node2idx,
                "nodes": nodes,
            },
            f
        )

    print(
        f"Number of nodes: {len(nodes):,}"
    )

    print(
        "Node mapping saved to:"
    )

    print(
        MAPPING_FILE
    )


    # ------------------------------------------------------------
    # Build adjacency matrix
    # ------------------------------------------------------------

    num_nodes = len(nodes)

    print(
        "\nCreating adjacency matrix..."
    )

    adj = torch.zeros(
        (num_nodes, num_nodes),
        dtype=torch.float32
    )

    edges_count = 0

    for u, v, d in G.edges(
        data=True
    ):

        i = node2idx[u]
        j = node2idx[v]

        weight = float(
            d.get(
                "weight",
                1.0
            )
        )

        adj[i][j] = weight
        adj[j][i] = weight

        edges_count += 1

    print(
        f"Adjacency matrix built."
    )

    print(
        f"Edges represented: {edges_count:,}"
    )

    print(
        f"Matrix shape: {tuple(adj.shape)}"
    )


    # ------------------------------------------------------------
    # Row-normalize adjacency
    #
    # This matches the original GastroGraph train.py.
    # ------------------------------------------------------------

    row_sum = torch.sum(
        adj,
        dim=1
    )

    row_sum[
        row_sum == 0
    ] = 1

    r_inv = 1.0 / row_sum

    adj_norm = (
        adj
        * r_inv.unsqueeze(1)
    )

    print(
        "Adjacency matrix row-normalized."
    )

    return (
        adj_norm,
        node2idx,
        list(G.edges())
    )


# ================================================================
# UNSUPERVISED LINK-PREDICTION LOSS
# ================================================================

def get_loss(
    z,
    edges,
    node2idx,
    batch_size=2048
):

    num_nodes = z.shape[0]

    # ------------------------------------------------------------
    # Positive edges
    # ------------------------------------------------------------

    indices = np.random.choice(
        len(edges),
        batch_size,
        replace=True
    )

    pos_edges = [
        edges[i]
        for i in indices
    ]

    u_pos_idx = [
        node2idx[u]
        for u, v in pos_edges
    ]

    v_pos_idx = [
        node2idx[v]
        for u, v in pos_edges
    ]

    u_pos = z[
        torch.tensor(
            u_pos_idx,
            dtype=torch.long
        )
    ]

    v_pos = z[
        torch.tensor(
            v_pos_idx,
            dtype=torch.long
        )
    ]

    pos_score = torch.sum(
        u_pos * v_pos,
        dim=1
    )


    # ------------------------------------------------------------
    # Negative edges
    # ------------------------------------------------------------

    u_neg_idx = np.random.randint(
        0,
        num_nodes,
        batch_size
    )

    v_neg_idx = np.random.randint(
        0,
        num_nodes,
        batch_size
    )

    u_neg = z[
        torch.tensor(
            u_neg_idx,
            dtype=torch.long
        )
    ]

    v_neg = z[
        torch.tensor(
            v_neg_idx,
            dtype=torch.long
        )
    ]

    neg_score = torch.sum(
        u_neg * v_neg,
        dim=1
    )


    # ------------------------------------------------------------
    # Binary link prediction loss
    # ------------------------------------------------------------

    pos_loss = (
        -torch.log(
            torch.sigmoid(
                pos_score
            ) + 1e-6
        ).mean()
    )

    neg_loss = (
        -torch.log(
            1
            - torch.sigmoid(
                neg_score
            )
            + 1e-6
        ).mean()
    )

    return (
        pos_loss
        + neg_loss
    )


# ================================================================
# TRAIN
# ================================================================

def train():

    print("\n" + "=" * 70)
    print("OUR GASTROGRAPH — GRAPHSAGE TRAINING")
    print("=" * 70)

    print(
        f"\nEpochs      : {EPOCHS}"
    )

    print(
        f"Learning rate : {LR}"
    )

    print(
        f"Hidden dim  : {HIDDEN_DIM}"
    )

    print(
        f"Output dim  : {OUTPUT_DIM}"
    )

    print(
        f"Batch size  : {BATCH_SIZE}"
    )

    print(
        f"Random seed : {SEED}"
    )


    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )


    # ------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------

    G = load_our_graph()


    # ------------------------------------------------------------
    # Prepare adjacency
    # ------------------------------------------------------------

    (
        adj_norm,
        node2idx,
        all_edges
    ) = prepare_data(G)


    # ------------------------------------------------------------
    # Create GraphSAGE
    #
    # THIS IS THE ORIGINAL GastroGraph model.py
    # ------------------------------------------------------------

    num_nodes = len(
        G.nodes()
    )

    model = GraphSAGE(
        num_nodes,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )


    print("\nModel architecture:")

    print(
        model
    )


    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    print("\n" + "-" * 70)
    print("STARTING TRAINING")
    print("-" * 70)

    model.train()

    for epoch in range(
        EPOCHS
    ):

        optimizer.zero_grad()

        z = model(
            adj_norm
        )

        loss = get_loss(
            z,
            all_edges,
            node2idx,
            batch_size=BATCH_SIZE
        )

        loss.backward()

        optimizer.step()


        if (
            epoch + 1
        ) % 10 == 0:

            print(
                f"Epoch "
                f"{epoch + 1:3d}/{EPOCHS}"
                f" | Loss: "
                f"{loss.item():.4f}"
            )


    print(
        "\nTraining complete."
    )


    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_FILE
    )


    # ------------------------------------------------------------
    # Generate final embeddings
    # ------------------------------------------------------------

    model.eval()

    with torch.no_grad():

        final_z = model(
            adj_norm
        )

    embeddings = (
        final_z
        .cpu()
        .numpy()
    )


    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )


    print(
        "\nEmbeddings saved to:"
    )

    print(
        EMBEDDINGS_FILE
    )

    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )


    # ------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING OUTPUT VERIFICATION")
    print("=" * 70)

    print(
        f"Nodes              : {num_nodes:,}"
    )

    print(
        f"Embedding dimension: {embeddings.shape[1]}"
    )

    print(
        f"Embedding rows     : {embeddings.shape[0]:,}"
    )

    print(
        "\nFiles created:"
    )

    print(
        f"1. {GRAPH_FILE}"
    )

    print(
        f"2. {MAPPING_FILE}"
    )

    print(
        f"3. {MODEL_FILE}"
    )

    print(
        f"4. {EMBEDDINGS_FILE}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "OUR GASTROGRAPH TRAINING COMPLETE."
    )

    print(
        "=" * 70
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    train()