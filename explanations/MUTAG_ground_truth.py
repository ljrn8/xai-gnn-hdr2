"""
Ground truth node and edge masks for MUTAG (graph classification, 188 graphs).

The mutagenic functional groups NO2 and NH2 are the ground truth explanations,
as established in the GNNExplainer paper (Ying et al., 2019).

Node features are 7-dim one-hot vectors: [C, N, O, F, I, Cl, Br]
Edge features are 4-dim one-hot vectors: [aromatic, single, double, triple]

A node mask value of 1 means the node is part of a mutagenic functional group.
An edge mask value of 1 means the edge connects nodes within a mutagenic group.
"""

import torch
import numpy as np
from torch_geometric.datasets import TUDataset

# MUTAG atom type encoding (one-hot column order)
ATOM_TYPES = ["C", "N", "O", "F", "I", "Cl", "Br"]
N_IDX = ATOM_TYPES.index("N")  # 1
O_IDX = ATOM_TYPES.index("O")  # 2


def get_atom_type(node_features: torch.Tensor) -> int:
    """Return the atom index (into ATOM_TYPES) for a one-hot node feature vector."""
    return int(node_features.argmax().item())


def build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> dict:
    """Return {node: set(neighbors)} from a COO edge_index."""
    adj = {i: set() for i in range(num_nodes)}
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        adj[s].add(d)
        adj[d].add(s)
    return adj


def find_mutagenic_nodes(x: torch.Tensor, edge_index: torch.Tensor) -> set:
    """
    Identify nodes belonging to NO2 or NH2 groups.

    NO2: a nitrogen (N) node whose neighbors are all oxygens (O), with exactly 2 O neighbors.
         The two O nodes are also included.

    NH2: a nitrogen (N) node with no oxygen neighbors and degree <= 2
         (the H atoms are implicit in MUTAG, so a low-degree N not bonded to O = NH2).
         Only the N node itself is flagged (no explicit H nodes exist).

    Returns a set of node indices that are part of a mutagenic group.
    """
    num_nodes = x.shape[0]
    adj = build_adjacency(edge_index, num_nodes)
    mutagenic = set()

    for node in range(num_nodes):
        atom = get_atom_type(x[node])
        if atom != N_IDX:
            continue

        neighbors = adj[node]
        neighbor_atoms = [get_atom_type(x[nb]) for nb in neighbors]
        o_neighbors = [nb for nb, a in zip(neighbors, neighbor_atoms) if a == O_IDX]
        non_o_neighbors = [nb for nb, a in zip(neighbors, neighbor_atoms) if a != O_IDX]

        # NO2: exactly 2 oxygen neighbors, no non-O non-implicit neighbors
        # (in practice: all neighbors are O, degree == 2)
        if len(o_neighbors) == 2 and len(non_o_neighbors) <= 1:
            # The N and its two O atoms form NO2
            mutagenic.add(node)
            mutagenic.update(o_neighbors)

        # NH2: nitrogen with no oxygen neighbors and degree <= 2
        # (implicit H, so N has degree 1 or 2 in the graph when it's NH2)
        elif len(o_neighbors) == 0 and len(neighbors) <= 2:
            mutagenic.add(node)

    return mutagenic


def compute_masks(data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ground truth node and edge masks for a single MUTAG graph.

    Returns:
        node_mask: BoolTensor of shape [num_nodes], True if node is in mutagenic group
        edge_mask: BoolTensor of shape [num_edges], True if edge is within mutagenic group
    """
    x = data.x  # [num_nodes, 7]
    edge_index = data.edge_index  # [2, num_edges]
    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]

    mutagenic_nodes = find_mutagenic_nodes(x, edge_index)

    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for n in mutagenic_nodes:
        node_mask[n] = True

    # An edge is in the ground truth mask if BOTH endpoints are mutagenic
    src, dst = edge_index
    edge_mask = torch.zeros(num_edges, dtype=torch.bool)
    for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
        if s in mutagenic_nodes and d in mutagenic_nodes:
            edge_mask[i] = True

    return node_mask, edge_mask


def compute_all_masks(dataset) -> list[dict]:
    """
    Compute masks for every graph in the dataset.

    Returns a list of dicts with keys:
        'graph_idx', 'label', 'node_mask', 'edge_mask',
        'num_mutagenic_nodes', 'has_explanation'
    """
    results = []
    for i, data in enumerate(dataset):
        node_mask, edge_mask = compute_masks(data)
        results.append(
            {
                "graph_idx": i,
                "label": int(data.y.item()),
                "node_mask": node_mask,
                "edge_mask": edge_mask,
                "num_mutagenic_nodes": int(node_mask.sum().item()),
                "has_explanation": bool(node_mask.any().item()),
            }
        )
    return results


if __name__ == "__main__":
    dataset = TUDataset(root="raw/TUDataset", name="MUTAG")
    print(f"Loaded {len(dataset)} graphs")
    print(f"Classes: {dataset.num_classes}, Node features: {dataset.num_node_features}")

    masks = compute_all_masks(dataset)

    # Summary statistics
    mutagenic_graphs = [m for m in masks if m["label"] == 1]
    nonmutagenic_graphs = [m for m in masks if m["label"] == 0]

    with_explanation = sum(1 for m in masks if m["has_explanation"])
    print(f"\nGraphs with at least one mutagenic node: {with_explanation}/{len(masks)}")
    print(f"Mutagenic graphs (label=1): {len(mutagenic_graphs)}")
    print(f"Non-mutagenic graphs (label=0): {len(nonmutagenic_graphs)}")

    # Per-graph example
    print("\nFirst 5 graphs:")
    for m in masks[:5]:
        print(
            f"  Graph {m['graph_idx']:3d} | label={m['label']} | "
            f"mutagenic_nodes={m['num_mutagenic_nodes']:2d} | "
            f"mutagenic_edges={int(m['edge_mask'].sum()):2d}"
        )

    # Access individual masks
    graph_idx = 0
    node_mask, edge_mask = masks[graph_idx]["node_mask"], masks[graph_idx]["edge_mask"]
    print(f"\nGraph 0 node_mask shape: {node_mask.shape}, dtype: {node_mask.dtype}")
    print(f"Graph 0 edge_mask shape: {edge_mask.shape}, dtype: {edge_mask.dtype}")
    print(f"Graph 0 node_mask: {node_mask.int()}")
    print(f"Graph 0 edge_mask: {edge_mask.int()}")
