import torch
from torch_geometric.data import Data
from torch_geometric.datasets import BA2MotifDataset, ExplainerDataset
from torch_geometric.datasets.graph_generator import TreeGraph
from torch_geometric.datasets.motif_generator import CycleMotif, HouseMotif
from pathlib import Path
import pickle


def _ba2motif_gt_masks(data: Data) -> Data:
    """BA2Motif ships without stored masks. By construction the last 5 nodes
    (indices 20-24) are always the motif; all edges with both endpoints in
    that range are motif-internal.
    """

    n = data.num_nodes
    motif_start = 20

    node_mask = torch.zeros(n)
    node_mask[motif_start:] = 1.0

    ei = data.edge_index
    edge_mask = ((ei[0] >= motif_start) & (ei[1] >= motif_start)).float()

    data.node_mask = node_mask
    data.edge_mask = edge_mask
    return data


def _build_treecircles(
    n_per_class: int = 500, tree_depth: int = 4, seed: int = 0
) -> list:
    """Generates n_per_class graphs per class:
      class 0 — binary tree + 6-node cycle motif
      class 1 — binary tree + house motif

    PyG's ExplainerDataset already produces node_mask and edge_mask.
    We add a graph-level y and a constant node feature vector.
    """

    torch.manual_seed(seed)
    generators = [
        (CycleMotif(6), 0),
        (HouseMotif(), 1),
    ]

    graphs = []
    for motif_gen, label in generators:
        ds = ExplainerDataset(
            graph_generator=TreeGraph(depth=tree_depth, branch=2),
            motif_generator=motif_gen,
            num_motifs=1,
            num_graphs=n_per_class,
        )
        for raw in ds:
            n = raw.num_nodes
            # Constant node features (same convention as BA2Motif)
            x = torch.full((n, 10), 0.1)
            graphs.append(
                Data(
                    x=x,
                    edge_index=raw.edge_index,
                    y=torch.tensor(label, dtype=torch.long),
                    node_mask=raw.node_mask.float(),
                    edge_mask=raw.edge_mask.float(),
                )
            )

    return graphs


if __name__ == '__main__':
    from .util import write_graph_iterable
    from ..config import DATASETS, SEED

    data_dir = Path(DATASETS['output'])
    raw_ba2 = BA2MotifDataset(root=data_dir / 'raw')
    ba2_graphs = [_ba2motif_gt_masks(raw_ba2[i]) for i in range(len(raw_ba2))]
    tc_graphs = _build_treecircles(n_per_class=500, tree_depth=4, seed=SEED)

    for name, graphs in (
        ('ba2_graphs', ba2_graphs),
        ('tc_graphs', tc_graphs)
    ):
        write_graph_iterable(
            graphs=graphs,
            destination= data_dir / 'processed' / f'{name}.pkl',
            overwrite=False,
            seed=SEED,
            test_fraction=DATASETS['test_split'],
            val_fraction=DATASETS['validation_split']
        )