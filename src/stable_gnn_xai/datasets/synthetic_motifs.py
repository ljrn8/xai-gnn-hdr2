import torch
from torch_geometric.data import Data
from torch_geometric.datasets import BA2MotifDataset, ExplainerDataset
from torch_geometric.datasets.graph_generator import TreeGraph
from torch_geometric.datasets.motif_generator import CycleMotif, HouseMotif
from pathlib import Path
import pickle


def _ba2motif_gt_masks(data: Data) -> Data:
    """
    BA2Motif ships without stored masks. By construction the last 5 nodes
    (indices 20–24) are always the motif; all edges with both endpoints in
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
    """
    Generates n_per_class graphs per class:
      class 0 — binary tree + 6-node cycle motif
      class 1 — binary tree + house motif

    PyG's ExplainerDataset already produces node_mask and mask.
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


def _class_counts(graphs: list) -> dict:
    labels = torch.tensor([g.y.item() for g in graphs])
    counts = torch.bincount(labels, minlength=2)
    return {0: counts[0].item(), 1: counts[1].item()}


def _motif_node_stats(graphs: list) -> tuple:
    """Returns (mean motif nodes per graph, motifs per graph — always 1 here)."""
    motif_counts = [g.node_mask.sum().item() for g in graphs]
    return sum(motif_counts) / len(motif_counts)


def summarise(name: str, splits: dict) -> None:
    all_graphs = splits["train"] + splits["val"] + splits["test"]
    sample = all_graphs[0]

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Total graphs   : {len(all_graphs)}")
    print(
        f"  Train / Val / Test : "
        f"{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}"
    )
    print(f"  Node feature dim   : {sample.x.shape[1]}")
    print(f"    -> all-constant (0.1); purely structural task")
    print(f"  Num classes        : 2  (binary)")

    for split_name, graphs in splits.items():
        cc = _class_counts(graphs)
        print(f"  {split_name:5s} class dist : label0={cc[0]}  label1={cc[1]}")

    sample_sizes = torch.tensor([g.num_nodes for g in all_graphs], dtype=torch.float)
    print(
        f"  Nodes per graph    : min={int(sample_sizes.min())}  "
        f"max={int(sample_sizes.max())}  mean={sample_sizes.mean():.1f}"
    )

    mean_motif = _motif_node_stats(all_graphs)
    print(f"  Motifs per graph   : 1  (GT mask: {mean_motif:.1f} motif nodes avg)")
    print(f"  GT masks           : node_mask + mask  (1=motif, 0=base)")


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