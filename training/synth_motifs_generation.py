import torch
from torch_geometric.data import Data
from torch_geometric.datasets import BA2MotifDataset, ExplainerDataset
from torch_geometric.datasets.graph_generator import TreeGraph
from torch_geometric.datasets.motif_generator import CycleMotif, HouseMotif


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


def _build_treecircles(n_per_class: int = 500, tree_depth: int = 4,
                       seed: int = 0) -> list:
    """
    Generates n_per_class graphs per class:
      class 0 — binary tree + 6-node cycle motif
      class 1 — binary tree + house motif

    PyG's ExplainerDataset already produces node_mask and edge_mask.
    We add a graph-level y and a constant node feature vector.
    """
    torch.manual_seed(seed)
    generators = [
        (CycleMotif(6), 0),
        (HouseMotif(),  1),
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
            graphs.append(Data(
                x=x,
                edge_index=raw.edge_index,
                y=torch.tensor(label, dtype=torch.long),
                node_mask=raw.node_mask.float(),
                edge_mask=raw.edge_mask.float(),
            ))

    return graphs


def _split(graphs: list, train: float, val: float, test: float,
           seed: int) -> dict:
    assert abs(train + val + test - 1.0) < 1e-6, "Fractions must sum to 1."
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(graphs), generator=g).tolist()
    n = len(graphs)
    n_train = int(n * train)
    n_val   = int(n * val)
    return {
        "train": [graphs[i] for i in idx[:n_train]],
        "val":   [graphs[i] for i in idx[n_train:n_train + n_val]],
        "test":  [graphs[i] for i in idx[n_train + n_val:]],
    }


def load_datasets(
    train: float = 0.8,
    val:   float = 0.1,
    test:  float = 0.1,
    seed:  int   = 42,
    ba2motif_root: str = "/tmp/ba2motif",
) -> tuple[dict, dict]:
    """
    Returns (ba2motif_splits, treecircles_splits), each a dict with keys
    'train', 'val', 'test' mapping to a list of torch_geometric.data.Data.

    Parameters
    ----------
    train / val / test : floats summing to 1.0
    seed               : controls both dataset generation (TreeCircles) and splits
    ba2motif_root      : cache directory for the BA2Motif pickle download
    """
    # --- BA2Motif ---
    raw_ba2 = BA2MotifDataset(root=ba2motif_root)
    ba2_graphs = [_ba2motif_gt_masks(raw_ba2[i]) for i in range(len(raw_ba2))]
    ba2_splits = _split(ba2_graphs, train, val, test, seed)

    # --- TreeCircles ---
    tc_graphs  = _build_treecircles(n_per_class=500, tree_depth=4, seed=seed)
    tc_splits  = _split(tc_graphs,  train, val, test, seed)

    return ba2_splits, tc_splits


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
    print(f"  Train / Val / Test : "
          f"{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
    print(f"  Node feature dim   : {sample.x.shape[1]}")
    print(f"    -> all-constant (0.1); purely structural task")
    print(f"  Num classes        : 2  (binary)")

    for split_name, graphs in splits.items():
        cc = _class_counts(graphs)
        print(f"  {split_name:5s} class dist : label0={cc[0]}  label1={cc[1]}")

    sample_sizes = torch.tensor([g.num_nodes for g in all_graphs], dtype=torch.float)
    print(f"  Nodes per graph    : min={int(sample_sizes.min())}  "
          f"max={int(sample_sizes.max())}  mean={sample_sizes.mean():.1f}")

    mean_motif = _motif_node_stats(all_graphs)
    print(f"  Motifs per graph   : 1  (GT mask: {mean_motif:.1f} motif nodes avg)")
    print(f"  GT masks           : node_mask + edge_mask  (1=motif, 0=base)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val",   type=float, default=0.1)
    p.add_argument("--test",  type=float, default=0.1)
    p.add_argument("--seed",  type=int,   default=42)
    args = p.parse_args()

    print(f"Split: train={args.train}  val={args.val}  test={args.test}  seed={args.seed}")

    ba2_splits, tc_splits = load_datasets(
        train=args.train, val=args.val, test=args.test, seed=args.seed
    )

    summarise("BA2Motif  (BA graph + house or 5-cycle)", ba2_splits)
    summarise("TreeCircles (binary tree + cycle or house)", tc_splits)

    print("\nSample access:")
    g = ba2_splits["train"][0]
    print(f"  g.x.shape={g.x.shape}  g.y={g.y.item()}  "
          f"node_mask.sum={int(g.node_mask.sum())}  edge_mask.sum={int(g.edge_mask.sum())}")
    
    from pathlib import Path; import pickle
    root = Path('../output/BA-5cycle')
    root.mkdir(exist_ok=True)
    with open(root / 'test_graphs.pkl', 'wb') as f:
        pickle.dump(ba2_splits['test'], f)

    with open(root / 'train_graphs.pkl', 'wb') as f:
        pickle.dump(ba2_splits['train'], f)

    with open(root / 'val_graphs.pkl', 'wb') as f:
        pickle.dump(ba2_splits['val'], f)

    root = Path('../output/6cycle-house')
    root.mkdir(exist_ok=True)
    with open(root / 'test_graphs.pkl', 'wb') as f:
        pickle.dump(ba2_splits['test'], f)

    with open(root / 'train_graphs.pkl', 'wb') as f:
        pickle.dump(ba2_splits['train'], f)

    with open(root / 'val_graphs.pkl', 'wb') as f:
        pickle.dump(ba2_splits['val'], f)