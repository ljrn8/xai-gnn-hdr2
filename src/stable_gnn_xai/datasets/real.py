"""
Ground truth node and edge masks for MUTAG (graph classification, 188 graphs).

The mutagenic functional groups NO2 and NH2 are the ground truth explanations,
as established in the GNNExplainer paper (Ying et al., 2019).

Node features are 7-dim one-hot vectors: [C, N, O, F, I, Cl, Br]
Edge features are 4-dim one-hot vectors: [aromatic, single, double, triple]

A node mask value of 1 means the node is part of a mutagenic functional group.
An edge mask value of 1 means the edge connects nodes within a mutagenic group.

build_real_world.py

Downloads and processes four additional binary graph-classification datasets
into the *same* `Data`-iterable format used by `build_synthetic.py`
(ba2_graphs / tc_graphs): every graph carries `x`, `edge_index`, `y`,
`node_mask`, `edge_mask`.

Datasets
--------
1. benzene              (Sanchez-Lengeling et al., 2020)   12,000 graphs
2. alkane_carbonyl       "         "          "              4,326 graphs
3. fluoride_carbonyl     "         "          "              8,671 graphs
4. reddit_binary        (Yanardag & Vishwanathan, 2015)      ~2,000 graphs



Sources
-------
(1)-(3) are pulled directly from the official `google-research/graph-attribution`
GitHub repo (raw.githubusercontent.com), which is where GraphXAI's own copies
of these three datasets originate. No GCS/Dataverse access needed.

(4) is a standard TUDataset and is fetched automatically by torch_geometric.

*** IMPORTANT — verified mapping, not taken from the repo's own labels ***
The graph-attribution repo stores Alkane-Carbonyl and Fluoride-Carbonyl under
the internal folder names `logic7` / `logic8`. The repo's own
`graph_attribution/visualizations.py` labels these as:
    'logic7': 'Fluoride AND Carbonyl'
    'logic8': 'Unbranched Alkane AND Carbonyl'
This is backwards relative to the actual data. Cross-checking graph counts
against the published paper (Alkane-Carbonyl = 4,326 graphs, Fluoride-Carbonyl
= 8,671 graphs) AND directly testing for fluorine-atom occurrence in the
molecules (100% of logic8's positive class contains a fluorine atom, vs ~15%
of logic7's positive class) confirms the correct mapping is:
    logic7  -> alkane_carbonyl   (4,326 graphs)
    logic8  -> fluoride_carbonyl (8,671 graphs)
This module uses the empirically-verified mapping below (_GA_TASK_TO_FOLDER).

*** Note - REDDIT-BINARY has no ground-truth explanation 
"""

import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from pathlib import Path
import io
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import torch
from torch_geometric.data import Data
from loguru import logger

GA_RAW_BASE = "https://raw.githubusercontent.com/google-research/graph-attribution/main/data"

# Empirically-verified mapping (see module docstring) — do NOT trust the
# graph_attribution repo's own visualizations.py labels for logic7/logic8.
_GA_TASK_TO_FOLDER = {
    "benzene": "benzene",
    "alkane_carbonyl": "logic7",
    "fluoride_carbonyl": "logic8",
}

_GA_FILES = ("y_true.npz", "x_true.npz", "true_raw_attribution_datadicts.npz")

def _cached_download(url: str, dest: Path) -> Path:
    """Download url to dest if dest doesn't already exist (idempotent cache,
    same pattern as PyG's own dataset download-caching)."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url) as resp:
            dest.write_bytes(resp.read())
    return dest


def _load_graph_attribution_task(task_name: str, data_dir: Path) -> list:
    """Loads one of {benzene, alkane_carbonyl, fluoride_carbonyl} from the
    graph-attribution repo and converts it into a list of `Data` objects with
    GT node_mask / edge_mask, matching the BA2Motif/TreeCircles convention.
    """
    folder = _GA_TASK_TO_FOLDER[task_name]
    raw_dir = data_dir / "raw" / "graph_attribution" / folder

    local_paths = {
        fname: _cached_download(f"{GA_RAW_BASE}/{folder}/{fname}", raw_dir / fname)
        for fname in _GA_FILES
    }

    y_all = np.load(local_paths["y_true.npz"], allow_pickle=True)["y"].reshape(-1)
    # x_true's datadict_list has shape (1, N); attribution's has shape (N, 1).
    x_dicts = np.load(local_paths["x_true.npz"], allow_pickle=True)["datadict_list"][0]
    att_dicts = np.load(
        local_paths["true_raw_attribution_datadicts.npz"], allow_pickle=True
    )["datadict_list"][:, 0]

    graphs = []
    for i in range(len(y_all)):
        xd = x_dicts[i]
        ad = att_dicts[i]

        # Node features: one-hot atom type (14-dim), directly from the
        # official featurization — these are real/informative, unlike the
        # constant features used in BA2Motif/TreeCircles, since atom
        # identity is exactly what a molecule classifier should rely on.
        x = torch.tensor(xd["nodes"], dtype=torch.float)

        # senders/receivers are already symmetric (both directions stored),
        # i.e. directly usable as PyG edge_index with no need to mirror.
        edge_index = torch.tensor(
            np.stack([xd["senders"], xd["receivers"]]), dtype=torch.long
        )

        # GT node mask: attribution is stored per-motif-instance (a molecule
        # with e.g. 2 separate benzene rings gets 2 (+1 union) columns).
        # Union across columns gives "is this atom part of ANY valid
        # explanation" -- the single ground-truth mask we want, matching
        # the papers' "any combinations of X and Y" phrasing.
        att_nodes = ad["nodes"]
        node_mask = torch.tensor(
            (att_nodes.sum(axis=1) > 0).astype("float32")
        )

        # No edge-level GT is stored for these datasets (att["edges"] is
        # None) -- reconstruct exactly like _ba2motif_gt_masks: an edge is
        # in the explanation iff both endpoints are.
        src, dst = edge_index
        edge_mask = ((node_mask[src] > 0) & (node_mask[dst] > 0)).float()

        graphs.append(
            Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor(int(y_all[i]), dtype=torch.long),
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        )

    return graphs


def _load_reddit_binary(data_dir: Path, max_degree: int = 500) -> list:
    """Loads REDDIT-BINARY via PyG's TUDataset (auto-downloaded) and adds
    one-hot node-degree features (the standard convention for this dataset,
    since it ships with no node features/attributes at all -- see the GIN
    paper, Xu et al. 2019, for the same convention).

    node_mask / edge_mask are all-zero PLACEHOLDERS -- see module docstring.
    This dataset has no real ground-truth explanation in the literature.
    """
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import degree

    raw = TUDataset(root=str(data_dir / "raw" / "reddit_binary"), name="REDDIT-BINARY")

    graphs = []
    for data in raw:
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes).long()
        deg = deg.clamp(max=max_degree)
        x = torch.nn.functional.one_hot(deg, num_classes=max_degree + 1).float()

        n_edges = data.edge_index.size(1)
        y = data.y.view(-1)[0].clone().long() if data.y.dim() else data.y.clone().long()

        graphs.append(
            Data(
                x=x,
                edge_index=data.edge_index.clone(),
                y=y,
                node_mask=torch.zeros(data.num_nodes),  # PLACEHOLDER -- no real GT
                edge_mask=torch.zeros(n_edges),          # PLACEHOLDER -- no real GT
            )
        )

    return graphs


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
    """Identify nodes belonging to NO2 or NH2 groups.

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
    """Compute ground truth node and edge masks for a single MUTAG graph.

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


def generate_annotated_dataset(data_raw_dir: Path):
    """Compute masks for every graph in the dataset."""

    dataset = TUDataset(root=data_raw_dir / "raw", name="MUTAG")
    print(f"Loaded {len(dataset)} graphs")
    print(f"Classes: {dataset.num_classes}, Node features: {dataset.num_node_features}")

    graphs = [g for g in dataset]
    for i, graph in enumerate(graphs):
        node_mask, edge_mask = compute_masks(graph)
        graph.edge_mask = edge_mask
        graph.node_mask = node_mask

    return graphs


if __name__ == "__main__":
    from .util import write_graph_iterable
    from ..config import DATASETS, SEED

    # Mutag
    logger.info('ds: mutag')
    write_graph_iterable(
        graphs=generate_annotated_dataset(DATASETS["output"] / "raw"),
        destination=DATASETS["output"] / "processed" / "mutag.pkl",
        seed=SEED,
        test_fraction=DATASETS["test_split"],
        val_fraction=DATASETS["validation_split"],
    )


    # others
    data_dir = Path(DATASETS["output"])
    tasks = {
        "benzene_graphs": lambda: _load_graph_attribution_task("benzene", data_dir),
        "alkane_carbonyl_graphs": lambda: _load_graph_attribution_task(
            "alkane_carbonyl", data_dir
        ),
        "fluoride_carbonyl_graphs": lambda: _load_graph_attribution_task(
            "fluoride_carbonyl", data_dir
        ),
        "reddit_binary_graphs": lambda: _load_reddit_binary(data_dir),
    }

    for name, loader_fn in tasks.items():
        logger.info(f'ds: {name}')
        graphs = loader_fn()
        write_graph_iterable(
            graphs=graphs,
            destination=data_dir / "processed" / f"{name}.pkl",
            overwrite=False,
            seed=SEED,
            test_fraction=DATASETS["test_split"],
            val_fraction=DATASETS["validation_split"],
        )
