import os
import sys
import torch
import torch.nn.functional as F
from loguru import logger
import pickle
import numpy as np
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.utils import from_networkx
from typing import Optional, Callable
import json
import pickle as pkl
from pathlib import Path

STANDARD_DATASETS = [
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliett",
    # NOTE: excluded due to 20x size
    # "oscar",
    # "november",
    # "kilo",
    # "lima",
    # "mike",
]


class OGXBenchmark(InMemoryDataset):
    """Class taken from Fontanesi et al. (2025), see https://github.com/OpenGraphXAI/benchmarks"""

    url = r"https://github.com/OpenGraphXAI/benchmarks/raw/refs/heads/main/data/raw/"

    def __init__(
        self,
        root: str,
        name: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):

        assert (
            split in ["train", "val", "test"] or split is None
        ), f'Unknown split: "{split}"'

        self.name_id = name
        self.name = f"OGX_{self.name_id}"

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

        if split == "train":
            self.load(self.processed_paths[1])
        elif split == "val":
            self.load(self.processed_paths[2])
        elif split == "test":
            self.load(self.processed_paths[3])
        else:
            self.load(self.processed_paths[0])

    def download(self):
        for raw_file in self.raw_file_names:
            download_url(f"{self.url}{raw_file}", self.raw_dir)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return [f"{self.name_id}.pkl", f"{self.name_id}_splits.pkl"]

    @property
    def processed_file_names(self):
        return [f"{self.name}.pt"] + [
            f"{self.name}_{split}.pt" for split in ["train", "val", "test"]
        ]

    def process(self):
        with open(self.raw_paths[0], "rb") as f:
            graphs = pkl.load(f)

        with open(self.raw_paths[1], "rb") as f:
            splits = pkl.load(f)

        data_list = []

        for class_idx in (0, 1):
            for graph in graphs[f"class{class_idx}"]:
                data = from_networkx(graph)
                data_list.append(
                    Data(
                        x=data.x,
                        edge_index=data.edge_index,
                        mask=(
                            data.mask
                            if hasattr(data, "mask")
                            else torch.zeros_like(data.x).bool()
                        ),
                        mask_root=(
                            data.mask_root
                            if hasattr(data, "mask_root")
                            else torch.zeros_like(data.x).bool()
                        ),
                        y=torch.tensor([class_idx]),
                    )
                )

        for i, split in enumerate(["train", "val", "test"]):
            split_data = [data_list[idx] for idx in splits[0][split]]
            if self.pre_filter is not None:
                split_data = [data for data in split_data if self.pre_filter(data)]
            if self.pre_transform is not None:
                split_data = [self.pre_transform(data) for data in split_data]
            self.save(split_data, self.processed_paths[i + 1])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def __repr__(self):
        return f"{self.name}({len(self)})"


def OHE_graph(G, num_classes):
    assert G.x.dim() == 1, "Expected node features to be 1D for OHE encoding"
    assert (
        int(G.x.max().item()) + 1 < 1000
    ), "Expected at less than 1000 classes for OHE encoding"
    G.x = F.one_hot(G.x, num_classes=num_classes).float()
    return G


def node_mask_to_edge_mask(edge_index, node_mask):
    src, dst = edge_index
    return node_mask[src] & node_mask[dst]


def yield_datasets(raw_directory: Path, datasets=STANDARD_DATASETS):
    for ds_name in datasets:
        logger.info(f" Dataset name: {ds_name}")
        ds = OGXBenchmark(root=raw_directory, name=ds_name)
        x_pool = torch.cat([G.x for G in ds], dim=0)
        num_classes = int(x_pool.max().item()) + 1

        print(f"statistics:")
        print(f"num graphs: {len(ds)}")
        print(f"x shape: {ds[0].x.shape}")
        print(f"x example: {ds[0].x}")
        print(f"num x classes: {num_classes}")
        
        graphs = [OHE_graph(G, num_classes=num_classes) for G in ds]

        for graph in graphs:
            node_mask = graph.mask.bool()
            if node_mask.dim() > 1:
                node_mask = node_mask.squeeze(-1)

            graph.edge_mask = node_mask_to_edge_mask(graph.edge_index, node_mask).float()

        yield graphs, ds_name


def main():
    from ..config import DATASETS, SEED
    from .util import write_graph_iterable
    data_dir = Path(DATASETS["output"])
    (data_dir / 'processed').mkdir(exist_ok=True, parents=True)
    (data_dir / 'raw').mkdir(exist_ok=True, parents=True)

    for graphs, ds_name in yield_datasets(raw_directory=Path(DATASETS['output'] / "raw")):
        write_graph_iterable(
            graphs=graphs,
            destination=data_dir / "processed" / f"{ds_name}.pkl",
            seed=SEED,
            test_fraction=DATASETS["test_split"],
            val_fraction=DATASETS["validation_split"],
        )
        logger.info(f'graphs[0] edge_mask shape: {graphs[0].edge_mask.shape} edge_index shape: {graphs[0].edge_index.shape}')
        logger.info(f'n nodes in graphs[0]: {graphs[0].num_nodes} n edges: {graphs[0].num_edges}')


if __name__ == "__main__":
    main()
