import numpy as np
import torch
from typing import Iterable
from torch_geometric.data import Data
from pathlib import Path
import pickle
from loguru import logger


def write_graph_iterable(
    graphs: Iterable[Data],
    destination: Path,
    seed,
    test_fraction,
    val_fraction,
    overwrite=False,
):
    """Attribute validation_mask, train_mask and write dataset to processed destination"""

    assert hasattr(graphs[0], "edge_mask")
    graphs = attribute_split_to_graph_list(graphs, seed, test_fraction, val_fraction)
    parent = destination.parent
    parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"writing dataset: {destination}")
    if (destination.exists() and overwrite) or not destination.exists():
        with open(destination, "wb") as f:
            pickle.dump(graphs, f)

    logger.info("done")


def attribute_split_to_graph_list(
    graphs: Iterable[Data], seed, test_fraction, val_fraction
):
    """Add integer test_mask and validation_mask attributes to each graph"""

    np.random.seed(seed)
    np.random.shuffle(graphs)
    num_graphs = len(graphs)
    test_size = int(num_graphs * test_fraction)
    val_size = int(num_graphs * val_fraction)
    val_mask, test_mask = torch.zeros(num_graphs), torch.zeros(num_graphs)
    test_mask[:test_size] = 1
    val_mask[test_size : test_size + val_size] = 1
    for i, graph in enumerate(graphs):
        graph.test_mask = test_mask[i]
        graph.validation_mask = val_mask[i]

    return graphs
