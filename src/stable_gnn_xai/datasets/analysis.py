import os
from pathlib import Path
import pickle
import torch

from ..config import DATASETS

datasets = os.listdir(DATASETS['output'] / 'processed')
for ds in datasets:
    with open(DATASETS['output'] / 'processed' / ds, 'rb') as f:
        graphs = pickle.load(f)

    # log details
    n_classes = len(set([g.y.item() for g in graphs]))
    n_graphs = len(graphs)

    # in th edgemask atttrute (0/1 per edge). whats the expectation per graph 
    average_edge_mask_proportion = sum([g.edge_mask.sum().item() / g.edge_mask.numel() for g in graphs]) / n_graphs

    print(f"Dataset: {ds} | n_graphs: {n_graphs} | n_classes: {n_classes} | average_edge_mask_proportion: {average_edge_mask_proportion:.4f}")

