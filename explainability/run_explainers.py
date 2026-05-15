import os
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch import nn
import torch.functional as F
import sys
from abc import ABC, abstractmethod
from torch_geometric.data import Data
from training.GNN_utils import *
from explainability.xAI_utils import *
from pathlib import Path
import argparse
from explainability.baselines.ORExplainer import ORExplainer

torch.autograd.set_detect_anomaly(True, check_nan=False)

parser = argparse.ArgumentParser()
parser.add_argument("--ds-root", description="Root directory containing dataset to evaluate on")
parser.add_argument('-n', '--node-level', action='store_true')
parser.add_argument('-g', '--graph-level', action='store_true')

# generic parameters
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--hidden-size", type=int, default=64)

args = parser.parse_args()
assert args.graph_level != args.node_level, "Please specify exactly one of --graph-level or --node-level"

for explainer_name, expl in [
    # ("PROXYExplainer", ProxyExplainerImpl(epochs=args.epochs)),
    ("ORExplainer", ORExplainer(hidden_channels=args.hidden_size, epochs=args.epochs, gamma=0.1)),
]:
    files = os.listdir(args.ds_root)
    for dataset in files:
        path = Path(args.ds_root) / dataset
        print(f"\n\n Dataset name: {dataset} \n")
        
        if args.graph_level:
            test_graphs = openpkl(path / "test_graphs.pkl")
            models_path = path / "models"
            graph_runs = {
                model: openpkl(models_path / model) for model in os.listdir(models_path)
            }
            for model_name, graph_run in graph_runs.items():
                print(f" --- Explaining model: {model_name} with explainer: {explainer_name} ")
                masks = expl.explain_graph_task(graph_run.task, test_graphs)
                explanation = Explanation(run=graph_run, edge_masks=masks)
                save_path = path / "explanations" / explainer_name
                save_path.mkdir(exist_ok=True, parents=True)
                with open(save_path / f"{model_name}_explanation.pkl", "wb") as f:
                    pickle.dump(explanation, f)

        if args.node_level:
            ...
