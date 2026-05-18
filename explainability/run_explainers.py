import os
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch import nn
import torch.functional as F
import sys
from abc import ABC, abstractmethod
from torch_geometric.data import Data
from training.GNN_utils import *
from explainability.explainer_utils import Explanation
from pathlib import Path
import argparse
from explainability.baselines.ORExplainer import ORExplainer
from explainability.baselines.ProxyExplainer import ProxyExplainerImpl

torch.autograd.set_detect_anomaly(True, check_nan=False)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ds-root", help="Root directory containing dataset to evaluate on"
)
parser.add_argument("-n", "--node-level", action="store_true")
parser.add_argument("-g", "--graph-level", action="store_true")

# generic parameters
parser.add_argument('-e', "--epochs", type=int, default=100)
parser.add_argument("--hidden-size", type=int, default=64)
parser.add_argument("--explainer", type=str, default=None, help=f'only run a specific explainer by name. For options see explainability.baselines')
args = parser.parse_args()

explainer_map = {
    "ORExplainer": ORExplainer(hidden_channels=args.hidden_size, epochs=args.epochs, gamma=0.1),
    # "PROXYExplainer": ProxyExplainerImpl(epochs=args.epochs),
}

assert (
    args.graph_level != args.node_level
), "Please specify exactly one of --graph-level or --node-level"

if args.explainer:
    explainer_map = {args.explainer: explainer_map[args.explainer]}

for explainer_name, expl in explainer_map.items():
    path = Path(args.ds_root)
    dataset = path.name
    models_path = path / "models"
    logger.info(f"\n\n Dataset name: {dataset} \n")

    runs = {model: openpkl(models_path / model) for model in os.listdir(models_path)}

    for model_name, run in runs.items():
        model_name = model_name.split(".pkl")[0]
        logger.info(f"Explaining model: {model_name} with explainer: {explainer_name} " )

        if args.graph_level:
            test_graphs = openpkl(path / "test_graphs.pkl")
            masks = expl.explain_graph_task(run.task, test_graphs)
            # ! expectes edge mask
            explanation = Explanation(run=run, edge_masks=masks, task_type="graph")

        elif args.node_level:
            test_graph = openpkl(path / "graph.pkl")
            masks = expl.explain_node_task(run.task, test_graph)
            # ! expects edge mask
            explanation = Explanation(run=run, edge_masks=masks, task_type="node")

        save_path = path / "explanations" / explainer_name
        save_path.mkdir(exist_ok=True, parents=True)
        with open(save_path / f"{model_name}_explanation.pkl", "wb") as f:
            pickle.dump(explanation, f)
