import os
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch import nn
import torch.functional as F
import sys
from abc import ABC, abstractmethod
from torch_geometric.data import Data
from model_training.GNN_training_utils import *
from explanations.xAI_utils import *
from pathlib import Path
import argparse

torch.autograd.set_detect_anomaly(True, check_nan=False)

parser = argparse.ArgumentParser()
parser.add_argument("--ds-root", default="./output/inductive_graph_tasks")
args = parser.parse_args()

for explainer_name, expl in [
    ("PROXYExplainer",  ProxyExplainerImpl(epochs=100, reg_coefs=[0.05, 1.0])),
]:
    files = os.listdir(args.ds_root)
    # is_inductive = 'test_graphs.pkl' in files
    for dataset in files:
        path = Path(args.ds_root) / dataset
        print(f"\n\n Dataset name: {dataset} \n")
        test_graphs = openpkl(path / "test_graphs.pkl")
        models_path = path / "models"
        graph_runs = {
            model: openpkl(models_path / model) for model in os.listdir(models_path)
        }
        for model_name, graph_run in graph_runs.items():
            print(
                f" --- Explaining model: {model_name} with explainer: {explainer_name} "
            )
            masks = expl.explain_graph_task(graph_run.task, test_graphs)
            explanation = Explanation(run=graph_run, edge_masks=masks)
            save_path = path / "explanations" / explainer_name
            save_path.mkdir(exist_ok=True, parents=True)
            with open(save_path / f"{model_name}_explanation.pkl", "wb") as f:
                pickle.dump(explanation, f)

