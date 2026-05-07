""" Optimizes a models in training_utils.MODEL_CONFIGS against each transductive/inductive binary graph classification task in the dataset folder
usage: python train_binary_graph_task.py <root_dir>

folder format anticipated (inductive tasks):
root_dir/
 - dataset1/
   - train_graphs.pkl 
   - test_graphs.pkl
   - val_graphs.pkl
 - dataset2/
    - train_graphs.pkl
    - test_graphs.pkl
    - val_graphs.pkl
  ...

folder format anticipated (transductive tasks):
root_dir/
 - dataset1/
   - graph.pkl  
 - dataset2/
    - graph.pkl
  ...

with pickle files representing torch_geometric graph object lists including G.x, G.edge_index and G.y,
transductive graphs also require G.train_mask, G.val_mask and G.test_mask

Output: 
    saves TrainingRun objects to <root_dir>/<dataset>/models
"""
import os
import sys
import torch
import torch.nn.functional as F
from loguru import logger
from models import NodeGCN, NodeGIN, GraphTaskFromNodeModel
from training_utils import train_binary_graph_task, TrainingRun, MODEL_CONFIGS
from pprint import pprint
import pickle
import numpy as np
from pathlib import Path
import gc
from training_utils import openpkl
import sys
import argparse

parser = argparse.ArgumentParser(description="Train GNNs on binary graph classification tasks")
parser.add_argument('--ds-root', default=Path("interm/") / "RW graph classification")
parser.add_argument('--transductive', action='store_true', help='Whether to train on transductive tasks instead of inductive tasks (default)' )
args = parser.parse_args()

root_dir = Path(args.ds_root)
for dset in os.listdir(root_dir):
    path = root_dir / dset
    logger.info(f"\n\n Dataset name: {path.name} \n")
    if args.transductive:
        graph = openpkl(path / "graph.pkl")
        num_features = graph.x.shape[1]
        assert hasattr(graph, 'train_mask') and hasattr(graph, 'val_mask') and hasattr(graph, 'test_mask'), 'transductive learning tasks requires these attributes'
    else:
        train_graphs = openpkl(path / "train_graphs.pkl")
        val_graphs = openpkl(path / "val_graphs.pkl")
        best_run: TrainingRun = None
        num_features = train_graphs[0].x.shape[1]
        
    for model_name, model_class, hyperparameter_candidates in MODEL_CONFIGS:
        logger.info(f" --- Training model: {model_name} ")
        for hp in hyperparameter_candidates:
            if args.transductive:
                model = model_class(
                    input_feat=num_features,
                    hidden_channels=hp["hidden_channels"],
                    num_layers=hp.get("num_layers"),
                    output_channels=1,
                    dropout=hp.get("dropout"),
                )
            else:
                model = GraphTaskFromNodeModel(
                    node_model=model_class(
                        input_feat=num_features,
                        hidden_channels=hp["hidden_channels"],
                        num_layers=hp.get("num_layers"),
                        output_channels=hp["hidden_channels"],
                    ),
                    incoming_channels=hp["hidden_channels"],
                    output_graph_channels=1,
                    dropout=None,
                )

            logger.info(f"Model: {model}")
            if args.transductive:
                run: TrainingRun = train_binary_graph_task(
                    graph=graph,
                    dataset_name=path.name,
                    model=model,
                    epochs=hp["epochs"],
                    lr=hp["lr"],
                    patience=20,
                    transductive=args.transductive
                )
            else:
                run: TrainingRun = train_binary_graph_task(
                    train_graphs=train_graphs,
                    val_graphs=val_graphs,
                    dataset_name=path.name,
                    model=model,
                    epochs=hp["epochs"],
                    lr=hp["lr"],
                    patience=20,
                    transductive=args.transductive
                )

            run.hyperparameter = hp

            # only keep the model with the best run.performance.f1
            # this ensures realistic sensativity (uncertainty, utilized by explainers) by enforcing the 0.5 threshold, unlike AUC metrics
            if best_run is None or run.val_performance.f1 > best_run.val_performance.f1:
                best_run = run

            logger.info(
                f"resulting F1 = {run.val_performance.f1:.4f}, ROC-AUC = {run.val_performance.roc_auc:.4f}," +
                f"PREC = {run.val_performance.prec:.4f}, REC = {run.val_performance.rec:.4f}"

            )
            logger.info("HP:")
            pprint(hp)
            logger.info(f"performance object:")
            pprint(run.val_performance)
            gc.collect()

        logger.info("\n === HPO complete, best run:")
        pprint(best_run.val_performance)
        logger.info("HP:")
        pprint(best_run.hyperparameter)

        location = path / "models"
        if not location.exists():
            os.mkdir(location)
        with open(location / f"{model_name}.pkl", "wb") as f:
            pickle.dump(best_run, f)
