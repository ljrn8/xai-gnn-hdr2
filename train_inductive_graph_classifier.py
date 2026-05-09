import os
import sys
import torch
import torch.nn.functional as F
from loguru import logger
from models import GraphTaskFromNodeModel
from training_utils import (
    GNN_task, TrainingRun, MODEL_CONFIGS, 
    evaluate_multiclass_predictions, evaluate_binary_predictions
)
from pprint import pprint
import pickle
import numpy as np
from pathlib import Path
import gc
from training_utils import openpkl
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds-root')
parser.add_argument('--multiclass', action='store_true')
args = parser.parse_args()

root_dir = Path(args.ds_root)
for dset in os.listdir(root_dir):
    path = root_dir / dset
    logger.info(f"\n\n Dataset name: {path.name} \n")

    train_graphs = openpkl(path / "train_graphs.pkl")
    val_graphs = openpkl(path / "val_graphs.pkl")
    best_run: TrainingRun = None
    num_features = train_graphs[0].x.shape[1]
        
    criterion = torch.nn.CrossEntropyLoss() if args.multiclass else torch.nn.BCEWithLogitsLoss()

    for model_name, model_class, hyperparameter_candidates in MODEL_CONFIGS:
        logger.info(f" --- Training model: {model_name} ")
        for hp in hyperparameter_candidates:

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
            logger.info(f'model: {model}')
            run: TrainingRun = GNN_task(
                train_graphs=train_graphs,
                val_graphs=val_graphs,
                dataset_name=path.name,
                model=model,
                epochs=hp["epochs"],
                lr=hp["lr"],
                patience=20,
                transductive=False,
                criterion=criterion
                )

            run.hyperparameter = hp

            if args.multiclass:
                run.val_performance = evaluate_multiclass_predictions(run.y_true, run.y_pred)
            else:
                run.val_performance = evaluate_binary_predictions(run.y_true, run.y_pred)

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
