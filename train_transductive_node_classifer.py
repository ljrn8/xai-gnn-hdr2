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
import json

parser = argparse.ArgumentParser()
parser.add_argument('--ds-root', default='output/transductive_node_tasks')
parser.add_argument('--multiclass', default=True, action='store_true')
parser.add_argument('--model-configurations', default="transductive citation datasets")
args = parser.parse_args()

model_configs = json.load(open("configuration.json"))["model configurations"][args.model_configurations]

root_dir = Path(args.ds_root)
for dset in os.listdir(root_dir):
    path = root_dir / dset
    logger.info(f"\n\n Dataset name: {path.name} \n")

    graph = openpkl(path / "graph.pkl")
    best_run: TrainingRun = None
    num_features = graph.x.shape[1]

    # both expect logits, NOTE: CE needs argmax on y_true
    criterion = torch.nn.CrossEntropyLoss() if args.multiclass else torch.nn.BCEWithLogitsLoss()
    n_classes = len(graph.y.unique()) if args.multiclass else 1

    for config in model_configs:

        logger.info(f" --- Training model: {config['model_name']} ")
        for hp in config['hyperparameter_candidates']:
            ...

            # NOTE:
            # pickup for next time
            # match the config string the model class
            # setup bayesion optimzation/ grid search

            model = model_class(
                    input_feat=num_features,
                    hidden_channels=hp["hidden_channels"],
                    num_layers=hp.get("num_layers"),
                    output_channels=n_classes,
                    dropout=hp.get("dropout"),
                )
            logger.info(f'model: {model}')
            run: TrainingRun = GNN_task(
                single_graph=graph,
                dataset_name=path.name,
                model=model,
                epochs=hp["epochs"],
                lr=hp["lr"],
                patience=20,
                criterion=criterion,
                transductive=True,
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