import os
import sys
import torch
import torch.nn.functional as F
from loguru import logger
from training.models import MODEL_ID, GraphTaskFromNodeModel
from training.GNN_utils import (
    TransductiveNodeClassification,
    openpkl,
    InductiveGraphClassification,
    train,
    ModelPerformance,
    TrainingRun,
    evaluate_multiclass_predictions,
    evaluate_binary_predictions,
)
from pprint import pprint
import pickle
import numpy as np
from pathlib import Path
import gc
import sys
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--ds-root", description="Root directory containing dataset to evaluate on")
parser.add_argument('-n', '--node-level', action='store_true')
parser.add_argument('-g', '--graph-level', action='store_true')
parser.add_argument("--model-configurations", description="Key in configuration.json specifying which model configurations to use for training for HPO")
parser.add_argument(
    "--optimization-iterations",
    default=10,
    type=int,
    help="number of Hyperparameter random searches",
)

args = parser.parse_args()
assert args.graph_level != args.node_level, "Please specify exactly one of --graph-level or --node-level"
model_configs = json.load(open("configuration.json"))["model configurations"][
    args.model_configurations
]

path = Path(args.ds_root)
logger.info(f"\n\n Dataset name: {path.name} \n")

if args.graph_level:
    train_graphs = openpkl(path / "train_graphs.pkl")
    val_graphs = openpkl(path / "val_graphs.pkl")
    test_graphs = openpkl(path / "test_graphs.pkl")
    num_features = train_graphs[0].x.shape[1]
    n_classes = len(train_graphs[0].y.unique())
    multiclass = n_classes > 2

elif args.node_level:
    graph = openpkl(path / "graph.pkl")
    num_features = graph.x.shape[1]
    n_classes = len(graph.y.unique())
    multiclass = n_classes > 2

criterion = (
    torch.nn.CrossEntropyLoss() if multiclass else torch.nn.BCEWithLogitsLoss()
)

for model_name, candidates in model_configs.items():
    best_run: TrainingRun = None
    logger.info(f" --- Training model: {model_name} ")

    for i in range(args.optimization_iterations):

        #  HP Optimization (simple random search)
        total_possible_combintations = np.prod(
            [len(v) for v in candidates.values()]
        )
        if args.optimization_iterations > total_possible_combintations:
            logger.warning(
                f"Number of optimization iterations ({args.optimization_iterations}) is greater than the total possible combinations of hyperparameters ({total_possible_combintations}). Consider reducing the number of iterations or increasing the hyperparameter search space to avoid redundant runs."
            )
            args.optimization_iterations = total_possible_combintations

        done_iterations = set()
        hp = {}
        for param, values in candidates.items():
            value = np.random.choice(values)
            hp[param] = value

        if tuple(hp.items()) in done_iterations:
            i -= 1
            continue

        done_iterations.add(tuple(hp.items()))

        # Task training
        model_class = MODEL_ID[model_name]
        if args.graph_level:
            model = GraphTaskFromNodeModel(
                node_model=model_class(
                    input_feat=num_features,
                    hidden_channels=hp["hidden_channels"],
                    num_layers=hp.get("num_layers"),
                    output_channels=hp["hidden_channels"],
                    dropout=hp.get("dropout"),
                ),
                incoming_channels=hp["hidden_channels"],
                output_graph_channels=n_classes,
            )
            logger.info(f"model: {model}")
            task = InductiveGraphClassification(
                model, criterion, train_graphs, val_graphs, test_graphs
            )

        elif args.node_level:
            model = model_class(
                input_feat=num_features,
                hidden_channels=hp["hidden_channels"],
                num_layers=hp.get("num_layers"),
                output_channels=n_classes,
                dropout=hp.get("dropout"),
            )
            logger.info(f"model: {model}")
            task = TransductiveNodeClassification(model, criterion, graph)

        run = train(
            task,
            epochs=hp["epochs"],
            lr=hp["lr"],
            dataset_name=path.name,
            patience=hp.get("patience", 20),
        )

        if multiclass:
            run.val_performance = evaluate_multiclass_predictions(
                run.y_true, run.y_pred
            )
        else:
            run.val_performance = evaluate_binary_predictions(
                run.y_true, run.y_pred
            )

        # only keep the model with the best run.performance.f1
        # this ensures realistic sensativity (uncertainty, utilized by explainers) by enforcing the 0.5 threshold, unlike AUC metrics
        if best_run is None or run.val_performance.f1 > best_run.val_performance.f1:
            best_run = run

        logger.info(
            f"resulting F1 = {run.val_performance.f1:.4f}, ROC-AUC = {run.val_performance.roc_auc:.4f},"
            + f"PREC = {run.val_performance.prec:.4f}, REC = {run.val_performance.rec:.4f}"
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
