from training.GNN_utils import (
    InductiveGraphClassification,
    TransductiveNodeClassification,
    evaluate_binary_predictions,
    evaluate_multiclass_predictions,
    openpkl,
    TrainingRun,
)
from pprint import pprint
import pickle
import os
from pathlib import Path
from loguru import logger
import torch
from pathlib import Path
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ds-root", help="Root directory containing dataset to evaluate on")
parser.add_argument('-n', '--node-level', action='store_true')
parser.add_argument('-g', '--graph-level', action='store_true')


args = parser.parse_args()
path = Path(args.ds_root)
logger.info(f" > Dataset name: {path.name}")
for model in os.listdir(path / "models"):
    logger.info(f"\n\n eval for model [{model}]")
    model_run: TrainingRun = openpkl(path / "models" / model)
    test_loss, y_true, y_scores = model_run.task.evaluate_test()

    performance = (
        evaluate_multiclass_predictions(y_true, y_scores)
        if y_true.ndim > 1 and y_true.shape[1] > 1
        else evaluate_binary_predictions(y_true, y_scores)
    )

    logger.info(f"model object: ")
    pprint(model_run.task.model)
    logger.info(f"f1: {performance.f1:.4f}")
    logger.info(f"Test performance:")
    pprint(performance)
    logger.info(f"hyperparameters: ")
    pprint(model_run.hyperparameter)
    logger.info(f"epochs trained: {model_run.epochs_trained}")

