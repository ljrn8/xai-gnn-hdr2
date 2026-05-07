""" Evaluates all models present in <root>/<dataset>/models against each transductive binary graph classification task in the dataset folder
usage: python eval_gnns.py <root_dir>

Evulations are restricted to standrad output.
"""
from training_utils import evaluate_binary_predictions, openpkl, TrainingRun
from pprint import pprint
import pickle
import os
from pathlib import Path
from loguru import logger
import torch
from pathlib import Path
import sys

if len(sys.argv) > 1:
    root_dir = Path(sys.argv[1])
else:
    root_dir = Path("interm/") / "RW graph classification"
    
for dsname in os.listdir(root_dir):
    path = root_dir / dsname

    logger.info(f"\n\n ===== Dataset name: {path.name}")

    for model in os.listdir(path / 'models'):
        logger.info(f'\n eval for model [{model}]')
        model_run: TrainingRun = openpkl(path / 'models' / model)

        # open datasets
        test_graphs = openpkl(path / "test_graphs.pkl")
        best_model = model_run.best_model

        # get x shape
        x_shape = test_graphs[0].x.shape[1]

        # get y_scores
        best_model.eval()
        y_true = []
        y_scores = []
        for graph in test_graphs:
            with torch.no_grad():
                out = best_model(graph.x, graph.edge_index).view(-1)
                y_scores.extend(out)
                y_true.extend(graph.y.view(-1))

        performance = evaluate_binary_predictions(y_true, y_scores)
        logger.info(f"f1: {performance.f1:.4f}")
        logger.info(f'Test performance: {performance}')
        logger.info(f'hyperparameters: {model_run.hyperparameter}')
        logger.info(f'epochs trained: {model_run.epochs_trained}')



    # NOTE:
    # ogx has a potential senstiivity issue garnering high auc but slightly lower f1
    # consider weighted BCE
    # or remove dropout
    # or go 128
