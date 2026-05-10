from GNN_training_utils import InductiveGraphClassification, TransductiveNodeClassification,evaluate_binary_predictions, evaluate_multiclass_predictions, openpkl, TrainingRun
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
parser.add_argument('--ds-root', default='output/ogx')
parser.add_argument('--multiclass', action='store_true')
args = parser.parse_args()


root_dir = Path(args.ds_root)

for dsname in os.listdir(root_dir):
    path = root_dir / dsname
    logger.info(f"\n\n ===== Dataset name: {path.name} ===== \n")

    for model in os.listdir(path / 'models'):
        logger.info(f'\n\n eval for model [{model}]')
        model_run: TrainingRun = openpkl(path / 'models' / model)

        test_loss, y_true, y_scores = model_run.task.evaluate_test()
        performance = evaluate_multiclass_predictions(y_true, y_scores) if args.multiclass else evaluate_binary_predictions(y_true, y_scores)

        logger.info(f'model object: ')
        pprint(model_run.task.model)
        logger.info(f"f1: {performance.f1:.4f}")
        logger.info(f'Test performance:')
        pprint(performance)
        logger.info(f'hyperparameters: ')
        pprint(model_run.hyperparameter)
        logger.info(f'epochs trained: {model_run.epochs_trained}')

        # !! they are the exact same for the models????





