import os
import sys
import torch
import torch.nn.functional as F
from loguru import logger 
from models import NodeGCN, NodeGIN, GraphTaskFromNodeModel
from training_utils import train_binary_graph_task, TrainingRun
from pprint import pprint
import pickle
import numpy as np
from pathlib import Path
import gc
from training_utils import openpkl


root_dir = Path('interm/') / "RW graph classification"
for dset in os.listdir(root_dir):
    path = root_dir / dset

    logger.info(f'\n\n ===== Dataset name: {path.name} =====\n')

    train_graphs = openpkl(path / 'train_graphs.pkl')
    val_graphs = openpkl(path / 'val_graphs.pkl')

    model_configurations = (
        ('GNC2', NodeGCN, [
            {'lr': 0.001, 'epochs': 40, 'hidden_channels': 32, 'num_layers': 3, 'dropout': None},
            {'lr': 0.0005, 'epochs': 100, 'hidden_channels': 32, 'num_layers': 2, 'dropout': None},
            {'lr': 0.0005, 'epochs': 100, 'hidden_channels': 64, 'num_layers': 3, 'dropout': None},
            {'lr': 0.0005, 'epochs': 200, 'hidden_channels': 64, 'num_layers': 3, 'dropout': 0.5},
            {'lr': 0.0003, 'epochs': 200, 'hidden_channels': 64, 'num_layers': 3, 'dropout': 0.5},
            {'lr': 0.001, 'epochs': 100, 'hidden_channels': 64, 'num_layers': 3, 'dropout': 0.5},
        ]),
        ('GIN', NodeGIN, [
            {'lr': 0.001, 'epochs': 40, 'hidden_channels': 32, 'num_layers': 3},
            {'lr': 0.0005, 'epochs': 100, 'hidden_channels': 32, 'num_layers': 2},
            {'lr': 0.0005, 'epochs': 100, 'hidden_channels': 64, 'num_layers': 4},
            {'lr': 0.0005, 'epochs': 200, 'hidden_channels': 64, 'num_layers': 5},
            {'lr': 0.0003, 'epochs': 200, 'hidden_channels': 64, 'num_layers': 5},
            {'lr': 0.001, 'epochs': 100, 'hidden_channels': 64, 'num_layers': 6},
        ])
    )

    best_run: TrainingRun = None

    for model_name, model_class, hyperparameter_candidates in model_configurations:
        logger.info(f'\n\n === Training model: {model_name} ===\n')
        for hp in hyperparameter_candidates:
            model = GraphTaskFromNodeModel(
                node_model=model_class(
                    input_feat=train_graphs[0].x.shape[1],
                    hidden_channels=hp['hidden_channels'], 
                    num_layers=hp.get('num_layers'),
                    output_channels=hp['hidden_channels'],
                ),
                incoming_channels=hp['hidden_channels'],
                output_graph_channels=1,
                dropout=None,
            )

            logger.info(f'Model: {model}')

            run: TrainingRun = train_binary_graph_task(
                train_graphs=train_graphs,
                val_graphs=val_graphs,
                dataset_name=path.name,
                model=model,
                epochs=hp['epochs'],
                lr=hp['lr'],
                patience=20,
            )

            run.hyperparameter = hp

            # only keep the one with the best run.performance.f1
            if best_run is None or run.val_performance.f1 > best_run.val_performance.f1:
                best_run = run

            # log info
            logger.info(f'resulting F1 = {run.val_performance.f1:.4f}, ROC-AUC = {run.val_performance.roc_auc:.4f}')
            logger.info('HP:')
            pprint(hp)
            logger.info(f'performance object:')
            pprint(run.val_performance)
            gc.collect()


        logger.info('\n === HPO complete, best run:')
        pprint(best_run.val_performance)
        logger.info('HP:')
        pprint(best_run.hyperparameter)

        location = path / 'models'
        if not location.exists():
            os.mkdir(location)
        with open(location / f'{model_name}.pkl', 'wb') as f:
            pickle.dump(best_run, f)
