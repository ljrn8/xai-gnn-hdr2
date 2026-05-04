import os
import sys
import torch
import torch.nn.functional as F
from loguru import logger 
from models import GraphTaskNodeGCN
from training_utils import train_binary_graph_task, TrainingRun
from pprint import pprint
import pickle
import numpy as np
from pathlib import Path

def openpkl(file):
    logger.info(f'Opening file: {file}')
    with open(file, 'rb') as f:
        file = pickle.load(f)
        logger.info(f'file size: {sys.getsizeof(file)} bytes')
        return file

root_dir = Path('interm/ogx')
for dset in os.listdir(root_dir):
    path = root_dir / dset
    logger.info(f'\n\n ===== Dataset name: {path.name} =====\n')

    # open datasets
    train_graphs = openpkl(path / 'train_graphs.pkl')
    val_graphs = openpkl(path / 'val_graphs.pkl')

    hyperparameter_candidates = [
        {'lr': 0.0003, 'epochs': 40, 'hidden_channels': 64, 'num_gcn_layers': 3},
        {'lr': 0.0001, 'epochs': 40, 'hidden_channels': 32, 'num_gcn_layers': 3},
        {'lr': 0.0001, 'epochs': 40, 'hidden_channels': 32, 'num_gcn_layers': 2},
        {'lr': 0.0001, 'epochs': 70, 'hidden_channels': 64, 'num_gcn_layers': 3},
        {'lr': 0.0001, 'epochs': 70, 'hidden_channels': 64, 'num_gcn_layers': 3},
    ]

    for hp in hyperparameter_candidates:
        model = GraphTaskNodeGCN(
            input_feat=train_graphs[0].x.shape[1],
            hidden_channels=hp['hidden_channels'], 
            num_gcn_layers=hp['num_gcn_layers'],
            output_graph_channels=1,
        )

        logger.info(f'Model: {model}')

        run: TrainingRun = train_binary_graph_task(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            dataset_name=path.name,
            model=model,
            epochs=hp['epochs'],
            lr=hp['lr'],
        )

        run.hyperparameter = hp

        # only keep the one with the best run.performance.f1
        if 'best_run' not in locals() or run.val_performance.f1 > best_run.val_performance.f1:
            best_run = run

    logger.info('best run:')
    pprint(best_run.val_performance)
    logger.info('HP:')
    pprint(best_run.hyperparameter)

    dset = path / 'GCN2.pkl'
    os.makedirs(dset, exist_ok=True)
    with open(f'{dset}/GCN2.pkl', 'wb') as f:
        pickle.dump(best_run, f)
