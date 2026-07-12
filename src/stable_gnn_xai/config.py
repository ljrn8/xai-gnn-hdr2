"""Run and experiment configurations"""

from .training.models import WeightedNodeGCN, WeightedNodeGIN
from .explainability.PGExplainer import PGExplainer
from .explainability.RandomExplainer import RandomExplainer
from .explainability.GNNExplainer import GNNExplainer
from pathlib import Path
import torch
from loguru import logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"config.py using device: {DEVICE}")

SEED = 0
FIGURES = Path("output/figures")
DATASETS = {"output": Path("Data"), "test_split": 0.2, "validation_split": 0.2}

MODELS = {
    "output": Path("output/runs"),
    "iterations": 10,
    "models":  [
        {
            "name": "GCN",
            "base_class": WeightedNodeGCN,
            "linear_search": {
                "lr": [0.0005, 0.0003, 0.001],
                "epochs": [100, 300, 400, 200],
                "hidden_channels": [32, 64],
                "num_layers": [1, 2, 3],
            }
        },
        {
            "name": "GIN",
            "base_class": WeightedNodeGIN,
            "linear_search": {
                "lr": [0.001, 0.0005, 0.0003],
                "epochs": [200, 500],
                "hidden_channels": [32, 64],
                "num_layers": [2, 3, 4, 5, 6],
            }
        }
    ]
}

EXPLAINERS = {
    "output": Path("output/explanations"), 
    "explainers": [
        {
            'name': 'debugging',
            'explainer': PGExplainer,
            'grid_search': {
                'tau':                          [0.5, 1, 0.1],
                'reparameterization_samples':   [30],
                'lr':                           [0.01, 0.5],
                'hidden_size':                  [64],
                'epochs':                       [100],
                'entropy_regularization':       [0.01],
                'mean_regularization':          [0.01],
                "explanation_module":           ['default'],
            },
        }
    ]
}
