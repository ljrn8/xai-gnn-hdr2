from .training.models import WeightedNodeGCN, WeightedNodeGIN
from .explainability.explainers.PGExplainers import (
    PGExplainer, 
    PGEExplanationModule,
    ComprehensiveMLPExplanationModule, 
    GRUExplanationModule
)
from .explainability.explainers.RandomExplainer import RandomExplainer
from .explainability.explainers.GNNExplainer import GNNExplainer
from pathlib import Path
import torch
from loguru import logger
from .explainability.proxy_generation import ProxyGraphGenerator
from functools import partial

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

SEED = 0

FIGURES = Path("output/figures")

DATASETS = {"output": Path("Data"), "test_split": 0.2, "validation_split": 0.2}

MODELS = {
    "output": Path("output/runs"),
    "iterations": 10,
    "configs":  {
        "WeightedNodeGCN": {
            "lr": [0.0005, 0.0003, 0.001],
            "epochs": [100, 300, 400, 200],
            "hidden_channels": [32, 64],
            "num_layers": [1, 2, 3],
        },
        "WeightedNodeGIN": {
            "lr": [0.001, 0.0005, 0.0003],
            "epochs": [200, 500],
            "hidden_channels": [32, 64],
            "num_layers": [2, 3, 4, 5, 6],
        }
    }
}

EXPLAINERS = {
    "output": Path("output/explanations"), 
    "explainers": [
        {
            'name': 'debugging',
            'explainer': PGExplainer,
            'grid_search': {
                'tau':                          [0.5],
                'reparameterization_samples':   [30],
                'lr':                           [0.01, 0.5],
                'hidden_size':                  [64],
                'epochs':                       [100],
                'entropy_regularization':       [0.1],
                'mean_regularization':          [0.1],
                "explanation_module_class":     [PGEExplanationModule],
            },
        }
    ]
}
