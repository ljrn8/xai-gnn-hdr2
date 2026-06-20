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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

SEED = 0

FIGURES = Path("output/figures")

DATASETS = {"output": Path("Data"), "test_split": 0.2, "validation_split": 0.2}

MODELS = {
    "output": Path("output/runs"),
    "iterations": 10,
    "random_search_configurations": {
        "GCN": {
            "base_class": WeightedNodeGCN,
            'hyperparameters': {
                "lr": [0.0005, 0.0003, 0.001],
                "epochs": [100, 300, 400, 200],
                "hidden_channels": [32, 64],
                "num_layers": [1, 2, 3],
            }
        },
        "GIN": {
            "base_class": WeightedNodeGIN,
            'hyperparameters': {
                 "lr": [0.001, 0.0005, 0.0003],
                "epochs": [200, 500],
                "hidden_channels": [32, 64],
                "num_layers": [2, 3, 4, 5, 6],
            }
        },
    },
}

EXPLAINERS = {
    "output": Path("output/explanations"), 
    "exhuastive_search_configurations": {
        'PGE': {
            'class': PGExplainer,
            'hyperparameters': {
                'tau': [0.5]*3,
                'reparameterization_samples': [30]*3,
                'lr': [0.01, 0.05, 0.1],
                'hidden_size': [64, 64, 32],
                'epochs': [300, 200, 100],
                'entropy_regularization': [0.1, 0.1, 0.1],
                'mean_regularization': [0.1, 0.1, 0.1],
            }
        },
        'PGE-ComprehensiveGRU': {
            'class': PGExplainer,
            'hyperparameters': {
                'tau': [0.5]*3,
                'reparameterization_samples': [30]*3,
                'explanation_module_class': [GRUExplanationModule]*3,
                'lr': [0.01, 0.05, 0.1],
                'hidden_size': [64, 64, 32],
                'epochs': [300, 200, 100],
                'entropy_regularization': [0.1, 0.1, 0.1],
                'mean_regularization': [0.1, 0.1, 0.1],
            }
        },
        'PGE-ComprehensiveMLP': {
            'class': PGExplainer,
            'hyperparameters': {
                'tau': [0.5]*3,
                'reparameterization_samples': [30]*3,
                'explanation_module_class': [ComprehensiveMLPExplanationModule]*3,
                'lr': [0.01, 0.05, 0.1],
                'hidden_size': [64, 64, 32],
                'epochs': [300, 200, 100],
                'entropy_regularization': [0.1, 0.1, 0.1],
                'mean_regularization': [0.1, 0.1, 0.1],
            }
            
        },
        'GNNExplainer': {
            'class': GNNExplainer,
            'hyperparameters': {
                'lr': [0.01, 0.05, 0.1],
                'hidden_size': [64, 64, 32],
                'epochs': [300, 200, 100],
            }
        },
        'Random Explainer': {
            'class': RandomExplainer
        }
    }
}