from .training.models import WeightedNodeGCN, WeightedNodeGIN
from .explainability.explainers.PGExplainers import PGExplainer
from .explainability.explainers.RandomExplainer import RandomExplainer
from .explainability.explainers.GNNExplainer import GNNExplainer
from pathlib import Path

SEED = 0

FIGURES = Path("output/figures")

DATASETS = {"output": Path("Data"), "test_split": 0.2, "validation_split": 0.2}

MODELS = {
    "output": Path("output/runs"),
    "iterations": 10,
    "random_search_configurations": {
        "GCN": {
            "base_class": WeightedNodeGCN,
            "lr": [0.0005, 0.0003, 0.001],
            "epochs": [100, 300, 400, 200],
            "hidden_channels": [32, 64],
            "num_layers": [1, 2, 3],
        },
        "GIN": {
            "base_class": WeightedNodeGIN,
            "lr": [0.001, 0.0005, 0.0003],
            "epochs": [200, 500],
            "hidden_channels": [32, 64],
            "num_layers": [2, 3, 4, 5, 6],
        },
    },
}

EXPLAINERS = {
    "output_directory": Path("output/explanations"), 
    "exhuastive_search_configurations": {
        'PGE': {
            'class': PGExplainer,
            'learning_rate': [0.01, 0.05, 0.1],
            'hidden_size': [64, 64, 32],
            'epochs': [300, 200, 100],
        },
        'GNNExplainer': {
            'class': GNNExplainer,
            'learning_rate': [0.01, 0.05, 0.1],
            'hidden_size': [64, 64, 32],
            'epochs': [300, 200, 100],
        },
        'Random Explainer': {
            'class': RandomExplainer
        }
    }
}
