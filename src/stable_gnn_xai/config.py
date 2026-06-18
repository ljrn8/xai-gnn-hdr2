from .training.models import WeightedNodeGCN, WeightedNodeGIN
from pathlib import Path

SEED = 0

DATASETS = {
    "output": Path("Data"),
    "test_split": 0.2,
    "validation_split": 0.2
}

MODELS = {
    "output": Path("output/runs"),
    "configurations": {
        "GCN": {
            'base_class': WeightedNodeGCN,
            "lr": [0.0005, 0.0003, 0.001],
            "epochs": [100, 300, 400, 200],
            "hidden_channels": [32, 64],
            "num_layers": [1, 2, 3],
            "dropout": [0.0, 0.5]
        },
        "GIN": {
            'base_class': WeightedNodeGIN,
            "lr": [0.001, 0.0005, 0.0003],
            "epochs": [200, 500],
            "hidden_channels": [32, 64],
            "num_layers": [2, 3, 4, 5, 6]
        }
    }
}

EXPLAINERS = {
    "output_directory": Path("output/explanations"),
    "configurations": {}
}