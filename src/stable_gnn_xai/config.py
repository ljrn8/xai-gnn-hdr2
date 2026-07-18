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

DATASETS = {
    "output": Path("/Data"), 
    "test_split": 0.2, 
    "validation_split": 0.2
}

MODELS = {
    "output": Path("output/runs"),
    "iterations": 5,
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
            'name': 'explainer_module',
            'explainer': PGExplainer,
            'grid_search': {
                'epochs':                       [1],
                'explanation_module':           ['default', 'comprehensive', 'contextual', 'auto-regressive'],
            },
        },
        {
            'name': 'sampler',
            'explainer': PGExplainer,
            'grid_search': {
                'epochs':                       [1],
                'sampler_method':               ['GS', 'IGR'],
            },
        },
        {
            'name': 'proxy_graphs',
            'explainer': PGExplainer,
            'grid_search': {
                'epochs':                       [1],
                'proxy_mode':                    [True],
            },
        },

        {
            'name': 'seed_ensembling_test',
            'explainer': PGExplainer,
            'grid_search': {
                'epochs':                       [1, 1, 1, 1],
            },
        },

        # TODO: GNNExplainer, Random Explainer other baselines
    ]
}
