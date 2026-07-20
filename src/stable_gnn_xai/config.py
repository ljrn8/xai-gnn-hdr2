"""Run and experimental configurations"""

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

# default epochs
epcs = 100

EXPLAINERS = {
    "output": Path("output/explanations"), 
    "explainers": [
        {
            'name': 'GNNE',
            'explainer': GNNExplainer,
            'grid_search': {
                'epochs': [epcs]
            }
        },
        {
            'name': 'Random',
            'explainer': RandomExplainer,
            'grid_search': {
                'seed': [0] 
            }
        },
        {
            'name': 'explainer_module',
            'explainer': PGExplainer,
            'grid_search': {
                'epochs':               [epcs],
                'explanation_module':   ['default', 'comprehensive', 
                                        'contextual', 'auto-regressive'],
            },
        },
        {
            'name': 'sampler_method',
            'explainer': PGExplainer,
            'grid_search': {
                'epochs':               [epcs],
                'sampler_method':   ['GS', 'IGR'],
            },
        },
        {
            'name': 'proxy_graphs',
            'explainer': PGExplainer,
            'grid_search': {
                'use_proxy_graphs': [True],
                'epochs': [epcs],
                'proxy_M': [1, 2, 5],
            }
        },
        {
            'name': 'seed_ensembling',
            'explainer': PGExplainer,
            'grid_search': {
                'epochs': [epcs, epcs, epcs, epcs, epcs]
            }
        },
        {
            'name': 'auto_regessive_seed_ensembling',
            'explainer': PGExplainer,
            'grid_search': {
                'epochs':               [epcs, epcs, epcs, epcs, epcs],
                'explanation_module':   ['auto-regressive'],
            },
        },
    ]
}


