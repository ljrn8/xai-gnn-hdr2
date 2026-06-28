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

EXPLAINER_GRIDS = {
    
    # ! For debugging (remove)
    "PGExplainer": {
        'tau':                          [0.5],
        'reparameterization_samples':   [30],
        'lr':                           [0.01, 0.5],
        'hidden_size':                  [64],
        'epochs':                       [100],
        'entropy_regularization':       [0.1],
        'mean_regularization':          [0.1],
        "explanation_module_class":     [PGEExplanationModule],
        "use_proxy":                    [False],
        "proxy_M":                      [1, 5],
        "proxy_lr":                     [0.01, 0.1],
        "proxy_lam":                    [0.25, 0.5],
        "proxy_latent":                 [64],
    },


    # "PGExplainer": {
    #     'tau':                          [0.5],
    #     'reparameterization_samples':   [30],
    #     'lr':                           [0.01, 0.5, 0.1],
    #     'hidden_size':                  [64],
    #     'epochs':                       [500],
    #     'entropy_regularization':       [0.1],
    #     'mean_regularization':          [0.1],
    #     "explanation_module_class":     [PGEExplanationModule, 
    #                                     ComprehensiveMLPExplanationModule, 
    #                                     GRUExplanationModule],
    #     "use_proxy":                    [False, True],
    #     "proxy_M":                      [1, 5],
    #     "proxy_lr":                     [0.01, 0.1],
    #     "proxy_lam":                    [0.25, 0.5],
    #     "proxy_latent":                 [64],
    # },

    "GNNExplainer": {
        'lr': [0.1, 0.01],
        'epochs': [1],
        'entropy_regularization': [0.1],
        'mean_regularization': [0.5],
    }

}

MODEL_GRIDS = {

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


FIGURES = Path("output/figures")

DATASETS = {"output": Path("Data"), "test_split": 0.2, "validation_split": 0.2}

MODELS = {
    "output": Path("output/runs"),
    "iterations": 10,
    "configs": MODEL_GRIDS
}

EXPLAINERS = {
    "output": Path("output/explanations"), 
    "explainers": EXPLAINER_GRIDS
}
