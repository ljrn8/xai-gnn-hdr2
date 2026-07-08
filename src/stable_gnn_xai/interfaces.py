from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from typing import Optional, Iterable


class GNN(ABC, nn.Module):
    """Interface for GNN models on node-featured static graphs"""

    @property
    def layers(self):
        """Iterable over all GNN layers (used for explanatory inspections)."""

    @abstractmethod
    def forward(self, x, edge_index, edge_weight=None, return_all_embeddings=False):
        """forward pass a node featured graph.

        Returns:
            x (Tensor): model output of graph input.
            embeddings_list (Iterable[Tensor]): If return_all_embeddings=True, retrieves layerwise model embeddings.
        """


class GraphLevelExplainer(ABC):
    """Simple interface for graph-classifier explainers"""

    def __init__(self, model: GNN, graphs: Iterable[Tensor]):
        self.model = model
        self.graphs = graphs

    @abstractmethod
    def explain_graph_task(self):
        """Perform graph explanation given abritrary initialization.

        Returns:
            edge_masks (Iterable[Tensor]): edge importance scores of all graphs
            objective_loss (float): the achieved minimization objective loss (abritrary representation) for downstream HPO
        """



@dataclass
class ModelEvaluation:
    y_pred_test: torch.Tensor
    metrics: Optional[dict]


@dataclass
class ModelRun:
    dataset_root: str
    model_name: str
    model: GNN
    model_configuration: dict
    details: Optional[dict]
    test_evaluation: Optional[ModelEvaluation] = None


@dataclass
class Explanation:
    name: str
    explainer: GraphLevelExplainer
    metadata: dict
    run: ModelRun
    task_type: Optional[str] = None
    edge_masks: Iterable[torch.Tensor] = None
    node_masks: Iterable[torch.Tensor] = None
    evaluation_metrics: Optional[dict] = field(default_factory=dict)
