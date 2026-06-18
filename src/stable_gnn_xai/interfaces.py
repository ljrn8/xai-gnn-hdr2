from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from typing import Optional


class GNN(ABC, nn.Module):
    """Interface for GNN models on node-featured static graphs"""

    @property
    def layers(self):
        pass

    @abstractmethod
    def forward(self, x, edge_index, edge_weight=None, return_all_embeddings=False):
        """forward pass a node featured graph.
        
        returns:
            x (Tensor): model output of graph input.
            embeddings_list (Iterable[Tensor]): If return_all_embeddings=True, retrieves layerwise model embeddings.
        """


@dataclass
class TrainingRun:
    dataset_root: str
    model: GNN
    model_configuration: dict
    y_pred: torch.Tensor
    details: Optional[dict]