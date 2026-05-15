from torch_geometric.nn import MessagePassing, global_mean_pool
from torch import nn
import torch.functional as F
import sys
from training.GNN_utils import *
from abc import ABC, abstractmethod
from torch_geometric.data import Data


class Explainer(ABC):
    @abstractmethod
    def explain_node_task(self, task, graph):
        """Returns edge importance scores for graph at each node"""

    @abstractmethod
    def explain_graph_task(self, task, graphs):
        """Returns edge importance scores of all graphs"""


@dataclass
class Explanation:
    run: TrainingRun
    edge_masks: Iterable[torch.Tensor] = None
    node_masks: Iterable[torch.Tensor] = None


# --------------------------------------------------------
# Convert Pretrained Models to Edge_weight handlable copies


class WeightedGINConv(MessagePassing):
    def __init__(self, nn, eps=0.0):
        super().__init__(aggr="add")
        self.nn = nn
        self.eps = eps

    def forward(self, x, edge_index, edge_weight=None):
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.nn(x * (1 + self.eps) + out)
        return out

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class WeightedNodeGIN(nn.Module):
    """GIN model variant that can handle an edge_weight in forward(),
    required for passing fractional subgraph explanations.
    """

    def __init__(
        self, input_feat, num_layers, hidden_channels, output_channels, dropout=None
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(
            WeightedGINConv(
                nn.Sequential(
                    nn.Linear(input_feat, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
            )
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                WeightedGINConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    )
                )
            )
        self.convs.append(
            WeightedGINConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, output_channels),
                )
            )
        )

    def forward(self, x, edge_index, edge_weight=None, return_embeds=False):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = x.relu()
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embeds:
            return x
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x


class WeightedNodeGCN(nn.Module):
    def __init__(
        self, input_feat, num_layers, hidden_channels, output_channels, dropout=None
    ):
        super(WeightedNodeGCN, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_feat, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, output_channels))

    def forward(self, x, edge_index, return_embeds=False, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = x.relu()
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embeds:
            return x
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x


def convert_node_model_state(trained_graph_model, empty_graph_model):
    for old_conv, new_conv in zip(
        trained_graph_model.node_model.convs, empty_graph_model.node_model.convs
    ):
        new_conv.nn.load_state_dict(old_conv.nn.state_dict())
    return empty_graph_model


def convert_graph_model_state(trained_graph_model, empty_graph_model):
    empty_graph_model = convert_node_model_state(trained_graph_model, empty_graph_model)
    empty_graph_model.lin.load_state_dict(trained_graph_model.lin.state_dict())
    return empty_graph_model


def _get_weighted_GCN(graph_level_model):
    first_conv = graph_level_model.node_model.convs[0]
    num_layers = len(graph_level_model.node_model.convs)
    hidden_channels = first_conv.out_channels
    empty_model = GraphTaskFromNodeModel(
        node_model=WeightedNodeGCN(
            input_feat=first_conv.in_channels,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            output_channels=hidden_channels,
        ),
        incoming_channels=hidden_channels,
        output_graph_channels=1,
    )
    for old_conv, new_conv in zip(
        graph_level_model.node_model.convs, empty_model.node_model.convs
    ):
        new_conv.load_state_dict(old_conv.state_dict())
    empty_model.lin.load_state_dict(graph_level_model.lin.state_dict())
    return empty_model


def _get_weighted_GIN(graph_level_model):
    first_nn = graph_level_model.node_model.convs[0].nn[0]
    hidden_channels = first_nn.out_features
    empty_model = GraphTaskFromNodeModel(
        node_model=WeightedNodeGIN(
            input_feat=first_nn.in_features,
            num_layers=len(graph_level_model.node_model.convs),
            hidden_channels=hidden_channels,
            output_channels=hidden_channels,
        ),
        incoming_channels=hidden_channels,
        output_graph_channels=1,
    )
    for old_conv, new_conv in zip(
        graph_level_model.node_model.convs, empty_model.node_model.convs
    ):
        new_conv.nn.load_state_dict(old_conv.nn.state_dict())
    empty_model.lin.load_state_dict(graph_level_model.lin.state_dict())
    return empty_model


def get_weighted_model(graph_level_model):
    if isinstance(graph_level_model.node_model, NodeGIN):
        return _get_weighted_GIN(graph_level_model)
    elif isinstance(graph_level_model.node_model, NodeGCN):
        return _get_weighted_GCN(graph_level_model)
