import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing
from abc import ABC, abstractmethod
from ..interfaces import GNN


class WeightedNodeGCN(GNN):
    """GCN model variant that can handle an edge_weight in forward()
    Required for passing fractional subgraph explanations.
    """

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

    @property
    def layers(self):
        return self.convs

    def forward(self, x, edge_index, edge_weight=None, return_all_embeddings=False):
        embeddings = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            if return_all_embeddings:
                embeddings.append(x)

            x = x.relu()
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        if return_all_embeddings:
            return x, embeddings

        return x


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


class WeightedNodeGIN(GNN):
    """GIN model variant that can handle an edge_weight in forward() 
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

    @property
    def layers(self):
        return self.convs

    def forward(self, x, edge_index, edge_weight=None, return_all_embeddings=False):
        embeddings = []
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            if return_all_embeddings:
                embeddings.append(x)

            x = x.relu()
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        if return_all_embeddings:
            return x, embeddings

        return x


class GraphGNNWrapper(GNN):
    """Attatched FC linear layer to node classifier for graph classification (includes batch pooling)"""

    def __init__(
        self, node_model: GNN, incoming_channels, output_graph_channels, dropout=None
    ):
        super(GraphGNNWrapper, self).__init__()
        self.node_model = node_model
        self.dropout = dropout
        self.lin = nn.Linear(incoming_channels, output_graph_channels)

    @property
    def layers(self):
        return self.node_model.layers + [self.lin]

    def forward(
        self, x, edge_index, edge_weight=None, return_all_embeddings=False, batch=None
    ):
        x = self.node_model(
            x,
            edge_index,
            edge_weight=edge_weight,
            return_all_embeddings=return_all_embeddings,
        )
        if return_all_embeddings:
            x, embeddings = x
            embeddings.append(x)

        x = x.relu()
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch=batch)

        x = self.lin(x)
        if return_all_embeddings:
            return x, embeddings

        return x


