import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing

class WeightedNodeGCN(nn.Module):
    """ GCN model variant that can handle an edge_weight in forward(),
    required for passing fractional subgraph explanations.
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
    """ GIN model variant that can handle an edge_weight in forward(),
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

class GraphTaskFromNodeModel(nn.Module):
    def __init__(
        self, node_model, incoming_channels, output_graph_channels, dropout=None
    ):
        super(GraphTaskFromNodeModel, self).__init__()
        self.node_model = node_model
        self.dropout = dropout
        self.lin = nn.Linear(incoming_channels, output_graph_channels)

    def forward(self, x, edge_index, return_embeds=False):
        # get the final output of the last GIN (considered as the embeddings here)
        x = self.node_model(x, edge_index, return_embeds=False)
        if return_embeds:
            return x

        x = x.relu()
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch=None)
        x = self.lin(x)
        return x


# for config JSON mapping
MODEL_ID = {
    "GCN": WeightedNodeGCN,
    "GIN": WeightedNodeGIN,
}
