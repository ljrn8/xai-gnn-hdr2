import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool


class NodeGIN(nn.Module):
    def __init__(
        self, input_feat, num_layers, hidden_channels, output_channels, dropout=None
    ):
        super(NodeGIN, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(input_feat, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
            )
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    )
                )
            )
        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, output_channels),
                    # no activation here
                )
            )
        )

    def forward(self, x, edge_index, return_embeds=False):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embeds:
            return x
        x = self.convs[-1](x, edge_index)
        return x


class NodeGCN(nn.Module):
    def __init__(
        self, input_feat, num_layers, hidden_channels, output_channels, dropout=None
    ):
        super(NodeGCN, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_feat, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, output_channels))

    def forward(self, x, edge_index, return_embeds=False):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embeds:
            return x
        x = self.convs[-1](x, edge_index)
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
