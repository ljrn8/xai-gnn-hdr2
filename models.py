import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class NodeGCN(nn.Module):
    def __init__(self, input_feat, num_layers, hidden_channels, output_channels, dropout=0.5):
        super(NodeGCN, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_feat, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, output_channels))


    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    

class GraphTaskNodeGCN(NodeGCN):
    def __init__(self,  input_feat, num_gcn_layers, hidden_channels, output_graph_channels):
        super(GraphTaskNodeGCN, self).__init__(input_feat, num_gcn_layers, hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, output_graph_channels)

    def forward(self, x, edge_index, batch=None):
        x = super().forward(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


class GraphTaskWrapper():
    def __init__(self, node_model, output_channels, output_graph_channels):
        super(GraphTaskWrapper, self).__init__()
        self.node_model = node_model
        self.lin = nn.Linear(output_channels, output_graph_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.node_model(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x
    
