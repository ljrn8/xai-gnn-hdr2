import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class NodeGCN2(nn.Module):
    def __init__(self, input_feat, hidden_channels, output_channels, dropout=0.5):
        super(NodeGCN2, self).__init__()
        self.conv1 = GCNConv(input_feat, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    

class GraphTaskNodeGCN2(NodeGCN2):
    def __init__(self,  input_feat, hidden_channels, output_graph_channels):
        super(GraphTaskNodeGCN2, self).__init__(input_feat, hidden_channels, hidden_channels)
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
    
