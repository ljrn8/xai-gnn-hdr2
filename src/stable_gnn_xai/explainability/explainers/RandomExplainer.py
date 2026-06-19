from ...interfaces import GNN, GraphLevelExplainer
import torch


class RandomExplainer(GraphLevelExplainer):
    """Random edge mask generator for benchmarking"""

    def __init__(self, graphs):
        self.graphs = graphs

    def explain_node_task(self):
        raise NotImplementedError()

    def explain_graph_task(self):
        return [torch.rand(size=(graph.edge_index.shape[1],)) for graph in self.graphs]
