from ..interfaces import GNN, GraphLevelExplainer
import torch


class RandomExplainer(GraphLevelExplainer):
    """Random edge mask generator for benchmarking"""

    def explain_graph_task(self,  model, graphs):
        return [
            (torch.rand(size=(graph.edge_index.shape[1],)), -1) 
            for graph in self.graphs
        ]
