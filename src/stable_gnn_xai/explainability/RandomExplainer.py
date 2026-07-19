from ..interfaces import GNN, GraphLevelExplainer
import torch


class RandomExplainer(GraphLevelExplainer):
    """Random edge mask generator for benchmarking"""

    def __init__(self, seed):
        self.seed = seed # NOTE: unused (1 param required for grid search)

    def explain_graph_task(self,  model, graphs):
        masks = [
            (torch.rand(size=(graph.edge_index.shape[1],))) 
            for graph in graphs
        ]
        penalties = [0.0 for _ in graphs]
        return masks, penalties
