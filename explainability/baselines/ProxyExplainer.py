
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch import nn
import torch.functional as F
import sys
from abc import ABC, abstractmethod
from torch_geometric.data import Data
from training.GNN_utils import *
from explainability.xAI_utils import *

# utilized original implementaion
sys.path.append("papercode/ProxyExplainer")
from ExplanationEvaluation.explainers.ProxyExplainer import PROXYExplainer

class ProxyExplainerImpl(Explainer):
    def __init__(self, epochs):
        self.epochs = epochs

    class ProxyExplainerModelWrapper(nn.Module):
        """Converts an inductive node classification model to the required model signature and input for ProxyExplainer's paper code."""

        def __init__(self, model):
            super().__init__()
            self.model = model
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.embedding_size = self.model.lin.in_features

        def embedding(self, x, edge_index, edge_weight=None):
            return self.model.node_model(
                x, edge_index, edge_weight=edge_weight, return_embeds=True
            )

        def forward(self, x, edge_index, batch=None, edge_weights=None):
            x = self.model.node_model(
                x, edge_index, edge_weight=edge_weights, return_embeds=False
            )
            x = x.relu()
            if self.model.dropout:
                x = F.dropout(x, p=self.model.dropout, training=self.training)
            x = global_mean_pool(x, batch=batch)
            x = self.model.lin(x)
            return x

    def explain_graph_task(self, task: InductiveGraphClassification, graphs):
        model = get_weighted_model(task.model) if isinstance(task.model.node_model, (NodeGIN, NodeGCN)) else task.model

        # freeze all layers
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        wrapper = self.ProxyExplainerModelWrapper(model)
        wrapper.eval()
        edge_indexes = [g.edge_index for g in graphs]
        features = [g.x for g in graphs]
        self.explainer = PROXYExplainer(
            wrapper, edge_indexes, features, 
            reg_coefs=[0.05, 1.0], # default, cannot be tuple 
            epochs=self.epochs
        )
        graph_masks = []
        idxs = list(range(len(graphs)))
        self.explainer.prepare(indices=idxs)
        for i in tqdm(idxs):
            graph, expl_edge_weights = self.explainer.explain(i)
            graph_masks.append(expl_edge_weights)
        
        return graph_masks

    def explain_node_task(self, task, graph):
        return super().explain_node_task(task, graph)