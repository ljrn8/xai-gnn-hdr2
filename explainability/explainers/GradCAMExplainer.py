import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from explainability.explainer_utils import Explainer
from training.models import GNN


class GradCAMExplainer(Explainer):
    def __init__(self, layer_idx: int = -1, node_agg: str = "sum"):
        self.layer_idx = layer_idx
        self.node_agg = node_agg

        self._activations = None
        self._gradients = None

    def _register_hooks(self, layer: nn.Module):
        """Attach forward/backward hooks to capture activations and gradients."""

        def forward_hook(module, input, output):
            # output shape: [N, d] — one embedding row per node
            self._activations = output

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] shape: [N, d]
            self._gradients = grad_output[0]

        h_fwd = layer.register_forward_hook(forward_hook)
        h_bwd = layer.register_full_backward_hook(backward_hook)
        return h_fwd, h_bwd

    def _remove_hooks(self, hooks):
        for h in hooks:
            h.remove()

    def _compute_edge_scores(self, model: GNN, x, edge_index,
                              edge_weight=None, target_node=None):
        """
        Runs one forward+backward pass and returns per-edge importance scores.

        Args:
            target_node: for node tasks, the node whose prediction we differentiate.
                         None means we differentiate the graph-level output directly.
        """
        layer = model.layers[self.layer_idx]
        hooks = self._register_hooks(layer)

        try:
            model.zero_grad()
            out = model(x, edge_index, edge_weight)  # [N, C] or [C]

            if target_node is not None:
                # Node task: scalar = predicted class logit for this node
                score = out[target_node].max()
            else:
                # Graph task: scalar = max logit over graph-level output
                score = out.max()

            score.backward()

        finally:
            self._remove_hooks(hooks)

        # activations: [N, d], gradients: [N, d]
        A = self._activations          # node embeddings
        G = self._gradients            # gradients w.r.t. those embeddings

        # Global-average-pool gradients over feature dim -> scalar weight per node
        # Then weight the activations and sum over features
        alpha = G  # shape [N, d] — per-feature gradient already gives us weights
        node_scores = torch.relu((alpha * A).sum(dim=-1))  # [N]

        # Map node scores -> edge scores
        src, dst = edge_index
        if self.node_agg == "sum":
            edge_scores = node_scores[src] + node_scores[dst]
        elif self.node_agg == "src":
            edge_scores = node_scores[src]
        elif self.node_agg == "max":
            edge_scores = torch.max(node_scores[src], node_scores[dst])
        else:
            raise ValueError(f"Unknown node_agg: {self.node_agg}")

        # Normalise to [0, 1]
        if edge_scores.max() > 0:
            edge_scores = edge_scores / edge_scores.max()

        return edge_scores.detach()

    def explain_node_task(self, model: GNN, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_weight = getattr(graph, "edge_weight", None)

        results = {}
        for node in range(x.size(0)):
            scores = self._compute_edge_scores(
                model, x, edge_index, edge_weight, target_node=node
            )
            results[node] = scores

        return results

    def explain_graph_task(self, model: GNN, graphs):
        results = []
        for graph in graphs:
            x = graph.x
            edge_index = graph.edge_index
            edge_weight = getattr(graph, "edge_weight", None)

            scores = self._compute_edge_scores(
                model, x, edge_index, edge_weight, target_node=None
            )
            results.append(scores)

        return results