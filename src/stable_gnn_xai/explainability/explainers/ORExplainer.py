from ..explainer_utils import Explainer
from src.stable_gnn_xai.training.train import GNN
import torch
import torch.nn as nn


class ORExplainer(Explainer):
    def __init__(
        self, hidden_channels=64, epochs=100, alpha=0.1, beta=0.1, gamma=0.1, bias=1e-10
    ):
        super(ORExplainer, self).__init__()
        self.hidden_channels = hidden_channels
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bias = bias
        self.hidden_channels = hidden_channels

    def explain_graph_task(self, model: GNN, test_graphs):
        raise NotImplementedError(
            "ORExplainer does not support graph-level explanations."
        )

    def WEP_value(self, model, graph, edge_mask): ...

    def gumbel_softmax_sampled_prediction(self, edge_mask, model, temperature=1.0): ...

    def explain_node_task(self, model: GNN, graph):
        assert hasattr(
            graph, "test_mask"
        ), "Graph must have a binary test_mask attribute for node-level explanations."
        model.eval()

        # freeze all layers
        for param in model.parameters():
            param.requires_grad_(False)

        # get node embeddings
        with torch.no_grad():
            y_logits, embeddings_list = model.forward(graph, return_all_embeddings=True)

        # only explain test nodes
        test_mask = graph.test_mask
        test_y_logits = y_logits[test_mask]
        test_embeddings = [emb[test_mask] for emb in embeddings_list]

        # concatenate embeddings from all layers for each node'
        concatenated_embeddings = torch.cat(test_embeddings, dim=1)

        # explanation network
        mlp = nn.Sequential(
            nn.Linear(concatenated_embeddings.size(1), self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, 1),
            nn.Sigmoid(),
        )

        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
        test_node_i = torch.where(test_mask)[0]
        for epoch in range(self.epochs):
            mlp.train()
            optimizer.zero_grad()
            for node_index in test_node_i:

                # predict edge importance scores
                edge_mask = mlp(concatenated_embeddings).squeeze()
                y_star = self.gumbel_softmax_sampled_prediction(
                    edge_mask, model, temperature=1.0, index=node_index
                )
                y = test_y_logits[node_index]

                # more efficiently produce node index predictions ..

            loss.backward()
            optimizer.step()
