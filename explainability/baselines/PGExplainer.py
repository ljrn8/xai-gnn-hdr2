from explainability.explainer_utils import Explainer
from training.models import GNN
import torch
from torch import Tensor
import torch.nn as nn
import random as rand
from torch_geometric.data import Data

class PGExplainer(Explainer):
    def __init__(self, epochs, hidden_size, lr, size_reg, edge_reg, tau, reparameterization_samples):
        self.epochs = epochs
        self.lr = lr
        self.alpha = size_reg
        self.edge_reg = edge_reg
        self.beta = tau
        self.hidden_size = hidden_size
        self.reparameterization_samples = reparameterization_samples

    def _estimate_masked_prediction(self, model: GNN, G: Data, edge_mask_logits: Tensor, samples: int):
        scores = []
        for _ in range(samples):
            hard_mask = self._binary_concrete_sample(edge_mask_logits, self.tau)
            scores.append(
                torch.sigmoid(
                    model(G.x, G.edge_index, edge_weight=hard_mask)
                )
            )

        scores = torch.cat(scores)
        return torch.mean(scores, axis=1)

    def _binary_concrete_sample(self, edge_mask_logits, tau):
        e = rand.uniform(0, 1)
        log_noise = torch.log(e) - torch.log(1 - e)
        return torch.sigmoid((edge_mask_logits + log_noise) / tau)

    def explain_graph_task(self, model: GNN, graphs):
        model.eval()
        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        y_preds, embeddings = [], []
        for G in graphs:
            if G.x.dim() == 1:
                G.x = G.x.unsqueeze(1)

            logit, embeddings = model(G.x, G.edge_index, return_all_embeddings=True)
            logit = logit.view(-1)
            y = G.y.float().view(-1)
            
        final_embeddings = [e[-1] for e in embeddings]
        embedding_size = final_embeddings[0].shape[1]
        mlp = nn.Sequential([
            nn.Linear(in_features=embedding_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=1),
        ])


        # TODO Questions to answer first
        # 1. are there any negative (class=0) toplogies in OGX or is it positive motif detection alone
        # 2. SGD vs batch descent


        optimizer = torch.optim.Adam(mlp.params, lr=self.lr)
        ...


        for epc in range(1, self.epochs+1):
            y_preds = []
            for G, final_embedding in zip(graphs, final_embeddings):
                edge_mask_logits = mlp(final_embedding)
                y_pred = self._estimate_masked_prediction(model, G, edge_mask_logits, samples=self.reparameterization_samples)
                y_preds.append(y_pred)

            ...




