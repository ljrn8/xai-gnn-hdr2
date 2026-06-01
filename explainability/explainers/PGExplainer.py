from explainability.explainer_utils import Explainer, elementwise_entropy
from training.models import GNN
import torch
from torch import Tensor
import torch.nn as nn
import random as rand
from torch_geometric.data import Data
from tqdm import tqdm
from collections.abc import Iterable
import numpy as np
from loguru import logger
from torch_geometric.data import Batch

class PGExplainer(Explainer):
    def __init__(
        self,
        epochs,
        hidden_size,
        lr,
        mean_regularization,
        entropy_regularization,
        tau,
        reparameterization_samples,
        loss_f=torch.nn.BCELoss(),
    ):
        self.epochs = epochs
        self.lr = lr
        self.mean_regularization = mean_regularization
        self.entropy_regularization = entropy_regularization
        self.tau = tau
        self.hidden_size = hidden_size
        self.reparameterization_samples = reparameterization_samples
        self.loss_f = loss_f

    def _batched_estimate_masked_prediction(self, model: GNN, G: Data, edge_mask_logits: Tensor, samples: int) -> float:
        u = torch.rand(samples, edge_mask_logits.shape[0]).clamp(1e-6, 1 - 1e-6)
        log_noise = torch.log(u) - torch.log(1 - u)
        hard_masks = torch.sigmoid(
            (edge_mask_logits.squeeze() + log_noise) / self.tau
        )
        batched_G = Batch.from_data_list([G] * samples)
        edge_weight = hard_masks.reshape(-1)
        scores = torch.sigmoid(model(batched_G.x, batched_G.edge_index, 
                                    edge_weight=edge_weight, batch=batched_G.batch))
        return scores.view(-1).mean()

    def explain_graph_task(self, model: GNN, graphs: Iterable[Data]):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        y_preds, final_embeddings = [], []
        for G in graphs:
            if G.x.dim() == 1:
                G.x = G.x.unsqueeze(1)

            logit, emb = model(G.x, G.edge_index, return_all_embeddings=True)
            logit = logit.view(-1)
            final_embeddings.append(emb[-1])
            y_preds.append(torch.sigmoid(logit))

        y_preds = torch.cat(y_preds).detach()  # ensure no backprop
        embedding_size = final_embeddings[0].shape[1]
        mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size*2, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=1),
        )
        optimizer = torch.optim.Adam(mlp.parameters(), lr=self.lr)

        pbar = tqdm(range(1, self.epochs + 1))
        for epc in pbar:
            optimizer.zero_grad()
            explanatory_y_preds = []
            masks = []
            mean_reg = 0
            entropy_reg = 0

            for G, final_embedding in zip(graphs, final_embeddings):
                src, dst = G.edge_index
                edge_concatenated_embeddings = torch.cat((final_embedding[src], final_embedding[dst]), dim=1) 
                edge_mask_logits = mlp(edge_concatenated_embeddings)
                y_pred = self._batched_estimate_masked_prediction(
                    model, G, edge_mask_logits, samples=self.reparameterization_samples
                )
                explanatory_y_preds.append(y_pred)
                soft_edge_mask = torch.sigmoid(edge_mask_logits)
                masks.append(soft_edge_mask)
                entropy_reg += (
                    self.entropy_regularization
                    * elementwise_entropy(soft_edge_mask).mean()
                )
                mean_reg += self.mean_regularization * soft_edge_mask.mean()

            stacked = torch.stack(explanatory_y_preds)
            loss = self.loss_f(stacked, y_preds)

            mean_reg /= len(graphs)
            entropy_reg /= len(graphs)
            loss += mean_reg + entropy_reg
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f"PGExplainer @ epc {epc} | BCEloss={loss.item():.5f} | entropy_reg={entropy_reg:.5f} | mean_reg={mean_reg:.5f}"
            )

        return masks


    def explain_node_task(self, model, graph):
        raise NotImplementedError()