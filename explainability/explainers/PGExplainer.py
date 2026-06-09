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
from abc import ABC, abstractmethod
from typing import Optional


def batched_estimate_masked_prediction(
    tau, model: GNN, G: Data, edge_mask_logits: Tensor, samples: int
) -> float:
    """Determine MC estimate of a masked prediction using Binary Concrete reparameterization"""
    u = torch.rand(samples, edge_mask_logits.shape[0]).clamp(1e-6, 1 - 1e-6)
    log_noise = torch.log(u) - torch.log(1 - u)
    hard_masks = torch.sigmoid((edge_mask_logits.squeeze() + log_noise) / tau)
    batched_G = Batch.from_data_list([G] * samples)
    edge_weight = hard_masks.reshape(-1)
    scores = torch.sigmoid(model(
        batched_G.x,
        batched_G.edge_index,
        edge_weight=edge_weight,
        batch=batched_G.batch,
    ))
    return scores.view(-1).mean()


class CustomExplanationModule(ABC, nn.Module):
    @abstractmethod
    def forward(
        self,
        model: GNN,
        graphs: Data,
    ) -> torch.Tensor:
        """ Takes a (batched) graph object and model, producing an edge importance mask as *logits. 
        """
        pass


class PGEExplanationModule(CustomExplanationModule):
    """ Default Explanation module for PGExplainer fitting an MLP over a model's final layer embeddings (concatenated for edge embeddings). 
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, model: GNN, graphs: Data):
        _, embeddings_list = model.forward(
            graphs.x,
            graphs.edge_index,
            return_all_embeddings=True,
        )
        final_embeddings = embeddings_list[-1]
        src, dst = graphs.edge_index
        edge_embeddings = torch.cat(
            [
                final_embeddings[src],
                final_embeddings[dst],
            ],
            dim=-1,
        )
        return self.layers(edge_embeddings)


class DeeperMLPExplanationModule(PGEExplanationModule):
    """ Extension of the default PGExplainer module that fits all latent model embeddings. 
    """
    def forward(self, model: GNN, graphs: Data):
        _, embeddings_list = model.forward(
            graphs.x,
            graphs.edge_index,
            return_all_embeddings=True,
        )
        node_embeddings = torch.stack(
            embeddings_list,
            dim=1,
        ).flatten(start_dim=1)
        src, dst = graphs.edge_index
        edge_embeddings = torch.cat(
            [
                node_embeddings[src],
                node_embeddings[dst],
            ],
            dim=-1,
        )
        return self.layers(edge_embeddings)


class GRUExplanationModule(CustomExplanationModule):
    """ Fits a GRU to all latent model embeddings, concatenating its own final node embeddings prior to a FC layer for mask output. 
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, model: GNN, graphs: Data):
        _, embeddings_list = model.forward(
            graphs.x,
            graphs.edge_index,
            return_all_embeddings=True,
        )
        node_sequences = torch.stack(
            embeddings_list,
            dim=1,
        )
        _, h_n = self.gru(node_sequences)
        node_representations = h_n[-1]
        src, dst = graphs.edge_index
        edge_representations = torch.cat(
            [
                node_representations[src],
                node_representations[dst],
            ],
            dim=-1,
        )
        return self.fc(edge_representations)
    

class PGExplainer(Explainer):
    """ Standrad PGExplainer Implementation for graph-level binary classification an abritrary explanation module.
    """
    def __init__(
        self,
        epochs,
        lr,
        mean_regularization,
        entropy_regularization,
        tau,
        reparameterization_samples,
        explanation_model: CustomExplanationModule,
        loss_f=torch.nn.BCELoss(),
    ):
        self.epochs = epochs
        self.lr = lr
        self.mean_regularization = mean_regularization
        self.entropy_regularization = entropy_regularization
        self.tau = tau
        self.reparameterization_samples = reparameterization_samples
        self.loss_f = loss_f
        self.explanation_model = explanation_model

    def explain_graph_task(self, model: GNN, graphs: Iterable[Data]):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # unmasked graph scores
        y_preds = [torch.sigmoid(model(G.x, G.edge_index).view(-1)) for G in graphs]

        y_preds = torch.cat(y_preds).detach()  # ensure no backprop
        optimizer = torch.optim.Adam(self.explanation_model.parameters(), lr=self.lr)

        pbar = tqdm(range(1, self.epochs + 1))
        for epc in pbar:
            optimizer.zero_grad()
            explanatory_y_preds = []
            masks = []
            mean_reg = 0
            entropy_reg = 0

            for G in graphs:
                edge_mask_logits = self.explanation_model(G=G, model=model)
                y_pred = batched_estimate_masked_prediction(
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
