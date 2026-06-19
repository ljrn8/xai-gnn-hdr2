from ..utils import elementwise_entropy, uniform_debug_log
from ...interfaces import GNN, GraphLevelExplainer
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm
from collections.abc import Iterable
from loguru import logger
from torch_geometric.data import Batch
from abc import ABC, abstractmethod


def parralel_MC_BCE_estimate(
    tau, model: GNN, G: Data, edge_mask_logits: Tensor, samples: int
) -> float:
    """Monte Carlo Binary Concrete estimation for probabilistic edge masks (parrelelized) on a binary graph classifier.

    Returns:
        scores (float): average MC estimate of sigmoided predictions across samples.
    """
    u = torch.rand(samples, edge_mask_logits.shape[0]).clamp(1e-6, 1 - 1e-6)
    log_noise = torch.log(u) - torch.log(1 - u)
    hard_masks = torch.sigmoid((edge_mask_logits.squeeze() + log_noise) / tau)
    batched_G = Batch.from_data_list([G] * samples)
    edge_weight = hard_masks.reshape(-1)
    scores = torch.sigmoid(
        model(
            batched_G.x,
            batched_G.edge_index,
            edge_weight=edge_weight,
            batch=batched_G.batch,
        )
    )
    return scores.view(-1).mean()


def get_model_embeddings_batched(model: GNN, graphs: Batch):
    """Detatched lists of the node embeddings for a batched forward for each intermediate layer.

    Returns:
        y_pred_logits (Tensor): logit predictions per graph
        embeddings_list (list): [[layer1_embeds, layer2_embeds], # graph 1
                                [layer1_embeds, layer2_embeds], ..] # graph 2

    """
    logits, embeddings_list = model.forward(
        graphs.x, graphs.edge_index, return_all_embeddings=True, batch=graphs.batch
    )
    num_graphs = graphs.batch.max().item() + 1
    embeddings_list = [
        [layer_emb[graphs.batch == g].detach() for layer_emb in embeddings_list]
        for g in range(num_graphs)
    ]
    return logits.detach(), embeddings_list


class CustomExplanationModule(ABC, nn.Module):
    """Graph classification interface for PGExplainer-style explanation modules"""

    def __init__(self, model: GNN, graphs: Iterable[Data], hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.model = model
        self.graphs = graphs
        batch_obj = Batch.from_data_list(graphs)
        self.logits, self.embeddings_list = get_model_embeddings_batched(
            model, batch_obj
        )
        self.embeddings_size = self.embeddings_list[0][0].shape[1]

        assert hasattr(graphs[0], "x"), "ill formated graphs"
        assert (
            self.embeddings_list[0][1].shape[1] == self.embeddings_size
        ), "embeddings size must be the same for all layers"

    def get_explanation(self) -> Iterable[torch.Tensor]:
        """Produce edge mask logits per graph"""


class PGEExplanationModule(CustomExplanationModule):
    """Default Explanation module for PGExplainer fitting an MLP over a model's final layer embeddings (concatenated for edge embeddings)."""

    def __init__(self, model: GNN, graphs: Iterable[Data], hidden_size):
        super().__init__(model, graphs, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.embeddings_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.final_embeddings = [e[-1] for e in self.embeddings_list]

    def get_explanation(self):
        masks = []
        for G, fe in zip(self.graphs, self.final_embeddings):
            src, dst = G.edge_index
            edge_embeddings = torch.cat([fe[src], fe[dst]], dim=-1)
            mask = self.mlp(edge_embeddings)
            assert mask.requires_grad, "grad issue"
            masks.append(mask)

        return masks


class ComprehensiveMLPExplanationModule(CustomExplanationModule):
    """Extension of the default PGExplainer module that fits all intermediate model embeddings."""

    def __init__(self, model: GNN, graphs: Iterable[Data], hidden_size):
        super().__init__(model, graphs, hidden_size)
        n_layers = len(self.embeddings_list[0])
        self.mlp = nn.Sequential(
            nn.Linear(self.embeddings_size * 2 * n_layers, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def get_explanation(self):
        concatenated_node_embeddings = [
            torch.concat(layer_embeddings_list, axis=1)
            for layer_embeddings_list in self.embeddings_list
        ]
        masks = []
        for G, node_emb in zip(self.graphs, concatenated_node_embeddings):
            src, dst = G.edge_index
            edge_embeddings = torch.cat(
                [
                    node_emb[src],
                    node_emb[dst],
                ],
                dim=-1,
            )
            mask = self.mlp(edge_embeddings)
            masks.append(mask)

        return masks


class GRUExplanationModule(CustomExplanationModule):
    """Fits a GRU to all latent model embeddings, concatenating its own final node embeddings prior to a FC layer for mask output."""

    def __init__(self, model: GNN, graphs: Iterable[Data], hidden_size):
        super().__init__(model, graphs, hidden_size)
        embeddings_size = self.embeddings_list[0][0].shape[1]
        self.gru = nn.GRU(
            input_size=embeddings_size,
            hidden_size=hidden_size,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def get_explanation(self):
        masks = []
        for G, embeddings_layer_list in zip(self.graphs, self.embeddings_list):
            src, dst = G.edge_index
            embeddings_layer_sequence = torch.stack(embeddings_layer_list)
            _, h_n = self.gru(embeddings_layer_sequence)
            node_representations = h_n[0]
            edge_representations = torch.cat(
                [
                    node_representations[src],
                    node_representations[dst],
                ],
                dim=-1,
            )
            mask = self.fc(edge_representations)
            masks.append(mask)

        return masks


class PGExplainer(GraphLevelExplainer):
    """Standrad PGExplainer Implementation for graph-level binary classification an abritrary explanation module."""

    def __init__(
        self,
        model: GNN,
        graphs: Iterable[Data],
        hidden_size,
        epochs,
        lr,
        mean_regularization,
        entropy_regularization,
        tau,
        reparameterization_samples,
        explanation_module_class: CustomExplanationModule = PGEExplanationModule,
        loss_f=torch.nn.BCELoss(),
    ):
        super().__init__(model, graphs)
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.mean_regularization = mean_regularization
        self.entropy_regularization = entropy_regularization
        self.tau = tau
        self.reparameterization_samples = reparameterization_samples
        self.loss_f = loss_f
        self.explanation_module = explanation_module_class(
            model=model, graphs=graphs, hidden_size=hidden_size
        )

    def explain_graph_task(self):
        model, graphs = self.explanation_module.model, self.explanation_module.graphs
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # unmasked graph scores
        y_preds = [torch.sigmoid(model(G.x, G.edge_index).view(-1)) for G in graphs]
        y_preds = torch.cat(y_preds).detach()
        assert not y_preds.requires_grad, "grad issue"
        optimizer = torch.optim.Adam(self.explanation_module.parameters(), lr=self.lr)

        pbar = tqdm(range(1, self.epochs + 1))
        for epc in pbar:
            optimizer.zero_grad()
            logit_edge_masks = self.explanation_module.get_explanation()

            # subgraph estimate
            explanatory_y_preds = torch.stack(
                [
                    parralel_MC_BCE_estimate(
                        tau=0.5,
                        model=model,
                        G=G,
                        edge_mask_logits=logit_edge_mask,
                        samples=self.reparameterization_samples,
                    )
                    for G, logit_edge_mask in zip(graphs, logit_edge_masks)
                ]
            )

            # regularization
            soft_edge_masks = [torch.sigmoid(m) for m in logit_edge_masks]
            entropy_reg = (
                self.entropy_regularization
                * torch.stack(
                    [elementwise_entropy(m).mean() for m in soft_edge_masks]
                ).mean()
            )
            mean_reg = (
                self.mean_regularization
                * torch.stack([m.mean() for m in soft_edge_masks]).mean()
            )

            loss = self.loss_f(explanatory_y_preds, y_preds)
            loss += mean_reg + entropy_reg
            loss.backward()

            optimizer.step()
            pbar.set_description(
                f"PGExplainer @ epc {epc} | BCEloss={loss.item():.5f} | entropy_reg={entropy_reg:.5f} | mean_reg={mean_reg:.5f}"
            )

        uniform_debug_log(soft_edge_masks)
        return soft_edge_masks, loss.item()
