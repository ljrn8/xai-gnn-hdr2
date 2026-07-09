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
from ..proxy_generation import ProxyGraphGenerator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_concrete_sample(logits: Tensor, tau: float, samples: int) -> Tensor:
    """Samples from the Binary Concrete distribution (Maddison et al., 2017) for a given logit tensor.

    Args:
        logits (Tensor): Logits for the Bernoulli distribution.
        tau (float): Temperature parameter for the Binary Concrete distribution.
        samples (int): Number of samples to draw.

    Returns:
        Tensor: Samples from the Binary Concrete distribution with shape (samples, *logits.shape).
    """
    u = torch.rand(samples, *logits.shape, device=logits.device).clamp(1e-6, 1 - 1e-6)
    log_noise = torch.log(u) - torch.log(1 - u)
    return torch.sigmoid((logits.unsqueeze(0) + log_noise) / tau)


def inverse_gaussian_sample(logits: Tensor, tau: float, samples: int) -> Tensor:
    ...


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

    def __init__(self, model: GNN, graphs: Iterable[Data], hidden_size, output_size):
        super().__init__()
        self.output_size = output_size
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

    def __init__(self, model: GNN, graphs: Iterable[Data], hidden_size, output_size):
        super().__init__(model, graphs, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.embeddings_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.final_embeddings = [e[-1] for e in self.embeddings_list]

    def get_explanation(self):
        # build all edge embeddings in one shot, run MLP once, split back by graph
        all_edge_embeddings = []
        edge_counts = []
        for G, fe in zip(self.graphs, self.final_embeddings):
            src, dst = G.edge_index
            all_edge_embeddings.append(torch.cat([fe[src], fe[dst]], dim=-1))
            edge_counts.append(G.edge_index.shape[1])

        all_masks = self.mlp(torch.cat(all_edge_embeddings, dim=0))
        return list(all_masks.split(edge_counts))


class ComprehensiveMLPExplanationModule(CustomExplanationModule):
    """Extension of the default PGExplainer module that fits all intermediate model embeddings."""

    def __init__(self, model: GNN, graphs: Iterable[Data], hidden_size, output_size):
        super().__init__(model, graphs, hidden_size)
        n_layers = len(self.embeddings_list[0])
        self.mlp = nn.Sequential(
            nn.Linear(self.embeddings_size * 2 * n_layers, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def get_explanation(self):
        all_edge_embeddings = []
        edge_counts = []
        for G, layer_embeddings_list in zip(self.graphs, self.embeddings_list):
            node_emb = torch.cat(layer_embeddings_list, dim=1)
            src, dst = G.edge_index
            all_edge_embeddings.append(torch.cat([node_emb[src], node_emb[dst]], dim=-1))
            edge_counts.append(G.edge_index.shape[1])

        all_masks = self.mlp(torch.cat(all_edge_embeddings, dim=0))
        return list(all_masks.split(edge_counts))


class GRUExplanationModule(CustomExplanationModule):
    """Fits a GRU to all latent model embeddings, concatenating its own final node embeddings prior to a FC layer for mask output."""

    def __init__(self, model: GNN, graphs: Iterable[Data], hidden_size, output_size):
        super().__init__(model, graphs, hidden_size)
        embeddings_size = self.embeddings_list[0][0].shape[1]
        self.gru = nn.GRU(
            input_size=embeddings_size,
            hidden_size=hidden_size,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def get_explanation(self):
        all_edge_embeddings = []
        edge_counts = []
        for G, embeddings_layer_list in zip(self.graphs, self.embeddings_list):
            src, dst = G.edge_index
            embeddings_layer_sequence = torch.stack(embeddings_layer_list)
            _, h_n = self.gru(embeddings_layer_sequence)
            node_representations = h_n[0]
            all_edge_embeddings.append(torch.cat([node_representations[src], node_representations[dst]], dim=-1))
            edge_counts.append(G.edge_index.shape[1])

        # GRU shares weights so we can batch the FC call across all graphs
        all_masks = self.fc(torch.cat(all_edge_embeddings, dim=0))
        return list(all_masks.split(edge_counts))


class PGExplainer(GraphLevelExplainer):
    """Ablation PGExplainer"""

    def __init__(
        self,
        hidden_size,
        epochs,
        lr,
        mean_regularization,
        entropy_regularization,
        tau,
        reparameterization_samples,
        use_proxy_graphs: bool = False,
        proxy_lr: float = 0.01,
        proxy_M: int = 1,
        sampler_method: str = 'GS',
        explanation_module: str = 'default',
        loss_f=torch.nn.BCELoss(),
    ):
        super().__init__()
        self.use_proxy_graphs = use_proxy_graphs
        self.proxy_lr = proxy_lr
        self.proxy_M = proxy_M
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.mean_regularization = mean_regularization
        self.entropy_regularization = entropy_regularization
        self.tau = tau
        self.reparameterization_samples = reparameterization_samples
        self.loss_f = loss_f
        self.example_loss_curves = {'BCELoss': [], 'entropy_regularization': [], 'mean_regularization': []}
        
        self.sampler_method = {
            'GS': binary_concrete_sample,
            'IGR': inverse_gaussian_sample
        }[sampler_method]

        self.explanation_module_class = {
            'default': PGEExplanationModule,
            'comprehensive': ComprehensiveMLPExplanationModule,
            'gru': GRUExplanationModule
        }[explanation_module]


    def _tile_graphs(self, graphs: Iterable[Data]):
        logger.info('Tiling graphs for Monte Carlo sampling')
        tiled_graphs = [G for G in graphs for _ in range(self.reparameterization_samples)]
        tiled = Batch.from_data_list(tiled_graphs)
        tiled.to(DEVICE)
        logger.info('done')
        return tiled

    def entropy_regularization(self, soft_edge_masks):
        return (
            self.entropy_regularization
                * torch.stack(
                    [elementwise_entropy(m).mean() for m in soft_edge_masks]
                ).mean()
        )
    
    def mean_regularization(self, soft_edge_masks):
        return (
            self.mean_regularization
                * torch.stack([m.mean() for m in soft_edge_masks]).mean()
        )

    def _proxy_generator_inner_optimization(self, soft_edge_masks):
        logger.debug('running inner optimization for proxy graph generator')
        for _ in range(self.proxy_M):
            self.proxy_optimizer.zero_grad()
            gen_loss = sum(
                self.proxy_generator(G, mask)[1]
                for G, mask in zip(self.graphs, soft_edge_masks)
            )
            gen_loss.backward()
            self.proxy_optimizer.step()

    def _proxy_generator_build_proxy_graphs(self, soft_edge_masks):
        logger.debug('running proxy graph construction for MC estimate')
        proxy_graphs = []
        is_exp_masks = []
        for G, mask in zip(self.graphs, soft_edge_masks):
            A_tilde, _ = self.proxy_generator(G, mask.detach())
            proxy_graphs.append(self.proxy_generator.build_proxy_data(G, A_tilde, mask.detach()))
            is_exp_masks.append((mask.detach().squeeze() > 0.5).view(-1))

        return proxy_graphs, is_exp_masks

    def _batched_estimation(
            self,
            model, 
            tiled_graphs_batch: Batch, 
            all_edge_mask_logits,
            n_graphs,
            hard_mask_generator=binary_concrete_sample
    ):
        """Parrallel Monte Carlo Binary Concrete Distribution estimation for 
        subgraph predictions using a logit (attribution) edge mask"""
        
        # for each graph, sample noise and compute hard masks
        edge_weights = []
        for logits in all_edge_mask_logits:
            hard = hard_mask_generator(logits.squeeze(), self.tau, self.reparameterization_samples)  
            edge_weights.append(hard.reshape(-1))  # (samples * n_edges,)
        
        edge_weight = torch.cat(edge_weights)  # matches batched edge_index ordering
        scores = torch.sigmoid(model(tiled_graphs_batch.x, 
                                    tiled_graphs_batch.edge_index, edge_weight=edge_weight, batch=tiled_graphs_batch.batch))

        return scores.view(n_graphs, self.reparameterization_samples).mean(dim=1)  # (N,)

    def _proxy_batched_estimation(
        self,
        model,
        tiled_graphs_batch: Batch,
        all_edge_mask_logits, 
        is_exp_masks, 
        proxy_edge_counts,
        n_graphs,
        hard_mask_generator=binary_concrete_sample
    ):
        """Same as _batched_estimation, but edge_weight is built to match the proxy
        graph's edge_index instead of the original graph's: edges kept from G_exp reuse
        their Binary Concrete sample (selected via is_exp_masks), and the remaining
        proxy-only (delta) edges get weight 1.0, since they were already a hard
        present/absent decision made when the proxy graph was built.
        """
        edge_weights = []
        for logits, is_exp, n_proxy_edges in zip(all_edge_mask_logits, is_exp_masks, proxy_edge_counts):
            hard = hard_mask_generator(logits.squeeze(), self.tau, self.reparameterization_samples)  
            exp_weight = hard[:, is_exp]  # (samples, n_exp_edges)
            n_new_edges = n_proxy_edges - exp_weight.shape[1]
            new_weight = torch.ones(self.reparameterization_samples, n_new_edges, device=logits.device)

            edge_weights.append(torch.cat([exp_weight, new_weight], dim=1).reshape(-1))

        edge_weight = torch.cat(edge_weights)  # matches batched proxy edge_index ordering
        scores = torch.sigmoid(model(tiled_graphs_batch.x,
                                    tiled_graphs_batch.edge_index, edge_weight=edge_weight, batch=tiled_graphs_batch.batch))

        return scores.view(n_graphs, self.reparameterization_samples).mean(dim=1)  # (N,)

    def explain_graph_task(self, model, graphs):
        self.explanation_module = self.explanation_module_class(
            model=model, graphs=graphs, hidden_size=self.hidden_size
        ).to(DEVICE)

        if self.use_proxy_graphs:
            logger.info('Proxy graph generator has been enabled')
            self.proxy_generator = ProxyGraphGenerator(node_feature_dim=graphs[0].x.shape[1])
            self.proxy_optimizer = torch.optim.Adam(self.proxy_generator.parameters(), lr=self.proxy_lr)
        else:
            # tile each graph `samples` times
            # proxy graphs have variable sizes and are handled differently
            tiled = self._tile_graphs(model, graphs)

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # unmasked graph scores — batch all graphs in one forward pass
        batch_obj = Batch.from_data_list(graphs)
        with torch.no_grad():
            y_preds = torch.sigmoid(
                model(batch_obj.x, batch_obj.edge_index, batch=batch_obj.batch).view(-1)
            ).detach()

        optimizer = torch.optim.Adam(self.explanation_module.parameters(), lr=self.lr)

        pbar = tqdm(range(1, self.epochs + 1))
        for epc in pbar:
            optimizer.zero_grad()
            logit_edge_masks = self.explanation_module.get_explanation()

            for m, g in zip(logit_edge_masks, self.graphs):
                assert m.shape[0] == g.edge_index.shape[1], f"mask shape {m.shape} does not match graph edge count {g.edge_index.shape}"

            soft_edge_masks = [torch.sigmoid(m) for m in logit_edge_masks]

            if self.use_proxy_graphs:
                self._proxy_generator_inner_optimization(soft_edge_masks)
                proxy_graphs, is_exp_masks = self._proxy_generator_build_proxy_graphs(soft_edge_masks)
                proxy_edge_counts = [G.edge_index.shape[1] for G in proxy_graphs]
                tiled_proxy = [G for G in proxy_graphs for _ in range(self.reparameterization_samples)]
                eval_batch = Batch.from_data_list(tiled_proxy).to(DEVICE)

                # Subgraph Estimate against the proxy graph
                explanatory_y_preds = self._proxy_batched_estimation(
                    model=model,
                    tiled_graphs_batch=eval_batch,
                    all_edge_mask_logits=logit_edge_masks,
                    is_exp_masks=is_exp_masks,
                    proxy_edge_counts=proxy_edge_counts,
                    n_graphs=len(self.graphs),
                    hard_mask_generator=self.sampler_method
                )

            else:
                # Subgraph Estimate
                explanatory_y_preds = self._batched_estimation(
                    model=model,
                    tiled_graphs_batch=tiled,
                    all_edge_mask_logits=logit_edge_masks,
                    n_graphs=len(self.graphs),
                    hard_mask_generator=self.sampler_method
                )

            # regularization
            entropy_reg = self.entropy_regularization(soft_edge_masks)
            mean_reg = self.entropy_regularization(soft_edge_masks)
            
            loss = self.loss_f(explanatory_y_preds, y_preds)
            loss += mean_reg + entropy_reg
            loss.backward()
            optimizer.step()

            self.example_loss_curves['BCELoss'].append(loss.item())
            self.example_loss_curves['entropy_regularization'].append(entropy_reg.item())
            self.example_loss_curves['mean_regularization'].append(mean_reg.item())

            pbar.set_description(
                f"PGExplainer @ epc {epc} | BCEloss={loss.item():.5f} | entropy_reg={entropy_reg:.5f} | mean_reg={mean_reg:.5f}"
            )

        uniform_debug_log(soft_edge_masks)
        return soft_edge_masks, loss.item()
