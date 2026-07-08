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
from abc import ABC, abstractmethode
from ..proxy_generation import ProxyGraphGenerator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch_MC_BCE_estimate(tau, model, 
                          tiled_graphs_batch: Batch, 
                          all_edge_mask_logits, samples,
                          n_graphs):
    """Parrallel Monte Carlo Binary Concrete Distribution estimation for 
    subgraph predictions using a logit (attribution) edge mask"""
    
    # for each graph, sample noise and compute hard masks
    edge_weights = []
    for logits in all_edge_mask_logits:
        u = torch.rand(samples, logits.shape[0], device=logits.device).clamp(1e-6, 1 - 1e-6)
        log_noise = torch.log(u) - torch.log(1 - u)
        hard = torch.sigmoid((logits.squeeze() + log_noise) / tau)
        edge_weights.append(hard.reshape(-1))  # (samples * n_edges,)
    
    edge_weight = torch.cat(edge_weights)  # matches batched edge_index ordering
    scores = torch.sigmoid(model(tiled_graphs_batch.x, 
                                 tiled_graphs_batch.edge_index, edge_weight=edge_weight, batch=tiled_graphs_batch.batch))

    return scores.view(n_graphs, samples).mean(dim=1)  # (N,)


def _batch_MC_BCE_estimate_proxy(tau, model,
                                tiled_graphs_batch: Batch,
                                all_edge_mask_logits, is_exp_masks, proxy_edge_counts,
                                samples, n_graphs):
    """Same as _batch_MC_BCE_estimate, but edge_weight is built to match the proxy
    graph's edge_index instead of the original graph's: edges kept from G_exp reuse
    their Binary Concrete sample (selected via is_exp_masks), and the remaining
    proxy-only (delta) edges get weight 1.0, since they were already a hard
    present/absent decision made when the proxy graph was built."""

    edge_weights = []
    for logits, is_exp, n_proxy_edges in zip(all_edge_mask_logits, is_exp_masks, proxy_edge_counts):
        u = torch.rand(samples, logits.shape[0], device=logits.device).clamp(1e-6, 1 - 1e-6)
        log_noise = torch.log(u) - torch.log(1 - u)
        hard = torch.sigmoid((logits.squeeze() + log_noise) / tau)  # (samples, n_edges)

        exp_weight = hard[:, is_exp]  # (samples, n_exp_edges)
        n_new_edges = n_proxy_edges - exp_weight.shape[1]
        new_weight = torch.ones(samples, n_new_edges, device=logits.device)

        edge_weights.append(torch.cat([exp_weight, new_weight], dim=1).reshape(-1))

    edge_weight = torch.cat(edge_weights)  # matches batched proxy edge_index ordering
    scores = torch.sigmoid(model(tiled_graphs_batch.x,
                                 tiled_graphs_batch.edge_index, edge_weight=edge_weight, batch=tiled_graphs_batch.batch))

    return scores.view(n_graphs, samples).mean(dim=1)  # (N,)


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

    def __init__(self, model: GNN, graphs: Iterable[Data], hidden_size):
        super().__init__(model, graphs, hidden_size)
        n_layers = len(self.embeddings_list[0])
        self.mlp = nn.Sequential(
            nn.Linear(self.embeddings_size * 2 * n_layers, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
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

    def __init__(self, model: GNN, graphs: Iterable[Data], hidden_size):
        super().__init__(model, graphs, hidden_size)
        embeddings_size = self.embeddings_list[0][0].shape[1]
        self.gru = nn.GRU(
            input_size=embeddings_size,
            hidden_size=hidden_size,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

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
    """PGExplainer: fits an explanation module to produce edge mask logits, evaluated
    against the original graph via Monte Carlo Binary Concrete subgraph sampling."""

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

        self.example_loss_curves = {'BCELoss': [], 'entropy_regularization': [], 'mean_regularization': []}
        self.explanation_module = explanation_module_class(
            model=model, graphs=graphs, hidden_size=hidden_size
        ).to(DEVICE)


    def explain_graph_task(self):
        model, graphs = self.explanation_module.model, self.explanation_module.graphs
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # tile each graph `samples` times
        logger.info('Tiling graphs for Monte Carlo sampling')
        tiled_graphs = [G for G in self.graphs for _ in range(self.reparameterization_samples)]
        batched = Batch.from_data_list(tiled_graphs)
        batched.to(DEVICE)
        logger.info('done')

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

            # Subgraph Estimate
            # (tile-parrelelized over graphs and graph samples)
            explanatory_y_preds = _batch_MC_BCE_estimate(
                tau=self.tau,
                model=model,
                tiled_graphs_batch=batched,
                all_edge_mask_logits=logit_edge_masks,
                samples=self.reparameterization_samples,
                n_graphs=len(self.graphs)
            )

            # regularization
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

            self.example_loss_curves['BCELoss'].append(loss.item())
            self.example_loss_curves['entropy_regularization'].append(entropy_reg.item())
            self.example_loss_curves['mean_regularization'].append(mean_reg.item())

            pbar.set_description(
                f"PGExplainer @ epc {epc} | BCEloss={loss.item():.5f} | entropy_reg={entropy_reg:.5f} | mean_reg={mean_reg:.5f}"
            )

        uniform_debug_log(soft_edge_masks)
        return soft_edge_masks, loss.item()


class ProxyExplainer(PGExplainer):
    """PGExplainer variant with the ProxyExplainer proxy-graph mode from Chen et al. (2026):
    trains a ProxyGraphGenerator alongside the explanation module and evaluates the explanation
    mask against the resulting proxy graph instead of the original graph (Alg. 2)."""

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
        proxy_generator: ProxyGraphGenerator,
        explanation_module_class: CustomExplanationModule = PGEExplanationModule,
        loss_f=torch.nn.BCELoss(),
        proxy_lr: float = 0.01,
        proxy_M: int = 1,
    ):
        super().__init__(
            model=model,
            graphs=graphs,
            hidden_size=hidden_size,
            epochs=epochs,
            lr=lr,
            mean_regularization=mean_regularization,
            entropy_regularization=entropy_regularization,
            tau=tau,
            reparameterization_samples=reparameterization_samples,
            explanation_module_class=explanation_module_class,
            loss_f=loss_f,
        )
        self.proxy_generator = proxy_generator
        self.proxy_M = proxy_M
        self.proxy_optimizer = torch.optim.Adam(self.proxy_generator.parameters(), lr=proxy_lr or lr)

    def explain_graph_task(self):
        model, graphs = self.explanation_module.model, self.explanation_module.graphs
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        logger.info('Tiling graphs for Monte Carlo sampling')
        batch_obj = Batch.from_data_list(graphs)
        with torch.no_grad():
            y_preds = torch.sigmoid(
                model(batch_obj.x, batch_obj.edge_index, batch=batch_obj.batch).view(-1)
            ).detach()
        logger.info('done')

        optimizer = torch.optim.Adam(self.explanation_module.parameters(), lr=self.lr)

        pbar = tqdm(range(1, self.epochs + 1))
        for epc in pbar:
            optimizer.zero_grad()

            logit_edge_masks = self.explanation_module.get_explanation()

            for m, g in zip(logit_edge_masks, self.graphs):
                assert m.shape[0] == g.edge_index.shape[1], f"mask shape {m.shape} does not match graph edge count {g.edge_index.shape}"

            soft_edge_masks = [torch.sigmoid(m) for m in logit_edge_masks]

            # inner optimization: train proxy generator proxy_M steps with explainer fixed
            logger.debug('running inner optimization for proxy graph generator')
            for _ in range(self.proxy_M):
                self.proxy_optimizer.zero_grad()
                gen_loss = sum(
                    self.proxy_generator(G, mask)[1]
                    for G, mask in zip(self.graphs, soft_edge_masks)
                )
                gen_loss.backward()
                self.proxy_optimizer.step()

            # build proxy graphs for this epoch's MC estimate, tracking which edges
            # came from G_exp so the MC estimate can reuse their mask weight
            logger.debug('running proxy graph construction for MC estimate')
            proxy_graphs = []
            is_exp_masks = []
            for G, mask in zip(self.graphs, soft_edge_masks):
                A_tilde, _ = self.proxy_generator(G, mask.detach())
                proxy_graphs.append(self.proxy_generator.build_proxy_data(G, A_tilde, mask.detach()))
                is_exp_masks.append((mask.detach().squeeze() > 0.5).view(-1))

            proxy_edge_counts = [G.edge_index.shape[1] for G in proxy_graphs]
            tiled_proxy = [G for G in proxy_graphs for _ in range(self.reparameterization_samples)]
            eval_batch = Batch.from_data_list(tiled_proxy).to(DEVICE)

            # Subgraph Estimate against the proxy graph
            # (tile-parrelelized over graphs and graph samples)
            explanatory_y_preds = _batch_MC_BCE_estimate_proxy(
                tau=self.tau,
                model=model,
                tiled_graphs_batch=eval_batch,
                all_edge_mask_logits=logit_edge_masks,
                is_exp_masks=is_exp_masks,
                proxy_edge_counts=proxy_edge_counts,
                samples=self.reparameterization_samples,
                n_graphs=len(self.graphs)
            )

            # regularization
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

            self.example_loss_curves['BCELoss'].append(loss.item())
            self.example_loss_curves['entropy_regularization'].append(entropy_reg.item())
            self.example_loss_curves['mean_regularization'].append(mean_reg.item())

            pbar.set_description(
                f"ProxyExplainer @ epc {epc} | BCEloss={loss.item():.5f} | entropy_reg={entropy_reg:.5f} | mean_reg={mean_reg:.5f}"
            )

        uniform_debug_log(soft_edge_masks)
        return soft_edge_masks, loss.item()


def grid_search(model, graphs, search_dict: dict) -> list[tuple]:
    from itertools import product
    keys, values = zip(*search_dict.items())
    configs = []
    for combo in product(*values):
        params = dict(zip(keys, combo))

        use_proxy = params.pop("use_proxy")
        proxy_lam = params.pop("proxy_lam", None)
        proxy_latent = params.pop("proxy_latent", None)
        proxy_M = params.pop("proxy_M", None)
        proxy_M = params.pop("proxy_lr", None)

        if use_proxy:
            params["proxy_generator"] = ProxyGraphGenerator(
                node_feature_dim=graphs[0].x.shape[1],
                latent_dim=proxy_latent,
                lam=proxy_lam,
                proxy_M=proxy_M
            ).to(DEVICE)
            explainer = ProxyExplainer(model=model, graphs=graphs, **params)
        else:
            explainer = PGExplainer(model=model, graphs=graphs, **params)

        configs.append((params, explainer))

    return configs