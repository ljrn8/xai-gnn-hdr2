from .utils import elementwise_entropy, uniform_debug_log
from ..interfaces import GNN, GraphLevelExplainer, CustomExplainerModule
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm
from collections.abc import Iterable
from loguru import logger
from torch_geometric.data import Batch
from abc import ABC, abstractmethod
from .proxy_generation import ProxyGraphGenerator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class _CachedCausalLayer(nn.Module):
    """One causal self-attention block with KV-caching: computes q/k/v for the
    new token only, appends k/v to the running cache, attends the new query
    over the full cache (all positions <= current step)."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model)
        )

    def step(self, x_new: Tensor, cache: dict):
        # x_new: (1, 1, d_model) -- this step's single new token, pre-norm input
        residual = x_new
        h = self.norm1(x_new)

        q = self.q_proj(h).view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(h).view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(h).view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)

        if cache["k"] is None:
            k, v = k_new, v_new
        else:
            k = torch.cat([cache["k"], k_new], dim=2)
            v = torch.cat([cache["v"], v_new], dim=2)
        cache["k"], cache["v"] = k, v

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (1, heads, 1, t)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = (attn_weights @ v).transpose(1, 2).reshape(1, 1, -1)
        attn_out = self.out_proj(attn_out)

        x = residual + attn_out
        x = x + self.ff(self.norm2(x))
        return x, cache


class AutoregressiveMaskExplanationModule(CustomExplainerModule):
    """Autoregressively generates a soft edge mask with causal, KV-cached
    self-attention. Edge order is a random permutation fixed by `seed`
    (offset per-graph), for downstream seed-ensembling to cancel order bias.

    Step i's input is [edge_i_embedding, m_{i-1}], where m_{i-1} is the REAL
    sigmoid mask value produced at step i-1 (m_0 = 0 baseline) -- carried
    forward explicitly as a scalar, separate from the KV-cache, which only
    ever stores k/v (valid to cache since causal masking guarantees they never
    change once computed).
    """

    def __init__(self, 
                 hidden_size, 
                 embedding_size,
                 output_size,
                 seed: int = 0, 
                 n_heads: int = 4, 
                 n_layers: int = 2):
        
        super().__init__(hidden_size, embedding_size, output_size)
        
        self.seed = seed
        self.n_layers = n_layers
        self.input_proj = nn.Linear(embedding_size * 2 + 1, hidden_size)
        self.layers = nn.ModuleList(
            [_CachedCausalLayer(hidden_size, n_heads) for _ in range(n_layers)]
        )
        self.output_head = nn.Linear(hidden_size, output_size)

    def forward(self, model, graphs):
        batch_obj = Batch.from_data_list(graphs).to(DEVICE)
        logits, embeddings_list = get_model_embeddings_batched(
            model, batch_obj
        )
        final_embeddings = [e[-1] for e in embeddings_list]
        all_logits = []

        for graph_idx, (G, fe) in enumerate(zip(graphs, final_embeddings)):
            src, dst = G.edge_index
            edge_embeddings = torch.cat([fe[src], fe[dst]], dim=-1)
            n_edges = edge_embeddings.shape[0]
            device = edge_embeddings.device

            gen = torch.Generator(device="cpu").manual_seed(self.seed + graph_idx)
            perm = torch.randperm(n_edges, generator=gen).to(device)
            inv_perm = torch.argsort(perm)
            shuffled_embeddings = edge_embeddings[perm]

            caches = [{"k": None, "v": None} for _ in range(self.n_layers)]
            logits_list = []
            m_prev = torch.zeros(1, device=device)  # m_0, baseline

            for i in range(n_edges):
                x_in = torch.cat([shuffled_embeddings[i], m_prev], dim=-1)
                x = self.input_proj(x_in).view(1, 1, -1)

                for layer_idx, layer in enumerate(self.layers):
                    x, caches[layer_idx] = layer.step(x, caches[layer_idx])

                logit_i = self.output_head(x.view(-1))       # o_i
                logits_list.append(logit_i)
                m_prev = torch.sigmoid(logit_i).mean().view(1)  # m_i, fed to step i+1

            logits = torch.stack(logits_list, dim=0)[inv_perm]  # back to edge_index order
            all_logits.append(logits)

        return all_logits


class PGEExplanationModule(CustomExplainerModule):
    """Default Explanation module for PGExplainer fitting an MLP over a model's final layer embeddings (concatenated for edge embeddings)."""

    def __init__(self, hidden_size, embedding_size, output_size):
        super().__init__(hidden_size, embedding_size, output_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, model, graphs):
        _, embeddings_list = get_model_embeddings_batched(
            model, Batch.from_data_list(graphs).to(DEVICE)
        )
        final_embeddings = [e[-1] for e in embeddings_list]

        # build all edge embeddings in one shot, run MLP once, split back by graph
        all_edge_embeddings = []
        edge_counts = []
        for G, fe in zip(graphs, final_embeddings):
            src, dst = G.edge_index
            all_edge_embeddings.append(torch.cat([fe[src], fe[dst]], dim=-1))
            edge_counts.append(G.edge_index.shape[1])

        all_masks = self.mlp(torch.cat(all_edge_embeddings, dim=0))
        return list(all_masks.split(edge_counts))


class ComprehensiveMLPExplanationModule(CustomExplainerModule):
    """Extension of the default PGExplainer module that fits all intermediate model embeddings."""

    def __init__(self, hidden_size, embedding_size, output_size):
        super().__init__(hidden_size, embedding_size, output_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, model, graphs):
        _, embeddings_list = get_model_embeddings_batched(
            model,  Batch.from_data_list(graphs).to(DEVICE)
        )

        all_edge_embeddings = []
        edge_counts = []
        for G, layer_embeddings_list in zip(graphs, embeddings_list):
            node_emb = torch.cat(layer_embeddings_list, dim=1)
            src, dst = G.edge_index
            all_edge_embeddings.append(torch.cat([node_emb[src], node_emb[dst]], dim=-1))
            edge_counts.append(G.edge_index.shape[1])

        all_masks = self.mlp(torch.cat(all_edge_embeddings, dim=0))
        return list(all_masks.split(edge_counts))


class ContextualGRUExplanationModule(CustomExplainerModule):
    """Fits a GRU over the sef of final edge (concatenated) embeddings to generate logits"""

    def __init__(self, hidden_size, embedding_size, output_size):
        super().__init__(hidden_size, embedding_size, output_size)
        self.gru = nn.GRU(
            input_size=embedding_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, model, graphs):
        _, embeddings_list = get_model_embeddings_batched(
            model,  Batch.from_data_list(graphs).to(DEVICE)
        )
        all_edge_embeddings = []
        edge_counts = []
        for G, embeddings_layer_list in zip(graphs, embeddings_list):
            src, dst = G.edge_index
            final_embeddings = embeddings_layer_list[-1]
            edge_embeds = torch.cat([final_embeddings[src], final_embeddings[dst]], dim=-1)
            all_edge_embeddings.append(edge_embeds.unsqueeze(0))  # Add batch dimension
            edge_counts.append(G.edge_index.shape[1])

        # Stack all edge embeddings into a single tensor for GRU processing
        all_edge_embeddings_tensor = torch.cat(all_edge_embeddings, dim=0)  # Shape: (num_graphs, num_edges, embedding_dim)
        gru_output, _ = self.gru(all_edge_embeddings_tensor)  # Shape: (num_graphs, num_edges, hidden_size)
        all_masks = self.fc(gru_output)  # Shape: (num_graphs, num_edges, output_size)

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
        self.sampler_method = sampler_method

        self.sampler = {
            'GS': self._GS_sample,
            'IGR': self._IGR_sample
        }[sampler_method]

        self.explanation_module_class = {
            'default': PGEExplanationModule,
            'comprehensive': ComprehensiveMLPExplanationModule,
            'contextual': ContextualGRUExplanationModule
        }[explanation_module]

    def _GS_sample(self, logits: Tensor, samples: int) -> Tensor:
        """Samples from the Binary Concrete distribution (Maddison et al., 2017) for a given logit tensor. (gumbel softmax for binary simplex)

        Args:
            logits (Tensor): Logits for the Bernoulli distribution.
            tau (float): Temperature parameter for the Binary Concrete distribution.
            samples (int): Number of samples to draw.

        Returns:
            Tensor: Samples from the Binary Concrete distribution with shape (samples, *logits.shape).
        """
        u = torch.rand(samples, *logits.shape, device=logits.device).clamp(1e-6, 1 - 1e-6)
        log_noise = torch.log(u) - torch.log(1 - u)
        masks = torch.sigmoid((logits.unsqueeze(0) + log_noise) / self.tau)
        return masks

    def _IGR_sample(self, logits: Tensor, samples: int) -> Tensor:
        # required to output std and variance
        assert logits.shape[1] == 2, "logits must have shape (n_edges, 2) for inverse gaussian sampling"
        mu = logits[:, 0]  
        # softplus to ensure std is positive
        std = torch.nn.functional.softplus(logits[:, 1]) + 1e-6 
        # acquire epislon from a normal distribution 
        guassian_noise = torch.randn(samples, logits.shape[0], device=logits.device)
        masks = torch.sigmoid(mu + std * guassian_noise)
        return masks

    def _tile_graphs(self, graphs: Iterable[Data]):
        logger.info('Tiling graphs for Monte Carlo sampling')
        tiled_graphs = [G for G in graphs for _ in range(self.reparameterization_samples)]
        tiled = Batch.from_data_list(tiled_graphs)
        tiled.to(DEVICE)
        logger.info('done')
        return tiled

    def entropy_regularizer(self, soft_edge_masks):
        return self.entropy_regularization * torch.stack(
            [elementwise_entropy(m).mean() for m in soft_edge_masks]
        ).mean()
        
    def mean_regularizer(self, soft_edge_masks):
        return self.mean_regularization* torch.stack(
            [m.mean() for m in soft_edge_masks]
        ).mean()
        
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
            hard_mask_generator=_GS_sample
    ):
        """Parrallel Monte Carlo Binary Concrete Distribution estimation for 
        subgraph predictions using a logit (attribution) edge mask"""
        
        # for each graph, sample noise and compute hard masks
        edge_weights = []
        for logits in all_edge_mask_logits:
            hard = hard_mask_generator(logits, self.reparameterization_samples)  
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
        hard_mask_generator=_GS_sample
    ):
        """Same as _batched_estimation, but edge_weight is built to match the proxy
        graph's edge_index instead of the original graph's: edges kept from G_exp reuse
        their Binary Concrete sample (selected via is_exp_masks), and the remaining
        proxy-only (delta) edges get weight 1.0, since they were already a hard
        present/absent decision made when the proxy graph was built.
        """
        edge_weights = []
        for logits, is_exp, n_proxy_edges in zip(all_edge_mask_logits, is_exp_masks, proxy_edge_counts):
            hard = hard_mask_generator(logits, self.reparameterization_samples)  
            exp_weight = hard[:, is_exp]  # (samples, n_exp_edges)
            n_new_edges = n_proxy_edges - exp_weight.shape[1]
            new_weight = torch.ones(self.reparameterization_samples, n_new_edges, device=logits.device)

            edge_weights.append(torch.cat([exp_weight, new_weight], dim=1).reshape(-1))

        edge_weight = torch.cat(edge_weights)  # matches batched proxy edge_index ordering
        scores = torch.sigmoid(model(tiled_graphs_batch.x,
                                    tiled_graphs_batch.edge_index, edge_weight=edge_weight, batch=tiled_graphs_batch.batch))

        return scores.view(n_graphs, self.reparameterization_samples).mean(dim=1)  # (N,)

    def explain_graph_task(self, model, graphs):

        logger.info('testing example prediction on graphs[0] ..')
        _, embs_list = model(graphs[0].x, graphs[0].edge_index, return_all_embeddings=True)
        layerwise_embedddings_dimensions = [e.shape[1] for e in embs_list]
        embedding_size = layerwise_embedddings_dimensions[-1]

        self.explanation_module = self.explanation_module_class(
            hidden_size=self.hidden_size, 
            output_size=2 if self.sampler_method == 'IGR' else 1,
            embedding_size=embedding_size
        ).to(DEVICE)

        if self.use_proxy_graphs:
            logger.info('Proxy graph generator has been enabled')
            self.proxy_generator = ProxyGraphGenerator(node_feature_dim=graphs[0].x.shape[1])
            self.proxy_optimizer = torch.optim.Adam(self.proxy_generator.parameters(), lr=self.proxy_lr)
        else:
            # tile each graph `samples` times
            # proxy graphs have variable sizes and are handled differently
            tiled = self._tile_graphs(graphs)

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
            logit_edge_masks = self.explanation_module.forward(model, graphs)

            for m, g in zip(logit_edge_masks, graphs):
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
                    n_graphs=len(graphs),
                    hard_mask_generator=self.sampler
                )

            else:
                # Subgraph Estimate
                explanatory_y_preds = self._batched_estimation(
                    model=model,
                    tiled_graphs_batch=tiled,
                    all_edge_mask_logits=logit_edge_masks,
                    n_graphs=len(graphs),
                    hard_mask_generator=self.sampler
                )

            # regularization
            entropy_reg = self.entropy_regularizer(soft_edge_masks)
            mean_reg = self.entropy_regularizer(soft_edge_masks)
            
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
