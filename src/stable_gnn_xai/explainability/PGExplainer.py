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
import random
from collections import defaultdict
from pprint import pprint
import gc

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
    """One causal self-attention block with a preallocated KV-cache, batched
    across graphs. Writes into a fixed (n_graphs, n_heads, max_edges, head_dim)
    buffer via index-assignment instead of growing it with torch.cat every step."""

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

    def init_cache(self, batch_size, max_len, device, dtype):
        return {
            "k": torch.zeros(batch_size, self.n_heads, max_len, self.head_dim, device=device, dtype=dtype),
            "v": torch.zeros(batch_size, self.n_heads, max_len, self.head_dim, device=device, dtype=dtype),
        }

    def step(self, x_new: Tensor, cache: dict, t: int):
        B = x_new.shape[0]
        residual = x_new
        h = self.norm1(x_new)

        q = self.q_proj(h).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(h).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(h).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)

        # buffer write stays O(1) in-place — fine, since nothing tracked by
        # autograd ever holds a live view into this storage (see clone below)
        cache["k"][:, :, t:t + 1, :] = k_new
        cache["v"][:, :, t:t + 1, :] = v_new

        # clone out of the buffer *now*, so this step's matmul operands are
        # immune to the in-place writes future steps make into `cache`.
        # gradients still flow (clone is differentiable), unlike detach.
        k = cache["k"][:, :, : t + 1, :].clone()
        v = cache["v"][:, :, : t + 1, :].clone()

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, 1, -1)
        attn_out = self.out_proj(attn_out)

        x = residual + attn_out
        x = x + self.ff(self.norm2(x))
        return x, cache


class AutoregressiveMaskExplanationModule(CustomExplainerModule):
    """Autoregressively generates a soft edge mask with causal, KV-cached
    self-attention, batched across graphs (not just within a graph). Edge
    order is a random permutation fixed by `seed` (offset per
    graph), for
    downstream seed-ensembling to cancel order bias.

    Step i's input is [edge_i_embedding, m_{i-1}] per graph, where m_{i-1} is
    the REAL sigmoid mask value produced at step i-1 (m_0 = 0 baseline).
    Graphs with fewer edges than max_edges are zero-padded and simply have
    their trailing steps discarded before returning -- they still burn compute
    on padded steps, but the whole batch shares one Python/kernel-launch loop
    of length max_edges instead of n_graphs independent loops of length n_edges_i.

    If max_edges for a graph_batch exceeds `max_edges_per_segment`, the edge
    dimension is further split into segments, each with its own freshly
    initialized KV cache. This bounds peak cache memory to
    max_edges_per_segment regardless of how large max_edges gets, at the
    cost of causal attention only seeing within-segment history (no
    cross-segment context).
    """

    def __init__(self,
                 hidden_size,
                 embedding_size,
                 output_size,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 graph_batch_size: int = 1_000,
                 max_edges_per_segment: int = 200,
                 **kwargs
    ):
        super().__init__(hidden_size, embedding_size, output_size)
        self.n_layers = n_layers
        self.input_proj = nn.Linear(embedding_size * 2 + 1, hidden_size)
        self.layers = nn.ModuleList(
            [_CachedCausalLayer(hidden_size, n_heads) for _ in range(n_layers)]
        )
        self.output_head = nn.Linear(hidden_size, output_size)
        self.graph_batch_size = graph_batch_size
        self.max_edges_per_segment = max_edges_per_segment

    def _run_autoregressive(self, padded, device, dtype):
        """Runs the step loop over `padded` edges, segmenting along the edge
        dimension whenever max_edges exceeds self.max_edges_per_segment.
        Cache is reset at each segment boundary -> bounds peak memory to
        max_edges_per_segment, at the cost of losing causal context across
        segment boundaries.
        """
        n_graphs, max_edges, edge_dim = padded.shape
        all_step_logits = torch.zeros(n_graphs, max_edges, self.output_size, device=device, dtype=dtype)
        m_prev = torch.zeros(n_graphs, 1, device=device, dtype=dtype)

        seg_len = self.max_edges_per_segment
        for seg_start in range(0, max_edges, seg_len):
            seg_end = min(seg_start + seg_len, max_edges)
            this_seg_len = seg_end - seg_start

            # fresh cache per segment -> bounded memory
            caches = [layer.init_cache(n_graphs, this_seg_len, device, dtype) for layer in self.layers]

            for i in range(this_seg_len):
                global_i = seg_start + i
                x_in = torch.cat([padded[:, global_i, :], m_prev], dim=-1)
                x = self.input_proj(x_in).unsqueeze(1)

                for layer_idx, layer in enumerate(self.layers):
                    x, caches[layer_idx] = layer.step(x, caches[layer_idx], t=i)  # t local to segment

                logit_i = self.output_head(x.squeeze(1))
                all_step_logits[:, global_i, :] = logit_i
                m_prev = torch.sigmoid(logit_i).mean(dim=-1, keepdim=True)

            del caches
            torch.cuda.empty_cache()
            gc.collect()

        return all_step_logits

    def forward(self, model, graphs):
        final_logits = []
        for start in range(0, len(graphs), self.graph_batch_size):
            group_graphs = graphs[start : start + self.graph_batch_size]
            n_graphs = len(group_graphs)
            logger.debug(f'fitting autoregressive explanation module for graphs {start}:{start+n_graphs}')

            batch_obj = Batch.from_data_list(group_graphs).to(DEVICE)
            logits, embeddings_list = get_model_embeddings_batched(model, batch_obj)
            final_embeddings = [e[-1] for e in embeddings_list]

            device = final_embeddings[0].device
            dtype = final_embeddings[0].dtype

            edge_counts = []
            permutations = []
            inv_perms = []
            shuffled_node_embeddings = []
            for G, fe in zip(group_graphs, final_embeddings):
                src, dst = G.edge_index
                edge_embeddings = torch.cat([fe[src], fe[dst]], dim=-1)
                n_edges = edge_embeddings.shape[0]
                edge_counts.append(n_edges)

                perm = torch.randperm(n_edges, device=device)
                permutations.append(perm)
                inv_perms.append(torch.argsort(perm))
                shuffled_node_embeddings.append(edge_embeddings[perm])

            max_edges = max(edge_counts)
            edge_dim = shuffled_node_embeddings[0].shape[-1]

            padded = torch.zeros(n_graphs, max_edges, edge_dim, device=device, dtype=dtype)
            for i, s in enumerate(shuffled_node_embeddings):
                padded[i, : s.shape[0]] = s

            all_step_logits = self._run_autoregressive(padded, device, dtype)

            for i in range(n_graphs):
                n_edges = edge_counts[i]
                final_logits.append(all_step_logits[i, :n_edges][inv_perms[i]])

        return final_logits


class PGEExplanationModule(CustomExplainerModule):
    """Default Explanation module for PGExplainer fitting an MLP over a model's final layer embeddings (concatenated for edge embeddings)."""

    def __init__(self, hidden_size, embedding_size, output_size, **kwargs):
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

    def __init__(self, hidden_size, embedding_size, output_size, n_layers, **kwargs):
        super().__init__(hidden_size, embedding_size, output_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2 * n_layers, self.hidden_size),
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

    def __init__(self, hidden_size, embedding_size, output_size, **kwargs):
        super().__init__(hidden_size, embedding_size, output_size)
        self.gru = nn.GRU(
            input_size=embedding_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, model, graphs):
        _, embeddings_list = get_model_embeddings_batched(
            model, Batch.from_data_list(graphs).to(DEVICE)
        )

        from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

        per_graph_edge_embeds = []
        edge_counts = []
        for G, embeddings_layer_list in zip(graphs, embeddings_list):
            src, dst = G.edge_index
            final_embeddings = embeddings_layer_list[-1]
            edge_embeds = torch.cat([final_embeddings[src], final_embeddings[dst]], dim=-1)
            per_graph_edge_embeds.append(edge_embeds)          # (n_edges_i, dim) -- no unsqueeze
            edge_counts.append(edge_embeds.shape[0])

        # pad to (num_graphs, max_edges, dim)
        padded = pad_sequence(per_graph_edge_embeds, batch_first=True)  # zero-padded
        lengths = torch.tensor(edge_counts)

        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        gru_output, _ = pad_packed_sequence(packed_out, batch_first=True)  # (num_graphs, max_edges, hidden)

        all_masks = self.fc(gru_output)  # (num_graphs, max_edges, output_size)

        # strip padding per graph before returning, to match the flat/list-of-tensors convention
        # used elsewhere (each entry shape (n_edges_i, output_size))
        return [all_masks[i, :edge_counts[i]] for i in range(len(graphs))]


class PGExplainer(GraphLevelExplainer):
    """Ablation PGExplainer"""

    def __init__(
        self,
        hidden_size=64,
        epochs=50,
        lr=0.01,
        mean_regularization=0.05,
        entropy_regularization=0.1,
        tau=1.0,
        reparameterization_samples=30,
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
            'contextual': ContextualGRUExplanationModule,
            'auto-regressive': AutoregressiveMaskExplanationModule
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
        tile_list = [G for G in graphs for _ in range(self.reparameterization_samples)]
        tiled_batch = Batch.from_data_list(tile_list)
        tiled_batch.to(DEVICE)
        logger.info('done')
        return tiled_batch, tile_list


    def _entropy_regularizer(self, soft_edge_masks):
        return self.entropy_regularization * torch.stack(
            [elementwise_entropy(m).mean() for m in soft_edge_masks]
        ).mean()
        

    def _mean_regularizer(self, soft_edge_masks):
        return self.mean_regularization * torch.stack(
            [m.mean() for m in soft_edge_masks]
        ).mean()
        

    def _proxy_generator_inner_optimization(self, graphs, edge_masks):
        batch_data = Batch.from_data_list(graphs).to(DEVICE)
        for _ in range(self.proxy_M):
            self.proxy_optimizer.zero_grad()
            _, gen_loss = self.proxy_generator.forward_batched(batch_data, edge_masks)
            gen_loss.backward()
            self.proxy_optimizer.step()


    def _proxy_generator_build_proxy_graphs(self, graphs, edge_masks):
        batch_data = Batch.from_data_list(graphs).to(DEVICE)
        A_tildes, _ = self.proxy_generator.forward_batched(batch_data, edge_masks)
        proxy_graphs = [
            self.proxy_generator.build_proxy_data(G, A_tilde)
            for G, A_tilde in zip(graphs, A_tildes)
        ]
        return proxy_graphs


    def _batched_reparameterization_estimate(
            self,
            model, 
            logit_edge_masks,
            tile_list, 
            hard_mask_sampler=_GS_sample,
    ):
        """Parrallel Monte Carlo Binary Concrete Distribution estimation for 
        subgraph predictions using a logit (attribution) edge mask, optionally generating proxy graphs on each sample"""
        
        tiled_graphs_batch = Batch.from_data_list(tile_list)
        n_graphs = len(logit_edge_masks)

        # for each graph, sample noise and compute hard masks
        hard_edge_weights = []
        per_graph_hard = []
        for logits in logit_edge_masks:
            hard = hard_mask_sampler(logits, self.reparameterization_samples)  
            hard_edge_weights.append(hard.reshape(-1))  # (samples * n_edges,)
            per_graph_hard.append(hard)

        # unroll into a list aligned with tile_list's ordering:
        # [G0_s0, G0_s1, ..., G0_s{S-1}, G1_s0, ...] -- matches _tile_graphs
        tiled_masks = [
            per_graph_hard[i][s]                      # (n_edges_i,)
            for i in range(n_graphs)
            for s in range(self.reparameterization_samples)
        ]

        if self.use_proxy_graphs:
            
            # detach masks during optimzaition (gradient stops at the GAE's)
            self._proxy_generator_inner_optimization(
                tile_list, [m.detach() for m in tiled_masks]
            ) 

            # keep masks attached to gradient when building proxy graphs (graident needs to flow through)
            proxy_graphs = self._proxy_generator_build_proxy_graphs(tile_list, tiled_masks)
            proxy_batch = Batch.from_data_list(proxy_graphs).to(DEVICE)
            preds = model(
                proxy_batch.x, 
                proxy_batch.edge_index,
                batch=proxy_batch.batch
            )

        else:
            hard_edge_weights = torch.cat(hard_edge_weights)
            preds = model(
                tiled_graphs_batch.x, 
                tiled_graphs_batch.edge_index,
                edge_weight=hard_edge_weights, 
                batch=tiled_graphs_batch.batch
            )

        scores = torch.sigmoid(preds)
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
            if hard.dim() == 3:
                hard = hard.squeeze(-1) 

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
        n_layers = len(embs_list)
        logger.debug(f'embeddings size: {embedding_size}')

        self.explanation_module = self.explanation_module_class(
            hidden_size=self.hidden_size, 
            output_size=2 if self.sampler_method == 'IGR' else 1,
            embedding_size=embedding_size,
            n_layers=n_layers
        ).to(DEVICE)

        if self.use_proxy_graphs:
            logger.info('Proxy graph generator has been enabled')
            self.proxy_generator = ProxyGraphGenerator(node_feature_dim=graphs[0].x.shape[1])
            self.proxy_optimizer = torch.optim.Adam(self.proxy_generator.parameters(), lr=self.proxy_lr)
        
        # tile each graph `samples` times
        tile_batch, tile_list = self._tile_graphs(graphs)

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


            # perturbed graph estimation (core entry point)
            explanatory_y_preds = self._batched_reparameterization_estimate(
                model=model,
                tile_list=tile_list,
                logit_edge_masks=logit_edge_masks,
                hard_mask_sampler=self.sampler
            )

            if self.sampler_method == 'IGR':
                # mu only
                soft_edge_masks = [torch.sigmoid(l[:, 0]) for l in logit_edge_masks]
            else:
                soft_edge_masks = [torch.sigmoid(m) for m in logit_edge_masks]


            # regularization
            entropy_reg = self._entropy_regularizer(soft_edge_masks)
            mean_reg = self._mean_regularizer(soft_edge_masks)

            loss = self.loss_f(explanatory_y_preds, y_preds)

            self.example_loss_curves['BCELoss'].append(loss.item())
            self.example_loss_curves['entropy_regularization'].append(entropy_reg.item())
            self.example_loss_curves['mean_regularization'].append(mean_reg.item())

            pbar.set_description(
                f"PGExplainer @ epc {epc} | BCEloss={loss.item():.5f} | entropy_reg={entropy_reg:.5f} | mean_reg={mean_reg:.5f}"
            )

            loss += mean_reg + entropy_reg
            loss.backward()
            optimizer.step()


        if sum(soft_edge_masks[0]) < 0.1 or 1 - sum(soft_edge_masks[0]) < 0.1:
            logger.debug('found uniform edge masks with delta 0.1')
        else:
            uniform_debug_log(soft_edge_masks)
        
        return soft_edge_masks, loss.item()
