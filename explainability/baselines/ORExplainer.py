from abc import ABC, abstractmethod
from typing import Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from torch_geometric.utils import degree
from tqdm import tqdm
from explainability.xAI_utils import Explainer

EPS = 1e-6

# ---------------------------------------------------------------------------
# Model adapter
#
# ORExplainer expects:
#   model.embed(x, edge_index, edge_weight=None) -> list of per-layer tensors
#   model.lin(last_embed)                        -> logits
#   model(data)                                  -> logits  (Data object)
#
# Your WeightedNodeGCN / WeightedNodeGIN have:
#   model.forward(x, edge_index, edge_weight=None, return_embeds=False) -> logits or embed
#   model.convs[-1]                              -> final conv (acts as linear head)
#
# The adapter below bridges that gap without touching the original model classes.
# ---------------------------------------------------------------------------


class _ModelAdapter(nn.Module):
    """
    Wraps WeightedNodeGCN / WeightedNodeGIN so ORExplainer can call
    .embed() and .lin() in addition to the normal forward pass.

    embed() returns a list of hidden-state tensors, one per intermediate
    conv layer (i.e. all layers except the last).  ORExplainer concatenates
    them, so the length drives the `num_hops` count and the `hidden` dimension
    used to size the explainer MLP input.
    """

    def __init__(self, model: nn.Module, hidden_channels: int):
        super().__init__()
        self._model = model
        self._hidden = hidden_channels

    # ---- delegate MessagePassing detection so num_hops still works ----
    def modules(self):
        return self._model.modules()

    def parameters(self):
        return self._model.parameters()

    # ---- interfaces ORExplainer calls ---------------------------------

    def forward(self, data: Data) -> torch.Tensor:
        """Called as model(data) inside ORExplainer."""
        ew = getattr(data, "edge_weight", None)
        return self._model(data.x, data.edge_index, edge_weight=ew)

    def embed(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        convs = self._model.convs
        hiddens = []
        h = x
        for conv in convs[:-1]:
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = h.relu()
            hiddens.append(h)
        return hiddens  # list of [N, hidden_channels]

    def lin(self, last_embed: torch.Tensor) -> torch.Tensor:
        n = last_embed.size(0)
        device = last_embed.device
        # self-loops: every node i -> i
        self_loops = torch.arange(n, device=device)
        ei = torch.stack([self_loops, self_loops], dim=0)
        return self._model.convs[-1](last_embed, ei)


# ---------------------------------------------------------------------------
# Core explainer logic  (ORExplainer, self-contained)
# ---------------------------------------------------------------------------


class _ORExplainerCore(nn.Module):
    """
    Stripped-down ORExplainer that works with _ModelAdapter.
    Key changes from the original:
      - in_channels computed dynamically from the adapter (no hard-coded x_dim dict)
      - num_hops inferred from model.modules() as before
      - no checkpoint save/load (kept simple; add back if needed)
    """

    def __init__(
        self,
        model: _ModelAdapter,
        x_dim: int,
        hidden: int,
        num_hops: int,
        epochs: int = 20,
        lr: float = 3e-3,
        coff_size: float = 1.0,
        coff_ent: float = 5e-4,
        t0: float = 1.0,
        t1: float = 1.0,
        gamma: float = 0.1,
        lamda: float = 0.5,
        temp: float = 1.0,
        sample_bias: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        self.gamma = gamma
        self.lamda = lamda
        self.temp = temp
        self.sample_bias = sample_bias
        self.num_hops = num_hops
        self.device = next(model._model.parameters()).device

        # explainer MLP input: [src_embed | dst_embed | target_embed]
        # each embed = x_dim + hidden * num_hops  (x || layer1 || layer2 || ...)
        embed_dim = x_dim + hidden * num_hops
        in_channels = embed_dim * 3

        self.elayers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU()),
                nn.Linear(64, 1),
            ]
        )
        self.elayers.to(self.device)

    # ------------------------------------------------------------------ #
    #  Graph utilities                                                     #
    # ------------------------------------------------------------------ #

    def _k_hop_subgraph(self, node_idx, edge_index, num_nodes):
        """Returns (subset, sub_edge_index, edge_mask) with relabelled nodes."""
        row, col = edge_index  # source_to_target convention used in original

        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        frontier = torch.tensor([node_idx], device=edge_index.device)
        subsets = [frontier]

        for _ in range(self.num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            edge_mask_hop = node_mask[row]
            subsets.append(col[edge_mask_hop])

        subset = torch.cat(subsets).unique()
        node_mask.fill_(False)
        node_mask[subset] = True
        edge_mask = node_mask[row] & node_mask[col]
        sub_edge_index = edge_index[:, edge_mask]

        # relabel
        relabel = torch.full(
            (num_nodes,), -1, dtype=torch.long, device=edge_index.device
        )
        relabel[subset] = torch.arange(subset.size(0), device=edge_index.device)
        sub_edge_index = relabel[sub_edge_index]

        return subset, sub_edge_index, edge_mask

    # ------------------------------------------------------------------ #
    #  Energy (OOD score propagation)                                     #
    # ------------------------------------------------------------------ #

    def _compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        return self.temp * -torch.logsumexp(logits / self.temp, dim=-1).detach()

    def _propagate_energy(self, energy, edge_index, edge_mask, num_nodes):
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=energy.dtype)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float("inf")] = 0
        norm = edge_mask * deg_inv[col]
        adj = SparseTensor(
            row=col, col=row, value=norm, sparse_sizes=(num_nodes, num_nodes)
        )
        e = energy.view(-1, 1)
        for _ in range(self.num_hops):
            e = self.lamda * e + (1 - self.lamda) * (adj @ e)
        return e.squeeze(1)

    # ------------------------------------------------------------------ #
    #  Loss                                                               #
    # ------------------------------------------------------------------ #

    def _loss(self, prob, ori_pred, edge_mask):
        ce = F.cross_entropy(prob.squeeze(), ori_pred.squeeze().to(self.device))
        size_loss = self.coff_size * edge_mask.mean()
        em = edge_mask * 0.99 + 5e-7
        ent = -(em * torch.log(em) + (1 - em) * torch.log(1 - em))
        ent_loss = self.coff_ent * ent.mean()
        return ce + size_loss + ent_loss, ce

    # ------------------------------------------------------------------ #
    #  Concrete relaxation                                                #
    # ------------------------------------------------------------------ #

    def _concrete_sample(self, w, beta=1.0, training=True):
        if training:
            bias = self.sample_bias + 1e-4
            eps = torch.rand_like(w) * (1 - 2 * bias) + bias
            gate = (torch.log(eps) - torch.log(1 - eps) + w) / beta
            return torch.sigmoid(gate)
        return torch.sigmoid(w)

    # ------------------------------------------------------------------ #
    #  Forward (edge mask prediction)                                     #
    # ------------------------------------------------------------------ #

    def forward(self, x, embed, edge_index, node_id, tmp, training=True):
        embed = embed.to(self.device)
        edge_index = edge_index.to(self.device)
        col, row = edge_index
        h = torch.cat(
            [embed[col], embed[row], embed[node_id].expand(col.size(0), -1)], dim=-1
        )
        for layer in self.elayers:
            h = layer(h)
        return self._concrete_sample(h.reshape(-1), beta=tmp, training=training)

    # ------------------------------------------------------------------ #
    #  Model helpers                                                      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _get_embeddings_and_logits(self, x, edge_index, edge_weight=None):
        self.model._model.eval()
        hiddens = self.model.embed(x, edge_index, edge_weight=edge_weight)
        last = hiddens[-1]
        logits = self.model.lin(last)
        return hiddens, logits

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def train_on_graph(self, x, edge_index, y, node_indices):
        """
        Train the explainer MLP on a set of node indices within a single graph.
        x, edge_index, y should already be on self.device.
        """
        num_nodes = x.size(0)

        # Pre-cache subgraphs and embeddings
        cache = {}
        with torch.no_grad():
            self.model._model.eval()
            full_logits = self.model(Data(x=x, edge_index=edge_index).to(self.device))
            energy = self._compute_energy(full_logits)

            for nid in tqdm(node_indices, desc="Caching subgraphs"):
                subset, sub_ei, _ = self._k_hop_subgraph(nid, edge_index, num_nodes)
                if sub_ei.size(1) == 0:
                    continue
                sub_x = x[subset]
                hiddens, logits = self._get_embeddings_and_logits(sub_x, sub_ei)
                local_idx = int((subset == nid).nonzero(as_tuple=True)[0])
                pred = logits.argmax(dim=-1)[local_idx]
                full_embed = torch.cat([sub_x] + hiddens, dim=-1)
                cache[nid] = dict(
                    subset=subset,
                    sub_ei=sub_ei,
                    sub_x=sub_x,
                    full_embed=full_embed,
                    local_idx=local_idx,
                    pred=pred,
                )

        optimizer = Adam(self.elayers.parameters(), lr=self.lr, weight_decay=5e-4)

        for epoch in range(self.epochs):
            self.elayers.train()
            optimizer.zero_grad()
            tmp = float(
                self.t0 * np.power(self.t1 / self.t0, epoch / max(self.epochs - 1, 1))
            )
            total_loss = 0.0

            for nid, c in cache.items():
                sub_ei = c["sub_ei"].to(self.device)
                sub_x = c["sub_x"].to(self.device)
                full_embed = c["full_embed"].to(self.device)

                edge_mask = self.forward(
                    sub_x, full_embed, sub_ei, c["local_idx"], tmp, training=True
                )
                sub_ei_und, edge_mask_und = to_undirected(
                    sub_ei, edge_attr=edge_mask, num_nodes=sub_x.size(0), reduce="mean"
                )

                pred_data = Data(
                    x=sub_x, edge_index=sub_ei_und, edge_weight=edge_mask_und
                ).to(self.device)
                logits = self.model(pred_data)

                loss, _ = self._loss(logits[c["local_idx"]], c["pred"], edge_mask_und)

                if self.gamma > 0:
                    sub_energy = energy[c["subset"]].to(self.device)
                    prop = self._propagate_energy(
                        sub_energy, sub_ei_und, edge_mask_und, sub_x.size(0)
                    )
                    loss = loss + self.gamma * prop[c["local_idx"]]

                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            print(f"Epoch {epoch:3d} | Loss {total_loss / max(len(cache), 1):.4f}")

    @torch.no_grad()
    def explain_node(self, x, edge_index, node_idx):
        """Returns edge_mask over edge_index for the given node."""
        num_nodes = x.size(0)
        subset, sub_ei, orig_edge_mask = self._k_hop_subgraph(
            node_idx, edge_index, num_nodes
        )
        if sub_ei.size(1) == 0:
            return torch.zeros(edge_index.size(1))

        sub_x = x[subset]
        local_idx = int((subset == node_idx).nonzero(as_tuple=True)[0])
        hiddens, _ = self._get_embeddings_and_logits(sub_x, sub_ei)
        full_embed = torch.cat([sub_x] + hiddens, dim=-1)

        self.elayers.eval()
        edge_mask = self.forward(
            sub_x, full_embed, sub_ei, local_idx, tmp=1.0, training=False
        )

        # Map back to full edge_index space (unselected edges get score 0)
        full_mask = torch.zeros(edge_index.size(1), device=self.device)
        full_mask[orig_edge_mask] = edge_mask
        return full_mask


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ORExplainer(Explainer):
    """
    Implements the OR (Out-of-distribution Robust) GNN explainer for both
    transductive node classification and inductive graph classification tasks
    using your WeightedNodeGCN / WeightedNodeGIN model family.

    Parameters
    ----------
    hidden_channels : int
        Hidden dimension used when training the GNN (must match the checkpoint).
    epochs : int
        Explainer MLP training epochs.
    lr : float
        Explainer MLP learning rate.
    node_sample_size : int
        For node tasks, how many training-mask nodes to train the explainer on.
        -1 means use all.
    gamma : float
        Weight of the energy-propagation loss term. 0 disables it.
    coff_size / coff_ent : float
        Coefficients for the edge-size and entropy regularisation losses.
    lamda : float
        Retention factor in the energy propagation (0 = full neighbour update,
        1 = no propagation).
    temp / t0 / t1 : float
        Temperature for energy scoring and concrete-relaxation annealing.
    """

    def __init__(
        self,
        hidden_channels: int = 16,
        epochs: int = 20,
        lr: float = 3e-3,
        node_sample_size: int = 300,
        gamma: float = 0.1,
        coff_size: float = 1.0,
        coff_ent: float = 5e-4,
        lamda: float = 0.5,
        temp: float = 1.0,
        t0: float = 1.0,
        t1: float = 1.0,
        sample_bias: float = 0.0,
        seed: int = 42,
    ):
        self.hidden_channels = hidden_channels
        self.epochs = epochs
        self.lr = lr
        self.node_sample_size = node_sample_size
        self.gamma = gamma
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.lamda = lamda
        self.temp = temp
        self.t0 = t0
        self.t1 = t1
        self.sample_bias = sample_bias
        self.seed = seed

    # ------------------------------------------------------------------ #
    #  Internal: build a fresh core for a given model + x_dim            #
    # ------------------------------------------------------------------ #

    def _build_core(self, raw_model: nn.Module, x_dim: int) -> _ORExplainerCore:
        adapter = _ModelAdapter(raw_model, self.hidden_channels)
        # count MessagePassing layers = num_hops
        num_hops = sum(1 for m in raw_model.modules() if isinstance(m, MessagePassing))
        # exclude the final layer (it's used as lin(), not for embedding)
        num_hops = max(num_hops - 1, 1)

        return _ORExplainerCore(
            model=adapter,
            x_dim=x_dim,
            hidden=self.hidden_channels,
            num_hops=num_hops,
            epochs=self.epochs,
            lr=self.lr,
            coff_size=self.coff_size,
            coff_ent=self.coff_ent,
            t0=self.t0,
            t1=self.t1,
            gamma=self.gamma,
            lamda=self.lamda,
            temp=self.temp,
            sample_bias=self.sample_bias,
            seed=self.seed,
        )

    # ------------------------------------------------------------------ #
    #  Node task  (transductive, single graph)                            #
    # ------------------------------------------------------------------ #

    def explain_node_task(self, task, graph):
        """
        Parameters
        ----------
        task : TransductiveNodeClassification
            Your trained task object. task.model is the GNN.
        graph : torch_geometric.data.Data
            The full graph with .x, .edge_index, .y, .train_mask, .test_mask.

        Returns
        -------
        dict mapping node_idx (int) -> edge_mask (FloatTensor, shape [num_edges])
            Scores in [0, 1]; higher = more important for the prediction at
            that node.  Only test-mask nodes are explained.
        """
        model = task.model
        device = next(model.parameters()).device
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        y = graph.y.to(device)
        x_dim = x.size(1)

        core = self._build_core(model, x_dim)

        # Use train-mask nodes to train the explainer MLP
        train_indices = graph.train_mask.nonzero(as_tuple=True)[0].tolist()
        if self.node_sample_size > 0:
            train_indices = train_indices[: self.node_sample_size]

        core.train_on_graph(x, edge_index, y, train_indices)

        # Explain test-mask nodes
        core.elayers.eval()
        test_indices = graph.test_mask.nonzero(as_tuple=True)[0].tolist()
        explanations = {}
        for nid in tqdm(test_indices, desc="Explaining nodes"):
            explanations[nid] = core.explain_node(x, edge_index, nid).cpu()

        return explanations

    # ------------------------------------------------------------------ #
    #  Graph task  (inductive, one explanation per graph)                 #
    # ------------------------------------------------------------------ #

    def explain_graph_task(self, task, graphs):
        """
        Parameters
        ----------
        task : InductiveGraphClassification
            Your trained task object. task.model is the GNN.
        graphs : list of torch_geometric.data.Data
            The graphs to explain (typically task.test_graphs).
            Each must have .x, .edge_index, .y.

        Returns
        -------
        list of FloatTensor, one per graph in `graphs`.
            Each tensor has shape [num_edges_in_graph] with scores in [0, 1].

        Notes
        -----
        For graph classification, ORExplainer's node-level explanation
        mechanism is applied to a synthetic "centre node" (node 0 of each
        graph, after adding a virtual global node connected to all others).
        The returned mask covers the original edges only — the virtual edges
        are stripped before returning.

        The explainer MLP is trained on task.train_graphs and then applied to
        `graphs`.
        """
        model = task.model
        device = next(model.parameters()).device

        # Infer x_dim from first graph
        x_dim = graphs[0].x.size(1)
        if graphs[0].x.dim() == 1:
            x_dim = 1

        core = self._build_core(model, x_dim)

        # ---- Train: treat each training graph as independent ----
        # We train the explainer on each graph's node 0 as the "target" node.
        # For graph-level tasks the model sees the whole graph, so we pass
        # the full graph each time.
        print("Training explainer on training graphs...")
        train_graphs = task.train_graphs

        # Build a merged view: concatenate all training graphs with block-
        # diagonal edge_index so the explainer sees diverse examples.
        # Simpler and avoids leaking test structure.
        all_x, all_ei, all_y, node_indices = [], [], [], []
        offset = 0
        for G in tqdm(train_graphs, desc="Merging train graphs"):
            gx = G.x.to(device)
            if gx.dim() == 1:
                gx = gx.unsqueeze(1)
            gy = G.y.to(device).expand(gx.size(0))
            gei = G.edge_index.to(device) + offset
            all_x.append(gx)
            all_ei.append(gei)
            all_y.append(gy)
            node_indices.append(offset)  # explain node 0 of each graph
            offset += gx.size(0)

        merged_x = torch.cat(all_x, dim=0)
        merged_ei = torch.cat(all_ei, dim=1)
        merged_y = torch.cat(all_y, dim=0)

        if self.node_sample_size > 0:
            node_indices = node_indices[: self.node_sample_size]

        core.train_on_graph(merged_x, merged_ei, merged_y, node_indices)

        # ---- Explain each target graph ----
        core.elayers.eval()
        explanations = []
        offset = 0
        print("Explaining test graphs...")
        for G in tqdm(graphs, desc="Explaining graphs"):
            gx = G.x.to(device)
            if gx.dim() == 1:
                gx = gx.unsqueeze(1)
            gei = G.edge_index.to(device)
            n = gx.size(0)

            # Add a virtual global node connected to all real nodes so the
            # explainer's k-hop neighbourhood covers the whole graph.
            vnode_idx = n
            v_src = torch.arange(n, device=device)
            v_dst = torch.full((n,), vnode_idx, device=device)
            virtual_ei = torch.stack(
                [
                    torch.cat([v_src, v_dst]),
                    torch.cat([v_dst, v_src]),
                ],
                dim=0,
            )
            aug_x = torch.cat([gx, gx.mean(0, keepdim=True)], dim=0)
            aug_ei = torch.cat([gei, virtual_ei], dim=1)

            # explain from the virtual node's perspective
            mask = core.explain_node(aug_x, aug_ei, vnode_idx)

            # strip virtual edges (last 2*n entries) and return original-graph scores
            orig_mask = mask[: gei.size(1)]
            explanations.append(orig_mask.cpu())

        return explanations
