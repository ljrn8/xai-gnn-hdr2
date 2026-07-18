"""
Proxy graph generator for ProxyExplainer (Chen et al. 2026, Section VI-C).

GAE on G_exp (eq. 13) + VGAE on G - G_exp (eq. 14-15).
Combined proxy adjacency A_tilde = A_tilde_exp + A_tilde_delta (eq. 16).
Loss L_gen = L_in + lambda * L_KL (eq. 17).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from loguru import logger
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_geometric.data import Batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(in_channels=out_channels, out_channels=out_channels) 

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        return self.conv2(x, edge_index, edge_weight=edge_weight)


def _decode(z: Tensor) -> Tensor:
    """Inner product decoder: sigmoid(Z Z^T) -> (n, n) edge probabilities."""
    return torch.sigmoid(z @ z.t())


def _adj(edge_index: Tensor, n: int) -> Tensor:
    A = torch.zeros(n, n, device=edge_index.device)
    if edge_index.shape[1] > 0:
        A[edge_index[0], edge_index[1]] = 1.0
    return A


class ProxyGraphGenerator(nn.Module):
    """Dual GAE/VGAE proxy graph generator per Section VI-C.

    Args:
        node_feature_dim: input node feature dimension
        latent_dim:       encoder output / latent dimension
        lam:              lambda weighting L_KL in L_gen (eq. 17)
        beta:             upweight for present edges in L_in (eq. 11)
    """
    def __init__(self, node_feature_dim: int, latent_dim: int = 32,
                 lam: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.lam = lam
        self.beta = beta

        # GAE branch for G_exp (eq. 13)
        self.gae_enc = _GCNEncoder(node_feature_dim, latent_dim).to(DEVICE)

        # VGAE branch for G_delta = G - G_exp (eq. 14)
        self.vgae_enc_mu = _GCNEncoder(node_feature_dim, latent_dim).to(DEVICE)
        self.vgae_enc_logvar = _GCNEncoder(node_feature_dim, latent_dim).to(DEVICE)


    def forward_batched(self, batch_data: Batch, edge_masks: list[Tensor]):
        """Encodes ALL graphs in one parallel GCN pass (this is the expensive part,
        now done once instead of per-graph), then decodes + computes loss per graph
        (cheap matmuls, not model forward calls)."""

        batch_data = batch_data.to(DEVICE)
        x = batch_data.x.float()
        edge_index = batch_data.edge_index.to()

        exp_weight = torch.cat([m.view(-1) for m in edge_masks]).to(DEVICE)
        delta_weight = 1.0 - exp_weight

        # one parallel encode pass across the whole batch -- was previously
        # n_graphs * proxy_M sequential calls, now 3 total calls regardless of n_graphs
        z_exp_all = self.gae_enc(x, edge_index, edge_weight=exp_weight)
        mu_all = self.vgae_enc_mu(x, edge_index, edge_weight=delta_weight)
        logvar_all = self.vgae_enc_logvar(x, edge_index, edge_weight=delta_weight)
        z_delta_all = mu_all + torch.exp(0.5 * logvar_all) * torch.randn_like(mu_all)

        # split back to per-graph node sets -- indexing only, no compute
        z_exp_list = unbatch(z_exp_all, batch_data.batch)
        z_delta_list = unbatch(z_delta_all, batch_data.batch)
        mu_list = unbatch(mu_all, batch_data.batch)
        logvar_list = unbatch(logvar_all, batch_data.batch)
        edge_index_list = unbatch_edge_index(edge_index, batch_data.batch)  # node idx reset to 0..n_i-1

        A_tildes = []
        total_loss = 0.0
        eps = 1e-8
        for z_exp, z_delta, mu, logvar, ei in zip(
            z_exp_list, z_delta_list, mu_list, logvar_list, edge_index_list
        ):
            n = z_exp.shape[0]
            A_tilde_exp = _decode(z_exp)
            A_tilde_delta = _decode(z_delta)
            A_tilde = (A_tilde_exp + A_tilde_delta).clamp(0, 1)
            A_tildes.append(A_tilde)

            A_orig = _adj(ei, n)
            pos = self.beta * (A_orig * torch.log(A_tilde + eps)).sum() / (A_orig.sum() + eps)
            neg = ((1 - A_orig) * torch.log(1 - A_tilde + eps)).sum() / ((1 - A_orig).sum() + eps)
            L_in = -(pos + neg)
            L_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            total_loss = total_loss + L_in + self.lam * L_kl

        return A_tildes, total_loss
    

    def build_proxy_data(self, G: Data, A_tilde: Tensor) -> Data:
        """Threshold A_tilde into a Data object, always keeping G_exp edges."""

        # threshold proxy, zero out exp region to avoid double-counting
        n = A_tilde.shape[0]
        edge_index = torch.stack(torch.meshgrid(
            torch.arange(n, device=A_tilde.device),
            torch.arange(n, device=A_tilde.device), indexing='ij'
        )).reshape(2, -1)
        edge_weight = A_tilde.reshape(-1)
        return Data(x=G.x, edge_index=edge_index, edge_attr=edge_weight, y=G.y)