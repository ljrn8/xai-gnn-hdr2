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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


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
        self.gae_enc = _GCNEncoder(node_feature_dim, latent_dim)

        # VGAE branch for G_delta = G - G_exp (eq. 14)
        self.vgae_enc_mu = _GCNEncoder(node_feature_dim, latent_dim)
        self.vgae_enc_logvar = _GCNEncoder(node_feature_dim, latent_dim)


    def forward(self, G: Data, soft_mask: Tensor):
        """
        Returns:
            A_tilde: (n, n) proxy adjacency probabilities
            loss:    L_gen scalar
        """
        n = G.x.shape[0]
        x = G.x.float()
        edge_index = G.edge_index

        is_exp = (soft_mask.detach().squeeze() > 0.5).view(-1).squeeze()
        exp_ei = edge_index[:, is_exp]
        non_exp_ei = edge_index[:, ~is_exp]

        # fallback to full graph if either split is empty (avoids empty GCN)
        enc_exp_ei = exp_ei if exp_ei.shape[1] > 0 else edge_index
        enc_non_ei = non_exp_ei if non_exp_ei.shape[1] > 0 else edge_index

        # GAE on G_exp (eq. 13)
        z_exp = self.gae_enc(x, enc_exp_ei)
        A_tilde_exp = _decode(z_exp)

        # VGAE on G_delta (eq. 14-15)
        mu = self.vgae_enc_mu(x, enc_non_ei)
        logvar = self.vgae_enc_logvar(x, enc_non_ei)
        z_delta = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        A_tilde_delta = _decode(z_delta)

        # eq. 16
        A_tilde = (A_tilde_exp + A_tilde_delta).clamp(0, 1)

        # L_in (eq. 11) — weighted BCE against original adjacency
        A_orig = _adj(edge_index, n)
        eps = 1e-8
        pos = self.beta * (A_orig * torch.log(A_tilde + eps)).sum() / (A_orig.sum() + eps)
        neg = ((1 - A_orig) * torch.log(1 - A_tilde + eps)).sum() / ((1 - A_orig).sum() + eps)
        L_in = -(pos + neg)

        # L_KL against N(0, I) prior
        L_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

        return A_tilde, L_in + self.lam * L_kl

    def build_proxy_data(self, G: Data, A_tilde: Tensor, soft_mask: Tensor,
                         threshold: float = 0.5) -> Data:
        """Threshold A_tilde into a Data object, always keeping G_exp edges."""
        edge_index = G.edge_index
        is_exp = (soft_mask.detach().squeeze() > 0.5).view(-1)
        exp_ei = edge_index[:, is_exp]

        # threshold proxy, zero out exp region to avoid double-counting
        proxy_adj = (A_tilde.detach() > threshold).float()
        if exp_ei.shape[1] > 0:
            proxy_adj[exp_ei[0], exp_ei[1]] = 0.0

        new_ei = proxy_adj.nonzero(as_tuple=False).t()
        proxy_ei = torch.cat([exp_ei, new_ei], dim=1) if exp_ei.shape[1] > 0 else new_ei
        return Data(x=G.x, edge_index=proxy_ei, y=G.y)