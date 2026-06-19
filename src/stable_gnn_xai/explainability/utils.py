from torch_geometric.nn import MessagePassing, global_mean_pool
from torch import nn
import torch.functional as F
import sys
from src.stable_gnn_xai.training.train import *
from abc import ABC, abstractmethod
from torch_geometric.data import Data
from collections.abc import Iterable
from kneed import KneeLocator


def uniform_debug_log(soft_edge_masks: torch.Tensor, check_parameters=None):
    if check_parameters:
        for name, p in check_parameters:
            grad_norm = p.grad.norm().item() if p.grad is not None else "None"
            logger.debug(f"{name} norm: {p.data.norm():.4f}, grad_norm: {grad_norm}")

    logger.debug(
        f"mask [0] bincounts = {np.histogram(soft_edge_masks[0].detach(), bins=10)}"
    )
    logger.debug(
        f"mask [1] bincounts = {np.histogram(soft_edge_masks[1].detach(), bins=10)}"
    )
    logger.debug(
        f"mask [2] bincounts = {np.histogram(soft_edge_masks[2].detach(), bins=10)}"
    )
    logger.debug(
        f"mask [3] bincounts = {np.histogram(soft_edge_masks[3].detach(), bins=10)}"
    )


def elementwise_entropy(p):
    eps = 1e-8
    return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))


def otsu_threshold(x, bins=256):
    x = np.asarray(x)
    hist, bin_edges = np.histogram(x, bins=bins)
    hist = hist.astype(float)
    prob = hist / hist.sum()
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    idx = np.argmax(sigma_b)
    return centers[idx]


def knee_threshold(x):
    x = np.sort(np.asarray(x))
    kneedle = KneeLocator(range(len(x)), x, curve="convex", direction="increasing")
    idx = kneedle.knee
    return x[idx]


def quantile_threshold(x, q=0.9, keep="above"):
    scores = np.asarray(x)
    threshold = np.quantile(scores, q)
    return threshold


def minimum_cluster_distance_threshhold(x):
    """Warning: n^2"""
    best_distance = float("inf")
    best_k = 0
    for k in range(1, len(x) + 1):
        top_k_indicies = torch.topk(x, k).indices
        neglected_indicies = torch.argsort(x)[:-k]
        distance = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if (i in top_k_indicies and j in top_k_indicies) or (
                    i in neglected_indicies and j in neglected_indicies
                ):
                    distance += abs(x[i] - x[j])
        if distance < best_distance:
            best_distance = distance
            best_k = k

    return best_k, best_distance
