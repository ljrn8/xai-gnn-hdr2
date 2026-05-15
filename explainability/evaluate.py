import argparse
import os

from torch import threshold
from explainability.xAI_utils import *
from pathlib import Path
from explainability.MUTAG_ground_truth import compute_all_masks
from loguru import logger
import numpy as np
from kneed import KneeLocator
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report


parser = argparse.ArgumentParser()
parser.add_argument("--ds-root", description="Root directory containing dataset to evaluate on")
parser.add_argument('-n', '--node-level', action='store_true')
parser.add_argument('-g', '--graph-level', action='store_true')
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
args = parser.parse_args()
assert args.graph_level != args.node_level, "Please specify exactly one of --graph-level or --node-level"


def otsu_threshold(x, bins=256):
    x = np.asarray(x)
    hist, bin_edges = np.histogram(x, bins=bins)
    hist = hist.astype(float)
    prob = hist / hist.sum()
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu)**2 / (
        omega * (1 - omega) + 1e-12
    )
    idx = np.argmax(sigma_b)
    return centers[idx]

def knee_threshold(x):
    x = np.sort(np.asarray(x))
    kneedle = KneeLocator(
        range(len(x)),
        x,
        curve="convex",
        direction="increasing"
    )

    idx = kneedle.knee
    return x[idx]

def quantile_threshold(x, q=0.9, keep="above"):
    scores = np.asarray(x)
    threshold = np.quantile(scores, q)
    return threshold

def minimum_cluster_distance_threshhold(x):
    """ Warning: n^2 """
    best_distance = float('inf')
    best_k = 0
    for k in range(1, len(x) + 1):
        top_k_indicies = torch.topk(x, k).indices
        neglected_indicies = torch.argsort(x)[:-k]
        distance = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if (i in top_k_indicies and j in top_k_indicies) or (i in neglected_indicies and j in neglected_indicies):
                    distance += abs(x[i] - x[j])
        if distance < best_distance:
            best_distance = distance
            best_k = k

    return best_k, best_distance

def evaluate_GT_edge_mask(GT_edge_mask: torch.Tensor, predicted_edge_mask: torch.Tensor, threshholder):
    print(f'\n > Classification report with {threshholder.__name__} thresholding ')
    threshhold = threshholder(predicted_edge_mask)
    thresholded_mask = predicted_edge_mask >= threshhold
    print(classification_report(GT_edge_mask, thresholded_mask))
    roc_auc = roc_auc_score(GT_edge_mask, thresholded_mask)
    pr_auc = average_precision_score(GT_edge_mask, thresholded_mask)
    print(f"ROC AUC: {roc_auc:.5f}, PR AUC: {pr_auc:.5f}")
    return GT_edge_mask, thresholded_mask


dataset_path = Path(args.ds_root)
dataset = dataset_path.name
print(f"\n\n Dataset name: {dataset} \n")

explanations_root = dataset_path / "explanations"
if not explanations_root.exists():
    logger.warning(f"No explanations found for dataset {dataset} at path {explanations_root}. Skipping evaluation.")
    exit(0)

for explainer_folder in os.listdir(explanations_root):
    explainer_path = explanations_root / explainer_folder

    explanations = {
        name: openpkl(explainer_path / name) 
        for name in os.listdir(explainer_path)
    }

    for name, explanation in explanations.items():
        logger.info(f"evaluating explanation: {name}")
        edge_masks = [e.detach() for e in explanation.edge_masks]

        if 'mutag' in str(dataset_path).lower():
            test_graphs = openpkl(dataset_path / "test_graphs.pkl")
            results = compute_all_masks(test_graphs)
            GT_edge_masks = [r["edge_mask"].detach() for r in results]
            GT_edge_masks = torch.cat(GT_edge_masks).float()

            for threshholder in otsu_threshold, knee_threshold, quantile_threshold, minimum_cluster_distance_threshhold:
                GT_edge_mask, thresholded_mask = evaluate_GT_edge_mask(GT_edge_masks, torch.cat(edge_masks).float(), threshholder)


        elif 'ogx' in str(dataset_path).lower():
            test_graphs = openpkl(dataset_path / "test_graphs.pkl")
            print(test_graphs[0].__dict__)
            GT_edge_masks = [g.mask for g in test_graphs] 
            GT_edge_masks = torch.cat(GT_edge_masks).float()

            for threshholder in otsu_threshold, knee_threshold, quantile_threshold:
                GT_edge_mask, thresholded_mask = evaluate_GT_edge_mask(GT_edge_masks, torch.cat(edge_masks).float(), threshholder)
                

            # !! G.mask is a node mask




                




