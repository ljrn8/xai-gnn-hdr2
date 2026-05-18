import argparse
import os
import torch
from torch import threshold
from explainability.explainer_utils import *
from pathlib import Path
from explainability.MUTAG_ground_truth import compute_all_masks
from loguru import logger
import numpy as np
from kneed import KneeLocator
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)

parser = argparse.ArgumentParser()
parser.add_argument(
"--ds-root", help="Root directory containing dataset to evaluate on"
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose logging"
)
parser.add_argument("-gt", "--dataset-for-GT", help="Specify the dataset to use for GT masks. Options: mutag, ogx, cora, citeseer", 
                    required=True)
args = parser.parse_args()

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


def evaluate_GT_edge_mask(
    GT_edge_mask: torch.Tensor, predicted_edge_mask: torch.Tensor, threshholder
):
    print(f"\n > Classification report with {threshholder.__name__} thresholding ")
    threshhold = threshholder(predicted_edge_mask)
    thresholded_mask = predicted_edge_mask >= threshhold

    if args.verbose:
        print(f'total explantory edges  : {thresholded_mask.sum()}')
        print(f'total non-explanatory edges  : {len(thresholded_mask) - thresholded_mask.sum()}')
        print(f'total GT explantory edges : {GT_edge_mask.sum()}')

        # histogram of the predicted edge mask
        import matplotlib.pyplot as plt
        plt.hist(predicted_edge_mask.cpu().numpy(), bins=50)
        plt.axvline(threshhold, color='red', linestyle='dashed', linewidth=1)
        plt.show()

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
    logger.warning(
        f"No explanations found for dataset {dataset} at path {explanations_root}. Skipping evaluation."
    )
    exit(0)

for explainer_folder in os.listdir(explanations_root):
    explainer_path = explanations_root / explainer_folder
    explanations = {
        name: openpkl(explainer_path / name) for name in os.listdir(explainer_path)
    }
    for name, explanation in explanations.items():
        logger.info(f"evaluating explanation: {name}")
        edge_masks = [e.detach() for e in explanation.edge_masks]

        # Ground truth masks are computed differently per dataset

        if args.dataset_for_GT == "mutag":
            test_graphs = openpkl(dataset_path / "test_graphs.pkl")
            results = compute_all_masks(test_graphs)
            GT_edge_masks = [r["edge_mask"].detach() for r in results]

        elif args.dataset_for_GT == "ogx":
            test_graphs = openpkl(dataset_path / "test_graphs.pkl")
            GT_node_masks = [g.mask for g in test_graphs]
            edge_indexes = [g.edge_index for g in test_graphs]

            # requires that both the source and destiation node are important for the edge mask
            GT_edge_masks = [
                mask[edge_index[0]] & mask[edge_index[1]] 
                for mask, edge_index in zip(GT_node_masks, edge_indexes)
            ]

            # print the first 3 edge masks and GT masks
            if args.verbose:
                for i in range(3):
                    print(f"\n\nGT edge mask {i}: {GT_edge_masks[i]}")
                    print(f"Predicted edge mask {i}: {edge_masks[i]}")


        elif args.dataset_for_GT == "citeseer" or args.dataset_for_GT == "cora":
            graph = openpkl(dataset_path / "graph.pkl")
            # no GT available, stick to Fidelity and stability
            ...

        


        # -------------------------------
        # fucked zone

        logger.info('NON-INVERTED')

        def fixed_threshhold(x):
            return 1e-10

        GT_edge_masks = torch.cat(GT_edge_masks).float()
        edge_masks = torch.cat(edge_masks).float()

        for threshholder in (
                otsu_threshold,
                knee_threshold,
                quantile_threshold,
                fixed_threshhold
            ):
                _ = evaluate_GT_edge_mask(
                    GT_edge_masks, edge_masks,
                    threshholder
                )

        edge_masks = 1 - edge_masks
        logger.info('INVERTED')

        for threshholder in (
                otsu_threshold,
                knee_threshold,
                quantile_threshold,
                fixed_threshhold
            ):
                _ = evaluate_GT_edge_mask(
                    GT_edge_masks, edge_masks,
                    threshholder
                )