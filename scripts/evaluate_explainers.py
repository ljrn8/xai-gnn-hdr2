import argparse
import os
import torch
from torch import threshold
from explainability.utils.explainer_utils import *
from pathlib import Path
from explainability.utils.mutag import compute_all_masks
from loguru import logger
import numpy as np
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)


def evaluate_GT_edge_mask(
    GT_edge_mask: torch.Tensor, predicted_edge_mask: torch.Tensor, threshholder
):
    print(f"\n > Results with {threshholder.__name__} thresholding")
    threshhold = threshholder(predicted_edge_mask)
    thresholded_mask = predicted_edge_mask >= threshhold

    if args.verbose:
        print(f"total explantory edges  : {thresholded_mask.sum()}")
        print(
            f"total non-explanatory edges  : {len(thresholded_mask) - thresholded_mask.sum()}"
        )
        print(f"total GT explantory edges : {GT_edge_mask.sum()}")

        # histogram of the predicted edge mask
        import matplotlib.pyplot as plt

        plt.hist(predicted_edge_mask.cpu().numpy(), bins=50)
        plt.axvline(threshhold, color="red", linestyle="dashed", linewidth=1)
        plt.show()

    if not args.auc:
        print(classification_report(GT_edge_mask, thresholded_mask))
    roc_auc = roc_auc_score(GT_edge_mask, thresholded_mask)
    pr_auc = average_precision_score(GT_edge_mask, thresholded_mask)
    print(f"ROC AUC: {roc_auc:.5f}, PR AUC: {pr_auc:.5f}")
    return GT_edge_mask, thresholded_mask


def main(args):
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

            elif args.dataset_for_GT == "motifs":
                test_graphs = openpkl(dataset_path / "test_graphs.pkl")
                GT_edge_masks = [g.edge_mask for g in test_graphs]

            logger.debug(
                f"gt.shape for gt in edge_masks {[gt.shape for gt in edge_masks]}"
            )
            logger.debug(
                f"total values in edge_masks {sum([gt.sum() for gt in edge_masks])}"
            )

            def fixed_threshhold(x):
                return 1e-10

            GT_edge_masks = torch.cat(GT_edge_masks).float()
            edge_masks = torch.cat(edge_masks).float()
            edge_masks = edge_masks.view(-1)

            for threshholder in (
                otsu_threshold,
                knee_threshold,
                quantile_threshold,
                # fixed_threshhold
            ):
                _ = evaluate_GT_edge_mask(GT_edge_masks, edge_masks, threshholder)

            # edge_masks = 1 - edge_masks
            # logger.info("INVERTED")

            # for threshholder in (
            #     otsu_threshold,
            #     knee_threshold,
            #     quantile_threshold,
            #     # fixed_threshhold
            # ):
            #     _ = evaluate_GT_edge_mask(GT_edge_masks, edge_masks, threshholder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds",
        "--ds-root",
        help="Root directory containing dataset to evaluate on",
        required=True,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-gt",
        "--dataset-for-GT",
        help="Specify the dataset to use for GT masks. Options: mutag, ogx, cora, citeseer",
        required=True,
    )

    parser.add_argument("-auc", help="only report auc metrics", action="store_true")
    args = parser.parse_args()
    main(args)
