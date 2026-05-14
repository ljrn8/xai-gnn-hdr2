import argparse
import os

from torch import threshold
from explanations.xAI_utils import *
from pathlib import Path
from explanations.MUTAG_ground_truth import compute_all_masks
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--ds-root", default="./output/inductive_graph_tasks")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
args = parser.parse_args()

for dataset in os.listdir(args.ds_root):
    dataset_path = Path(args.ds_root) / dataset
    print(f"\n\n Dataset name: {dataset} \n")

    explanations_root = dataset_path / "explanations"
    if not explanations_root.exists():
        logger.warning(f"No explanations found for dataset {dataset} at path {explanations_root}. Skipping evaluation.")
        continue

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

                # show examples of edge masks where training_graph.y == 1
                if args.verbose:
                    postive_graphs = [g for g in test_graphs if g.y.item() == 1]
                    for i in range(min(3, len(postive_graphs))):
                        g = postive_graphs[i]
                        mask = edge_masks[i]
                        print(f" * Graph {i} with label {g.y.item()}, gave expl edge mask: {mask}")
                        print(f'Corresponding GT is {GT_edge_masks[i]}')

                # classification report of GT_edge_masks vs edge_masks (variable size per mask btw)
                from sklearn.metrics import classification_report
                print(f'\n === Classification reports with thresholding === ')

                # TODO now:
                # elbow method, inverstigate how evals where done for proxy in paper

                threshold = 0.5e-6
                all_gt_masks = torch.cat(GT_edge_masks).cpu().numpy()
                all_masks = torch.cat(edge_masks).cpu().numpy()
                print(classification_report(all_gt_masks, all_masks > threshold))

                print(f'\n === AUC === ')
                # get roc AUC and pr AUC
                from sklearn.metrics import roc_auc_score, average_precision_score
                roc_auc = roc_auc_score(all_gt_masks, all_masks)
                pr_auc = average_precision_score(all_gt_masks, all_masks)
                print(f"ROC AUC: {roc_auc}, PR AUC: {pr_auc}")



            elif ...:
                ...
                




