import argparse
import os
from explanations.xAI_utils import *
from pathlib import Path
from explanations.MUTAG_ground_truth import compute_all_masks
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--ds-root", default="./output/inductive_graph_tasks")
args = parser.parse_args()

for dataset in os.listdir(args.ds_root):
    dataset_path = Path(args.ds_root) / dataset
    print(f"\n\n Dataset name: {dataset} \n")

    explanations_root = dataset_path / "explanations"
    for explainer_folder in os.listdir(explanations_root):
        explainer_path = explanations_root / explainer_folder

        explanations = {
            name: openpkl(explainer_path / name) 
            for name in os.listdir(explainer_path)
        }

        for name, explanation in explanations.items():
            logger.info(f"evaluating explanation: {name}")
            edge_masks = explanation.edge_masks
            edge_masks = [e.detach() for e in edge_masks]

            # show examples of edge_masks
            for i, mask in enumerate(edge_masks[:3]):
                print(f"Example {i} edge mask: {mask.cpu().numpy()}")
                print(f"Number of edges in explanation: {(mask > 0.5).sum().item()} / {mask.shape[0]}")

            if 'mutag' in str(dataset_path).lower():
                test_graphs = openpkl(dataset_path / "test_graphs.pkl")
                results = compute_all_masks(test_graphs)
                GT_edge_masks = [r["edge_mask"].detach()for r in results]

                # classification report of GT_edge_masks vs edge_masks (variable size per mask btw)
                from sklearn.metrics import classification_report
                all_gt_masks = torch.cat(GT_edge_masks).cpu().numpy()
                all_masks = torch.cat(edge_masks).cpu().numpy()
                print(classification_report(all_gt_masks, all_masks > 0.5))


            elif ...:
                ...
                




