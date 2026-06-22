import os
from torch import nn
import torch.functional as F
import torch
from torch_geometric.data import Data
from loguru import logger
from pathlib import Path
from ..interfaces import GNN, Explanation, ModelRun
from ..util import openpkl, savepkl
from ..config import EXPLAINERS, MODELS, FIGURES, DEVICE
from typing_extensions import Iterable
import matplotlib.pyplot as plt


def run_explainers_from_config(
        model_run: ModelRun, 
        output_path: Path, 
        explainers_search: dict = EXPLAINERS['exhuastive_search_configurations'],
        specify_explainer: str = None
):
    """Applies, optimizes and saves all explainers specified in explainers_search towards the model run 
    
    Produces:
        [output_path]/
            [explainer_name].pkl
    """
    graphs = openpkl(model_run.dataset_root)
    GT_test_edge_masks = [g.edge_mask for g in graphs if g.test_mask == 1]

    test_graphs = [g for g in graphs if g.test_mask == 1]
    model = model_run.model
    
    logger.info(f'moving model and graphs to {DEVICE}')
    model = model_run.model.to(DEVICE)
    test_graphs = [g.to(DEVICE) for g in test_graphs]
    logger.info('done')
    
    dataset = Path(model_run.dataset_root).stem

    logger.info(f"\n\n Dataset name: {dataset} \n")
    logger.info(f"total graphs in test set: {len(test_graphs)}")
    logger.info("passing a test graph for embedding dimensions etc")
    y_pred, embeddings_list = model(
        test_graphs[0].x, test_graphs[0].edge_index, return_all_embeddings=True
    )
    embedding_sizes = [e.shape[1] for e in embeddings_list]

    logger.info(f"embeddings sizes: {embedding_sizes}")
    for e in embedding_sizes:
        assert (
            e == embedding_sizes[0]
        ), f"all latent embeddings must be the same size for an RNN Module"

    # Exhuastive explainer configuration search loop
    for explainer_name, explainer_config in explainers_search.items():
        if specify_explainer and explainer_name != specify_explainer:
            logger.info(f'skipping {explainer_name} as unspecified explainer')
            continue

        logger.info(f'EXPLAINER: {explainer_name}')
        hyperparameters_config = explainer_config['hyperparameters']
        expl_class = explainer_config['class']
        hp_names = list(hyperparameters_config.keys())
        n_configurations = len(hyperparameters_config[hp_names[0]])
        best_penalty = float('inf')

        for config_idx in range(n_configurations):
            explainer = expl_class(
                model=model, 
                graphs=test_graphs, 
                **{k: hyperparameters_config[k][config_idx] for k in hyperparameters_config.keys() if k != 'class'}
            )
            masks, final_objective_loss = explainer.explain_graph_task()
            if final_objective_loss < best_penalty:
                best_penalty = final_objective_loss
                best_explanation = Explanation(
                    explainer=explainer,
                    run=model_run,
                    task_type='graph',
                    edge_masks=masks,
                    name=explainer_name
                )

            evaluate_explanation(masks, GT_test_edge_masks, figures_id=f"{dataset}_{explainer_name}_{config_idx}")

        best_explanation.evaluation_metrics['roc_auc'] = evaluate_explanation(
            best_explanation.edge_masks, GT_test_edge_masks, figures_id=f"{dataset}_{explainer_name}_best"
        )
        output_path.mkdir(exist_ok=True, parents=True)
        savepkl(best_explanation, output_path / f"{explainer_name}.pkl")


def evaluate_explanation(
    edge_masks: Iterable[torch.Tensor],
    GT_edge_masks: Iterable[torch.Tensor],
    figures_id: str,
    figures_dir: Path = FIGURES / 'explanations'
):
    """"Evaluates ROC-AUC against binary GT edge masks & mask histogram
    
    Returns:
        roc_auc (float)
    """
    from sklearn.metrics import roc_auc_score

    # flatten list of per-graph tensors -> single 1D numpy arrays
    masks_flat = torch.cat([m.detach().cpu().view(-1) for m in edge_masks]).numpy()
    gt_flat = torch.cat([m.detach().cpu().view(-1) if isinstance(m, torch.Tensor) else torch.tensor(m).view(-1) for m in GT_edge_masks]).numpy()

    roc_auc = roc_auc_score(gt_flat, masks_flat)
    logger.info(f"ROC AUC of explanation: {roc_auc:.4f}")

    figures_dir.mkdir(exist_ok=True, parents=True)
    plt.hist(masks_flat, bins=50)
    plt.savefig(figures_dir / f"{figures_id}_explanation_histogram.png")
    plt.clf()

    return roc_auc


def main(args):
    # explain a specifc run
    if args.model_run_path:
        model_run = openpkl(args.model_run_path)
        model_name = Path(args.model_run_path).parent.name
        dataset_name = Path(model_run.dataset_root).stem

        logger.info(f"\n --> explaining specific model run at path: {args.model_run_path} \n")
        run_explainers_from_config(
            model_run, 
            output_path=EXPLAINERS['output'] / model_name / dataset_name,
            specify_explainer=args.explainer)

    # evaluate existing explanations
    elif args.evaluate:
        explanations_dir = EXPLAINERS['output']
        models = os.listdir(explanations_dir)
        for model_name in models:
            model_explanations_dir = explanations_dir / model_name
            datasets = os.listdir(model_explanations_dir)

            for dataset_name in datasets:
                explanation_files = os.listdir(model_explanations_dir / dataset_name)
                for explanation_file in explanation_files:
                    explanation = openpkl(model_explanations_dir / dataset_name / explanation_file)
                    GT_test_edge_masks = [g.edge_mask for g in explanation.explainer.graphs if g.test_mask == 1]
                    model_name = Path(explanation.run.dataset_root).parent.name
                    dataset_name = Path(explanation.run.dataset_root).stem
                    logger.info(f'eval for {explanation.name} on model {model_name} for dataset {dataset_name}')
                    explanation.evaluation_metrics['roc_auc'] = evaluate_explanation(
                        explanation.edge_masks, GT_test_edge_masks, figures_dir =  FIGURES / 'explanations' / model_name, 
                        figures_id=f"{dataset_name}_{explanation.name}best"
                    )

    # explain all runs
    else:
        root = MODELS['output']
        model_names = os.listdir(root)

        for model_name in model_names:
            run_names = os.listdir(root / model_name)
            logger.info(f"found {run_names} model runs to explain")

            for model_run_file in run_names:
                model_run = openpkl(root / model_name / model_run_file)
                dataset_name = Path(model_run_file).stem

                logger.info(f"\n --> explaining model [{model_name}] for dataset [{dataset_name}]\n")
                run_explainers_from_config(
                    model_run, 
                    output_path=EXPLAINERS['output'] / model_name / dataset_name,
                    specify_explainer=args.explainer)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '-m', 
        '--model-run-path', 
        help='specific model run to explain (expects [some_path]/[model_name]/[dataset_name].pkl), as a path to the ModelRun pkl file'
    )
    parser.add_argument(
        '-e', 
        '--evaluate', 
        action='store_true',
        help='evaluate existing explanations instead'
    )
    parser.add_argument(
        '-ex', 
        '--explainer', 
        help='only apply a single explainer (names defined under config.py)'
    )
    main(parser.parse_args())