import os
from torch import nn
import torch.functional as F
import torch
from torch_geometric.data import Data
from loguru import logger
from pathlib import Path
from ..interfaces import GNN, Explanation, ModelRun
from ..util import openpkl, savepkl
from ..config import EXPLAINERS, MODELS, FIGURES
from typing_extensions import Iterable
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True, check_nan=False)

def run_explainers_from_config(
        model_run: ModelRun, 
        output_path: Path, 
        explainers_search: dict = EXPLAINERS['exhuastive_search_configurations']
):
    """Applies, optimizes and saves all explainers specified in explainers_search towards the model run 
    
    Produces:
        [output_explainer_directory]/
            [dataset_name]/
                [model_name]/
                    [explainer_name].pkl
    """
    graphs = openpkl(model_run.dataset_root)
    GT_test_edge_masks = [g.edge_mask for g in graphs if g.test_mask == 1]

    test_graphs = [g for g in graphs if g.test_mask == 1]
    G = test_graphs[0]
    model = model_run.model
    dataset = Path(model_run.dataset_root).stem

    output_path.mkdir(exist_ok=True, parents=True)

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
        expl_class = explainer_config['class']
        explainer_config.remove('class')
        n_configurations = len(explainer_config[explainer_config.keys()[0]])
        best_penalty = float('inf')
        for config_idx in range(n_configurations):
            explainer = explainer_config['class'](
                model=model, 
                graphs=test_graphs, 
                **{k: explainer_config[k][config_idx] for k in explainer_config.keys() if k != 'class'}
            )
            masks, final_objective_loss = explainer.explain_graph_task()
            if final_objective_loss < best_penalty:
                best_penalty = final_objective_loss
                best_explanation = Explanation(
                    explainer=explainer,
                    run=model_run,
                    task_type='graph',
                    edge_masks=masks,
                )

            evaluate_explanation(masks, GT_test_edge_masks, figures_id = f"{dataset}_{explainer_name}_{config_idx}")

        best_explanation.evaluation_metrics['roc_auc'] = evaluate_explanation(best_explanation.edge_mask, GT_test_edge_masks)
        savepkl(best_explanation, output_path / f"{explainer_name}.pkl")


def evaluate_explanation(edge_masks: Iterable[torch.Tensor], GT_edge_masks: Iterable[torch.Tensor], figures_id, figures_dir = FIGURES / 'explanations'):
    """"Evaluates per-sparsity level fidelity and ROC-AUC against binary GT edge masks
    
    Returns:
        roc_auc (float)
    """
    edge_masks = edge_masks
    edge_masks = edge_masks.view(-1)
    GT_edge_masks = GT_edge_masks.view(-1)

    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(GT_edge_masks.cpu(), edge_masks.cpu())
    logger.info(f"ROC AUC of explanation: {roc_auc:.4f}")

    # create a histogram of the edge_masks
    figures_dir.mkdir(exist_ok=True, parents=True)
    plt.hist(edge_masks.cpu().numpy(), bins=50)
    plt.savefig(figures_dir / f"{figures_id}_explanation_histogram.png")
    plt.clf()

    # TODO: fidelity/sparsity levels ?
    return roc_auc


def main(args):
    if args.model_run_path:
        logger.info(f"explaining specific model run at path: {args.model_run_path}")
        model_run = openpkl(args.model_run_path)
        model = model_run.path.parent
        run_explainers_from_config(model_run, output_path = EXPLAINERS['output'] / model)

    else:
        files = os.listdir(MODELS['output'])
        logger.info(f"found {len(files)} model runs to explain")
        for model_run_file in files:
            model_run = openpkl(MODELS['output'] / model_run_file)
            model = model_run.path.parent
            run_explainers_from_config(model_run, output_path = EXPLAINERS['output'] / model)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', '--model-run-path', help='specific model run to explain, as a path to a pkl file')
    main(parser.parse_args())
