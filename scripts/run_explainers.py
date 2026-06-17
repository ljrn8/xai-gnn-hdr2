import os
from torch import nn
import torch.functional as F
from torch_geometric.data import Data
from ..training.GNN_utils import *
from ..explainability.utils.explainer_utils import Explanation
from pathlib import Path
import argparse
from src.explainability.explainers.PGExplainers import (
    PGExplainer,
    PGEExplanationModule,
    ComprehensiveMLPExplanationModule,
    GRUExplanationModule,
)
from ..explainability.explainers.GNNExplainer import GNNExplainer
from ..explainability.utils.explainer_utils import Explainer

# TODO:
# seperate args for each explainer
# evaluate explainers for different HPOs (high PGE variance)

class RandomExplainer(Explainer):
    """Random edge mask generator for benchmarking"""

    def __init__(self, graphs: Iterable[Data]):
        self.graphs = graphs

    def explain_node_task(self):
        raise NotImplementedError()

    def explain_graph_task(self):
        return [torch.rand(size=(graph.edge_index.shape[1],)) for graph in self.graphs]


def main(args):
    torch.autograd.set_detect_anomaly(True, check_nan=False)

    logger.debug("opening test graphs")
    test_graphs = openpkl(Path(args.ds_root) / "test_graphs.pkl")
    G = test_graphs[0]
    n_features = G.x.shape[1]

    path = Path(args.ds_root)
    dataset = path.name
    models_path = path / "models"
    logger.info(f"\n\n Dataset name: {dataset} \n")
    runs = {model: openpkl(models_path / model) for model in os.listdir(models_path)}

    logger.info(f'total graphs in test set: {len(test_graphs)}')

    for model_name, run in runs.items():
        model = run.task.model
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

        explainer_map = {
            "Random Explainer": RandomExplainer(graphs=test_graphs),
            'GNNExplainer': GNNExplainer(
                model=model,
                graphs=test_graphs,
                epochs=args.epochs,
                lr=0.05, # !!!
                mean_regularization=0.1,
                entropy_regularization=0.05,
            )
        }
        for pge_module in (
            PGEExplanationModule,
            ComprehensiveMLPExplanationModule,
            GRUExplanationModule,
        ):
            explainer_map[f"PGExplainer with {pge_module.__name__}"] = PGExplainer(
                epochs=args.epochs,
                lr=args.learning_rate,
                mean_regularization=0.1,
                entropy_regularization=0.05,
                tau=0.3,
                reparameterization_samples=30,
                explanation_module=pge_module(
                    graphs=test_graphs, model=run.task.model, hidden_size=args.hidden_size
                ),
            )

        if args.explainer:
            explainer_map = {args.explainer: explainer_map[args.explainer]}

        for explainer_name, expl in explainer_map.items():
            model_name = model_name.split(".pkl")[0]
            logger.info(f"Explaining model: {model_name} with explainer: {explainer_name} ")

            if args.graph_level:
                masks = expl.explain_graph_task()
                explanation = Explanation(run=run, edge_masks=masks, task_type="graph")

            elif args.node_level:
                masks = expl.explain_node_task()
                explanation = Explanation(run=run, edge_masks=masks, task_type="node")

            save_path = path / "explanations" / explainer_name
            save_path.mkdir(exist_ok=True, parents=True)
            with open(save_path / f"{model_name}_explanation.pkl", "wb") as f:
                pickle.dump(explanation, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds",
        "--ds-root",
        help="Root directory containing dataset to evaluate on",
        required=True,
    )
    parser.add_argument("-n", "--node-level", action="store_true")
    parser.add_argument("-g", "--graph-level", action="store_true", help='(default)')
    parser.add_argument("-e", "--epochs", type=int,             default=300)
    parser.add_argument("-lr", "--learning-rate", type=float,   default=0.01)
    parser.add_argument("--hidden-size", type=int,              default=64)
    parser.add_argument(
        "--explainer",
        type=str,
        default=None,
        help=f"only run a specific explainer by name. For options see explainability.baselines",
    )
    args = parser.parse_args()
    if not args.node_level:
        args.graph_level = True

    main(args)