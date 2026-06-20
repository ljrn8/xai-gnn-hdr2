import torch
from torch import nn
import numpy as np
from typing import Optional, Iterable
from copy import deepcopy
import torch.optim as optim
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from torch_geometric.data import Data
from pathlib import Path
import os
from ..interfaces import GNN, ModelRun, ModelEvaluation
from ..config import MODELS, DATASETS, SEED
from ..util import openpkl, savepkl
from .models import GraphGNNWrapper


def _run_graphs(graphs: Iterable[Data], model: GNN, criterion, optimizer=None):
    """Run a training or evaluation loop over a set of graphs.
    Training is performed if an optimizer is provided, otherwise evaluation is performed.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    y_true, y_scores = [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for G in graphs:
            if training:
                optimizer.zero_grad()

            if G.x.dim() == 1:
                G.x = G.x.unsqueeze(1)

            logit = model(G.x, G.edge_index).view(-1)
            y = G.y.float().view(-1)
            loss = criterion(logit, y)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            prob = torch.sigmoid(logit)
            y_true.extend(y.cpu().tolist())
            y_scores.extend(prob.detach().cpu().tolist())

    return total_loss / len(graphs), y_true, y_scores


def train(
    model: GNN,
    graphs: Iterable[Data],
    epochs: int,
    lr: float,
    patience: int = 20,
    delta: float = 0.001,
    verbose: bool = True,
):
    """Generic training loop for binary graph classifiers.
    (Uses early stopping on validation loss)

    Returns:
        train_loss (Float): Average training loss
        val_loss (Float): Average validation loss
    """
    train_losses, val_losses = [], []
    best_model = None
    best_loss = float("inf")
    early_stopping_counter = 0
    best_epoch = 0
    pbar = tqdm(range(1, epochs + 1))

    val_graphs = [g for g in graphs if g.validation_mask == 1]
    test_graphs = [g for g in graphs if g.test_mask == 1]
    train_graphs = [g for g in graphs if g.validation_mask == 0 and g.test_mask == 0]

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info("Training and validating...")
    for epc in pbar:

        # train
        train_loss, y_true, y_scores = _run_graphs(
            graphs=train_graphs, model=model, criterion=criterion, optimizer=optimizer
        )
        train_losses.append(train_loss)

        # validation
        val_loss, y_true, y_scores = _run_graphs(
            graphs=val_graphs, model=model, criterion=criterion
        )
        val_losses.append(val_loss)
        validation_metrics = evaluate_binary_predictions(y_true, y_scores)

        # early stopping
        if best_loss - val_loss < delta:
            if early_stopping_counter >= patience:
                logger.info(
                    f"Early stopping at epoch {epc} | best epoch {best_epoch} | best loss {best_loss:.4f}"
                )
                break
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epc

        pbar.set_description(
            f"Epoch {epc}/{epochs}: train_l={train_loss:.4f} val_l={val_loss:.4f} val_f1={validation_metrics['f1']:.4f}" 
        )

    return train_loss, val_loss


def evaluate_binary_predictions(y_true, y_scores) -> dict:
    """Evaluate binary classification predictions using various metrics."""

    y_pred_bin = [1 if p > 0.5 else 0 for p in y_scores]
    return {
        "roc_auc": roc_auc_score(y_true, y_scores),
        "pr_auc": average_precision_score(y_true, y_scores),
        "rec": recall_score(y_true, y_pred_bin),
        "prec": precision_score(y_true, y_pred_bin),
        "acc": accuracy_score(y_true, y_pred_bin),
        "f1": f1_score(y_true, y_pred_bin),
    }


def summarize_metrics(metrics_dict):
    for metric, value in metrics_dict.items():
        logger.info(f"{metric}: {value:.4f}")


def evaluate(model, test_graphs):
    """Evaluate a trained model on test graphs."""

    model.eval()
    test_loss, y_true, y_scores = _run_graphs(
        graphs=test_graphs, model=model, criterion=nn.BCEWithLogitsLoss()
    )
    metrics = evaluate_binary_predictions(y_true, y_scores)
    summarize_metrics(metrics)
    return y_scores, metrics


def configurations_random_search(dataset_path: Path, models_config: Path = MODELS, overwrite: bool = False):
    """Perform random search over model hyperparameters and save results for a single dataset.

    Args:
        dataset_path (Path): Path to dataset pkl file to train on.
        models_config (dict): Model hyperparameter configuration dictionary, containing hyperparameter value options and number of iterations for random search,
            for examples see config.py.

    Returns:
        runs (ModelRun): List of model runs with best model and hyperparameters for each model type.
    """
    ds_name = dataset_path.stem
    output_dir = models_config["output"]
    iterations = models_config["iterations"]
    best_val_loss = float("inf")
    best_model = None
    runs = []
    logger.info(f" > Dataset name: {ds_name}")
    graphs = openpkl(dataset_path)
    n_features = graphs[0].x.shape[1]

    for model_name, model_config in models_config["random_search_configurations"].items():
        hyperparameter_config = model_config['hyperparameters']
        hp_keys = [k for k in hyperparameter_config.keys() if k != "base_class"]
        model_class = model_config["base_class"]
        done_iterations = set()
        
        save_path = output_dir / model_name / f"{ds_name}.pkl"
        if save_path.exists() and not overwrite:
            logger.info(f"Model run already exists for model [{model_name}] on dataset [{ds_name}] at {save_path}, skipping...")
            run = openpkl(save_path)
            runs.append(run)
            continue

        for i in range(iterations):
            hp = {k: np.random.choice(hyperparameter_config[k]) for k in hp_keys}
            if tuple(hp.items()) in done_iterations:
                continue

            done_iterations.add(tuple(hp.items()))

            model: GNN = GraphGNNWrapper(
                node_model=model_class(
                    input_feat=n_features,
                    hidden_channels=hp["hidden_channels"],
                    num_layers=hp["num_layers"],
                    output_channels=hp["hidden_channels"],
                ),
                incoming_channels=hp["hidden_channels"],
                output_graph_channels=1,
            )

            logger.info(f"model: {model}")

            logger.info(f" > Training model [{model_name}] with hyperparameters: {hp}")
            train_loss, val_loss = train(
                model=model,
                graphs=graphs,
                epochs=hp["epochs"],
                lr=hp["lr"],
                patience=hp.get("patience", 20),
                delta=hp.get("delta", 0.001),
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = deepcopy(model)

        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Saving model and results for model [{model_name}] on dataset [{ds_name}] to {save_path}"
        )
        run = ModelRun(
            dataset_root=str(dataset_path),
            model=best_model,
            model_configuration=hp,
            details={
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            },
        )
        savepkl(run, save_path)
        runs.append(run)

    return runs


def evaluate_model_directory(model_directory: Path):
    """Evaluate all models in a directory on their respective test sets and save results."""

    for model_file in os.listdir(model_directory):
        if not model_file.endswith(".pkl"):
            continue

        model_run: ModelRun = openpkl(model_directory / model_file)
        test_graphs = [g for g in openpkl(model_run.dataset_root) if g.test_mask == 1]
        logger.info(
            f"Evaluating model {model_file} on test set of dataset {model_run.dataset_root}"
        )
        y_scores, metrics = evaluate(model_run.model, test_graphs)
        model_run.test_evaluation = ModelEvaluation(
            y_pred_test=y_scores, metrics=metrics
        )
        savepkl(model_run, model_directory / model_file)


def main(args):
    if args.evaluate:
        logger.info(f"running test evals on {args.model_directory}")
        evaluate_model_directory(args.model_directory, overwrite=args.overwrite)

    else:
        datasets = os.listdir(args.data_directory)
        datasets = [ds for ds in datasets if ds.endswith(".pkl")]
        if args.exclude:
            exclude = set(args.exclude.split(","))
            datasets = [ds for ds in datasets if ds.split(".")[0] not in exclude]
        if args.specifically:
            specifically = args.specifically.split(",")
            datasets = [ds for ds in datasets if ds.split(".")[0] in specifically]

        logger.info(
            f"running model optimizations on {args.data_directory} on datasets: {datasets}"
        )

        for ds_name in datasets:
            dataset_path = args.data_directory / ds_name
            runs = configurations_random_search(dataset_path, overwrite=args.overwrite)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-directory",
        help="Root directory containing datasets to train on",
        default=DATASETS["output"] / "processed",
    )
    parser.add_argument(
        "--model-directory",
        help="Root directory containing datasets to train on",
        default=MODELS["output"],
    )
    parser.add_argument(
        "-e", "--evaluate", action="store_true", help="run evaluation on test set only"
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="overwrite pkl model runs (redo all)"
    )
    parser.add_argument(
        "-ex", "--exclude", help="exclude comma seperated datasets (eg -ex delta,echo)"
    )
    parser.add_argument(
        "-sp",
        "--specifically",
        help="only train on specified comma seperated datasets (eg -sp delta,echo)",
    )
    main(parser.parse_args())
