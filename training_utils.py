import torch
from torch import nn
import numpy as np
import pickle
import sys
from dataclasses import dataclass
from typing import Optional, Iterable
from models import NodeGCN2
from torch_geometric.data import Data
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from copy import deepcopy
import torch.optim as optim
from tqdm import tqdm
from pprint import pprint

@dataclass
class ModelPerformance:
    roc_auc: float
    pr_auc: float
    rec: float
    prec: float
    acc: float
    f1: float
    y_pred: Optional[np.ndarray] = None
    y_true: Optional[np.ndarray] = None


@dataclass
class TrainingRun:
    best_model: nn.Module
    current_model: nn.Module
    dataset: str
    train_losses: list
    node_level_task: bool
    test_losses: list
    performance: Optional[ModelPerformance] = None
    hyperparameter: Optional[dict] = None 
    test_indexes: Optional[np.ndarray] = None

def evaluate_binary_predictions(y_true, y_scores):
    y_pred_bin = [1 if p > 0.5 else 0 for p in y_scores]
    return ModelPerformance(
        roc_auc=roc_auc_score(y_true, y_scores),
        pr_auc=average_precision_score(y_true, y_scores),
        rec=recall_score(y_true, y_pred_bin),
        prec=precision_score(y_true, y_pred_bin),
        acc=accuracy_score(y_true, y_pred_bin),
        f1=f1_score(y_true, y_pred_bin),
        y_pred=np.array(y_pred_bin),
        y_true=np.array(y_true)
    )


def train_binary_graph_task(
    graphs: Iterable,
    model: nn.Module,
    epochs: int,
    lr: float,
    dataset_name="",
) -> TrainingRun:
    """Graphs iterable requires G.y G.edge_index and G.x"""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, test_losses = [], []
    best_model = None
    best_loss = float("inf")

    # do a random split of 80-20 for train and test
    # but save the indexes from the original sequence used for testing
    graphs = list(graphs)
    num_graphs = len(graphs)
    indices = np.arange(num_graphs)
    np.random.shuffle(indices)
    split_idx = int(0.8 * num_graphs)
    train_graphs = [graphs[i] for i in indices[:split_idx]]
    test_indices =  indices[split_idx:]
    test_graphs = [graphs[i] for i in test_indices]

    for epc in range(1, epochs + 1):

        model.train()
        total_loss = 0.0
        for G in tqdm(train_graphs):
            optimizer.zero_grad()

            # x has a shape of 1 so add an empty feature dimension
            if G.x.dim() == 1:
                G.x = G.x.unsqueeze(1)

            logit = model(G.x, G.edge_index)
            logit = logit.view(-1)
            y = G.y.float().view(-1)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_graphs))

        # ---- eval ----
        model.eval()
        total_test_loss = 0.0
        y_true, y_scores = [], []
        with torch.no_grad():
            for G in tqdm(test_graphs):
                logit = model(G.x, G.edge_index).view(-1)
                y = G.y.float().view(-1)
                loss = criterion(logit, y)
                total_test_loss += loss.item()
                prob = torch.sigmoid(logit)
                y_true.extend(y.cpu().tolist())
                y_scores.extend(prob.cpu().tolist())

        avg_test_loss = total_test_loss / len(test_graphs)
        test_losses.append(avg_test_loss)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model = deepcopy(model)

        logger.info(f"[{dataset_name}] Epoch {epc}/{epochs} | train {train_losses[-1]:.4f} | test {avg_test_loss:.4f}")

    performance = evaluate_binary_predictions(y_true, y_scores)

    return TrainingRun(
        best_model=best_model,
        current_model=model,
        dataset=dataset_name,
        train_losses=train_losses,
        test_losses=test_losses,
        performance=performance,
        node_level_task=False,
        hyperparameter={"lr": lr, "epochs": epochs},
        test_indexes=test_indices,
    )

        







