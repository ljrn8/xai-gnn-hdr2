import torch
from torch import nn
import numpy as np
import pickle
import sys
from dataclasses import dataclass
from typing import Optional, Iterable
from models import NodeGCN
from torch_geometric.data import Data
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from copy import deepcopy
import torch.optim as optim
from tqdm import tqdm
from pprint import pprint

def openpkl(file):
    logger.info(f'Opening file: {file}')
    with open(file, 'rb') as f:
        file = pickle.load(f)
        logger.info(f'file size: {sys.getsizeof(file)} bytes')
        return file
    
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
    node_level_task: bool
    train_losses: list
    val_losses: list
    val_performance: Optional[ModelPerformance] = None
    hyperparameter: Optional[dict] = None 
    test_indexes: Optional[np.ndarray] = None
    epochs_trained: int = 0

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
    train_graphs: Iterable,
    val_graphs: Iterable,
    model: nn.Module,
    epochs: int,
    lr: float,
    patience: int = 10,
    delta: float = 0.001,
    dataset_name="",
    criterion=None,
) -> TrainingRun:
    """ Trains a binary graph classification model.
    (Graphs iterable require pyGeometric style G.y G.edge_index and G.x alone)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not criterion:
        # crierion is weight BCE loss to unbalanced y (y is 0 or 1 per graph btw)
        all_labels = torch.cat([G.y.view(-1) for G in train_graphs], dim=0)
        pos_weight = (all_labels == 0).sum() / (all_labels == 1).sum()
        logger.info(f'Using weighted BCE loss with pos_weight: {pos_weight:.4f}')
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses, val_losses = [], []
    best_model = None
    best_loss = float("inf")
    early_stopping_timer = 0
    best_epoch = 0


    pbar = tqdm(range(1, epochs + 1), desc=f"[{dataset_name}] w/ {epochs} epochs")
    for epc in pbar:

        # --- training ---

        model.train()
        total_loss = 0.0
        for G in train_graphs:
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

        # ---- validation ----

        model.eval()
        total_val_loss = 0.0
        y_true, y_scores = [], []
        with torch.no_grad():
            for G in val_graphs:
                logit = model(G.x, G.edge_index).view(-1)
                y = G.y.float().view(-1)
                loss = criterion(logit, y)
                total_val_loss += loss.item()
                prob = torch.sigmoid(logit)
                y_true.extend(y.cpu().tolist())
                y_scores.extend(prob.cpu().tolist())

        avg_val_loss = total_val_loss / len(val_graphs)
        val_losses.append(avg_val_loss)

        # Early stopping check
        if  best_loss - avg_val_loss < delta:
            if early_stopping_timer >= patience:
                logger.info(f"Early stopping at epoch {epc} with best epoch {best_epoch} and best loss {best_loss:.4f}")
                break
            
            early_stopping_timer += 1
        else:
            early_stopping_timer = 0

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = deepcopy(model)
            best_epoch = epc

        description = f"[{dataset_name}] Epoch {epc}/{epochs} | train {train_losses[-1]:.4f} | val {avg_val_loss:.4f}"
        pbar.set_description(description)


    return TrainingRun(
        best_model=best_model,
        current_model=model,
        dataset=dataset_name,
        train_losses=train_losses,
        val_losses=val_losses,
        val_performance=evaluate_binary_predictions(y_true, y_scores),
        epochs_trained=epc,
        node_level_task=False,
    )

        







