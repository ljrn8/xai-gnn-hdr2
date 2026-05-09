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
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from copy import deepcopy
import torch.optim as optim
from tqdm import tqdm
from pprint import pprint
from models import *


def openpkl(file):
    logger.info(f"Opening file: {file}")
    with open(file, "rb") as f:
        file = pickle.load(f)
        logger.info(f"file size: {sys.getsizeof(file)} bytes")
        return file

@dataclass
class ModelPerformance:
    roc_auc: float
    pr_auc: float
    rec: float
    prec: float
    acc: float
    f1: float

@dataclass
class TrainingRun:
    best_model: nn.Module
    current_model: nn.Module
    dataset_name: str
    train_losses: list
    val_losses: list
    y_pred: torch.Tensor 
    y_true: torch.Tensor
    criterion: nn.Module
    transductive: Optional[bool] = None
    val_performance: Optional[ModelPerformance] = None
    hyperparameter: Optional[dict] = None
    epochs_trained: int = 0

def evaluate_multiclass_predictions(y_true, y_scores):
    y_pred_bin = np.argmax(y_scores, axis=1)
    
    # what labels do not exist in y_true or y_pred_bin, but are a dimension in y_scores?
    training_only_labels = set(y_pred_bin.cpu().numpy()) & set(y_true.cpu().numpy())
    excess_dimension = set(range(y_scores.shape[1])) -  training_only_labels
    y_scores = np.delete(y_scores.cpu(), list(excess_dimension), axis=1)

    # reapply softmax
    y_scores = torch.softmax(torch.tensor(y_scores), axis=1).numpy()

    return ModelPerformance(
        roc_auc=roc_auc_score(y_true, y_scores, multi_class="ovr"),
        pr_auc=average_precision_score(y_true, y_scores, average="macro"),
        rec=recall_score(y_true, y_pred_bin, average="macro"),
        prec=precision_score(y_true, y_pred_bin, average="macro"),
        acc=accuracy_score(y_true, y_pred_bin),
        f1=f1_score(y_true, y_pred_bin, average="macro"),
    )

def evaluate_binary_predictions(y_true, y_scores):
    y_pred_bin = [1 if p > 0.5 else 0 for p in y_scores]
    return ModelPerformance(
        roc_auc=roc_auc_score(y_true, y_scores),
        pr_auc=average_precision_score(y_true, y_scores),
        rec=recall_score(y_true, y_pred_bin),
        prec=precision_score(y_true, y_pred_bin),
        acc=accuracy_score(y_true, y_pred_bin),
        f1=f1_score(y_true, y_pred_bin),
    )

def fit_inductive_graphs(Graphs, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for G in Graphs:
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

    return total_loss / len(Graphs)

def eval_inductive_graphs(Graphs, model, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_scores = [], []
    with torch.no_grad():
        for G in Graphs:
            logit = model(G.x, G.edge_index).view(-1)
            y = G.y.float().view(-1)
            loss = criterion(logit, y)
            total_loss += loss.item()
            prob = torch.sigmoid(logit)
            y_true.extend(y.cpu().tolist())
            y_scores.extend(prob.cpu().tolist())

    return total_loss / len(Graphs), y_true, y_scores

def fit_transductive_node_task(graph, model, criterion, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    logit = model(graph.x, graph.edge_index)
    # logger.warning(f"logit shape: {logit.shape}, graph.x shape: {graph.x.shape}, graph.y shape: {graph.y.shape}, train_mask shape: {train_mask.shape}")  # Debugging line
    logit = logit[train_mask, :]
    y = graph.y.long()[train_mask]
    loss = criterion(logit, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_transductive_node_task(graph, model, criterion, eval_mask):
    model.eval()
    with torch.no_grad():
        logit = model(graph.x, graph.edge_index)[eval_mask, :]
        y = graph.y.long()[eval_mask]
        loss = criterion(logit, y)
        prob = torch.softmax(logit, axis=1)
        y_true = y.cpu()
        y_scores = prob.cpu()

    return loss.item(), y_true, y_scores

def GNN_task(model: nn.Module, epochs: int, lr: float, criterion, single_graph = None, 
             train_graphs: Optional[Iterable] = False, val_graphs: Optional[Iterable] = False, 
             patience: int = 20, delta: float = 0.001, dataset_name="", transductive=False) -> TrainingRun:

    if transductive:
        assert single_graph, "Transductive learning requires a single graph with train_mask, val_mask and test_mask attributes"
    else:
        assert train_graphs and val_graphs, "Inductive learning requires train and val graph iterables"

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    best_model = None
    best_loss = float("inf")
    early_stopping_timer = 0
    best_epoch = 0

    pbar = tqdm(range(1, epochs + 1), desc=f"[{dataset_name}] w/ {epochs} epochs")
    for epc in pbar:

        if transductive:
            train_loss = fit_transductive_node_task(single_graph, model, criterion, optimizer, train_mask=single_graph.train_mask)
            val_loss, y_true, y_scores = eval_transductive_node_task(single_graph, model, criterion, eval_mask=single_graph.val_mask)
        else:
            train_loss = fit_inductive_graphs(train_graphs, model, criterion, optimizer)
            val_loss, y_true, y_scores = eval_inductive_graphs(val_graphs, model, criterion)

        val_losses.append(val_loss)
        train_losses.append(train_loss)

        # Early stopping check
        if best_loss - val_loss < delta:
            if early_stopping_timer >= patience:
                logger.info(
                    f"Early stopping at epoch {epc} with best epoch {best_epoch} and best loss {best_loss:.4f}"
                )
                break

            early_stopping_timer += 1
        else:
            early_stopping_timer = 0

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(model)
            best_epoch = epc

        description = f"[{dataset_name}] Epoch {epc}/{epochs} | train {train_losses[-1]:.4f} | val {val_loss:.4f}"
        pbar.set_description(description)

    return TrainingRun(
        best_model=best_model,
        current_model=model,
        dataset_name=dataset_name,
        train_losses=train_losses,
        val_losses=val_losses,
        epochs_trained=epc,
        transductive=transductive,
        criterion=criterion,
        y_true=y_true,
        y_pred=y_scores,
    )

