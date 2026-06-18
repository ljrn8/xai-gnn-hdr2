import torch
from torch import nn
import numpy as np
import pickle
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
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
from ..interfaces import TrainingRun

from src.gnn_xai.training.models import *




def evaluate_binary_predictions(y_true, y_scores) -> dict:
    y_pred_bin = [1 if p > 0.5 else 0 for p in y_scores]
    return {
        'roc_auc': roc_auc_score(y_true, y_scores),
        'pr_auc': average_precision_score(y_true, y_scores),
        'rec': recall_score(y_true, y_pred_bin),
        'prec': precision_score(y_true, y_pred_bin),
        'acc': accuracy_score(y_true, y_pred_bin),
        'f1': f1_score(y_true, y_pred_bin),
    }


def run_graphs(graphs: Iterable[Data], optimizer=None):
    training = optimizer is not None
    self.model.train() if training else self.model.eval()
    total_loss = 0.0
    y_true, y_scores = [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for G in graphs:
            if G.x.dim() == 1:
                G.x = G.x.unsqueeze(1)

            logit = self.model(G.x, G.edge_index).view(-1)
            y = G.y.float().view(-1)
            loss = self.criterion(logit, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            prob = torch.sigmoid(logit)
            y_true.extend(y.cpu().tolist())
            y_scores.extend(prob.detach().cpu().tolist())

    return total_loss / len(graphs), y_true, y_scores


def train(
    model: GNN,
    dataset_root: str,
    epochs: int,
    lr: float,
    patience: int = 20,
    delta: float = 0.001,
) -> TrainingRun:
    
    """
    Generic training loop that works with any GraphTask subclass.
    Uses early stopping on validation loss.
    """
    optimizer = optim.Adam(task.model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    best_task = None
    best_loss = float("inf")
    early_stopping_counter = 0
    best_epoch = 0

    pbar = tqdm(range(1, epochs + 1))
    for epc in pbar:

        train-loss, y_true, y_scores = run_graphs(
            graphs=...
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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
            best_task = deepcopy(task)
            best_epoch = epc

        pbar.set_description(
            f"[epoch {epc}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f}"
        )

    # Evaluate best model on test split
    task = best_task
    _, y_true, y_scores = task.evaluate_test()

    return y_scores, train_loss, val_loss
