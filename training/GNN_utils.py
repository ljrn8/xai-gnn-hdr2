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

from training.models import *


def openpkl(file):
    logger.info(f"Opening file: {file}")
    with open(file, "rb") as f:
        data = pickle.load(f)
        logger.info(f"file size: {sys.getsizeof(data)} bytes")
        return data


# --------------
# Task Interfaces


class GNN(ABC, nn.Module):
    @abstractmethod
    def forward(self, x, edge_index, return_all_embeddings=False, edge_weight=None):
        """Returns node-level logits"""

    # TODO: add this
    # enforce things like .lin?


class GraphTask(ABC):
    def __init__(self, model: nn.Module, criterion: nn.Module):
        self.model = model
        self.criterion = criterion

    @abstractmethod
    def fit(self, optimizer: optim.Optimizer) -> float:
        """Run one training step. Returns scalar loss."""

    @abstractmethod
    def evaluate(self) -> tuple[float, any, any]:
        """Run inference. Returns (loss, y_true, y_scores)."""

    @abstractmethod
    def evaluate_test(self) -> tuple[float, any, any]:
        """Run inference on the held-out test split."""


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
    task: GraphTask
    dataset_name: str
    train_losses: list
    val_losses: list
    y_pred: torch.Tensor
    y_true: torch.Tensor
    criterion: nn.Module
    val_performance: Optional[ModelPerformance] = None
    hyperparameter: Optional[dict] = None
    epochs_trained: int = 0


# --------------
# Evaluation


def evaluate_binary_predictions(y_true, y_scores) -> ModelPerformance:
    y_pred_bin = [1 if p > 0.5 else 0 for p in y_scores]
    return ModelPerformance(
        roc_auc=roc_auc_score(y_true, y_scores),
        pr_auc=average_precision_score(y_true, y_scores),
        rec=recall_score(y_true, y_pred_bin),
        prec=precision_score(y_true, y_pred_bin),
        acc=accuracy_score(y_true, y_pred_bin),
        f1=f1_score(y_true, y_pred_bin),
    )


def evaluate_multiclass_predictions(y_true, y_scores) -> ModelPerformance:
    y_pred_bin = np.argmax(y_scores, axis=1)

    # Drop score dimensions for classes absent from both y_true and y_pred
    present_labels = set(y_pred_bin.cpu().numpy()) & set(y_true.cpu().numpy())
    excess_dims = set(range(y_scores.shape[1])) - present_labels
    y_scores = np.delete(y_scores.cpu(), list(excess_dims), axis=1)
    y_scores = torch.softmax(torch.tensor(y_scores), dim=1).numpy()

    return ModelPerformance(
        roc_auc=roc_auc_score(y_true, y_scores, multi_class="ovr"),
        pr_auc=average_precision_score(y_true, y_scores, average="macro"),
        rec=recall_score(y_true, y_pred_bin, average="macro"),
        prec=precision_score(y_true, y_pred_bin, average="macro"),
        acc=accuracy_score(y_true, y_pred_bin),
        f1=f1_score(y_true, y_pred_bin, average="macro"),
    )


class InductiveGraphClassification(GraphTask):
    def __init__(
        self,
        model,
        criterion,
        train_graphs: Iterable,
        val_graphs: Iterable,
        test_graphs: Iterable,
    ):
        super().__init__(model, criterion)
        self.train_graphs = list(train_graphs)
        self.val_graphs = list(val_graphs)
        self.test_graphs = list(test_graphs)

    def _run_graphs(self, graphs, optimizer=None) -> tuple[float, list, list]:
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

    def fit(self, optimizer) -> float:
        loss, _, _ = self._run_graphs(self.train_graphs, optimizer=optimizer)
        return loss

    def evaluate(self):
        return self._run_graphs(self.val_graphs)

    def evaluate_test(self):
        return self._run_graphs(self.test_graphs)


class TransductiveNodeClassification(GraphTask):
    def __init__(self, model, criterion, graph):
        super().__init__(model, criterion)
        self.graph = graph

    def _run_mask(self, mask, optimizer=None) -> tuple[float, any, any]:
        training = optimizer is not None
        self.model.train() if training else self.model.eval()

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            logit = self.model(self.graph.x, self.graph.edge_index)
            logit_masked = logit[mask, :]
            y = self.graph.y.long()[mask]
            loss = self.criterion(logit_masked, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            prob = torch.softmax(logit_masked, dim=1)

        return loss.item(), y.cpu(), prob.detach().cpu()

    def fit(self, optimizer) -> float:
        loss, _, _ = self._run_mask(self.graph.train_mask, optimizer=optimizer)
        return loss

    def evaluate(self):
        return self._run_mask(self.graph.val_mask)

    def evaluate_test(self):
        return self._run_mask(self.graph.test_mask)


def train(
    task: GraphTask,
    epochs: int,
    lr: float,
    dataset_name: str = "",
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
    last_epoch = 0

    pbar = tqdm(range(1, epochs + 1), desc=f"[{dataset_name}]")
    for epc in pbar:
        last_epoch = epc
        train_loss = task.fit(optimizer)
        val_loss, y_true, y_scores = task.evaluate()

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
            f"[{dataset_name}] epoch {epc}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f}"
        )

    # Evaluate best model on test split
    task = best_task
    _, y_true, y_scores = task.evaluate_test()

    return TrainingRun(
        task=task,
        dataset_name=dataset_name,
        train_losses=train_losses,
        val_losses=val_losses,
        epochs_trained=last_epoch,
        criterion=task.criterion,
        y_true=y_true,
        y_pred=y_scores,
    )
