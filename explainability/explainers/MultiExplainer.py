from .PGExplainer import PGExplainer
import torch
from explainability.explainer_utils import (
    knee_threshold,
    otsu_threshold,
    quantile_threshold,
    minimum_cluster_distance_threshhold,
)


class MultiExplainer(PGExplainer):

    def __init__(
        self,
        epochs,
        hidden_size,
        lr,
        mean_regularization,
        entropy_regularization,
        tau,
        reparameterization_samples,
        loss_f=torch.nn.BCELoss,
        threshholder_function=otsu_threshold,
    ):

        super().__init__(
            epochs,
            hidden_size,
            lr,
            mean_regularization,
            entropy_regularization,
            tau,
            reparameterization_samples,
            loss_f,
        )

        self.threshholder_function = threshholder_function

    def explain_graph_task(self, model, graphs):
        while True:
            masks = super().explain_graph_task(model, graphs)
            hard_masks = [m > self.threshholder_function(m) for m in masks]

            ...
