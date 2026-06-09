from explainability.explainer_utils import Explainer
from training.GNN_utils import GNN
from torch_geometric.data import Data
from typing_extensions import Iterable
import torch


class GNNExplainer(Explainer):
    def __init__(
        self,
        epochs,
        lr,
        mean_regularization,
        entropy_regularization,
        reparameterization_samples,
        loss_f=torch.nn.BCELoss(),
    ):
        self.reparameterization_samples = reparameterization_samples
        self.epochs = epochs
        self.lr = lr
        self.mean_regularization = mean_regularization
        self.entropy_regularization = entropy_regularization
        self.loss_f = loss_f

    def _get_predictions(self, model, graphs):
        y_preds = []
        for G in graphs:
            if G.x.dim() == 1:
                G.x = G.x.unsqueeze(1)

            logit, emb = model(G.x, G.edge_index, return_all_embeddings=False)
            y_preds.append(torch.sigmoid(logit))

        return torch.cat(y_preds)

    def _run_mask_epoch(self, W, G: Data):
        src, dst = G.edge_index
        edge_concatenated_embeddings = torch.cat(
            (final_embedding[src], final_embedding[dst]), dim=1
        )
        edge_mask_logits = mlp(edge_concatenated_embeddings)
        y_pred = self._batched_estimate_masked_prediction(
            model, G, edge_mask_logits, samples=self.reparameterization_samples
        )
        explanatory_y_preds.append(y_pred)
        soft_edge_mask = torch.sigmoid(edge_mask_logits)
        masks.append(soft_edge_mask)
        entropy_reg += (
            self.entropy_regularization * elementwise_entropy(soft_edge_mask).mean()
        )
        mean_reg += self.mean_regularization * soft_edge_mask.mean()

    def explain_graph_task(self, model: GNN, graphs: Iterable[Data]):
        n_features = graphs[0].x.shape[1]
        W = torch.nn.Linear(in_features=n_features, out_features=1)
        optimizer = torch.optim.Adam(W, lr=self.lr)
        y_preds = self._get_predictions(model, graphs).detach()

        pbar = tqdm(range(1, self.epochs + 1))
        for epc in pbar:
            optimizer.zero_grad()
