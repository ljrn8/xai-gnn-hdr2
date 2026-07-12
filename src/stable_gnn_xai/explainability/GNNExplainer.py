from .utils import elementwise_entropy, uniform_debug_log
from ..interfaces import GNN, GraphLevelExplainer
from torch_geometric.data import Data
from typing_extensions import Iterable
import torch
from tqdm import tqdm
from loguru import logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNExplainer(GraphLevelExplainer):
    """Classic GNNExplainer Implementation for binary graph classifiers (does not include feature extractor)"""

    def __init__(
        self,
        epochs,
        lr,
        mean_regularization,
        entropy_regularization,
        loss_f=torch.nn.BCELoss(),
    ):
        self.model.eval()
        self.epochs = epochs
        self.lr = lr
        self.mean_regularization = mean_regularization
        self.entropy_regularization = entropy_regularization
        self.loss_f = loss_f
        self.example_loss_curves = {'BCELoss': [], 'entropy_regularization': [], 'mean_regularization': []}

    def explain_graph_task(self, model, graphs):
        for p in model.parameters():
            p.requires_grad_(False)

        model.to(DEVICE)

        logger.info(f"GNNE About to sequentially fit {len(graphs)} graphs")
        pbar = tqdm(graphs)
        masks = []

        for i, G in enumerate(pbar):
            num_edges = G.edge_index.size(1)
            parameterized_edge_mask = torch.nn.Parameter(torch.randn(num_edges).to(DEVICE))
            optimizer = torch.optim.Adam([parameterized_edge_mask], lr=self.lr)

            original_y_pred = (
                model.forward(G.x, G.edge_index).detach().view(-1).sigmoid()
            )
            
            for epoch in range(1, self.epochs + 1):
                optimizer.zero_grad()
                explanatatory_y_preds = (
                    model.forward(G.x, G.edge_index, edge_weight=parameterized_edge_mask.sigmoid())
                    .view(-1)
                    .sigmoid()
                )
                loss = self.loss_f(explanatatory_y_preds, original_y_pred) 
                entr_reg = (
                    self.entropy_regularization * elementwise_entropy(parameterized_edge_mask.sigmoid()).mean()
                )
                mean_reg = self.mean_regularization * parameterized_edge_mask.sigmoid().mean()
                loss += entr_reg + mean_reg
                loss.backward()

                # diagnostics    
                if i == 0:            
                    self.example_loss_curves['BCELoss'].append(loss.item())
                    self.example_loss_curves['entropy_regularization'].append(entr_reg.item())
                    self.example_loss_curves['mean_regularization'].append(mean_reg.item())
                
                pbar.set_description(
                    f"GNNE: Graph={i}/{len(self.graphs)} Epoch={epoch}/{self.epochs} | "
                    + f"params.grad[0]={parameterized_edge_mask.grad[0]:.5f} | "
                    + f"BCE loss={loss.item():.5f} entr_reg={entr_reg.item():.5f} mean_reg={mean_reg.item():.5f} "
                )
                optimizer.step()

            masks.append(torch.Tensor(parameterized_edge_mask.sigmoid().detach()))

        uniform_debug_log(masks)
        return masks, loss.item()

