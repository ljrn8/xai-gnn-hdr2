from ..explainer_utils import Explainer, elementwise_entropy, uniform_debug_log
from src.stable_gnn_xai.training.train import GNN
from torch_geometric.data import Data
from typing_extensions import Iterable
import torch
from  tqdm import tqdm
from loguru import logger

class GNNExplainer(Explainer):
    """Classic GNNExplainer Implementation for binary graph classifiers (does not include feature extractor)"""

    def __init__(
        self,
        model: GNN,
        graphs: list[Data],
        epochs,
        lr,
        mean_regularization,
        entropy_regularization,
        loss_f=torch.nn.BCELoss(),
    ):
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.graphs = graphs
        self.epochs = epochs
        self.lr = lr
        self.mean_regularization = mean_regularization
        self.entropy_regularization = entropy_regularization
        self.loss_f = loss_f
        

    def explain_graph_task(self):
        logger.info(f'GNNE About to sequentially fit {len(self.graphs)} graphs')
        pbar = tqdm(self.graphs)
        masks = []
        for i, G in enumerate(pbar):
            num_edges = G.edge_index.size(1)
            parameterized_edge_mask = torch.nn.Parameter(
                torch.randn(num_edges) * 0.1
            )
            original_y_pred = self.model.forward(G.x, G.edge_index).detach().view(-1).sigmoid()
            optimizer = torch.optim.Adam([parameterized_edge_mask], lr=self.lr)
            for epoch in range(1, self.epochs+1):
                edge_mask = parameterized_edge_mask.sigmoid()
                optimizer.zero_grad()
                explanatatory_y_preds = self.model.forward(
                    G.x, G.edge_index, edge_weight=edge_mask
                ).view(-1).sigmoid()
                loss = self.loss_f(explanatatory_y_preds, original_y_pred)
                entr_reg = self.entropy_regularization * elementwise_entropy(edge_mask).mean()
                mean_reg = self.mean_regularization * edge_mask.mean()
                loss += entr_reg + mean_reg
                loss.backward()
                pbar.set_description(
                    f"GNNE: Graph={i}/{len(self.graphs)} Epoch={epoch}/{self.epochs} | " + 
                    f"params.grad={edge_mask.grad}" +
                    f"BCE loss={loss.item():.5f} entr_reg={entr_reg.item():.5f} mean_reg={mean_reg.item():.5f} " 
                )
                optimizer.step()

            masks.append(torch.Tensor(parameterized_edge_mask.sigmoid().detach()))

        uniform_debug_log(masks)
        return masks



    def explain_node_task(self):
        raise NotImplementedError()



