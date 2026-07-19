from .utils import elementwise_entropy, uniform_debug_log
from ..interfaces import GNN, GraphLevelExplainer
from torch_geometric.data import Data
from typing_extensions import Iterable
import torch
from tqdm import tqdm
from loguru import logger
from torch_geometric.data import Data, Batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNExplainer(GraphLevelExplainer):
    """Classic GNNExplainer, parallelized across the graph dimension.

    Instead of a separate nn.Parameter + optimizer per graph run sequentially,
    all edge masks are packed into one flat parameter aligned with a single
    `Batch.from_data_list(graphs)`, and every epoch does exactly one batched
    forward/backward over all graphs at once. This is a pure parallelization
    of the same optimization problem, not an algorithmic change: Adam's moment
    estimates are elementwise, so one optimizer over the concatenated
    parameter is identical per-edge to `len(graphs)` independent optimizers.
    Regularizers are still computed per-graph (split by edge_counts) and
    averaged across graphs, matching the original's equal per-graph weighting
    rather than weighting by edge count.
    """

    def __init__(
        self,
        epochs=50,
        lr=0.01,
        mean_regularization=0.1,
        entropy_regularization=0.05,
        loss_f=torch.nn.BCELoss(),
    ):
        self.epochs = epochs
        self.lr = lr
        self.mean_regularization = mean_regularization
        self.entropy_regularization = entropy_regularization
        self.loss_f = loss_f
        self.example_loss_curves = {'BCELoss': [], 'entropy_regularization': [], 'mean_regularization': []}


    def explain_graph_task(self, model, graphs):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(DEVICE)

        n_graphs = len(graphs)
        edge_counts = [G.edge_index.shape[1] for G in graphs]
        total_edges = sum(edge_counts)

        logger.info(f"GNNE: batching {n_graphs} graphs ({total_edges} edges total) into one forward per epoch")

        batch_obj = Batch.from_data_list(graphs).to(DEVICE)

        # one flat mask, segments in the same order Batch.from_data_list concatenates edge_index
        parameterized_edge_mask = torch.nn.Parameter(torch.randn(total_edges, device=DEVICE))
        optimizer = torch.optim.Adam([parameterized_edge_mask], lr=self.lr)

        with torch.no_grad():
            original_y_pred = torch.sigmoid(
                model.forward(batch_obj.x, batch_obj.edge_index, batch=batch_obj.batch).view(-1)
            ).detach()  # (n_graphs,)

        pbar = tqdm(range(1, self.epochs + 1))
        for epoch in pbar:
            optimizer.zero_grad()

            soft_mask = parameterized_edge_mask.sigmoid()  # (total_edges,)

            explanatory_y_preds = torch.sigmoid(
                model.forward(
                    batch_obj.x,
                    batch_obj.edge_index,
                    edge_weight=soft_mask,
                    batch=batch_obj.batch,
                ).view(-1)
            )  # (n_graphs,)

            loss = self.loss_f(explanatory_y_preds, original_y_pred)

            # split flat mask back into per-graph segments for equally-weighted regularization
            per_graph_masks = torch.split(soft_mask, edge_counts)
            entr_reg = self.entropy_regularization * torch.stack(
                [elementwise_entropy(m).mean() for m in per_graph_masks]
            ).mean()
            mean_reg = self.mean_regularization * torch.stack(
                [m.mean() for m in per_graph_masks]
            ).mean()

            self.example_loss_curves['BCELoss'].append(loss.item())
            self.example_loss_curves['entropy_regularization'].append(entr_reg.item())
            self.example_loss_curves['mean_regularization'].append(mean_reg.item())

            loss += entr_reg + mean_reg
            loss.backward()

            pbar.set_description(
                f"GNNE: Epoch={epoch}/{self.epochs} | "
                + f"mask.grad[0]={parameterized_edge_mask.grad[0]:.5f} | "
                + f"BCE loss={loss.item():.5f} entr_reg={entr_reg.item():.5f} mean_reg={mean_reg.item():.5f}"
            )
            optimizer.step()

        masks = [m.detach().clone() for m in torch.split(parameterized_edge_mask.sigmoid(), edge_counts)]
        uniform_debug_log(masks)
        return masks, loss.item()

