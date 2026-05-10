import argparse
from pathlib import Path
from notebooks_archive.training_utils import TrainingRun
import os
from loguru import logger
from notebooks_archive.training_utils import openpkl
from models import NodeGCN, NodeGIN, GraphTaskFromNodeModel

parser = argparse.ArgumentParser(description="Train GNNs on binary graph classification tasks")
parser.add_argument('--ds-root', default=Path("interm/") / "RW graph classification")
args = parser.parse_args()

root_dir = Path(args.ds_root)
for dset in os.listdir(root_dir):
    path = root_dir / dset
    logger.info(f"\n\n Dataset name: {path.name} \n")
    train_graphs = openpkl(path / "train_graphs.pkl")
    val_graphs = openpkl(path / "val_graphs.pkl")
    best_run: TrainingRun = None
    num_features = train_graphs[0].x.shape[1]
    
    for model_name, model_class, hyperparameter_candidates in MODEL_CONFIGS:
        logger.info(f" --- Training model: {model_name} ")
        for hp in hyperparameter_candidates:
            model = GraphTaskFromNodeModel(
                node_model=model_class(
                    input_feat=num_features,
                    hidden_channels=hp["hidden_channels"],
                    num_layers=hp.get("num_layers"),
                    output_channels=hp["hidden_channels"],
                ),
                incoming_channels=hp["hidden_channels"],
                output_graph_channels=1,
                dropout=None,
            )

            logger.info(f"Model: {model}")

