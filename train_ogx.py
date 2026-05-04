import os
import sys
import torch
sys.path.append('../benchmarks'); import ogx_datasets_pyg
import torch.nn.functional as F
from loguru import logger 
from models import GraphTaskNodeGCN2
from training_utils import train_binary_graph_task, TrainingRun
from pprint import pprint
import pickle

for ds_name in [
    'delta', 
    'echo', 
    'foxtrot',
    'golf',
    'hotel',
    'india',
    'juliett',
    'kilo',
    'lima',
    'mike',
    'november',
    'oscar',
]:

    logger.info(f'\n\n ===== Dataset name: {ds_name} =====\n')
    ds = ogx_datasets_pyg.OGXBenchmark(root='/tmp', name=ds_name)

    # ---  OHE

    x_pool = torch.cat([G.x for G in ds], dim=0)
    num_classes = int(x_pool.max().item()) + 1

    def OHE_graph(G):
        assert G.x.dim() == 1, "Expected node features to be 1D for OHE encoding"
        assert int(G.x.max().item()) + 1 < 1000, "Expected at less than 1000 classes for OHE encoding"
        G.x = F.one_hot(G.x, num_classes=num_classes).float()
        return G

    logger.info('applying OHE')
    logger.info(f'Original graph x shape: {ds[0].x.shape}')
    process_graphs = [OHE_graph(G) for G in ds]
    logger.info(f'Processed graph x shape: {process_graphs[0].x.shape}')

    # --- train

    model = GraphTaskNodeGCN2(
        input_feat=process_graphs[0].x.shape[1],
        hidden_channels=64, 
        output_graph_channels=1,
    )

    logger.info(f'Model: {model}')

    run: TrainingRun = train_binary_graph_task(
        graphs=process_graphs,
        dataset_name=ds_name,
        model=model,
        epochs=30,
        lr=0.001,
    )

    logger.info('run:')
    pprint(run.performance)

    root = f'output/OGX/{ds_name}'
    os.makedirs(root, exist_ok=True)
    with open(f'{root}/GCN2.pkl', 'wb') as f:
        pickle.dump(run, f)
