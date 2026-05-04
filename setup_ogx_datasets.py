"""
Assembles consistant Train/val/test split and OHE for OGX datasets
"""
import os
import sys
import torch
sys.path.append('../benchmarks'); import ogx_datasets_pyg
import torch.nn.functional as F
from loguru import logger 
import pickle
import numpy as np

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
    
    logger.info(f' Dataset name: {ds_name}')
    ds = ogx_datasets_pyg.OGXBenchmark(root='/tmp', name=ds_name)

    # TODO: try without OHE aswell

    x_pool = torch.cat([G.x for G in ds], dim=0)
    num_classes = int(x_pool.max().item()) + 1

    def OHE_graph(G):
        assert G.x.dim() == 1, "Expected node features to be 1D for OHE encoding"
        assert int(G.x.max().item()) + 1 < 1000, "Expected at less than 1000 classes for OHE encoding"
        G.x = F.one_hot(G.x, num_classes=num_classes).float()
        return G

    logger.info('applying OHE')
    logger.info(f'Original graph x shape: {ds[0].x.shape}')
    graphs = [OHE_graph(G) for G in ds]
    logger.info(f'Processed graph x shape: {graphs[0].x.shape}')

    # split graphs randomly into train val test 70/20/10, never touching test indexes
    num_graphs = len(graphs)
    indices = list(range(num_graphs))
    np.random.shuffle(indices)
    split_idx1 = int(0.7 * num_graphs)
    split_idx2 = int(0.9 * num_graphs)
    train_indexes = indices[:split_idx1]
    val_indexes = indices[split_idx1:split_idx2]
    test_indexes = indices[split_idx2:]
    train_graphs = [graphs[i] for i in train_indexes]
    val_graphs = [graphs[i] for i in val_indexes]
    test_graphs = [graphs[i] for i in test_indexes]

    root = f'./interm/ogx/{ds_name}'
    os.makedirs(root, exist_ok=True)

    with open(os.path.join(root, 'train_graphs.pkl'), 'wb') as f:
        pickle.dump(train_graphs, f)
    with open(os.path.join(root, 'val_graphs.pkl'), 'wb') as f:
        pickle.dump(val_graphs, f)
    with open(os.path.join(root, 'test_graphs.pkl'), 'wb') as f:
        pickle.dump(test_graphs, f)
