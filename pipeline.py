import sys
from pprint import pprint
sys.path.append('../benchmarks')

import ogx_datasets_pyg

dataset = ogx_datasets_pyg.OGXBenchmark(root='/tmp', name='alfa')
print(dataset)
print(dataset.mask_root)
print(dataset.mask)
print(dataset.y)
pprint(dataset.__dict__)
print('\n')

dataset = ogx_datasets_pyg.OGXBenchmark(root='/tmp', name='bravo')
print(dataset)
print(dataset.mask_root)
print(dataset.mask)
print(dataset.y)
pprint(dataset.__dict__)
print('\n')

dataset = ogx_datasets_pyg.OGXBenchmark(root='/tmp', name='india')
print(dataset)
print(dataset.mask_root)
print(dataset.mask)
print(dataset.y)
pprint(dataset.__dict__) 
print('\n')

dataset = ogx_datasets_pyg.OGXBenchmark(root='/tmp', name='oscar')
print(dataset)
print(dataset.mask_root)
print(dataset.mask)
print(dataset.y)
pprint(dataset.__dict__)
print('\n')
