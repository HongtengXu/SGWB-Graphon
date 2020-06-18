import torch
import Methods.Baselines as baselines
import time

graphs = torch.rand(100, 100, 10)

since = time.time()
graphon1 = baselines.sort_smooth(graphs)
print('Sort and smooth: {:.2f}sec'.format(time.time() - since))

since = time.time()
graphon2 = baselines.largest_gap(graphs, num_blocks=10)
print('Largest gap: {:.2f}sec'.format(time.time() - since))

since = time.time()
graphon3 = baselines.matrix_completion(graphs)
print('Matrix completion: {:.2f}sec'.format(time.time() - since))

since = time.time()
graphon4 = baselines.universal_svd(graphs)
print('USVD: {:.2f}sec'.format(time.time() - since))

since = time.time()
graphon5 = baselines.estimate_blocks_directed(graphs, threshold=0.1)
print('SBA: {:.2f}sec'.format(time.time() - since))