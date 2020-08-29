# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
# from torch_geometric.transforms import ToDense

# data_names = ['COLLAB', 'deezer_ego_net', 'IMDB-BINARY', 'IMDB-MULTI',
#               'REDDIT_BINARY', 'REDDIT_MULTI', 'reddit_threads', 'twitch_egos']
#
# for name in data_names:
#     dataset = TUDataset('data', name=name)

# dataloader = DataLoader(dataset, batch_size=dataloader_batch_size)
#         get_adjacency = ToDense() # this is a class which takes edge_index as input and output adjacency matrix

import os
import methods.loader as loader

data_names = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']

# extract dataset
data_path = 'data'

for d in data_names:
    dataset = loader.GraphDataset()
    # dataset.extract_folder(os.path.join(data_path, '{}.zip'.format(d)), data_path)
    graphs, labels = dataset.read_dataset(d, os.path.join(data_path, d + '/'))
    print(len(graphs), len(labels))
