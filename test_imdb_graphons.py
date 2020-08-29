import argparse
import networkx as nx
import os

import methods.learner as learner
import methods.loader as loader
import methods.simulator as simulator


parser = argparse.ArgumentParser(description='Comparison for various methods on real-world data')
parser.add_argument('--f-result', type=str, default='results',
                    help='the root path saving results')
parser.add_argument('--f-data', type=str, default='data',
                    help='the root path loading data')
parser.add_argument('--r', type=int, default=1000,
                    help='the resolution of graphon')
parser.add_argument('--threshold', type=float, default=0.1,
                    help='the destination folder saving face landmarks')
parser.add_argument('--alpha', type=float, default=0.001,
                    help='the weight of smoothness regularizer')
parser.add_argument('--beta', type=float, default=5e-3,
                    help='the weight of proximal term')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='the weight of gw term')
parser.add_argument('--inner-iters', type=int, default=50,
                    help='the number of inner iterations')
parser.add_argument('--outer-iters', type=int, default=20,
                    help='the number of outer iterations')
parser.add_argument('--dataset', type=str, default='IMDB-BINARY')
args = parser.parse_args()


data_names = ['IMDB-BINARY', 'IMDB-MULTI']  # , 'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']

for d in data_names:
    dataset = loader.GraphDataset()
    graph_list, label_dict = dataset.read_dataset(d, os.path.join(args.f_data, d + '/'))

    print(d, len(graph_list), len(label_dict))

    graphs = {}
    for i in range(len(graph_list)):
        label = label_dict[i]
        graph = graph_list[i].convert_to_nx()
        adj = nx.to_numpy_array(graph)
        if label not in graphs.keys():
            graphs[label] = [adj]
        else:
            graphs[label].append(adj)

    for method in ['fgwb']:
        for key in graphs.keys():
            print(key, len(graphs[key]))
            estimation = learner.estimate_graphon(graphs[key], method=method, args=args)
            simulator.visualize_graphon(estimation,
                                        save_path=os.path.join(args.f_result,
                                                               'estimation_{}_{}_{}.pdf'.format(method, d, key)))
