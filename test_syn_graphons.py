import os

import methods.simulator as simulator


for i in range(10):
    graphon = simulator.synthesize_graphon(r=1000, type_idx=i)
    graphs = simulator.simulate_graphs(graphon, num_graphs=10, num_nodes=200, graph_size='fixed')
    simulator.visualize_graphon(graphon, save_path=os.path.join('results', 'graphon_{}.pdf'.format(i)))
    simulator.visualize_unweighted_graph(graphs[0], save_path=os.path.join('results', 'adj_{}.pdf'.format(i)))

