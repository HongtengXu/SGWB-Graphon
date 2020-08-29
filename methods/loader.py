import functools
import networkx as nx
import numpy as np
import operator
import os
import zipfile
from collections import Counter


class GraphStruct(object):
    def __init__(self, edges, node_labels, node_attributes,
                 edge_labels, edge_attributes):
        """
        Graph representation in simple python data structures
        :param edges: list of edges, e.g. [(0,1), (0,2), (1,0), ...]
        :param node_labels: dictionary of labels, e.g. {(0: 'A', 1: 'B', ...}
        :param node_attributes: dictionary of attributes, e.g. {(0: '0.25', 1: '0.33', ...}
        :param edge_labels: dictionary of edge labels, e.g. {(0,1): 'Q', (0,2): 'W', (1,0}: 'Q', ...}
        :param edge_attributes: dictionary of edge attributes, e.g. {(0,1): '0.1', (0,2): '0.3', (1,0}: '0.1', ...}
        """
        self.edges = edges
        self.edge_labels = edge_labels
        self.edge_attributes = edge_attributes
        self.nodes = set(functools.reduce(operator.iconcat, self.edges, []))
        self.node_labels = {node: node_labels[node] for node in self.nodes} if len(node_labels) else dict()
        self.node_attributes = {node: node_attributes[node] for node in self.nodes} if len(node_attributes) else dict()

    def is_edge_label_directed(self):
        el = self.edge_labels
        for e in el:
            if el[e] != el[(e[1], e[0])]:
                return True

        return False

    def is_edge_attribute_directed(self):
        ea = self.edge_attributes
        for e in ea:
            if ea[e] != ea[(e[1], e[0])]:
                return True

        return False

    def convert_to_nx(self):
        graph = nx.Graph()
        if self.is_edge_label_directed() or self.is_edge_attribute_directed():
            graph = nx.DiGraph()
        graph.add_edges_from(self.edges)
        nx.set_edge_attributes(graph, self.edge_labels, 'edge_label') if len(self.edge_labels) else None
        nx.set_edge_attributes(graph, self.edge_attributes, 'edge_attribute') if len(self.edge_attributes) else None
        nx.set_node_attributes(graph, self.node_labels, 'node_label') if len(self.node_labels) else None
        nx.set_node_attributes(graph, self.node_attributes, 'node_attribute') if len(self.node_attributes) else None
        return graph


class GraphDataset(object):
    @staticmethod
    def extract_folder(zip_folder, output):
        with zipfile.ZipFile(zip_folder, 'r') as f:
            f.extractall(output)

    @staticmethod
    def get_filenames(input_folder):
        fns = os.listdir(input_folder)
        graphs_fn = indicator_fn = graph_labels_fn = \
            node_labels_fn = edge_labels_fn = \
            edge_attributes_fn = node_attributes_fn = graph_attributes_fn = None
        for fn in fns:
            if 'A.txt' in fn:
                graphs_fn = input_folder + fn
            elif '_graph_indicator.txt' in fn:
                indicator_fn = input_folder + fn
            elif '_graph_labels.txt' in fn:
                graph_labels_fn = input_folder + fn
            elif '_node_labels.txt' in fn:
                node_labels_fn = input_folder + fn
            elif '_edge_labels.txt' in fn:
                edge_labels_fn = input_folder + fn
            elif '_node_attributes.txt' in fn:
                node_attributes_fn = input_folder + fn
            elif '_edge_attributes.txt' in fn:
                edge_attributes_fn = input_folder + fn
            elif '_graph_attributes.txt' in fn:
                graph_attributes_fn = input_folder + fn
        return graphs_fn, indicator_fn, graph_labels_fn, node_labels_fn, edge_labels_fn, \
            edge_attributes_fn, node_attributes_fn, graph_attributes_fn

    def read_graphs(self, input_folder):
        graphs_fn, indicator_fn, graph_labels_fn, node_labels_fn, edge_labels_fn, \
            edge_attributes_fn, node_attributes_fn, graph_attributes_fn = self.get_filenames(input_folder)

        edge_labels_f = []
        edge_attributes_f = []

        if edge_labels_fn:
            edge_labels_f = open(edge_labels_fn)
        if edge_attributes_fn:
            edge_attributes_f = open(edge_attributes_fn)

        with open(indicator_fn) as f:
            nodes2graph = dict()
            for i, line in enumerate(f):
                nodes2graph[i + 1] = int(line.strip())

        node_labels = dict()
        if node_labels_fn:
            with open(node_labels_fn) as f:
                for i, line in enumerate(f):
                    node_labels[i + 1] = line.strip()

        node_attributes = dict()
        if node_attributes_fn:
            with open(node_attributes_fn) as f:
                for i, line in enumerate(f):
                    node_attributes[i + 1] = line.strip()

        if graph_attributes_fn:
            graph_attributes = dict()
            with open(graph_attributes_fn) as f:
                for i, line in enumerate(f):
                    graph_attributes[i + 1] = line.strip()

        new_graphs = []
        with open(graphs_fn) as f:
            current_graph = 1
            edges = []
            edge_labels = dict()
            edge_attributes = dict()
            for i, line in enumerate(f):
                ls = line.strip().split(',')
                u, v = int(ls[0]), int(ls[1])
                g1, g2 = nodes2graph[u], nodes2graph[v]
                assert g1 == g2, 'Nodes should be connected in the same graph. Line {}, graphs {} {}'. \
                    format(i, g1, g2)

                if g1 != current_graph:  # assumes indicators are sorted
                    # print(g1, current_graph, edges)
                    graph = GraphStruct(edges, node_labels, node_attributes, edge_labels, edge_attributes)

                    new_graphs.append(graph)

                    edges = []
                    edge_labels = dict()
                    edge_attributes = dict()
                    current_graph += 1
                    # if current_graph % 1000 == 0:
                    #     print('Finished {} dataset'.format(current_graph - 1))

                edges.append((u, v))
                if edge_labels_fn:
                    edge_labels[(u, v)] = next(edge_labels_f).strip()
                if edge_attributes_fn:
                    edge_attributes[(u, v)] = next(edge_attributes_f).strip()

        # last graph
        if len(edges) > 0:
            graph = GraphStruct(edges, node_labels, node_attributes, edge_labels, edge_attributes)
            new_graphs.append(graph)

        if edge_labels_fn:
            edge_labels_f.close()
        if edge_attributes_fn:
            edge_attributes_f.close()

        return new_graphs

    @staticmethod
    def read_labels(dataset, input_folder):
        graph_labels = dict()
        with open(input_folder + dataset + '_graph_labels.txt') as f:
            for i, label in enumerate(f):
                graph_labels[i] = label.strip()
        return graph_labels

    def read_dataset(self, dataset, input_folder):
        assert os.path.exists(input_folder), f'Path to dataset should contain folder {dataset}'
        graphs = self.read_graphs(input_folder)
        labels = self.read_labels(dataset, input_folder)
        return graphs, labels

    @staticmethod
    def compute_stats(graphs, labels):
        if len(graphs) > 0:
            num_nodes = [len(g.nodes) for g in graphs]
            num_edges = [len(g.edges) / 2 for g in graphs]
            c = Counter(labels.values())
            least, most = c.most_common()[-1][1], c.most_common()[0][1]
            return len(graphs), np.mean(num_nodes), np.mean(num_edges), len(c), least, most
        return 0, 0, 0, 0, 0, 0

    @staticmethod
    def convert_to_nx_graphs(graphs):
        return [g.convert_to_nx() for g in graphs]

    def save_graphs_graphml(self, graphs, output_folder):
        nx_graphs = self.convert_to_nx_graphs(graphs)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        for ix, g in enumerate(nx_graphs):
            nx.write_graphml(g, output_folder + f'{ix}.graphml')

    @staticmethod
    def save_graphs_edgelist(graphs, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        for ix, g in enumerate(graphs):
            fn = f'{ix}.edgelist'
            with open(output_folder + fn, 'w+') as f:
                for e in g.edges:
                    f.write(f"{e[0]} {e[1]}\n")
