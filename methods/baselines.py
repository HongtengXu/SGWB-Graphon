import torch
import numpy as np


def sort_smooth(aligned_graphs: torch.Tensor):
    """
    Estimate a graphon by a sorting and smoothing method

    Reference:
    S. H. Chan and E. M. Airoldi,
    "A Consistent Histogram Estimator for Exchangeable Graph Models",
    Proceedings of International Conference on Machine Learning, 2014.

    :param aligned_graphs:
    :return:
    """
    num_nodes = aligned_graphs.size(0)
    num_graphs = aligned_graphs.size(2)
    block_size = int(np.log(num_nodes))
    if num_graphs > 1:
        sum_graph = torch.sum(aligned_graphs, dim=2)
    else:
        sum_graph = aligned_graphs[:, :, 0]

    # sort node degrees
    degree = torch.sum(sum_graph - torch.diag(torch.diag(sum_graph)), dim=1)
    normalized_degree = degree / (num_nodes - 1)
    idx = torch.argsort(normalized_degree, descending=False)
    sum_graph = sum_graph[idx, :]
    sum_graph = sum_graph[:, idx]
    sorted_graph = sum_graph / num_graphs
    sorted_graph = torch.unsqueeze(torch.unsqueeze(sorted_graph, 0), 0)
    kernel = torch.ones(1, 1, block_size, block_size) / (block_size ** 2)
    # smoothing
    graphon = torch.nn.functional.conv2d(sorted_graph, kernel, padding=int(block_size / 2))
    graphon = graphon[0, 0, :, :]
    return graphon


def largest_gap(aligned_graphs: torch.Tensor, num_blocks: int, permute: bool = True):
    """
    Estimate a graphon by a stochastic block model based n empirical degrees

    Reference:
    Channarond, Antoine, Jean-Jacques Daudin, and Stéphane Robin.
    "Classification and estimation in the Stochastic Blockmodel based on the empirical degrees."
    Electronic Journal of Statistics 6 (2012): 2574-2601.

    :param aligned_graphs: a (N, N, K) a torch tensor, contains K N-node graphs
    :param num_blocks: thee number of blocks we predefined
    :param permute: output graphon with explicit block structure (permute nodes) or not
    :return: graphon: the estimated graphon model
    """
    num_nodes = aligned_graphs.size(0)
    num_graphs = aligned_graphs.size(2)
    if num_graphs > 1:
        sum_graph = torch.sum(aligned_graphs, dim=2)
    else:
        sum_graph = aligned_graphs[:, :, 0]

    # sort node degrees
    degree = torch.sum(sum_graph - torch.diag(torch.diag(sum_graph)), dim=1)
    normalized_degree = degree / (num_nodes - 1)
    idx = torch.argsort(normalized_degree, descending=False)
    sorted_degree = normalized_degree[idx]

    # find num_blocks-1 largest gap of the node degrees
    diff_degree = sorted_degree[1:] - sorted_degree[:-1]
    _, index = torch.topk(diff_degree, k=num_blocks - 1)
    sorted_index, _ = torch.sort(index + 1, descending=False)
    blocks = {}
    for b in range(num_blocks):
        if b == 0:
            blocks[b] = idx[0:sorted_index[b]]
        elif b == num_blocks - 1:
            blocks[b] = idx[sorted_index[b - 1]:num_nodes]
        else:
            blocks[b] = idx[sorted_index[b - 1]:sorted_index[b]]

    # derive the graphon by stochastic block model
    probability = torch.zeros(num_blocks, num_blocks)
    if permute:
        graphon = None
        for i in range(num_blocks):
            rows = blocks[i]
            tmp_graphon = None
            for j in range(num_blocks):
                cols = blocks[j]
                tmp = sum_graph[rows, :]
                # print(rows, cols, tmp.size())
                tmp = tmp[:, cols]
                probability[i, j] = torch.sum(tmp) / (num_graphs * rows.size(0) * cols.size(0))
                one_block = probability[i, j] * torch.ones(rows.size(0), cols.size(0))
                if j == 0:
                    tmp_graphon = one_block
                else:
                    tmp_graphon = torch.cat((tmp_graphon, one_block), dim=1)

            if i == 0:
                graphon = tmp_graphon
            else:
                graphon = torch.cat((graphon, tmp_graphon), dim=0)
    else:
        graphon = torch.zeros(num_nodes, num_nodes)
        for i in range(num_blocks):
            for j in range(num_blocks):
                rows = blocks[i]
                cols = blocks[j]
                tmp = sum_graph[rows, :]
                tmp = tmp[:, cols]
                probability[i, j] = torch.sum(tmp) / (num_graphs * rows.size(0) * cols.size(0))
                for r in range(rows.size(0)):
                    for c in range(cols.size(0)):
                        graphon[rows[r], cols[c]] = probability[i, j]

    return graphon, probability, blocks


def universal_svd(aligned_graphs: torch.Tensor, threshold: float = 2.02):
    """
    Estimate a graphon by universal singular value thresholding.

    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param aligned_graphs: a (N, N, K) a torch tensor, contains K N-node graphs
    :param threshold: the threshold for singular values
    :return: graphon: the estimated graphon model
    """
    num_nodes = aligned_graphs.size(0)
    num_graphs = aligned_graphs.size(2)
    if num_graphs > 1:
        average_graph = torch.mean(aligned_graphs, dim=2)
    else:
        average_graph = aligned_graphs[:, :, 0]
    u, s, v = torch.svd(average_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s == True] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    return graphon


def matrix_completion(aligned_graphs: torch.Tensor, rank: int = None):
    """
    Estimate the graphon by matrix completion

    Reference:
    Keshavan, Raghunandan H., Andrea Montanari, and Sewoong Oh.
    "Matrix completion from a few entries."
    IEEE transactions on information theory 56.6 (2010): 2980-2998.

    :param aligned_graphs: a (N, N, K) a torch tensor, contains K N-node graphs
    :param rank: the rank of adjacency matrix
    :return: graphon: the estimated graphon model
    """
    num_nodes = aligned_graphs.size(0)
    num_graphs = aligned_graphs.size(2)
    if num_graphs > 1:
        average_graph = torch.mean(aligned_graphs, dim=2)
    else:
        average_graph = aligned_graphs[:, :, 0]

    # low-rank approximation via svd
    average_graph = 2 * (average_graph - 0.5)
    if rank is None:
        rank = int(num_nodes / int(np.log(num_nodes)))
        # rank = guess_rank(average_graph)
        # print(rank)
    u, s, v = torch.svd(average_graph)
    graphon = (u[:, :rank] @ torch.diag(s[:rank]) @ torch.t(v[:, :rank]) + 1) / 2
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    return graphon


def guess_rank(matrix: torch.Tensor):
    """
    A function to guess the rank of a matrix
    :param matrix: a torch.Tensor matrix
    :return:
    """
    n = matrix.size(0)
    m = matrix.size(1)
    epsilon = torch.sum(matrix != 0) / ((n * m) ** 0.5)

    u, s, v = torch.svd(matrix, compute_uv=False)
    max_num = min([100, s.size(0)])
    s = s[:max_num]
    s, _ = torch.sort(s, descending=True)
    diff_s1 = s[:-1] - s[1:]
    diff_s1 = diff_s1 / torch.mean(diff_s1[-10:])
    r1 = torch.zeros(1)
    gamma = 0.05
    while r1.item() <= 0:
        cost = torch.zeros(diff_s1.size(0))
        for i in range(diff_s1.size(0)):
            cost[i] = gamma * torch.max(diff_s1[i:]) + i + 1

        idx = torch.argmin(cost)
        r1 = torch.argmax(idx)
        gamma += 0.05

    cost = torch.zeros(diff_s1.size(0))
    for i in range(diff_s1.size(0)):
        cost[i] = s[i + 1] + ((i + 1) * epsilon ** 0.5) * s[0] / epsilon

    idx = torch.argmin(cost)
    r2 = torch.max(idx)
    return max([r1.item(), r2.item()])


def estimate_blocks_directed(aligned_graphs: torch.Tensor, threshold: float, permute: bool = True):
    """
    Estimate a graphon by stochastic block approximation.

    Reference:
    E. M. Airoldi, T. B. Costa, and S. H. Chan,
    "Stochastic blockmodel approximation of a graphon: Theory and consistent estimation",
    Advances in Neural Information Processing Systems 2013.

    :param aligned_graphs: a (N, N, K) a torch tensor, contains K N-node graphs
    :param threshold: the threshold for singular values
    :param permute: output graphon with explicit block structure (permute nodes) or not
    :return: graphon: the estimated graphon model
    """
    num_nodes = aligned_graphs.size(0)
    num_graphs = aligned_graphs.size(2)
    num_half_graphs = int(num_graphs / 2)
    w = 1/(num_half_graphs * (num_nodes - num_half_graphs))
    if num_graphs > 1:
        sum_graph = torch.sum(aligned_graphs, dim=2)
        if num_half_graphs > 1:
            sum_half_graph1 = torch.sum(aligned_graphs[:, :, :num_half_graphs], dim=2)
        else:
            sum_half_graph1 = aligned_graphs[:, :, 0]

        if num_graphs - num_half_graphs > 1:
            sum_half_graph2 = torch.sum(aligned_graphs[:, :, num_half_graphs:], dim=2)
        else:
            sum_half_graph2 = aligned_graphs[:, :, -1]
    else:
        sum_half_graph1 = aligned_graphs[:, :, 0]
        sum_half_graph2 = aligned_graphs[:, :, 0]
        sum_graph = aligned_graphs[:, :, 0]

    pivot_idx = [int(num_nodes * np.random.RandomState(seed=42).rand())]
    blocks = dict()
    blocks[0] = [pivot_idx[0]]
    not_assigned_idx = list(range(num_nodes))
    not_assigned_idx.remove(pivot_idx[0])
    while len(not_assigned_idx) > 0:
        if len(not_assigned_idx) == 1:
            i = not_assigned_idx[0]
        else:
            idx = np.random.permutation(len(not_assigned_idx))
            i = not_assigned_idx[idx[0]]
        not_assigned_idx.remove(i)

        dhat = torch.zeros(len(pivot_idx))
        for j in range(len(pivot_idx)):
            bj = pivot_idx[j]

            set_idx = list(range(num_nodes))
            set_idx.remove(i)
            set_idx.remove(bj)

            term1 = torch.sum(w * sum_half_graph1[i, set_idx] * sum_half_graph2[i, set_idx])
            term2 = torch.sum(w * sum_half_graph1[bj, set_idx] * sum_half_graph2[bj, set_idx])
            term3 = torch.sum(w * sum_half_graph1[i, set_idx] * sum_half_graph2[bj, set_idx])
            term4 = torch.sum(w * sum_half_graph1[bj, set_idx] * sum_half_graph2[i, set_idx])

            term5 = torch.sum(w * sum_half_graph1[set_idx, i] * sum_half_graph2[set_idx, i])
            term6 = torch.sum(w * sum_half_graph1[set_idx, bj] * sum_half_graph2[set_idx, bj])
            term7 = torch.sum(w * sum_half_graph1[set_idx, i] * sum_half_graph2[set_idx, bj])
            term8 = torch.sum(w * sum_half_graph1[set_idx, bj] * sum_half_graph2[set_idx, i])

            dhat[j] = (0.5*(torch.abs(term1 + term2 - term3 - term4) +
                            torch.abs(term5 + term6 - term7 - term8)) / len(set_idx)) ** 0.5

        if dhat.size(0) == 1:
            value = dhat[0]
            idx = 0
        else:
            value, idx = torch.min(dhat)

        if value < threshold:
            blocks[idx].append(i)
        else:
            blocks[len(pivot_idx) + 1] = [i]
            pivot_idx.append(i)

    num_blocks = len(blocks)
    for key in blocks.keys():
        blocks[key] = torch.LongTensor(blocks[key])

    # derive the graphon by stochastic block model
    probability = torch.zeros(num_blocks, num_blocks)
    if permute:
        graphon = None
        for i in range(num_blocks):
            rows = blocks[i]
            tmp_graphon = None
            for j in range(num_blocks):
                cols = blocks[j]
                tmp = sum_graph[rows, :]
                tmp = tmp[:, cols]
                probability[i, j] = torch.sum(tmp) / (num_graphs * rows.size(0) * cols.size(0))
                one_block = probability[i, j] * torch.ones(rows.size(0), cols.size(0))
                if j == 0:
                    tmp_graphon = one_block
                else:
                    tmp_graphon = torch.cat((tmp_graphon, one_block), dim=1)

            if i == 0:
                graphon = tmp_graphon
            else:
                graphon = torch.cat((graphon, tmp_graphon), dim=0)
    else:
        graphon = torch.zeros(num_nodes, num_nodes)
        for i in range(num_blocks):
            for j in range(num_blocks):
                rows = blocks[i]
                cols = blocks[j]
                tmp = sum_graph[rows, :]
                tmp = tmp[:, cols]
                probability[i, j] = torch.sum(tmp) / (num_graphs * rows.size(0) * cols.size(0))
                for r in range(rows.size(0)):
                    for c in range(cols.size(0)):
                        graphon[rows[r], cols[c]] = probability[i, j]

    return graphon, probability, blocks
