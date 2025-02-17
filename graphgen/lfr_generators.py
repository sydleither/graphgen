from graphgen.weighted_undirected_graph_generator import GenerateWeightedUndirectedGraph
from graphgen.unweighted_undirected_graph_generator import GenerateUnweightedUndirectedGraph
from graphgen.weighted_directed_graph_generator import GenerateWeightedDirectedGraph
from graphgen.unweighted_directed_graph_generator import GenerateUnweightedDirectedGraph
from graphgen.klemm_graph_generator import GenerateKlemmGraph
import networkx as nx
import numpy as np
import inspect


DEFAULT_FLOAT = np.float32
DEFAULT_INT = np.int64


def weighted_undirected_lfr_graph(num_nodes, average_k, max_degree, mut,
                                  muw, com_size_min, com_size_max, seed, beta=1.5,
                                  tau=2.0, tau2=1.0, overlapping_nodes=0,
                                  overlap_membership=0, fixed_range=True,
                                  excess=False, defect=False, randomf=False,
                                  avg_clustering=0.0, edge_dtype=None,
                                  weight_dtype=None):
    """
    Nodes start at 0 and are contiguous
    Return Ex2 numpy array, tuple of community memberships for each node,
    and a numpy array of weights corresponding to each edge in edge list (i.e
    the ith weight belongs to the ith edge)

    :param num_nodes: Number of nodes in the network (starts id at 0)
    :param average_k: average degree of the nodes
    :param max_degree: largest degree of the nodes
    :param mut: mixing parameter, fraction of bridges
    :param muw: weight mixing parameter
    :param beta: minus exponent for weight distribution
    :param com_size_min: smallest community size
    :param com_size_max: largest community size
    :param seed: for rng
    :param tau: minus exponent for degree sequence
    :param tau2: minus exponent for community size distribution
    :param overlapping_nodes: number of overlapping nodes
    :param overlap_membership: number of memberships of overlapping nodes
    :param fixed_range: If True, uses com_size_min/max, else distribution
        determines the range
    :param excess: -
    :param defect: -
    :param randomf: -
    :param avg_clustering: the average clustering coefficient
    :param edge_dtype: dtype of edge array. Default: DEFAULT_INT
    :param weight_dtype: dtype of weights. Default: DEFAULT_FLOAT
    :return: (Ex2 numpy array, tuple community memberships for each node, E numpy array)
     * Order of resulting Numpy array is: Ex2 with major axis as [0]=tail, [1]=head
     * Row major format, so [edge#][0]=tail, [edge#][1]=head
    """

    if edge_dtype is None:
        edge_dtype = DEFAULT_INT
    if weight_dtype is None:
        weight_dtype = DEFAULT_FLOAT

    edge_array, community_memberships, weights = GenerateWeightedUndirectedGraph(
        num_nodes, average_k, max_degree, mut, muw, com_size_min, com_size_max,
        seed, tau, tau2, overlapping_nodes, overlap_membership, fixed_range,
        excess, defect, randomf, beta, avg_clustering)

    if edge_array.dtype != edge_dtype:
        edge_array = edge_array.astype(edge_dtype)
    if weights.dtype != weight_dtype:
        weights = weights.astype(weight_dtype)

    return edge_array, community_memberships, weights


def weighted_undirected_lfr_as_nx(*args, **kwargs):
    """
    Nodes start at 0 and are contiguous
    Calls weighted_undirected_lfr_graph and converts to a networkx graph
    :return: networkx graph
    """
    edge_array, community_memberships, weights = weighted_undirected_lfr_graph(*args, **kwargs)

    nx_graph = nx.Graph()
    # Add nodes and attributes to graph
    nodes_and_memberships = []
    for node, node_memberships in enumerate(community_memberships):
        attributes = {'communities': node_memberships}
        for com_level, membership in enumerate(node_memberships):
            attributes['com_level'] = membership

        nodes_and_memberships.append((node, attributes))
    nx_graph.add_nodes_from(nodes_and_memberships)

    # Add edges to graph
    nx_graph.add_edges_from(edge_array)
    nx.set_edge_attributes(nx_graph, 'weight', {tuple(edge): weights[i]
                                                for i, edge in enumerate(edge_array)})

    return nx_graph


def weighted_undirected_lfr_as_adj(*args, **kwargs):
    """
    Calls weighted_undirected_lfr_graph and converts to a numpy matrix
    :param transpose: transpose the matrix representation
    :return: NxN float32 numpy array, and community membership
        adj matrix: axis1 (minor) is tail, axis2 (major) is head
        or (transpose): axis1 is head, axis2 is tail
    """

    graph_pars = {key: value for key, value in kwargs.items()
                  if key in inspect.getfullargspec(weighted_undirected_lfr_graph).args}

    converter_pars = {key: value for key, value in kwargs.items()
                      if key in inspect.getfullargspec(convert_weighted_to_numpy_matrix).args}

    edge_array, community_membership, weights = weighted_undirected_lfr_graph(*args,
                                                                              **graph_pars)
    return (convert_weighted_to_numpy_matrix(edge_array,
                                             weights=weights, **converter_pars),
            community_membership)


def weighted_directed_lfr_graph(num_nodes, average_k, max_degree, mut,
                                muw, com_size_min, com_size_max, seed, beta=1.5,
                                tau=2.0, tau2=1.0, overlapping_nodes=0,
                                overlap_membership=0, fixed_range=True,
                                excess=False, defect=False, randomf=False,
                                edge_dtype=None, weight_dtype=None):
    """
    Nodes start at 0 and are contiguous
    Return Ex2 numpy array, tuple of community memberships for each node,
    and a numpy array of weights corresponding to each edge in edge list (i.e
    the ith weight belongs to the ith edge)

    :param num_nodes: Number of nodes in the network (starts id at 0)
    :param average_k: average degree of the nodes
    :param max_degree: largest degree of the nodes
    :param mut: mixing parameter, fraction of bridges
    :param muw: weight mixing parameter
    :param beta: minus exponent for weight distribution
    :param com_size_min: smallest community size
    :param com_size_max: largest community size
    :param seed: for rng
    :param tau: minus exponent for degree sequence
    :param tau2: minus exponent for community size distribution
    :param overlapping_nodes: number of overlapping nodes
    :param overlap_membership: number of memberships of overlapping nodes
    :param fixed_range: If True, uses com_size_min/max, else distribution
        determines the range
    :param excess: -
    :param defect: -
    :param randomf: -
    :param edge_dtype: dtype of edge array. Default: DEFAULT_INT
    :param weight_dtype: dtype of weights. Default: DEFAULT_FLOAT
    :return: (Ex2 numpy array, tuple community memberships for each node, E numpy array)
     * Order of resulting Numpy array is: Ex2 with major axis as [0]=tail, [1]=head
     * Row major format, so [edge#][0]=tail, [edge#][1]=head
    """

    if edge_dtype is None:
        edge_dtype = DEFAULT_INT
    if weight_dtype is None:
        weight_dtype = DEFAULT_FLOAT

    edge_array, community_memberships, weights = GenerateWeightedDirectedGraph(
        num_nodes, average_k, max_degree, mut, muw, com_size_min, com_size_max,
        seed, tau, tau2, overlapping_nodes, overlap_membership, fixed_range,
        excess, defect, randomf, beta)

    if edge_array.dtype != edge_dtype:
        edge_array = edge_array.astype(edge_dtype)
    if weights.dtype != weight_dtype:
        weights = weights.astype(weight_dtype)

    return edge_array, community_memberships, weights


def weighted_directed_lfr_as_nx(*args, **kwargs):
    """
    Nodes start at 0 and are contiguous
    Calls weighted_directed_lfr_graph and converts to a networkx graph
    :return: networkx graph
    """
    edge_array, community_memberships, weights = weighted_directed_lfr_graph(*args,
                                                                             **kwargs)

    nx_graph = nx.DiGraph()
    # Add nodes and attributes to graph
    nodes_and_memberships = []
    for node, node_memberships in enumerate(community_memberships):
        attributes = {'communities': node_memberships}
        for com_level, membership in enumerate(node_memberships):
            attributes['com_level'] = membership

        nodes_and_memberships.append((node, attributes))
    nx_graph.add_nodes_from(nodes_and_memberships)

    # Add edges to graph
    nx_graph.add_edges_from(edge_array)
    nx.set_edge_attributes(nx_graph, 'weight', {tuple(edge): weights[i]
                                                for i, edge in enumerate(edge_array)})

    return nx_graph


def weighted_directed_lfr_as_adj(*args, **kwargs):
    """
    Calls weighted_directed_lfr_graph and converts to a numpy matrix
    :param transpose: transpose the matrix representation
    :return: NxN float32 numpy array, and community membership
        adj matrix: axis1 (minor) is tail, axis2 (major) is head
        or (transpose): axis1 is head, axis2 is tail
    """

    graph_pars = {key: value for key, value in kwargs.items()
                  if key in inspect.getfullargspec(weighted_directed_lfr_graph).args}

    converter_pars = {key: value for key, value in kwargs.items()
                      if key in inspect.getfullargspec(convert_weighted_to_numpy_matrix).args}

    edge_array, community_membership, weights = weighted_directed_lfr_graph(*args,
                                                                            **graph_pars)
    return (convert_weighted_to_numpy_matrix(edge_array,
                                             weights=weights, **converter_pars),
            community_membership)


def klemm_graph(num_nodes, clique_size, clique_linkage, muw, beta, seed,
                edge_dtype=None, weight_dtype=None):
    if edge_dtype is None:
        edge_dtype = DEFAULT_INT
    if weight_dtype is None:
        weight_dtype = DEFAULT_FLOAT

    edge_array, community_memberships, weights = GenerateKlemmGraph(
        num_nodes, clique_size, clique_linkage, muw, beta, seed)

    if edge_array.dtype != edge_dtype:
        edge_array = edge_array.astype(edge_dtype)
    if weights.dtype != weight_dtype:
        weights = weights.astype(weight_dtype)

    return edge_array, community_memberships, weights


def klemm_as_adj(*args, **kwargs):
    graph_pars = {key: value for key, value in kwargs.items()
                  if key in inspect.getfullargspec(klemm_graph).args}

    converter_pars = {key: value for key, value in kwargs.items()
                      if key in inspect.getfullargspec(convert_weighted_to_numpy_matrix).args}

    edge_array, community_membership, weights = klemm_graph(*args, **graph_pars)
    return (convert_weighted_to_numpy_matrix(edge_array,
                                             weights=weights, **converter_pars),
            community_membership)


def unweighted_undirected_lfr_graph(num_nodes, average_k, max_degree, mu,
                                    com_size_min, com_size_max, seed, tau=2.0,
                                    tau2=1.0, overlapping_nodes=0,
                                    overlap_membership=0, fixed_range=True,
                                    excess=False, defect=False, randomf=False,
                                    avg_clustering=0.0, edge_dtype=None):
    """
    Nodes start at 0 and are contiguous
    Return Ex2 numpy array and tuple of community memberships for each node

    :param num_nodes: Number of nodes in the network (starts id at 0)
    :param average_k: average degree of the nodes
    :param max_degree: largest degree of the nodes
    :param mu: mixing parameter, fraction of bridges
    :param com_size_min: smallest community size
    :param com_size_max: largest community size
    :param seed: for rng
    :param tau: minus exponent for degree sequence
    :param tau2: minus exponent for community size distribution
    :param overlapping_nodes: number of overlapping nodes
    :param overlap_membership: number of memberships of overlapping nodes
    :param fixed_range: If True, uses com_size_min/max, else distribution
        determines the range
    :param excess: -
    :param defect: -
    :param randomf: -
    :param avg_clustering: the average clustering coefficient
    :param edge_dtype: return type of edge list. Default: DEFAULT_INT
    :return: (Ex2 numpy array, tuple community memberships for each node)
    """

    if edge_dtype is None:
        edge_dtype = DEFAULT_INT

    edge_array, community_memberships = GenerateUnweightedUndirectedGraph(
        num_nodes, average_k, max_degree, mu, com_size_min, com_size_max, seed,
        tau, tau2, overlapping_nodes, overlap_membership, fixed_range, excess,
        defect, randomf, avg_clustering)

    if edge_array.dtype != edge_dtype:
        edge_array = edge_array.astype(edge_dtype)

    return edge_array, community_memberships


def unweighted_undirected_lfr_as_nx(*args, **kwargs):
    """
    Nodes start at 0 and are contiguous
    Calls unweighted_undirected_lfr_graph and converts to a networkx graph
    :return: networkx graph
    """
    edge_array, community_memberships = unweighted_undirected_lfr_graph(*args, **kwargs)

    nx_graph = nx.Graph()
    # Add nodes and attributes to graph
    nodes_and_memberships = []
    for node, node_memberships in enumerate(community_memberships):
        attributes = {'communities': node_memberships}
        for com_level, membership in enumerate(node_memberships):
            attributes['com_level'] = membership

        nodes_and_memberships.append((node, attributes))
    nx_graph.add_nodes_from(nodes_and_memberships)

    # Add edges to graph
    nx_graph.add_edges_from(edge_array)

    return nx_graph


def unweighted_undirected_lfr_as_adj(*args, **kwargs):
    """
    Nodes start at 0 and are contiguous
    Calls unweighted_undirected_lfr_graph and converts to an adjacency matrix
    :param transpose: transpose the matrix representation
    :return: Return a adj matrix: axis1 (minor) is tail, axis2 (major) is head
        and return community membership
        or (transpose): axis1 is head, axis2 is tail
    """

    graph_pars = {key: value for key, value in kwargs.items()
                  if key in inspect.getfullargspec(unweighted_undirected_lfr_graph).args}

    converter_pars = {key: value for key, value in kwargs.items()
                      if key in inspect.getfullargspec(convert_unweighted_to_numpy_matrix).args}

    edge_array, community_memberships = unweighted_undirected_lfr_graph(*args, **graph_pars)
    return (convert_unweighted_to_numpy_matrix(edge_array,
                                               **converter_pars),
            community_memberships)


def unweighted_directed_lfr_graph(num_nodes, average_k, max_degree, mu,
                                  com_size_min, com_size_max, seed, tau=2.0,
                                  tau2=1.0, overlapping_nodes=0,
                                  overlap_membership=0, fixed_range=True,
                                  excess=False, defect=False, randomf=False,
                                  edge_dtype=None):
    """
    Nodes start at 0 and are contiguous
    Return Ex2 numpy array and tuple of community memberships for each node

    :param num_nodes: Number of nodes in the network (starts id at 0)
    :param average_k: average degree of the nodes
    :param max_degree: largest degree of the nodes
    :param mu: mixing parameter, fraction of bridges
    :param com_size_min: smallest community size
    :param com_size_max: largest community size
    :param seed: for rng
    :param tau: minus exponent for degree sequence
    :param tau2: minus exponent for community size distribution
    :param overlapping_nodes: number of overlapping nodes
    :param overlap_membership: number of memberships of overlapping nodes
    :param fixed_range: If True, uses com_size_min/max, else distribution
        determines the range
    :param excess: -
    :param defect: -
    :param randomf: -
    :param edge_dtype: dtype of edge array. Default: DEFAULT_INT
    :return: (Ex2 numpy array, tuple community memberships for each node)
     * Order of resulting Numpy array is: Ex2 with major axis as [0]=tail, [1]=head
     * Row major format, so [edge#][0]=tail, [edge#][1]=head
    """

    if edge_dtype is None:
        edge_dtype = DEFAULT_INT

    edge_array, community_memberships = GenerateUnweightedDirectedGraph(
        num_nodes, average_k, max_degree, mu, com_size_min, com_size_max, seed, 
        tau, tau2, overlapping_nodes, overlap_membership, fixed_range, excess, 
        defect, randomf)

    if edge_array.dtype != edge_dtype:
        edge_array = edge_array.astype(edge_dtype)

    return edge_array, community_memberships


def unweighted_directed_lfr_as_nx(*args, **kwargs):
    """
    Nodes start at 0 and are contiguous
    Calls unweighted_directed_lfr_graph and converts to a networkx graph
    :return: networkx graph
    """
    edge_array, community_memberships = unweighted_directed_lfr_graph(*args, **kwargs)

    nx_graph = nx.DiGraph()
    # Add nodes and attributes to graph
    nodes_and_memberships = []
    for node, node_memberships in enumerate(community_memberships):
        attributes = {'communities': node_memberships}
        for com_level, membership in enumerate(node_memberships):
            attributes['com_level'] = membership

        nodes_and_memberships.append((node, attributes))
    nx_graph.add_nodes_from(nodes_and_memberships)

    # Add edges to graph
    nx_graph.add_edges_from(edge_array)

    return nx_graph


def unweighted_directed_lfr_as_adj(*args, **kwargs):
    """
    Nodes start at 0 and are contiguous
    Calls unweighted_directed_lfr_graph and converts to an adjacency matrix
    :param transpose: transpose the matrix representation
    :return: Return a adj matrix: axis1 (minor) is tail, axis2 (major) is head
        and return community membership
        or (transpose): axis1 is head, axis2 is tail
    """

    graph_pars = {key: value for key, value in kwargs.items()
                  if key in inspect.getfullargspec(unweighted_directed_lfr_graph).args}

    converter_pars = {key: value for key, value in kwargs.items()
                      if key in inspect.getfullargspec(convert_unweighted_to_numpy_matrix).args}

    edge_array, community_memberships = unweighted_directed_lfr_graph(*args, **graph_pars)
    return (convert_unweighted_to_numpy_matrix(edge_array,
                                               **converter_pars),
            community_memberships)


def convert_weighted_to_numpy_matrix(edge_array, num_nodes, weights, transpose=False,
                                     weight_dtype=None):
    """
    :param edge_array: Ex2 numpy array
    :param num_nodes: N
    :param weights: E np.float32 array
    :param transpose: transposes output matrix to reverse representation order
        default: False
    :param weight_dtype: dtype of return matrix
    :return: dtype=np.float32 NxN matrix
    """

    if weight_dtype is None:
        weight_dtype = DEFAULT_FLOAT

    matrix = np.zeros((num_nodes, num_nodes), dtype=weight_dtype)
    for i, edge in enumerate(edge_array):
        matrix[edge[0], edge[1]] = weights[i]

    if transpose:
        return matrix.transpose().copy()

    return matrix


def convert_unweighted_to_numpy_matrix(edge_array, num_nodes, transpose=False,
                                       edge_dtype=None):
    """
    :param edge_array: Ex2 numpy array
    :param num_nodes: N
    :param transpose: transposes output matrix to reverse representation order
        default: False
    :param edge_dtype: dtype of return matrix
    :return: dtype=np.uint64 NxN matrix
    """

    if edge_dtype is None:
        edge_dtype = DEFAULT_INT

    matrix = np.zeros((num_nodes, num_nodes), dtype=edge_dtype)
    for edge in edge_array:
        matrix[edge[0], edge[1]] = 1

    if transpose:
        return matrix.transpose().copy()

    return matrix


if __name__ == '__main__':
    """
    """
    pass
