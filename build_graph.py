import torch
import numpy
import math
from sklearn.ensemble import IsolationForest

from scdata_metrics import GraphTopology

def build_graph(scdata, k=10, data_type="svg", graph_name='graph', self_edge = False, prune=True):
    """
    Build graph by using single cell gene expression matrix.

    :param scdata: Single cell data
    :param k: (int) – The number of neighbors
    :param data_type: (str, "fae", "svg" or "raw") - Using feature auto-encoder output, selected variable or raw
                                                gene expression matrix to build graph.
    :param batch: check the param "batch" in package function - torch_geometric.nn.knn_graph.
    :param self_loop: (bool, optional) – If True, the graph will contain self-loops. (default: False)
    :param flow: (str, optional) – The flow direction when using in combination with message passing
                                    ("source_to_target" or "target_to_source"). (default: "source_to_target")
    :param distance_type: ("euclidean" or "cosine") - use the euclidean distance or the cosine distance instead
                                                        to find nearest neighbors. (default: "euclidean")
    :param num_workers: check the param "num_workers" in package function - torch_geometric.nn.knn_graph.
    :return: (scdata) - Single cell data with graph information contained.
    """
    scdata.to_torch()
    scdata.to_device("cpu")

    # Select data to build graph according to 'data_type'
    if data_type=="raw":
        feature_matrix = scdata.raw.log
    elif data_type=="svg":
        feature_matrix = scdata.svg.log
    elif data_type=="gae":
        feature_matrix = scdata.gae.output
    else:
        raise Exception(f'data_type should be one of ("raw", "svg", "gae") but got {data_type}.')
        
    graph_object = hidden_build_graph(feature_matrix, k=k, self_edge=self_edge, prune=prune)
    
    if graph_name == "graph":
        scdata.graph = graph_object
    elif graph_name == "raw_graph":
        scdata.raw_graph = graph_object
    elif graph_name == "dif_graph":
        scdata.dif_graph = graph_object
    elif graph_name == "graph_le":
        scdata.graph_le = graph_object
    elif graph_name == "graph_ge":
        scdata.graph_ge = graph_object
    else:
        raise Exception(f'graph_name should be one of ("graph", "raw_graph", "dif_graph", "graph_le", "graph_ge") but got {graph_name}.')
    
    return scdata

def build_gnd_graph(scdata, k=10, self_edge = False, prune=True):
    
    scdata.to_torch()
    scdata.to_device("cpu")
    
    scdata.gae.gnd_graph = []
    for feature_matrix in scdata.gae.gnd:
        graph_object = hidden_build_graph(feature_matrix, k=k, self_edge=self_edge, prune=prune)
        scdata.gae.gnd_graph.append(graph_object)
        
    return scdata
    

def hidden_build_graph(feature_matrix, k=10, self_edge=False, prune=True):
    num_of_nodes = feature_matrix.size()[0]
    # construct graph
    edge_index = knn_graph(feature_matrix, k, self_edge=self_edge)
    
    if prune:
        edge_index = prune_fn(edge_index, feature_matrix)
        
    graph_object = GraphTopology()
        
    graph_object.edge_index = edge_index
    # construct adjacency matrix from edge index
    graph_object.adj = edge_index_to_adj(graph_object.edge_index, num_of_nodes = num_of_nodes)
    # construct edge list from edge index
    graph_object.edge_list = list(zip(graph_object.edge_index[0].tolist(), graph_object.edge_index[1].tolist()))
    # construct edge dictionary from edge index
    graph_object.edge_dict = edge_index_to_edge_dict(graph_object.edge_index)
    
    return graph_object

def knn_graph(feature_matrix, k, self_edge = False):
    """
    Compute the k-nearest neighbors graph from a feature matrix.

    Args:
        feature_matrix (torch.Tensor): A tensor of shape (num_points, num_features).
        k (int): The number of nearest neighbors to consider.

    Returns:
        networkx.Graph: A NetworkX graph representing the k-nearest neighbors graph.
    """
    # Calculate the pairwise squared distances between points
    dist_matrix = torch.cdist(feature_matrix, feature_matrix, p=2)

    # Find the indices of the k nearest neighbors for each point
    if self_edge:
        knn_indices = torch.argsort(dist_matrix, dim=1)[:, :k]  # Exclude the point itself (at index 0)
        # construct edge index from knn_indices
        edge_index = knn_indices_to_edge_index(knn_indices)
    
    else:
        knn_indices = torch.argsort(dist_matrix, dim=1)[:, :k+1] # the point itself may not at index 0(overlapped points)
        # construct edge index from knn_indices
        edge_index = knn_indices_to_edge_index(knn_indices)
        
        edge_index = numpy.array(edge_index)
        # remove self-edges
        filt = []
        for i in range(len(edge_index[0])):
            if edge_index[0,i]==edge_index[1,i]:
                filt.append(False)
            else:
                filt.append(True)
        edge_index = edge_index[:,filt]
        
        edge_index = torch.from_numpy(edge_index)

    return edge_index

def knn_indices_to_edge_index(knn_indices):
    """
    Convert a knn_indices tensor to an edge_index tensor.

    Args:
        knn_indices (torch.Tensor): A tensor of shape (num_points, k) containing the indices of 
                                    the k-nearest neighbors for each point.

    Returns:
        torch.Tensor: An edge_index tensor of shape (2, num_edges) representing the edges in the graph.
    """
    num_points, k = knn_indices.shape

    # Create source and target node index tensors
    src_nodes = torch.arange(num_points).view(-1, 1).repeat(1, k).view(-1)
    trg_nodes = knn_indices.reshape(-1)

    # Concatenate the source and target node index tensors to create the edge_index tensor
    edge_index = torch.stack([src_nodes, trg_nodes], dim=0)

    return edge_index

def edge_index_to_adj(edge_index, num_of_nodes):
    """
    construct adjacency matrix from edge index
    """
    adjacency_matrix = torch.zeros((num_of_nodes, num_of_nodes), dtype=edge_index.dtype, device=edge_index.device)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1

    return adjacency_matrix

def edge_index_to_edge_dict(edge_index):
    """
    construct edge dictionary from edge index
    """
    edge_dict = {}
    for src, tgt in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if src not in edge_dict:
            edge_dict[src] = []
        edge_dict[src].append(tgt)

    return edge_dict



def prune_fn(edge_index, feature_matrix):
    
    edge_index = numpy.array(edge_index)
    feature_matrix = numpy.array(feature_matrix)
    
    N_nodes = feature_matrix.shape[0]
    
    N_edges = edge_index.shape[1]
    
    NEPN = int(N_edges/N_nodes)
    
    N_least = math.ceil(NEPN/10) # The least number of edges every node should keep 
    
    # Fit an IsolationForest model
    clf = IsolationForest(random_state=0).fit(feature_matrix)

    # Get the anomaly labels for each data point
    node_IF_labels = clf.predict(feature_matrix)

    filt = []
    for i in range(len(edge_index[0])):
        if i % NEPN <= N_least:
            filt.append(True)
        elif node_IF_labels[int(edge_index[1,i])]==-1:
            if node_IF_labels[int(edge_index[0,i])]==1:
                filt.append(False)
            else:
                filt.append(True)
        else:
            filt.append(True)

    edge_index = edge_index[:,filt]
    
    edge_index = torch.from_numpy(edge_index)
        
    return edge_index



    