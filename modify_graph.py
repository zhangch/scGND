import torch
import numpy
import random

from build_graph import edge_index_to_adj, edge_index_to_edge_dict


def modify_graph(scdata, use_graph="graph", pct_outward_edges=0.3):
    """
    """
    scdata.to_device('cpu')
    scdata.to_numpy()
    
    # list of cell cluster labels, length = number of cells
    assert len(scdata.cluster) > 3, f'Clustering results is requested. Please do clustering first.'
    
    clustering_label = scdata.cluster
    num_of_nodes = scdata.raw.expr.shape[0]
    
    # select graph to build trajectory
    if use_graph == "graph":
        graph_object = scdata.graph
    elif use_graph == "raw_graph":
        graph_object = scdata.raw_graph
    elif use_graph == "dif_graph":
        graph_object = scdata.dif_graph
    elif use_graph == "graph_le":
        graph_object = scdata.graph_le
    elif use_graph == "graph_ge":
        graph_object = scdata.graph_ge
    else:
        raise Exception(f'use_graph should be one of ("graph", "raw_graph", "dif_graph", "graph_le", "graph_ge") but got {use_graph}.')
        
    
    # list of clusters, length = number of clusters
    cluster_list = list(range(max(clustering_label)+1))
    
    #
    # step 1: cluster nodes
    #
    
    # numpy.array(), from 0 to (number of cells -1)
    all_nodes = numpy.array(range(len(clustering_label)))
    
    
    cluster_inside_node_labels_list = []
    for cluster in cluster_list:
        filter_array = numpy.where(clustering_label == cluster, True, False)
        cluster_inside_node_labels_list.append(all_nodes[filter_array])
    
    #
    # step 2: cluster edages
    #
    
    edge_index = graph_object.edge_index
    source = edge_index[0]

    num_total_edges = edge_index.shape[1]
    num_inward_edges = 0
    for cluster in cluster_list:
        
        # numpy.isin(array_1, array_2) return a bool valued array with the same shape as array_1. 
        # True if the coresponding element in array_1 is included in array_2 otherwise False.
        filter_src = numpy.isin(source, cluster_inside_node_labels_list[cluster])
        cluster_edge_index = edge_index[:,filter_src]
        
        cluster_target = cluster_edge_index[1]
        filter_trg = numpy.isin(cluster_target, cluster_inside_node_labels_list[cluster])
        num_inward_edges = num_inward_edges + numpy.count_nonzero(filter_trg == True)
        
    inward_ratio = num_inward_edges/num_total_edges
    
    print("Raw inward ratio: ", inward_ratio)
    
    if inward_ratio > (1-pct_outward_edges):
        
        prune_ratio = 1.0 - (((1-pct_outward_edges)*(num_total_edges-num_inward_edges))/(pct_outward_edges*num_inward_edges))

        edge_index = prune_fn(edge_index, clustering_label, prune_ratio)

        graph_object.edge_index = edge_index
        # construct adjacency matrix from edge index
        graph_object.adj = edge_index_to_adj(graph_object.edge_index, num_of_nodes = num_of_nodes)
        # construct edge list from edge index
        graph_object.edge_list = list(zip(graph_object.edge_index[0].tolist(), graph_object.edge_index[1].tolist()))
        # construct edge dictionary from edge index
        graph_object.edge_dict = edge_index_to_edge_dict(graph_object.edge_index)

        if use_graph == "graph":
            scdata.graph=graph_object
        elif use_graph == "raw_graph":
            scdata.raw_graph=graph_object
        elif use_graph == "dif_graph":
            scdata.dif_graph=graph_object
        elif use_graph == "graph_le":
            scdata.graph_le=graph_object
        elif use_graph == "graph_ge":
            scdata.graph_ge=graph_object
        else:
            raise Exception(f'use_graph should be one of ("graph", "raw_graph", "dif_graph", "graph_le", "graph_ge") but got {use_graph}.')

    scdata.to_device('cpu')
    scdata.to_numpy()
    
    return scdata

def prune_fn(edge_index, cluster, prune_ratio):
    
    num_edges = edge_index.shape[1]
    
    random_array = numpy.random.rand(num_edges)
    random_array[random_array <= prune_ratio] = 0
    random_array[random_array > prune_ratio] = 1

    filt = []
    for i in range(len(edge_index[0])):
        if cluster[edge_index[0,i]] == cluster[edge_index[1,i]]:
            filt.append(0)
        else:
            filt.append(1)

    filt = numpy.array(filt)
    filt = filt + random_array
    filt[filt>0.5]=1
    filt = filt.astype(bool)
    
    edge_index = edge_index[:,filt]
    
    edge_index = torch.from_numpy(edge_index)
        
    return edge_index



            
