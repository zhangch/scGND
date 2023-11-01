"""
"""

import numpy as np
from sklearn.cluster import KMeans #, AffinityPropagation

import igraph as ig
import leidenalg as la

import community as community_louvain
import networkx as nx

import sklearn.metrics as mt

import func.info_log as info_log

def kmeans(scdata, num_clusters=10, data_type = 'gae'):

    scdata.to_numpy()
    
    # prepare clustering data
    if data_type == "gae":
        feature_matrix = scdata.gae.output
    elif data_type == "fae":
        feature_matrix = scdata.fae.output
    elif data_type == "svg":
        feature_matrix = scdata.svg.log
    elif data_type == "raw":
        feature_matrix = scdata.raw.log
    else:
        raise Exception(f'data_type should be one of ("svg", "all") but got {data_type}.')
    
    # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    #feature_matrix_np = feature_matrix.detach().numpy() 
    
    # Perform k-means clustering using scikit-learn
    kmresult = KMeans(n_clusters=num_clusters, random_state=0).fit(feature_matrix)

    # Get the cluster labels
    scdata.cluster  = rank_clusters(kmresult.labels_) 

    # Print the cluster labels
    return scdata

def leiden(scdata, resolution=0.5):
    
    # Create a simple graph
    edges = scdata.graph.edge_list.copy()
    G = ig.Graph(edges)

    # Find the partition with the Leiden algorithm
    partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=resolution)

    cluster_labels = np.array(partition.membership)

    # Get the cluster labels
    scdata.cluster  = rank_clusters(cluster_labels) 

    # Print the cluster labels
    return scdata

def louvain(scdata, resolution=1.0):
    
    # Create a simple graph
    edge_list=scdata.graph.edge_list.copy()

    # Create a graph
    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    # compute the best partition
    partition = community_louvain.best_partition(G.to_undirected(), resolution=resolution)

    num_nodes = len(G.nodes())
    labels = [0]*num_nodes
    # fill in the list with community IDs
    for node, comm in partition.items():
        labels[node] = comm

    cluster_labels = np.array(labels)

    # Get the cluster labels
    scdata.cluster  = rank_clusters(cluster_labels) 

    # Print the cluster labels
    return scdata



def clustering(scdata, use_graph="graph", resolution=0.5, initial_membership=None, rank_labels=True):
    
    
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
       
    weighted_edge_list = graph_object.weighted_edge_list

    # Create the graph
    vertices = list(set([vertex for edge in weighted_edge_list for vertex in edge[:2]]))
    edges = [(edge[0], edge[1]) for edge in weighted_edge_list]
    weights = [edge[2] for edge in weighted_edge_list]

    G = ig.Graph()
    G.add_vertices(vertices)
    G.add_edges(edges)
    G.es['weight'] = weights

    # Find the optimal partition with Leiden algorithm
    partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=resolution, 
                                  weights='weight', initial_membership=initial_membership)


    cluster_labels = np.array(partition.membership)

    # Get the cluster labels
    if rank_labels:
        scdata.cluster = rank_clusters(cluster_labels) 
    else:
        scdata.cluster = cluster_labels

    # Print the cluster labels
    return scdata


def rank_clusters(cluster):
    cluster_label_list = list(range(max(cluster)+1))
    num_nodes_list = []
    for it in cluster_label_list:
        N_nodes = np.count_nonzero(cluster==it)
        num_nodes_list.append(N_nodes)
        
    num_nodes_list = make_unique(num_nodes_list)
    
    rank = rank_list(num_nodes_list)
        
    mapping = dict(zip(cluster_label_list, rank))
    
    # Define a function to replace the values
    def replace_values(x):
        return mapping.get(x, x)

    # Vectorize the function to apply it to the entire NumPy array
    vectorized_replace = np.vectorize(replace_values)

    # Replace the values in the NumPy array
    cluster = vectorized_replace(cluster)
    
    return cluster

def rank_list(data):
    sorted_data = sorted(data)
    sorted_data.reverse()
    rank = [sorted_data.index(x) for x in data]
    return rank

def make_unique(data):
    copy_list = data.copy()
    copy_list = sorted(copy_list)
    copy_list.reverse()
    
    number = len(data)
    for value in copy_list:
        for it in range(len(data)):
            if value == data[it]:
                data[it] = value + number
                number -= 1
    return data



def record_cluster(scdata):
    
    scdata.to_numpy()
    
    scdata.record.cluster=[]
    record_cluster = export_cluster(scdata)
    scdata.record.cluster.append(record_cluster)
    
    return scdata

def export_cluster(scdata):
    
    scdata.to_numpy()
    
    record_cluster={}
    record_cluster['umap'] = scdata.gae.umap_out
    record_cluster['edge_list'] = scdata.graph.edge_list
    record_cluster['cluster'] = scdata.cluster
    
    return record_cluster


def evaluate_clustering(cluster, label):
    
    adjusted_rand_score = mt.adjusted_rand_score(cluster, label)
    normalized_mutual_info_score = mt.normalized_mutual_info_score(cluster, label)
    fowlkes_mallows_score = mt.fowlkes_mallows_score(cluster, label)
    #jaccard_score = mt.jaccard_score(cluster, label)
    
    return adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
