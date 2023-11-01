import networkx as nx
import numpy
import matplotlib.pyplot as plt


def call_modularity(scdata, use_graph="graph", edge_weight=True):

    # select graph to call modularity
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

    modularity = hidden_call_modularity(graph_object, cluster_labels=scdata.cluster, edge_weight=edge_weight)
        
    return modularity

def call_gnd_modularity(scdata, edge_weight=True):

    modularity_gnd = []
    for graph_object in scdata.gae.gnd_graph:
        modularity = hidden_call_modularity(graph_object, cluster_labels=scdata.cluster, edge_weight=edge_weight)
        modularity_gnd.append(modularity)

    return modularity_gnd


def hidden_call_modularity(graph_object, cluster_labels, edge_weight=True):

    # Create the graph
    G = nx.Graph()
    
    if edge_weight:
        G.add_weighted_edges_from(graph_object.weighted_edge_list)
    else:
        G.add_edges_from(graph_object.edge_list)

    # Convert the cluster labels to the appropriate format for the modularity function
    # The modularity function expects a list of sets, where each set contains the nodes in one community.
    existing_partition = []
    for i in numpy.unique(cluster_labels):
        community = {node for node, community in enumerate(cluster_labels, start=0) if community == i}
        existing_partition.append(community)

    # Compute the modularity of the existing partition
    if edge_weight:
        modularity = nx.algorithms.community.modularity(G, existing_partition, weight='weight')
    else:
        modularity = nx.algorithms.community.modularity(G, existing_partition)
        
    return modularity


def view_modularity(Weighted_modularity, save_fig=None):
    x = list(range(len(Weighted_modularity)))

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=True, sharey=False)

    axs[0].plot(x, Weighted_modularity, color='orange')
    axs[0].set_title('Attention weighted modularity')
    axs[0].set_xlabel('gnd steps')
    axs[0].set_ylabel('Modularity')

    if save_fig is not None:
        plt.savefig(save_fig, dpi=500)

    plt.show()


def view_gnd_modularity(Weighted_modularity, Unweighted_modularity, save_fig=None):
    pass
    

