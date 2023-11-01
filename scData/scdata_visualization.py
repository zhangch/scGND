import matplotlib.pyplot as plt
import networkx as nx


def umap_view(scdata, data='gae.output',clusters=False, edges=False, edge_scale=0.02, node_scale=0.5, save_fig=None):
    
    scdata.to_numpy()
    
    cluster = scdata.cluster if clusters else None
    edge_list = scdata.graph.edge_list if edges else None

    if data=='gae.output':
        feature_mtx = scdata.gae.umap_out
        umap_view_hidden(feature_mtx, cluster=cluster, edge_list=edge_list, 
                         edge_scale=edge_scale, node_scale=node_scale, save_fig=save_fig)
    elif data=='gae.input':
        feature_mtx = scdata.gae.umap_in
        umap_view_hidden(feature_mtx, cluster=cluster, edge_list=edge_list, 
                         edge_scale=edge_scale, node_scale=node_scale, save_fig=save_fig) 
    elif data=='gae':
        print('Before diffusion: ')
        feature_mtx = scdata.gae.umap_in
        umap_view_hidden(feature_mtx, cluster=cluster, edge_list=edge_list, 
                         edge_scale=edge_scale, node_scale=node_scale, save_fig=save_fig)
        print('Ather diffusion: ')
        feature_mtx = scdata.gae.umap_out
        umap_view_hidden(feature_mtx, cluster=cluster, edge_list=edge_list, 
                         edge_scale=edge_scale, node_scale=node_scale, save_fig=save_fig)
    elif data=='gae.gnd':
        for it in range(len(scdata.gae.gnd_embed)):
            print(f'Diffusion {it}: ')
            feature_mtx = scdata.gae.gnd_embed[it]
            umap_view_hidden(feature_mtx, cluster=cluster, edge_list=edge_list, 
                             edge_scale=edge_scale, node_scale=node_scale, save_fig=save_fig) 
    
def umap_view_hidden(feature_mtx, cluster=None, edge_list=None, edge_scale=0.02, node_scale=0.5, save_fig=None, show=True):
    
    num_nodes = feature_mtx.shape[0]
    
    plt.rcParams['figure.dpi'] = 500

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    
    pos = {n: feature_mtx[n] for n in G.nodes()}
    nodecolor = 'blue' if cluster is None else [cluster[n] for n in G.nodes()]
    
    if edge_list is None:
        nx.draw_networkx_nodes(G, pos, nodelist=None, node_size=node_scale, node_color=nodecolor)
    else:
        G.add_edges_from(edge_list)
    
        nx.draw_networkx(G, pos=pos, width=edge_scale, with_labels=None, arrows=0, node_size=node_scale, 
                     node_color=nodecolor, edge_color='gray', connectionstyle="arc3,rad=0.1",
                    nodelist = None)
    plt.axis('equal')
    if save_fig is not None:  
        plt.savefig(save_fig, dpi=500)
    if show:
        plt.show()



def similarity_view(scdata, edge_threshold=0.2, edge_scale=0.5, node_scale=0.1, node_labels=True,
                       node_pos=True, show_umap=True, resolution=1.0, fig_size=(12,12),
                       umap_args={'edge_scale': 0.05,
                             'node_scale': 0.5},
                       save_fig=None):
    scdata.to_numpy()

    feature_mtx = scdata.traj.expr
    weighted_edge_list = scdata.traj.smlr_edge_list
    cluster = list(range(len(scdata.traj.nodes)))
    node_weight = scdata.traj.nodes
    
    if show_umap:
        umap_args['feature_mtx']=scdata.gae.umap_out
        umap_args['edge_list']=scdata.graph.edge_list
        umap_args['cluster'] = scdata.cluster
        
        trajectory_view_umap(feature_mtx, cluster=cluster, 
                         weighted_edge_list=weighted_edge_list, edge_scale=edge_scale, 
                            arrows=False, edge_threshold=edge_threshold,
                         node_scale=node_scale, node_weight=scdata.traj.nodes, node_labels=True, 
                         fig_size =fig_size, node_pos=node_pos,
                         umap_args=umap_args, save_fig=save_fig, title = "Similarity")
        
    else:
        trajectory_view_hidden(feature_mtx, cluster, 
                         edge_list=None, weighted_edge_list=weighted_edge_list, 
                         edge_scale=edge_scale, edge_weight=True, arrows=False, edge_threshold=edge_threshold,
                         node_scale=node_scale, node_weight=node_weight, node_labels=True, node_pos=node_pos,
                               save_fig=save_fig)


    
def trajectory_view(scdata, use_traj=None, edge_threshold=0.2, edge_scale=0.5, node_scale=0.1, node_labels=True,
                       node_pos=True, show_umap=True, resolution=1.0, fig_size=(12,12),
                       umap_args={'edge_scale': 0.05,
                             'node_scale': 0.5},
                       save_fig=None):
    scdata.to_numpy()

    feature_mtx = scdata.traj.expr
    if use_traj is None:
        weighted_edge_list = scdata.traj.traj_edge_list
    else:
        weighted_edge_list = use_traj
    cluster = list(range(len(scdata.traj.nodes)))
    node_weight = scdata.traj.nodes
    
    if show_umap:
        umap_args['feature_mtx']=scdata.gae.umap_out
        umap_args['edge_list']=scdata.graph.edge_list
        umap_args['cluster'] = scdata.cluster
        
        trajectory_view_umap(feature_mtx, cluster=cluster, 
                         weighted_edge_list=weighted_edge_list, edge_scale=edge_scale, 
                            arrows=True, edge_threshold=edge_threshold,
                         node_scale=node_scale, node_weight=scdata.traj.nodes, node_labels=True, 
                         fig_size =fig_size, node_pos=node_pos,
                         umap_args=umap_args, save_fig=save_fig, title = "TRAJ")
        
    else:
        trajectory_view_hidden(feature_mtx, cluster, 
                         edge_list=None, weighted_edge_list=weighted_edge_list, 
                         edge_scale=edge_scale, edge_weight=True, arrows=True, edge_threshold=edge_threshold,
                         node_scale=node_scale, node_weight=node_weight, node_labels=True, node_pos=node_pos,
                               save_fig=save_fig)


def trajectory_view_hidden(feature_mtx, cluster=None,
                     edge_list=None, weighted_edge_list=None, edge_scale=0.5, edge_weight=False, 
                           arrows=True, edge_threshold=0.2,
                     node_scale=0.1, node_weight=None, node_labels=False, node_pos=True, save_fig=None):
    
    num_nodes = feature_mtx.shape[0]
    
    plt.rcParams['figure.dpi'] = 500

    
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    if edge_list is not None:
        G.add_edges_from(edge_list)
    else:
        new_weighted_edge_list = []
        for w_edge in weighted_edge_list:
            if w_edge[2]>= edge_threshold:
                new_weighted_edge_list.append(w_edge)
            else:
                new_weighted_edge_list.append((w_edge[0], w_edge[1], 0.0))
        
        G.add_weighted_edges_from(new_weighted_edge_list)
    
    pos = {n: feature_mtx[n] for n in G.nodes()} if node_pos else None

    nodecolor = 'blue' if cluster is None else [cluster[n] for n in G.nodes()]
    
    if edge_weight:
        weights = [edge[2]['weight'] for edge in G.edges(data=True)]
        weights = [x * edge_scale for x in weights]
    else:
        weights = edge_scale

    if node_weight is not None:
        node_size = [x * node_scale for x in node_weight]
    else:
        node_size =node_scale
    
    nx.draw_networkx(G, pos=pos, width=weights, with_labels=node_labels, 
                     arrows=arrows, arrowstyle='->',
                     node_size=node_size, 
                     node_color=nodecolor, edge_color='gray', connectionstyle="arc3,rad=0.1",
                    nodelist = range(num_nodes))
    plt.axis('equal')
    if save_fig is not None:  
        plt.savefig(save_fig, dpi=500)
    plt.show()
    

def trajectory_view_umap(feature_mtx, cluster=None,
                         weighted_edge_list=None, edge_scale=0.5, arrows=True, edge_threshold=0.2,
                         node_scale=0.1, node_weight=None, node_labels=False, fig_size=(12,12),
                         umap_args=None, node_pos=True, save_fig=None, title = "TRAJ"):
    
    #
    # trajectory graph:
    #
    
    num_nodes = feature_mtx.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    new_weighted_edge_list = []
    for w_edge in weighted_edge_list:
        if w_edge[2]>= edge_threshold:
            new_weighted_edge_list.append(w_edge)
        else:
            new_weighted_edge_list.append((w_edge[0], w_edge[1], 0.0))
    G.add_weighted_edges_from(new_weighted_edge_list)
    
    pos = {n: feature_mtx[n] for n in G.nodes()} if node_pos else None
    nodecolor = 'blue' if cluster is None else [cluster[n] for n in G.nodes()]
    weights = [edge[2]['weight'] for edge in G.edges(data=True)]
    weights = [x * edge_scale for x in weights]
    node_size = [x * node_scale for x in node_weight]
    
    #
    # umap graph:
    #
    if umap_args is not None:
        num_nodes_umap = umap_args['feature_mtx'].shape[0]
        G_umap = nx.DiGraph()
        G_umap.add_nodes_from(range(num_nodes_umap))
        G_umap.add_edges_from(umap_args['edge_list'])

        pos_umap = {n: umap_args['feature_mtx'][n] for n in G_umap.nodes()}
        nodecolor_umap = 'blue' if umap_args['cluster'] is None else [umap_args['cluster'][n] for n in G_umap.nodes()]
        weights_umap = umap_args['edge_scale']
        node_size_umap = umap_args['node_scale']
        
    plt.rcParams['figure.dpi'] = 500
    fig, axes = plt.subplots(1, 2, figsize=fig_size, sharex=True, sharey=True)
    
    # plot umap
    ax = axes[0]
    nx.draw_networkx(G_umap, pos=pos_umap, width=weights_umap, with_labels=None, arrows=0, 
                     node_size=node_size_umap, node_color=nodecolor_umap, edge_color='gray', 
                     connectionstyle="arc3,rad=0.1", nodelist = None, ax=ax)
    ax.set_title('UMAP')
    ax.axis("off")
    
    # plot traj
    ax = axes[1]
    nx.draw_networkx(G, pos=pos, width=weights, with_labels=node_labels, 
                     arrows=arrows, arrowstyle='->', 
                     node_size=node_size, 
                     node_color=nodecolor, edge_color='gray', connectionstyle="arc3,rad=0.1",
                    nodelist = range(num_nodes), ax=ax)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    if save_fig is not None:  
        plt.savefig(save_fig, dpi=500)
    plt.show()
    
    
def diffusion_view(scdata, layout = [3,3], fig_size = (12,12), 
                   clusters=False, edges=False, 
                   edge_scale=0.02, node_scale=0.5, title=True, save_fig=None):
    
    scdata.to_numpy()
    
    pos_list = scdata.gae.gnd_embed
    
    sub_title = []
    list_edges = []
    list_cluster = []
    for i in range(len(pos_list)):
        sub_title.append(f'Diffusion {i}')
        list_edges.append(scdata.graph.edge_list)
        list_cluster.append(scdata.cluster)
    
    list_cluster = list_cluster if clusters else None
    list_edges = list_edges if edges else None
    
    sub_title = sub_title if title else None

    umap_subplots_view_hiden(pos_list=pos_list, edges=list_edges, clusters=list_cluster, sub_title=sub_title,
                   layout = layout, fig_size = fig_size,  
                   edge_scale=edge_scale, node_scale=node_scale, save_fig=save_fig)
 

    
def gnd_view(scdata, fig_size = (8,4), 
                   clusters=False, edges=False, 
                   edge_scale=0.02, node_scale=0.5, save_fig=None):
    
    scdata.to_numpy()
    
    pos_list = []
    pos_list.append(scdata.gae.umap_in)
    pos_list.append(scdata.gae.umap_out)
    
    sub_title = []
    sub_title.append('Before diffusion')
    sub_title.append('After diffusion')
    
    if edges:
        list_edges = []
        list_edges.append(scdata.graph.edge_list)
        list_edges.append(scdata.graph.edge_list)
    else:
        list_edges=None
    if clusters:
        list_cluster = []
        list_cluster.append(scdata.cluster)
        list_cluster.append(scdata.cluster)
    else:
        list_cluster=None

    
    umap_subplots_view_hiden(pos_list=pos_list, edges=list_edges, clusters=list_cluster, sub_title=sub_title,
                   layout = [1,2], fig_size = fig_size,  
                   edge_scale=edge_scale, node_scale=node_scale, save_fig=save_fig)
    
def clustering_comparation_view(scdata, other=None, fig_size = (8,4), 
                                edges=False, edge_scale=0.02, node_scale=0.5, save_fig=None):
    
    scdata.to_numpy()
    if other is not None:
        other.to_numpy()
    
    pos_list = []
    umap_pos = other.gae.umap_out if other is not None else scdata.record.cluster[-1]['umap']
    pos_list.append(scdata.gae.umap_out)
    pos_list.append(umap_pos)
    pos_list.append(scdata.gae.umap_out)
    pos_list.append(umap_pos)
    
    sub_title = []
    sub_title.append('UMP-Round 1, Clustering-Round 1')
    sub_title.append('UMP-Round 2, Clustering-Round 1')
    sub_title.append('UMP-Round 1, Clustering-Round 2')
    sub_title.append('UMP-Round 2, Clustering-Round 2')
    
    if edges:
        list_edges = []
        edge_data = other.graph.edge_list if other is not None else scdata.record.cluster[-1]['edge_list']
        list_edges.append(scdata.graph.edge_list)
        list_edges.append(edge_data)
        list_edges.append(scdata.graph.edge_list)
        list_edges.append(edge_data)
    else:
        list_edges=None

    list_cluster = []
    cluster_data = other.cluster if other is not None else scdata.record.cluster[-1]['cluster']
    list_cluster.append(scdata.cluster)
    list_cluster.append(scdata.cluster)
    list_cluster.append(cluster_data)
    list_cluster.append(cluster_data)

    umap_subplots_view_hiden(pos_list=pos_list, edges=list_edges, clusters=list_cluster, sub_title=sub_title,
                   layout = [2,2], fig_size = fig_size,  
                   edge_scale=edge_scale, node_scale=node_scale, save_fig=save_fig)
    
    
    
def umap_subplots_view_hiden(pos_list, edges=None, clusters=None, sub_title=None,
                   layout = [3,3], fig_size = (12,12),  
                   edge_scale=0.02, node_scale=0.5, save_fig=None):
    
    plt.rcParams['figure.dpi'] = 500
    fig, axes = plt.subplots(layout[0], layout[1], figsize=fig_size, sharex=True, sharey=True)
    
    for i in range(len(pos_list)):
        pos = pos_list[i]
        
        cluster = clusters[i] if clusters is not None else None
        edge_list = edges[i] if edges is not None else None
        
        edge_size = edge_scale[i] if isinstance(edge_scale, list) else edge_scale
        node_size = node_scale[i] if isinstance(node_scale, list) else node_scale
        
        title = sub_title[i] if isinstance(sub_title, list) else sub_title

        num_nodes = pos.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        if edges:
            G.add_edges_from(edge_list)
        nodecolor = 'blue' if cluster is None else [cluster[n] for n in G.nodes()]

        row = i // layout[1]
        col = i % layout[1]
        ax =axes[col] if layout[0]==1 else axes[row,col]
        
        if edge_list is None:
            nx.draw_networkx_nodes(G, pos, nodelist=None, node_size=node_size, 
                                   node_color=nodecolor, ax=ax)
        else:
            nx.draw_networkx(G, pos=pos, width=edge_size, with_labels=None, arrows=0, node_size=node_size, 
                         node_color=nodecolor, edge_color='gray', connectionstyle="arc3,rad=0.1",
                        nodelist = None, ax=ax) 
        if title is not None:
            ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    if save_fig is not None:  
        plt.savefig(save_fig, dpi=500)
    plt.show()
    
    
    
# def diffusion_view(scdata, layout = [3,3], fig_size = (12,12), 
#                    clusters=False, edges=False, 
#                    edge_scale=0.02, node_scale=0.5, save_fig=None):
    
#     scdata.to_numpy()
    
#     cluster = scdata.cluster if clusters else None
#     edge_list = scdata.graph.edge_list if edges else None

#     num_nodes = scdata.gae.gnd_embed[0].shape[0]
#     G = nx.DiGraph()
#     G.add_nodes_from(range(num_nodes))
#     if edges:
#         G.add_edges_from(edge_list)
#     nodecolor = 'blue' if cluster is None else [cluster[n] for n in G.nodes()]
    
#     plt.rcParams['figure.dpi'] = 500
#     fig, axes = plt.subplots(layout[0], layout[1], figsize=fig_size, sharex=True, sharey=True)
    
#     for i in range(len(scdata.gae.gnd_embed)):
        
#         pos = scdata.gae.gnd_embed[i]

#         row = i // layout[1]
#         col = i % layout[1]
#         ax = axes[row,col]
        
#         if edge_list is None:
#             nx.draw_networkx_nodes(G, pos, nodelist=None, node_size=node_scale, 
#                                    node_color=nodecolor, ax=ax)
#         else:
#             nx.draw_networkx(G, pos=pos, width=edge_scale, with_labels=None, arrows=0, node_size=node_scale, 
#                          node_color=nodecolor, edge_color='gray', connectionstyle="arc3,rad=0.1",
#                         nodelist = None, ax=ax)    
#         ax.set_title(f'Diffusion: {i}')
#         ax.axis("off")

#     plt.tight_layout()
#     if save_fig is not None:  
#         plt.savefig(save_fig, dpi=500)
#     plt.show()
    

