import numpy
import torch

from GND.gnd import Attention_Weight_Sum, Attention_Inner_Product, Attention_Distance


def call_attention(scdata, data_type="gae_output", use_graph="graph", distance_adjust=False, attention_type=None, num_heads_diffusion=None, dropout=None):
    
    scdata.to_torch()
    
    if data_type=="gae_output":
        nodes_features = scdata.gae.output
    elif data_type=="gae_input":
        nodes_features = scdata.gae.input
    else:
        pass
    
    # select graph to call attention
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
        
    model_dict = scdata.gae.model
    
    graph_object = hidden_call_attention(nodes_features, graph_object, model_dict, distance_adjust=distance_adjust, attention_type=attention_type, num_heads_diffusion=num_heads_diffusion, dropout=dropout, gae_args=scdata.gae.args)
    
    if use_graph == "graph":
        scdata.graph = graph_object
    elif use_graph == "raw_graph":
        scdata.raw_graph = graph_object
    elif use_graph == "dif_graph":
        scdata.dif_graph = graph_object
    elif use_graph == "graph_le":
        scdata.graph_le = graph_object
    elif use_graph == "graph_ge":
        scdata.graph_ge = graph_object
    else:
        raise Exception(f'use_graph should be one of ("graph", "raw_graph", "dif_graph", "graph_le", "graph_ge") but got {graph_name}.')
    
    return scdata

def call_gnd_attention(scdata, distance_adjust=False, attention_type=None, num_heads_diffusion=None, dropout=None):
    
    assert len(scdata.gae.gnd) == len(scdata.gae.gnd_graph), f'Please run build_gnd_graph first.'
    
    model_dict = scdata.gae.model
    
    for i in range(len(scdata.gae.gnd)):
        nodes_features = scdata.gae.gnd[i]
        graph_object = scdata.gae.gnd_graph[i]
        graph_object = hidden_call_attention(nodes_features, graph_object, model_dict, distance_adjust=distance_adjust, attention_type=attention_type, num_heads_diffusion=num_heads_diffusion, dropout=dropout, gae_args=scdata.gae.args)
        scdata.gae.gnd_graph[i] = graph_object
        
    return scdata
        


def hidden_call_attention(nodes_features, graph_object, model_dict, distance_adjust=False, attention_type=None, num_heads_diffusion=None, dropout=None, gae_args=None):
    
    num_features = nodes_features.shape[1]
    nodes_features = nodes_features.view(-1, 1, num_features)
    data = (nodes_features, graph_object.edge_index)
    
    if num_heads_diffusion is None:
        num_heads_diffusion = gae_args["num_heads_diffusion"] 
    if attention_type is None:
        attention_type = gae_args["attention_type"] 
    if dropout is None:
        dropout = gae_args["dropout"]
    
    # attention shape = (E, NH, 1)
    if attention_type == "sum":
        attention = att_weight_sum(data, model_dict, num_features, num_heads_diffusion)
    elif attention_type == "prod":
        attention = att_inner_product(data, model_dict, num_features, num_heads_diffusion)
    elif attention_type == "dist":
        attention = att_distance(data, model_dict, num_features, num_heads_diffusion)
    else:
        raise Exception(f'No such attention type {attention_type}.')
        
    nodes_features = nodes_features.view(-1, num_features)
    
    graph_object.heads_attention = attention.squeeze(-1)
    graph_object.attention = graph_object.heads_attention.mean(dim=1,keepdim=False)
    

    adjusted_attention, edge_distance_inverse, edge_distance = adjust_attention(nodes_features, graph_object.edge_index, graph_object.attention, distance_adjust=distance_adjust)
    
    graph_object.adjusted_attention = adjusted_attention
    graph_object.edge_distance_inverse = edge_distance_inverse
    graph_object.edge_distance = edge_distance
    
    return graph_object


def att_weight_sum(data, model_dict, num_features, num_heads):
    
    scoring_fn_target = model_dict['gnn.gnd_layer.attention_layer.scoring_fn_target']
    scoring_fn_source = model_dict['gnn.gnd_layer.attention_layer.scoring_fn_source']
    
    attention_layer = Attention_Weight_Sum(num_features_diffusion = num_features, 
                                           num_of_heads = num_heads, 
                                           recover=True, 
                                           scoring_fn_target=scoring_fn_target, 
                                           scoring_fn_source=scoring_fn_source)
        
    attention, other = attention_layer(data)
    
    return attention

        
def att_inner_product(data, model_dict, num_features, num_heads):
    
    metric_weights = model_dict['gnn.gnd_layer.attention_layer.metric_weights']
    
    attention_layer = Attention_Weight_Sum(num_features_diffusion = num_features, 
                                           num_of_heads = num_heads, 
                                           recover=True, 
                                           metric_weights=metric_weights)
        
    attention, other = attention_layer(data)
    
    return attention
        

def att_distance(data, model_dict, num_features, num_heads):
    
    edge_dims_weights = model_dict['gnn.gnd_layer.attention_layer.edge_dims_weights']
    distance_dims_weights = model_dict['gnn.gnd_layer.attention_layer.distance_dims_weights']
    
    attention_layer = Attention_Weight_Sum(num_features_diffusion = num_features, 
                                           num_of_heads = num_heads, 
                                           recover=True, 
                                           edge_dims_weights=edge_dims_weights, 
                                           distance_dims_weights=distance_dims_weights)
        
    attention, other = attention_layer(data)
    
    return attention


def adjust_attention(nodes_features, edge_index, attention, distance_adjust=False):
    
    num_of_nodes = nodes_features.shape[0]
    
    edge_trg = edge_index[0]
    
    adjusted_attention = attention
    for i in range(num_of_nodes):
        count = edge_trg.eq(i).sum().item()
        adjusted_attention = adjusted_attention.where(edge_trg!=i, adjusted_attention*count)
        
    
    edge_distance = edge_distance_fn(nodes_features, edge_index)
    edge_distance_ad = edge_distance + edge_distance.mean()
    edge_distance_inverse = edge_distance_ad.reciprocal()
    
    if distance_adjust:   
        adjusted_attention = adjusted_attention * edge_distance_inverse
        
    return adjusted_attention, edge_distance_inverse, edge_distance
    


def edge_distance_fn(nodes_features, edge_index):
    """
    Lifts i.e. duplicates certain vectors depending on the edge index.
    One of the tensor dims goes from N -> E (that's where the "lift" comes from).
    Compute edge vectors.

    """
    src_nodes_index = edge_index[1]
    trg_nodes_index = edge_index[0]


    # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
    nodes_features_sources = nodes_features.index_select(0, src_nodes_index)
    nodes_features_target = nodes_features.index_select(0, trg_nodes_index)
    edge_vectors_matrix = nodes_features_target - nodes_features_sources
    
    edge_distance = edge_vectors_matrix * edge_vectors_matrix
    edge_distance = edge_distance.sum(dim=1,keepdim=False)
    
    return edge_distance



    
    