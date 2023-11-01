"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import info_log

from auto_encoders.gnd import GND


def graph_diffusion(scdata, D_out, data_type='gae', use_graph='graph', recover_adj_name=None,
                      max_epoch=1000, lr=1e-3, device='cpu',
                           num_features_diffusion=128,
                           num_heads_diffusion=6,
                           num_steps_diffusion=8, 
                           time_increment_diffusion=0.5,
                           attention_type = 'sum', 
                           activation=nn.ELU(),
                           dropout=0.0, 
                           log_attention=False, 
                           log_diffusion=False,
                           encoder=None, 
                           decoder=None,
                           save_model = True,
                           load_model_state = False,
                           loss_adj=0.0,
                           loss_reduction = "sum",
                           rebuild_graph=False,
                           k=100):
    scdata.to_torch()
    scdata.to_device('cpu')
    
    # Record arguments
    scdata.gae.args = {"D_out": D_out, 
                       "data_type": data_type, 
                       "use_graph": use_graph,
                       "recover_adj_name": recover_adj_name,
                       "max_epoch": max_epoch, 
                       "lr": lr, 
                       "device": device,
                           "num_features_diffusion": num_features_diffusion,
                           "num_heads_diffusion": num_heads_diffusion,
                           "num_steps_diffusion": num_steps_diffusion, 
                           "time_increment_diffusion": time_increment_diffusion,
                           "attention_type": attention_type, 
                           "activation": activation,
                           "dropout": dropout, 
                           "log_attention": log_attention, 
                           "log_diffusion": log_diffusion,
                           "encoder": encoder, 
                           "decoder": decoder,
                           "save_model": save_model,
                           "load_model_state": load_model_state,
                           "loss_adj": loss_adj}
    
    info_log.print('--------> Starting Graph AE ...')
    
    # Select data to do diffusion according to 'data_type'
    if data_type=="svg":
        feature_matrix = scdata.svg.log
    elif data_type=="raw":
        feature_matrix = scdata.raw.log
    else:
        raise Exception(f'data_type should be one of ("fae", "svg", "raw") but got {data_type}.')
        
    # Select graph to do diffusion according to 'use_graph'
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
        
    # Select adjacency matrix to recover according to 'recover_adj_name'
    if recover_adj_name is None:
        adj_graph = graph_object
    elif recover_adj_name == "graph":
        adj_graph = scdata.graph
    elif recover_adj_name == "raw_graph":
        adj_graph = scdata.raw_graph
    elif recover_adj_name == "dif_graph":
        adj_graph = scdata.dif_graph
    elif recover_adj_name == "graph_le":
        adj_graph = scdata.graph_le
    elif recover_adj_name == "graph_ge":
        adj_graph = scdata.graph_ge
    else:
        raise Exception(f'recover_adj_name should be one of ("None", "graph", "raw_graph", "dif_graph", "graph_le", "graph_ge") but got {use_graph}.')
    

    D_in = feature_matrix.shape[1]
    
    if encoder is None:
        encoder = None if D_in==num_features_diffusion else [D_in, num_features_diffusion]
    else:
        encoder = [D_in] + encoder + [num_features_diffusion]
    
    if decoder is None:
        decoder = None if D_out==num_features_diffusion else [num_features_diffusion, D_out]
    else:
        decoder = [num_features_diffusion] + decoder + [D_out]
        

    model_gae = Graph_DIF(num_features_diffusion = num_features_diffusion, 
                           num_heads_diffusion=num_heads_diffusion,
                           num_steps_diffusion= num_steps_diffusion, 
                           time_increment_diffusion=time_increment_diffusion,
                           attention_type = attention_type, 
                           activation=activation,
                           dropout=dropout, 
                           log_attention=log_attention, 
                           log_diffusion=log_diffusion,
                           encoder=encoder, 
                           decoder=decoder,
                           rebuild_graph=rebuild_graph).to(device)
#     model_gae= nn.DataParallel(model_gae)
#     model_gae.to(device)
    if load_model_state:
        try: 
            model_gae.load_state_dict(scdata.gae.model)
        except:
            print("Graph autoencoder failed to load model state.")
                            
    optimizer = torch.optim.Adam(model_gae.parameters(), lr=lr)

    for epoch in range(max_epoch):
        model_gae.train()
        optimizer.zero_grad()
        
        if rebuild_graph:
            data = (feature_matrix.to(device), k)
        else:
            data = (feature_matrix.to(device), graph_object.edge_index.to(device))

        out_nodes_features, recon_adj, dif_input, dif_output = model_gae(data)
        
        target_1 = torch.tensor(adj_graph.adj.to(device), dtype = recon_adj.dtype)
        target_2 = torch.tensor(feature_matrix.to(device), dtype = out_nodes_features.dtype)
        
        if loss_adj==1.0:
            loss = F.binary_cross_entropy_with_logits(recon_adj, target_1, reduction=loss_reduction)
        elif loss_adj==0.0:
            loss = F.mse_loss(out_nodes_features, target_2, reduction=loss_reduction)
        else:
            loss_1 = F.binary_cross_entropy_with_logits(recon_adj, target_1, reduction=loss_reduction)
            loss_2 = F.mse_loss(out_nodes_features, target_2, reduction=loss_reduction)
            
            fold = loss_1.item()/loss_2.item()
            loss = loss_adj*loss_1 + (1.0-loss_adj)*fold*loss_2
            
        
        

        # Backprop and Update
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()
        
        if epoch%50 == 0:
            info_log.interval_print(f"----------------> Epoch: {epoch+1}/{max_epoch}, Current loss: {cur_loss:.4f}")
    
    info_log.interval_print(f"----------------> Epoch: {epoch+1}/{max_epoch}, Current loss: {cur_loss:.4f}")
    
    
    # Remove grad and save data
    scdata.gae.input = dif_input.cpu()
    scdata.gae.output = dif_output.detach().cpu()
    
    scdata.gae.indata = feature_matrix.to('cpu')
    scdata.gae.outdata = out_nodes_features.detach().cpu()

    # save model state
    if save_model:
        scdata.gae.model = model_gae.state_dict()
        
    if log_attention:
        scdata.gae.attention = model_gae.attention_weights
        
    if log_diffusion:
        scdata.gae.gnd = []
        for it in range(len(model_gae.diffusion_step_outputs)):
            scdata.gae.gnd.append(model_gae.diffusion_step_outputs[it].detach().cpu())
        scdata.gae.gnd.append(dif_output.detach().cpu())
        
        
    
    return scdata 


class Graph_DIF(nn.Module):
    def __init__(self, num_features_diffusion,
                           num_heads_diffusion,
                           num_steps_diffusion, 
                           time_increment_diffusion,
                           attention_type = 'sum', 
                           activation=nn.ELU(),
                           dropout=0.0, 
                           log_attention=False, 
                           log_diffusion=False,
                           encoder=None, 
                           decoder=None,
                           rebuild_graph=False):
        super().__init__()
        
        self.log_attention = log_attention
        self.log_diffusion=log_diffusion
        
        self.attention_weights = None
        self.diffusion_step_outputs = None
        
        self.gnn = GND(num_features_diffusion = num_features_diffusion, 
                           num_heads_diffusion=num_heads_diffusion,
                           num_steps_diffusion= num_steps_diffusion, 
                           time_increment_diffusion=time_increment_diffusion,
                           attention_type = attention_type, 
                           activation=activation,
                           dropout=dropout, 
                           log_attention=log_attention, 
                           log_diffusion=log_diffusion,
                           encoder=encoder, 
                           decoder=decoder,
                           rebuild_graph=rebuild_graph)

        self.decode = InnerProductDecoder(0, act=lambda x: x)
        #self.decode = InnerProductDecoder(0, act=torch.sigmoid)


    def forward(self, data):
        # [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        
        data, dif_input, dif_output = self.gnn(data)
        
        out_nodes_features, edge_index = data
        
        recon_adj = self.decode(out_nodes_features)
        
        if self.log_attention:
            self.attention_weights = self.gnn.attention_weights
            
        if self.log_diffusion:
            self.diffusion_step_outputs = self.gnn.diffusion_step_outputs
        
        return out_nodes_features, recon_adj, dif_input, dif_output
        
    
    
class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super().__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

