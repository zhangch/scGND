import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import func.info_log as info_log

import umap

def umap_embedding(scdata, data = 'gae.output', 
                   umap_args = {'n_neighbors': 15,
                                'min_dist': 0.3,
                                'metric': 'correlation',
                                'random_state': 2021}):
    scdata.to_numpy()
    
    if data == 'gae.output':
        feature_mtx = scdata.gae.output.copy()
        scdata.gae.umap_out = umap_hidden(feature_mtx, umap_args=umap_args)
    elif data == 'gae.input':
        feature_mtx = scdata.gae.input.copy()
        scdata.gae.umap_in = umap_hidden(feature_mtx, umap_args=umap_args)
    elif data == 'gae.gnd':
        scdata.gae.gnd_embed = []
        for it in scdata.gae.gnd:
            feature_mtx = it.copy()
            scdata.gae.gnd_embed.append(umap_hidden(feature_mtx, umap_args=umap_args))
    
    return scdata

def umap_hidden(data, umap_args = {'n_neighbors': 15,
                                'min_dist': 0.3,
                                'metric': 'correlation',
                                'random_state': 2021}):
 
    reducer = umap.UMAP(n_neighbors=umap_args['n_neighbors'],
                      min_dist=umap_args['min_dist'],
                      metric=umap_args['metric'],
                      random_state=umap_args['random_state'])
    embedding= reducer.fit_transform(data) 
    
    return embedding
   

def special_umap_embedding(scdata, data = 'gae.output', other_1=None, other_2=None, 
                           template=None, return_template=False, concat_1by1=False,
                           umap_args = {'n_neighbors': 15,
                                        'min_dist': 0.3,
                                        'metric': 'correlation',
                                        'random_state': 2021}):
    scdata.to_numpy()
    
    # get other_1 data
    if other_1 is None:
        pass
    elif other_1=="gae.input":
        other_1 = scdata.gae.input
    elif other_1=="gae.output":
        other_1 = scdata.gae.output
    else:
        raise Exception(f'other_1 should be one of ["gae.input", "gae.output"] but got other_1.')
    
    # get other_2 data
    if other_2 is None:
        pass
    elif other_2=="gae.input":
        other_2 = scdata.gae.input
    elif other_2=="gae.output":
        other_2 = scdata.gae.output
    else:
        raise Exception(f'other_2 should be one of ["gae.input", "gae.output"] but got other_2.')
    
    # do umap
    if data == 'gae.output':
        feature_mtx = scdata.gae.output.copy()
        scdata.gae.umap_out = special_umap_hidden(feature_mtx, other_1=other_1, other_2=other_2, 
                                                  template=template, return_template=return_template, 
                                                  umap_args=umap_args)
    elif data == 'gae.input':
        feature_mtx = scdata.gae.input.copy()
        scdata.gae.umap_in = special_umap_hidden(feature_mtx, other_1=other_1, other_2=other_2, 
                                                  template=template, return_template=return_template, 
                                                  umap_args=umap_args)
    if data == 'gae':
        feature_mtx = scdata.gae.output.copy()
        scdata.gae.umap_out = special_umap_hidden(feature_mtx, other_1=other_1, other_2=other_2, 
                                                  template=template, return_template=return_template, 
                                                  umap_args=umap_args)
        feature_mtx = scdata.gae.input.copy()
        scdata.gae.umap_in = special_umap_hidden(feature_mtx, other_1=other_1, other_2=other_2, 
                                                  template=template, return_template=return_template, 
                                                  umap_args=umap_args)
    if data == 'gae.gnd':
        scdata.gae.gnd_embed = []
        feature_mtx = scdata.gae.gnd[0]
        special_umap_hidden_return = special_umap_hidden(feature_mtx, 
                                                              other_1=other_1, other_2=other_2, 
                                                              template=template, return_template=True, 
                                                              concat_1by1=concat_1by1, umap_args=umap_args)
        scdata.gae.gnd_embed.append(special_umap_hidden_return[0])
        umap_template = special_umap_hidden_return[1]
        for it in range(1,len(scdata.gae.gnd)):
            feature_mtx = scdata.gae.gnd[it]
            embedding= special_umap_hidden(feature_mtx, other_1=None, other_2=None, 
                                          template=umap_template, return_template=False, 
                                          concat_1by1=concat_1by1,umap_args=None)
            scdata.gae.gnd_embed.append(embedding)
    
    return scdata


def special_umap_hidden(data, other_1=None, other_2=None, template=None, return_template=False, 
                        concat_1by1=False, use_three=False,
                        umap_args = {'n_neighbors': 15,
                                        'min_dist': 0.3,
                                        'metric': 'correlation',
                                        'random_state': 2021}):
    """
    Do umap embedding for data, return the embeded data and umap template(optional).
    args:
        data:   data to do umap embedding
        (other_1, other_2):   set template_data to make umap template
                              template_data = data                if (None, None)
                              template_data = other_1             if (other_1, None)
                              template_data = (data + other_2)    if (None, other_2)
                              template_data = (other_1 + other_2) if (other_1, other_2)
                            
    """
    if template is not None:
        umaper = template
    else:
        reducer = umap.UMAP(n_neighbors=umap_args['n_neighbors'],
                          min_dist=umap_args['min_dist'],
                          metric=umap_args['metric'],
                          random_state=umap_args['random_state'])
        if use_three:
            if concat_1by1:
                data_list = []
                for i in range(data.shape[0]):
                    data_list.append(other_1[i,:])
                    data_list.append(data[i,:])
                    data_list.append(other_2[i,:])

                temp_data=numpy.array(data_list)


            else:
                temp_data = numpy.concatenate((other_1, data), axis=0)
                temp_data = numpy.concatenate((temp_data, other_2), axis=0)
            
        else:
            temp_data = data if other_1 is None else other_1
            if other_2 is not None:
                if concat_1by1:
                    data_list = []
                    for i in range(data.shape[0]):
                        data_list.append(temp_data[i,:])
                        data_list.append(other_2[i,:])

                    temp_data=numpy.array(data_list)


                else:
                    temp_data = numpy.concatenate((temp_data, other_2), axis=0)
            else:
                pass
        umaper = reducer.fit(temp_data) 
    embedding = umaper.transform(data)
    
    if return_template:
        return embedding, umaper  
    else:
        return embedding
    

def diffusion_umap_embedding(scdata, concat_1by1=False, use_three=False,
                           umap_args = {'n_neighbors': 15,
                                        'min_dist': 0.3,
                                        'metric': 'correlation',
                                        'random_state': 2021}):
    scdata.to_numpy()
    
    scdata.gae.gnd_embed = []
    embedding = umap_hidden(data=scdata.gae.gnd[0], umap_args = umap_args)
    scdata.gae.gnd_embed.append(embedding)
    for it in range(1, len(scdata.gae.gnd)-1):
        feature_mtx = scdata.gae.gnd[it]
        other_1 = scdata.gae.gnd[it-1]
        other_2 = scdata.gae.gnd[it+1]
        embedding = special_umap_hidden(feature_mtx, other_1=other_1, other_2=other_2, 
                                      template=None, return_template=False, 
                                      concat_1by1=concat_1by1, use_three=use_three, umap_args=umap_args)
        scdata.gae.gnd_embed.append(embedding)
        
    embedding = umap_hidden(data=scdata.gae.gnd[-1], umap_args = umap_args)
    scdata.gae.gnd_embed.append(embedding)
    
    return scdata

    
def diffusion_ML_embedding(scdata, data="gae.output", other=True, batch=None, max_epoch=50000, lr=1e-3, dropout=0.0, device='cpu'):
    
    scdata.to_torch()
    scdata.to_device(device)
    
    if data=="gae.output":
        data = scdata.gae.output
        target = scdata.gae.umap_out
        
        if other:
            data = torch.cat((data, scdata.gae.input), 0)
            target = torch.cat((target, scdata.gae.umap_in), 0) 
    elif data=="gae.input":
        data = scdata.gae.input
        target = scdata.gae.umap_in
        
        if other:
            data = torch.cat((data, scdata.gae.output), 0)
            target = torch.cat((target, scdata.gae.umap_out), 0)
            
    data = F.dropout(data, p=dropout)
    
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad() # initialize model's grad as 0
        
        encoded = model(data) 

        # Calculate loss
        loss = F.mse_loss(encoded, target, reduction='sum')

        # Backprop and Update
        loss.backward()
        optimizer.step()

        if (epoch)%2000 == 0:
            info_log.print(f'--------> lr: {lr}, Epoch: {epoch+1}/{max_epoch}, Current loss: {loss}')

               
    info_log.print(f'--------> lr: {lr}, Epoch: {max_epoch}/{max_epoch}, Current loss: {loss}')
    
    # Remove grad and save data
    scdata.gae.gnd_embed = []
    
    # encode diffusion data
    
    for it in scdata.gae.gnd:
        encoded_dif = model(it).detach()
        scdata.gae.gnd_embed.append(encoded_dif)
        
    return scdata

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


   