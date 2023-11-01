import numpy
import pandas as pd
import os

import scipy.io
from scipy.sparse import csr_matrix
import gzip

def export_data(scdata, data=['cluster'], output_path=None, add_folder=True, gzip=True):

    if isinstance(data, str):
        export_single_data(scdata, data=data, output_path=output_path, add_folder=add_folder)
    elif isinstance(data, list):
        for it in data:
            export_single_data(scdata, data=it, output_path=output_path, add_folder=add_folder)
    else:
        raise Exception(f'data should be str or a list of str but got {data}.')
        
    
def export_single_data(scdata, data='cluster', output_path=None, add_folder=True):
    
    scdata.to_device('cpu')
    scdata.to_numpy()

    if output_path is not None:
        if add_folder:
            save_path = output_path + "scdata_exported_data/" 
        else:
            save_path = output_path
    else: 
        save_path = "scdata_exported_data/" 
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_path = save_path + data + ".csv"

    if data=="cluster":
        cluster = scdata.cluster.copy()
        numpy.savetxt(save_path, cluster, delimiter=',')
    elif data=="cell":
        cell = scdata.raw.cell.copy()
        cell_df = pd.DataFrame(cell)
        cell_df.to_csv(save_path, index=False, header=False)
    
    elif data=="gene":
        gene = scdata.svg.gene.copy()
        gene_df = pd.DataFrame(gene)
        gene_df.to_csv(save_path, index=False, header=False)
        
    elif data=="umap":
        umap = scdata.gae.umap_out.copy()
        umap_df = pd.DataFrame(umap)
        umap_df.to_csv(save_path, index=False, header=False)
    
    elif data=="gnd":
        gndif = scdata.gae.output.copy()
        gndif_df = pd.DataFrame(gndif)
        gndif_df.to_csv(save_path, index=False, header=False)
    
    elif data=="svg.expr":
        svg_expr = scdata.svg.expr.copy()
        svg_expr_df = pd.DataFrame(svg_expr)
        svg_expr_df.to_csv(save_path, index=False, header=False)
        
    else:
        raise Exception(f'Can NOT find data name {data}.')
        

def export_10X_data(scdata, path, data_type='raw', use_gzip=True):
    """
    args:   path, data. data included three numpy array valued components (counts_matrix, genes, cells).
    """
    
    if data_type=="raw":     
        data_10x = scdata.raw.expr.copy(), scdata.raw.gene.copy(), scdata.raw.cell.copy() 
        
    elif data_type=="svg":     
        data_10x = scdata.svg.expr.copy(), scdata.svg.gene.copy(), scdata.raw.cell.copy() 
    
    
    counts_matrix, genes, cells = data_10x # 
    
    # compress counts_matrix
    counts_matrix = csr_matrix(counts_matrix).T
    
    if use_gzip:
        with gzip.open(path+'matrix.mtx.gz', 'wb') as f:
            scipy.io.mmwrite(f, counts_matrix)

        with gzip.open(path+'features.tsv.gz', 'wt') as f:
            numpy.savetxt(f, genes, delimiter='\t', fmt='%s')

        with gzip.open(path+'barcodes.tsv.gz', 'wt') as f:
            numpy.savetxt(f, cells, delimiter='\t', fmt='%s')
            
    else:
        scipy.io.mmwrite(path+'matrix.mtx', counts_matrix)

        numpy.savetxt(path+'features.tsv', genes, delimiter='\t', fmt='%s')

        numpy.savetxt(path+'barcodes.tsv', cells, delimiter='\t', fmt='%s')
        
        
        