"""

"""

import numpy as np
import pandas as pd
import scipy.io
from scipy.sparse import csr_matrix
import scipy.sparse as sp

import gzip

import os
import pickle as pkl

import func.info_log as info_log

from scData.scdata_metrics import SCData

def load_10X_data(data_path, is_cell_by_gene=True, is_genes=False):
    """
    args:   data_path
    return: counts_matrix, genes, cells
    """
    info_log.print('--------> Loading from 10X data ...')
    try:
        counts_matrix = scipy.io.mmread(data_path + 'matrix.mtx.gz').T.tocsc().toarray()
    except:
        counts_matrix = scipy.io.mmread(data_path + 'matrix.mtx').T.tocsc().toarray()
    if is_genes:
        try:
            features = pd.read_csv(data_path + 'genes.tsv.gz', header=None, sep='\t')
        except:
            features = pd.read_csv(data_path + 'genes.tsv', header=None, sep='\t')
    else:
        try:
            features = pd.read_csv(data_path + 'features.tsv.gz', header=None, sep='\t')
        except:
            features = pd.read_csv(data_path + 'features.tsv', header=None, sep='\t')   
                    
    genes = np.array(features)
    try: 
        barcodes = pd.read_csv(data_path + 'barcodes.tsv.gz', header=None, sep='\t')
    except:
        barcodes = pd.read_csv(data_path + 'barcodes.tsv', header=None, sep='\t')
    cells = np.squeeze(np.array(barcodes), axis=1)

    scdata = SCData(counts_matrix, cells, genes)
    
    # Record load data information
    scdata.dlog.load = {"cell": len(scdata.raw.cell),
                       "gene": len(scdata.raw.gene)
                      }
    
    info_log.print(f"----------------> Matrix has {len(cells)} cells and {len(genes)} genes")
    
    return scdata

def load_CSV_data(data_path, is_cell_index=True):
    """
    args:   data_path
    return: counts_matrix, genes, cells
    """
    info_log.print('--------> Loading from CSV data ...')
    
    matrix = pd.read_csv(data_path,index_col=0)
    
    if is_cell_index:
        cells = np.array(matrix.index)
        genes = np.array(matrix.columns)
        matrix_np = np.array(matrix)
    else:
        cells = np.array(matrix.columns)
        genes = np.array(matrix.index)
        counts_matrix = np.array(matrix).T

    scdata = SCData(counts_matrix, cells, genes)
    
    # Record load data information
    scdata.dlog.load = {"cell": len(scdata.raw.cell),
                       "gene": len(scdata.raw.gene)
                      }
    
    info_log.print(f"----------------> Matrix has {len(cells)} cells and {len(genes)} genes")
    
    return scdata


def load_CSV_to_10X(path_CSV, path_10X, is_cell_index=True):
    """
    args:   data_path
    return: counts_matrix, genes, cells
    """
    info_log.print('--------> Loading from CSV data ...')
    
    matrix = pd.read_csv(path_CSV,index_col=0)
    
    if is_cell_index:
        cells = np.array(matrix.index)
        genes = np.array(matrix.columns)
        matrix_np = np.array(matrix).T
    else:
        cells = np.array(matrix.columns)
        genes = np.array(matrix.index)
        counts_matrix = np.array(matrix)
        
    info_log.print(f"----------------> Matrix has {len(cells)} cells and {len(genes)} genes")

    matrix_csr = csr_matrix(counts_matrix)
    
    with gzip.open(PATH+'outputs/matrix.mtx.gz', 'wb') as f:
        scipy.io.mmwrite(f, matrix_csr)
        
    with gzip.open(PATH+'outputs/features.tsv.gz', 'wt') as f:
        np.savetxt(f, genes, delimiter='\t', fmt='%s')

    with gzip.open(PATH+'outputs/barcodes.tsv.gz', 'wt') as f:
        np.savetxt(f, cells, delimiter='\t', fmt='%s')
        
    
    return counts_matrix, cells, counts_matrix

