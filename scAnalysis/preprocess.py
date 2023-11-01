
import numpy as np
import func.info_log as info_log
import matplotlib.pyplot as plt

def fast(scdata):
    info_log.print('--------> Preprocessing sc data ...')

    scdata = percentile_filter(scdata, cell_cutoff=10, gene_cutoff=10)
    scdata = varibale_gene_select(scdata, num_select=2000)
    scdata = log_transform(scdata, data_type = "svg")

    return scdata


def percentile_filter(scdata, cell_cutoff=10, gene_cutoff=10):
    """
    Truncate barely expressed genes and cells
    """
    info_log.print('--------> Truncating genes and cells ------- Method: Percentile ...')

    scdata.raw.expr = np.asarray(scdata.raw.expr)
    
    gene_non_zero_counts = np.count_nonzero(scdata.raw.expr, axis=0)
    if gene_cutoff ==0:
        gene_mask = np.ones_like(gene_non_zero_counts, dtype=bool)
    else:    
        gene_mask = gene_non_zero_counts > np.percentile(gene_non_zero_counts, gene_cutoff)

    cell_non_zero_counts = np.count_nonzero(scdata.raw.expr, axis=1)
    if cell_cutoff ==0:
        cell_mask = np.ones_like(cell_non_zero_counts, dtype=bool)
    else:   
        cell_mask = cell_non_zero_counts > np.percentile(cell_non_zero_counts, cell_cutoff)
  
   
    scdata.raw.expr = scdata.raw.expr[cell_mask,:][:,gene_mask]
    scdata.raw.gene = scdata.raw.gene[gene_mask]
    scdata.raw.cell = scdata.raw.cell[cell_mask]
    
    # Record filtered data information
    scdata.dlog.raw = {"cell": len(scdata.raw.cell),
                       "gene": len(scdata.raw.gene)
                      }
    
    info_log.print(f'--------> Loaded (cell, gene): ({ scdata.dlog.load["cell"]}, '+
                   f'{scdata.dlog.load["gene"]}), '+
                   f'Remove: ({scdata.dlog.load["cell"] - scdata.dlog.raw["cell"]}, '+ 
                   f'{scdata.dlog.load["gene"] - scdata.dlog.raw["gene"]}),' + 
                   f'Left: ({ scdata.dlog.raw["cell"]}, '+
                   f'{scdata.dlog.raw["gene"]}).')
    
    return scdata


def key_filter(scdata, gene_key=None, cell_key=None):
    """
    Truncate barely expressed genes and cells
    """
    info_log.print('--------> Truncating genes and cells ------- Method: Key ...')

    gene_mask = [g in gene_key for g in scdata.raw.gene] if gene_key is not None else range(len(scdata.raw.gene))
    cell_mask = [c in cell_key for c in scdata.raw.cell] if cell_key is not None else range(len(scdata.raw.cell))
        
    scdata.raw.expr = scdata.raw.expr[cell_mask,:][:,gene_mask]
    scdata.raw.gene = scdata.raw.gene[gene_mask]
    scdata.raw.cell = scdata.raw.cell[cell_mask]
                       
    # Record filtered data information
    scdata.dlog.raw = {"cell": len(scdata.raw.cell),
                       "gene": len(scdata.raw.gene)
                      }
    
    info_log.print(f'--------> Loaded (cell, gene): ({ scdata.dlog.load["cell"]}, '+
                   f'{scdata.dlog.load["gene"]}), '+
                   f'Remove: ({scdata.dlog.load["cell"] - scdata.dlog.raw["cell"]}, '+ 
                   f'{scdata.dlog.load["gene"] - scdata.dlog.raw["gene"]}),' + 
                   f'Left: ({ scdata.dlog.raw["cell"]}, '+
                   f'{scdata.dlog.raw["gene"]}).')
    
    return scdata


def varibale_gene_select(scdata, data_type="raw.log", num_select=2000, method='product', plot=True):
    info_log.print('---------> Sorting and selecting top genes ...')

    if num_select == -1:
        pass
    else:
        if data_type == "raw.log":
            expr = scdata.raw.log.copy()
        elif data_type == "raw.expr":
            expr = scdata.raw.expr.copy()
        else:
            raise Exception(f'data_type should be one of ("raw.log","raw.expr") but got {data_type}.')
            
        variation = expr.var(axis=0)
        mean = expr.mean(axis=0)
        product = mean*variation
        
        if method=='product':
            idx_data = product
        elif method=='variation':
            idx_data = variation
        elif method=='mean':
            idx_data = mean
            
        gene_idx = idx_data.argsort()[::-1][:num_select]
        
        scdata.svg.expr = expr[:, gene_idx]
        if data_type == "raw.log":
            scdata.svg.log = expr[:, gene_idx]
        
        scdata.svg.gene = scdata.raw.gene[gene_idx]
                       
        # Record filtered data information
        scdata.dlog.svg = {"cell": len(scdata.raw.cell),
                           "gene": len(scdata.svg.gene)
                      }
        if plot:
            plot_variable_gene(variation, mean, gene_idx)
        
    return scdata

def plot_variable_gene(variation, mean, index):

#     mean_n = np.log(mean + 1)
#     variation_n = np.log(variation + 1)

#     mean_n_s = mean_n[index]
#     variation_n_s = variation_n[index]
    
    mean_s = mean[index]
    variation_s = variation[index]

    plt.scatter(mean, variation)

    plt.scatter(mean_s, variation_s, color='red')

    plt.title('Variable gene plot')
    plt.xlabel('mean/log')
    plt.ylabel('variation/log')

    plt.show()


    
def log_transform(scdata, data_type = "raw", log=True, scale=True, scale_factor = 10000):
    info_log.print('---------> Log-transforming data ...')
    
    data = scdata.raw.expr if data_type == "raw" else scdata.svg.expr
    
    data = data.astype(np.float64)
    
    if scale:
        for row in range(data.shape[0]):
            data[row] /= np.sum(data[row])/scale_factor
    
    if log:
        data = np.log(data + 1)
    
    if data_type == "svg":
        scdata.svg.log = data

    elif data_type == "raw":
        scdata.raw.log = data
        
    
    elif data_type == "all":
        scdata.raw.log = np.log(scdata.raw.expr + 1)
        scdata.svg.log = np.log(scdata.svg.expr + 1)
        
    else:
        raise Exception(f'data_type should be one of ("svg","raw","all") but got {data_type}.')
    
    return scdata









