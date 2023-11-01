import sys
from pathlib import Path

from scipy.stats import mode
import sklearn
import warnings

sys.path.insert(0, "../")

import scgpt as scg
import scanpy as sc
import scvi
import pandas as pd
import numpy as np

cellfile = './data/Tabula_Muris/barcodes.tsv'
genefile = './data/Tabula_Muris/features.tsv'
celltypefile = './data/Tabula_Muris/labels.csv'
matrixfile = './data/Tabula_Muris/matrix.mtx'
out_embeding_file = 'Tabula_Muris_embedding.h5ad'
out_orin_file = 'Tabula_Muris_original.h5ad'

adata = sc.read(matrixfile)
cell = pd.read_csv(cellfile, sep='\t', header=None)
gene = pd.read_csv(genefile, sep='\t', header=None)
genedata = []
celldata = []
for i in range(len(gene.values)):
    genedata.append(gene.values[i][0].upper())
for i in range(len(cell.values)):
    celldata.append(cell.values[i][0])

celltype = pd.read_csv(celltypefile, header=None)

obs = pd.DataFrame(celltype.values.astype(int), columns=['Celltype'], index=celldata)
var = pd.DataFrame(gene.values[:,0], columns=['Gene Symbol'], index=genedata)
adata_my = sc.AnnData(X=(adata.X).T, obs=obs, var=var)

warnings.filterwarnings("ignore", category=ResourceWarning)

model_dir = Path("./save/scGPT_human")
cell_type_key = "Celltype"
gene_col = "index"


ref_embed_adata = scg.tasks.embed_data(
    adata_my,
    model_dir,
    cell_type_key=cell_type_key,
    gene_col=gene_col,
    batch_size=64,
    return_new_adata=True,
)

# Optional step to visualize the reference dataset using the embeddings
sc.pp.neighbors(ref_embed_adata, use_rep="X")
sc.tl.umap(ref_embed_adata)
sc.pl.umap(ref_embed_adata, color=cell_type_key, frameon=False, wspace=0.4)

sc.tl.leiden(ref_embed_adata, resolution=0.2)
sc.pl.umap(ref_embed_adata, color=['leiden'])

ref_embed_adata.write_h5ad(out_embeding_file)
adata_my.write_h5ad(out_orin_file)

print('done')
