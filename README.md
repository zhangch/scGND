# About:

scGND(Single-cell Graph Neural Diffusion) is a physics-informed graph generative model to do scRNA-seq analysis. scGND investigates cellular dynamics utilizing an attention-based neural network. Unlike methods focusing solely on gene expression in individual cells, scGND targets cell-cell association graph by incorporating two distinct physical effects: local and global equilibrium. It has great potential to apply to multiple scenarios in scRNA-seq data analysis. To better understand the model, we implement it to clustering analysis guided by attention-weighted modularity and trajectory prediction directed by inter-cluster attention network. We demonstrate the balance between local and global equilibrium effects are particularly beneficial for clustering and trajectory determination. Within latent clusters, the local equilibrium effect amplifies the attention-weighted modularity during the diffusion process, resulting to improved clustering accuracy. Simultaneously, the global equilibrium effect strengthens inter-relationships among different clusters, aiding in the accurate prediction of trajectories. As a deep learning neural network with solid mathematical foundations and rich physical explanations, scGND provided a comprehensive generative model based on cell graph diffusion and showed great potential in scRNA-seq data analysis both theoretically and practically.

This repository contains the source code for the paper "scGND: Graph neural diffusion model enhances single-cell RNA-seq analysis", Yu-Chen Liu, Anqi Zou, Simon Liang Lu, Jou-Hsuan Lee, Juexin Wang*, and Chao Zhang*.

# Installation:

Grab this source codes:
```
git clone https://github.com/zhangch/scGND.git
cd scGND
```
Python=3.9.9 is required. See other requirements in the file requirements.txt.

# Tutorials:

For clustering tasks, please check the notebook file "scGND_clustering.ipynb". 

For trajectory tasks, please check the notebook file "scGND_trajectory.ipynb".

To view the UMAP and modularity changes during the diffusion process, please check the notebook file "scGND_view_diffusion.ipynb".
