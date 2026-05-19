<div align="center">
  <img src="https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo_text.svg?sanitize=true width="250" height="150">
  <h1>
    <bold> Graph-Mining-and-Application-Project </bold>
  </h1>
  <p>
    <a href = "https://www.python.org/downloads/release/python-3110/">
    <img src="https://img.shields.io/badge/Python-3.11-darkviolet.svg" alt="Python"/>
    </a>
    <a href = "https://www.uniroma1.it/en/pagina-strutturale/home">
    <img src="https://img.shields.io/badge/Sapienza-Università_di_Roma-822433" alt="Sapienza"/>
    </a>
  </p>
  </p>
</div>

This repository branch contains an analysis of several explanation methods applied to Graph Neural Networks for the fraud graph dataset [DGraphFin](https://arxiv.org/pdf/2207.03579), developed as a final project of the Graph Mining and Applications course at Sapienza University of Rome.
To use different explainers (GGNExplainers, MaskGraphExplainer), we relied on the `torch.geometric` library.

Additionally, some explainers were custom-defined based on their respective papers, while others were imported from their official repositories.

## DGraphFin Description

<div align="center">
  <img src="https://dgraph.xinye.com/static/img/new_diagram2.97eee42d.jpg">
</div>

It is a directed, unweighted dynamic graph consisting of millions of nodes and edges, representing a realistic user-to-user social network in financial industry. Node represents a Finvolution user, and an edge from one user to another means that the user regards the other user as the emergency contact person. Each edge is associated with a timestamp ranging from 1 to 821 and a type of emergency contact ranging from 0 to 11.

### Graph Stats

| Number of nodes      |    Number of edges     |  Number of features      |    Number of classes     | 
|:-------:|:-------:|:-------:|:-------:| 
|3,700,550| 4,300,999| 17 | 4| 
 
## Requirements

Install dependencies with:

```bash
pip install -r requirement.txt
```

## Repository Structure 

- `GNNShap`: [Modified version of Official epository](https://github.com/HipGraph/GNNShap) related to GNNShap s for explainability in Graph Neural Networks.

- `Notebooks`: Collection of Jupyter notebooks for experiments with Graph Neural Networks, including GCN and GAT models, used for training, evaluation, and visualization.

- `geco_explainer`: [Modified version of Official epository](https://github.com/salvatorecalderaro/geco_explainer) of the GECO explanation method , used to interpret GNN predictions.

- `CausGNN.py`: Script with our custom implementation of a causal-based explanation approach for Graph Neural Networks.

- `GraphExt.py`: Module containing our custom implementations of graph explanation techniques and supporting utilities for model interpretation.

- `Model.py`: File with the GNN architectures (e.g., GCN, GAT) and integrating explanation methods.

- `OREExplainer.py`: Script with our custom implementation of the ORE (Optimized Rule Extraction) explainer adapted for Graph Neural Networks.

- `utils_graph.py`: Utility functions for explanation evaluation, including keep/deletion fidelity.

## Import and initialize models

### Graph Convolutional Network

```bash
!git clone https://github.com/elenadigrigoli/Graph-Mining-and-Application-Project.git

from Model import GCN

GCN_PARAMS = {
    "hidden_channels": 64,
    "dropout": 0.2,
    "batchnorm": False,
    "lr": 0.01,
    "weight_decay": 5e-7,
}

# Initialize the GCN model
model = GCN(
    in_channels=graph_data.num_node_features,
    hidden_channels=GCN_PARAMS["hidden_channels"],
    out_channels=2,
    dropout=GCN_PARAMS["dropout"],
    batchnorm=GCN_PARAMS["batchnorm"],
).to(device)
```
---

### Graph Attention Network

```bash
!git clone https://github.com/elenadigrigoli/Graph-Mining-and-Application-Project.git

from Model import GAT

GAT_PARAMS = {
    "hidden_channels": 16,
    "dropout": 0.2,
    "batchnorm": False,
    "lr": 0.01,
    "weight_decay": 5e-7,
    "number of heads": 2,
}

# Initialize the GAT model
model = GAT(
    in_channels=graph_data.num_node_features,
    hidden_channels=GAT_PARAMS["hidden_channels"],
    out_channels=2,
    dropout=GAT_PARAMS["dropout"],
    heads=GAT_PARAMS["number of heads"],
).to(device)
```

## Sources

> 1. [Huang et al., *DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection*, arXiv:2207.03579 (2022).](https://arxiv.org/abs/2207.03579)
> 2. [Amato et al., *GECo: a community-based graph neural network explainer*, Computing, 108(4), 55 (2026). Springer.](https://link.springer.com/article/10.1007/s00607-026-01642-z)
> 3. [Akkas, S., & Azad, A. (2024). GNNShap: Scalable and accurate GNN explanation using Shapley values. arXiv:2401.04829](https://arxiv.org/pdf/2401.04829)










