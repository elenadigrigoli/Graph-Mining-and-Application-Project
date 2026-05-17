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

This repository contains an analysis of several explanation methods applied to Graph Neural Networks, developed as a final project of the Graph Mining and Applications course at Sapienza University of Rome.

## Repository Structure 

- `GNNShap`: [Modified version of Official epository](https://github.com/HipGraph/GNNShap) related to GNNShap s for explainability in Graph Neural Networks.

- `Notebooks`: Collection of Jupyter notebooks for experiments with Graph Neural Networks, including GCN and GAT models, used for training, evaluation, and visualization.

- `geco_explainer`: [Modified version of Official epository](https://github.com/salvatorecalderaro/geco_explainer) of the GECO explanation method , used to interpret GNN predictions.

- `CausGNN.py`: Script with our custom implementation of a causal-based explanation approach for Graph Neural Networks.

- `GraphExt.py`: Module containing our custom implementations of graph explanation techniques and supporting utilities for model interpretation.

- `Model.py`: File with the GNN architectures (e.g., GCN, GAT) and integrating explanation methods.

- `OREExplainer.py`: Script with our custom implementation of the ORE (Optimized Rule Extraction) explainer adapted for Graph Neural Networks.

- `utils_graph.py`: Utility functions for fidelity computation. 

