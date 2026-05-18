<div align="center">
<img src="https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo_text.svg?sanitize=true width="250" height="150">
  <h1><b>Graph Mining and Application Project</b></h1>
  <p>
    <a href="https://www.python.org/downloads/release/python-3110/">
      <img src="https://img.shields.io/badge/Python-3.11-darkviolet.svg" alt="Python"/>
    </a>
    <a href="https://www.uniroma1.it/en/pagina-strutturale/home">
      <img src="https://img.shields.io/badge/Sapienza-Università_di_Roma-822433" alt="Sapienza"/>
    </a>
  </p>
</div>

<div align="center">

## Team Members

|      STUDENT       |   ID    |
|:------------------:|:-------:|
| Luca De Ruggiero   | 2174783 |
| Elena Di Grigoli   | 2011814 |
| Flavio Mangione    | 2201201 |

</div>

This repository contains the code for the final project of **Graph Mining and Applications** of the Master's Degree in Data Science at Sapienza University of Rome.
The main goal of this project is to study the interpretability of Graph Neural Network models, using different explainers for fraud detection and molecular data.

## Repository Description

The repository is divided into two different branches. In the first branch, called [**DGraphFin**](https://github.com/elenadigrigoli/Graph-Mining-and-Application-Project/tree/DGraphFin), you can find our implementation for the fraud graph DGraphFin.
In the second branch, called [**B-XAIC**](https://github.com/elenadigrigoli/Graph-Mining-and-Application-Project/tree/B-XAIC), you can find our implementation for the molecular graph dataset B-XAIC.

```bash
git clone https://github.com/elenadigrigoli/Graph-Mining-and-Application-Project.git
# Switch to DGraphFin branch
git checkout DGraphFin
```

## DGraphFin Description

<div align="center">
  <img src="https://dgraph.xinye.com/static/img/new_diagram2.97eee42d.jpg">
</div>

### How to Use

```bash
git clone https://github.com/elenadigrigoli/Graph-Mining-and-Application-Project.git
# Switch to B-XAIC branch
git checkout B-XAIC
```

DGraphFin is a directed, unweighted dynamic graph consisting of millions of nodes and edges, representing a realistic user-to-user social network in the financial industry. Each node represents a Finvolution user, and an edge from one user to another indicates that the user has listed the other as an emergency contact. Each edge is associated with a timestamp ranging from 1 to 821 and an emergency contact type ranging from 0 to 11.

### Graph Stats

| Number of nodes | Number of edges | Number of features | Number of classes |
|:--------------:|:--------------:|:-----------------:|:----------------:|
| 4,300,999      | 3,700,550      | 17                | 2                |

## B-XAIC Description

B-XAIC (Benchmark for eXplainable AI in Chemistry) consists of real-world chemical graphs (~50K molecules) and multiple tasks with ground-truth explanations, enabling direct and reliable assessment of explanation quality. It provides a standardized framework to measure both predictive performance and the faithfulness of explanations, supporting the development of more interpretable and trustworthy GNN models in applications such as drug discovery and cheminformatics.

## Sources

> 1. [Huang et al., DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection, NeurIPS 2023.](https://arxiv.org/pdf/2207.03579)
> 2. [B-XAIC, Benchmarking Explainable AI for Graph Neural Networks Using Chemical Data, arXiv:2505.22252, 2025.](https://arxiv.org/pdf/2505.22252)
