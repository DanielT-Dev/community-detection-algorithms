# Community Detection using Genetic Algorithms

## Description
Python application for community detection in graphs using a genetic algorithm (locus-based encoding). Supports multiple fitness functions (modularity, conductance, coverage-performance), compares with NetworkX algorithms, and includes visualization of communities and GA convergence.

---

![img1](https://github.com/DanielT-Dev/community-detection-algorithms/blob/main/results/communities_krebs.png?raw=true)
![img2](https://github.com/DanielT-Dev/community-detection-algorithms/blob/main/results/convergence_krebs_modularity.png?raw=true)
![img3](https://github.com/DanielT-Dev/community-detection-algorithms/blob/main/results/communities_dolphins.png?raw=true)
![img4](https://github.com/DanielT-Dev/community-detection-algorithms/blob/main/results/convergence_dolphins_modularity.png?raw=true)

## Features
- Genetic Algorithm for community detection
- Multiple fitness functions:
  - Modularity (Newman–Girvan)
  - Conductance
  - Coverage + Performance
- Built-in NetworkX baseline algorithms
- Graph visualization of communities
- Convergence plots for GA evolution
- Supports GML / GraphML / edge list formats

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-2.2.2-blue?logo=numpy)
![NetworkX](https://img.shields.io/badge/NetworkX-3.6.1-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-orange)

---

## Requirements

```bash
pip install numpy networkx matplotlib
