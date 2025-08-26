# 🔬 Thesis Code and Analysis
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Academic](https://img.shields.io/badge/license-academic-lightgrey)](#-license)
[![Last Commit](https://img.shields.io/github/last-commit/ScoopySnack/thesis-code-and-analysis)](https://github.com/ScoopySnack/thesis-code-and-analysis/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/ScoopySnack/thesis-code-and-analysis)](https://github.com/ScoopySnack/thesis-code-and-analysis)

Machine Learning methods for establishing links between **symmetries** and **physical properties** of molecular systems.  
This repository accompanies my MSc thesis (European University Cyprus, 2025) and contains the code, data, and analysis pipelines.

## 📖 Project Overview
Molecular structures can be represented as **graphs**, where atoms correspond to vertices and bonds to edges.  
Using graph theory concepts such as **quotient graphs**, **equitable partitions**, and **spectral descriptors**, this project explores how structural **symmetries** relate to measurable **physical properties** of alkanes.  
By integrating **graph-theoretic descriptors** with **machine learning models**, the project establishes predictive links between molecular structure and macroscopic characteristics such as density, boiling point, dielectric constant, and more.


```

## 🔬 Methodology
1. **Graph Representation**
   - Molecules modeled as graphs (atoms → vertices, bonds → edges).
   - Structural descriptors: adjacency matrices, Laplacians, eigenvalues, quotient graphs.  
2. **Feature Engineering**
   - Graph-based descriptors: Perron–Frobenius eigenvalue, Fiedler eigenvalue, compression ratio.
   - Physical properties: density, boiling/melting point, dielectric constant, dipole moment.  
3. **Machine Learning Models**
   - Regression & classification models for property prediction.
   - Dimensionality reduction (PCA, kernel PCA) and spectral methods.  
4. **Evaluation**
   - Predicted vs. experimental comparisons.
   - Structure–property relationship analysis.

## 🚀 Getting Started
### Requirements
- Python 3.10+
- Recommended packages:
  ```bash
  pip install pandas numpy scikit-learn networkx matplotlib
  ```
### Installation
Clone the repo:
```bash
git clone https://github.com/ScoopySnack/thesis-code-and-analysis.git
cd thesis-code-and-analysis
```
### Usage
Extract features:
```bash
python scripts/feature_extraction.py
```
Train a regression model:
```bash
python scripts/train_model.py
```

## 📊 Dataset Example
Example entry from `alkanesStenutz.json`:
```json
"pentane": {
  "number_ofC": 5,
  "molecular_weight": 72.15,
  "Density": 0.626,
  "boiling_point": 36,
  "dielectric_constant": 1.80,
  "dipole_moment": 0.0,
  "logP": 3.23
}
```
The dataset includes:
- **Structural properties** (C count, molecular weight, molar volume)  
- **Thermodynamic properties** (boiling/melting points, critical values)  
- **Spectral descriptors** (dielectric constant, dipole moment, refractive index)  


## 👩‍💻 Author
**Angeliki Nikolaou**  
MSc in Artificial Intelligence, European University Cyprus  
🔗 [LinkedIn](https://www.linkedin.com/in/angelikinikolaou) · [GitHub](https://github.com/ScoopySnack)

## 📜 License
This repository is for **academic and research purposes only**.  
If you use it, please cite this work appropriately.
"""

