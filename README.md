# Machine-Learning Surrogate Models for View Factor Estimation in Benchmark Fire Configurations

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

> **Paper submitted to:** *International Journal of Heat and Mass Transfer*

## Overview

This repository contains the code and data accompanying the paper *"View Factor Estimation via Supervised Learning: A Machine Learning and Deep Learning Approach"*. The work presents a data-driven framework for estimating radiative view factors (VF) using machine learning (ML) and deep learning (DL) surrogate models, benchmarked against high-fidelity contour-integral solutions derived from Stokes' theorem.

Four regression approaches are evaluated (Polynomial Regression, Ridge Regression, Support Vector Regression (SVR), and Multilayer Perceptron (MLP)) across three geometric configurations of increasing complexity relevant to fire science and thermal radiation.

## Key Results

| Case Study | Best Model | R² | MAPE (%) | Speed-up |
|---|---|---|---|---|
| 1 — Plate → Disk | MLP | 0.9999 | 0.31 | 10⁵ |
| 2 — Cylindrical Heater → Element | MLP | 0.9998 | 0.58 | 1.6 × 10⁵ |
| 3 — Cone Calorimeter → Element | MLP | 0.9949 | 0.34 | 10⁶ |

## Repository Structure

```
├── case_1/                          # Rectangular plate and a disk
│   ├── code/
│   │   ├── Neural_network_case_1.ipynb          # MLP surrogate (h, k, d)
│   │   ├── Neural_network_case_1_dimensionless.ipynb  # MLP with π-groups (h/d, k/d, L/d, P/d, R/d)
│   │   ├── Polynomial_case_1.ipynb
│   │   ├── Ridge_case_1.ipynb
│   │   ├── SVR_case_1.ipynb
│   │   └── VF_PLACA_DISCO.py                    # Reference contour-integral solver
│   └── data/
│       └── Case_1.csv
│
├── case_2/                          # Cylindrical heater (I-FIT) and differential element
│   ├── code/
│   │   ├── Neural_network_case_2.ipynb          # MLP surrogate (R, b)
│   │   ├── Neural_network_case_2_dimensionless.ipynb  # MLP with π-groups (R/H, b/H)
│   │   ├── Ridge_case_2.ipynb
│   │   └── VF_IFIT_ALPHA.py                     # Reference contour-integral solver
│   └── data/
│       └── Case_2.csv
│
├── case_3/                          # Cone calorimeter and differential element
│   ├── code/
│   │   ├── Neural_network_case_3.ipynb          # MLP surrogate (h, k, p, d)
│   │   ├── Neural_network_case_3_dimensionless.ipynb  # MLP with π-groups (h/d, k/d)
│   │   └── VF_CONE_CALORIMETER_MAURO.py         # Reference contour-integral solver
│   └── data/
│       └── Case_3.csv
│
├── requirements.txt                 # Python dependencies
├── CITATION.cff                     # Machine-readable citation metadata
├── CONTRIBUTING.md                  # Contribution policy (read-only repository)
├── LICENSE                          # CC BY-NC 4.0
└── README.md
```

## Case Studies

### Case 1: Rectangular Plate and a Disk

View factor F_pl→disk between a rectangular plate (L × P = 100 × 80 mm) and a disk of radius R centered at (h, k) at separation distance d. Dataset: 37,500 samples generated via contour integration. All four ML models are evaluated.

### Case 2: Cylindrical Heater and a Differential Element

View factor F_dA→cyl for the I-FIT (Idealized-Firebrand Ignition Test) configuration. A cylindrical heater of radius R and height H=46 mm, with a differential element at axial position b. Dataset: 40,000 samples. Ridge and MLP models are evaluated.

### Case 3: Cone Calorimeter and a Differential Element

View factor F_dA→HC between a differential element and the helical coil geometry of a cone calorimeter heater (ISO 5660-1 / ASTM E1354). Dataset: 40,401 samples. MLP model only.

## Notebook Contents

Each `Neural_network_case_N.ipynb` notebook includes:

- GPU-accelerated training via PyTorch (CUDA)
- Hyperparameter optimization (Optuna in Case 2)
- Performance metrics: MSE, RMSE, R², MAPE
- Tolerance band analysis (percentage of predictions within ±1%, ±2%, ±5% of true value)
- Comparison against a KNN interpolation baseline (Shepard's method)
- Uncertainty analysis via Monte Carlo dropout
- SHAP analysis (GradientExplainer on GPU): bar plot, beeswarm plot, dependence plots

Each `Neural_network_case_N_dimensionless.ipynb` notebook includes:

- Dimensionless feature formulation via Buckingham Pi theorem
- Same training and evaluation pipeline as the dimensional notebooks
- Comparison table: dimensional vs. dimensionless model performance
- SHAP analysis in dimensionless feature space

### Dimensionless parameter sets

| Case | Reference length | Dimensionless features |
|---|---|---|
| 1 | d (separation distance) | h/d, k/d, L/d, P/d, R/d |
| 2 | H (heater height, 46 mm) | R/H, b/H |
| 3 | d (vertical distance, 30 mm) | h/d, k/d |

## Getting Started

### Requirements

- Python ≥ 3.10
- Jupyter Notebook or JupyterLab
- NVIDIA GPU with CUDA (recommended; notebooks fall back to CPU automatically)
- Dependencies listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/ignaciovermon/IJHMT_ViewFactor_ML_estimation.git
cd IJHMT_ViewFactor_ML_estimation
pip install -r requirements.txt
```

### Usage

Each case study is self-contained. Navigate to the corresponding `case_N/code/` directory and run the Jupyter notebooks:

```bash
cd case_1/code
jupyter notebook Neural_network_case_1.ipynb
```

The notebooks load data from their respective `data/` directories.

## Methodology

1. **View factor computation** — High-fidelity reference values via contour integrals (Stokes' theorem) using adaptive quadrature in Python (`VF_*.py` scripts).
2. **Data generation** — Synthetic datasets generated by sampling geometric parameters over physically meaningful domains.
3. **Model training** — Polynomial, Ridge, SVR, and MLP models trained on 80/20 train-test splits with standardized features.
4. **Hyperparameter optimization** — Grid Search, Bayesian Optimization (BayesSearchCV), Optuna, and information criteria (AIC/BIC).
5. **Validation** — Performance assessed via MSE, RMSE, R², MAPE, and APE on held-out test sets.
6. **Dimensionless analysis** — Buckingham Pi theorem applied to identify physically motivated dimensionless groups; MLP retrained in dimensionless feature space for all three cases.
7. **Explainability** — SHAP values (GradientExplainer) computed for all MLP models to quantify input feature importance.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{mercado2026viewfactor,
  title   = {Machine-Learning Surrogate Models for View Factor Estimation in Benchmark Fire Configurations},
  author  = {Mercado, M. and Pinto, P.E. and Verdugo, I. and Littin, M.
             and Escudero, F. and Demarco, R. and Fuentes, A.},
  journal = {International Journal of Heat and Mass Transfer},
  year    = {2026},
  note    = {Manuscript submitted}
}
```

See also [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## Acknowledgments

This research was funded by Agencia Nacional de Investigación y Desarrollo de Chile (ANID) through grants FONDECYT/REGULAR 1252096 and 1252119, and FONDECYT/INICIACIÓN 11241102.

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). You are free to share and adapt the material for non-commercial purposes, provided you give appropriate credit. See [LICENSE](LICENSE) for details.

## Contact

- **Rodrigo Demarco** (Corresponding author) — rodrigo.demarco@usm.cl
- Departamento de Industrias, Universidad Técnica Federico Santa María, Valparaíso, Chile

---

*This repository accompanies a research paper submitted to the International Journal of Heat and Mass Transfer. Contents will be updated as the manuscript progresses through peer review.*
