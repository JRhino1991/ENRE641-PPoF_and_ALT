📘 ENRE641 – Physics of Failure (PPoF) and Accelerated Life Testing (ALT)

This repository contains Python modules, Bayesian fitters, and validation notebooks developed as part of ENRE641 – Physics of Failure and Accelerated Life Testing (University of Maryland).
The project focuses on accelerated degradation testing (ADT), accelerated life testing (ALT), and Bayesian reliability modelling, including full reproduction and extension of Modarres et al., 2020 – Chapter 5 examples.

🔧 Repository Structure
ENRE641-PPoF_and_ALT/
│
├── adt_fitters/                     # ADT fitting models (LSQ, MLE, Bayesian)
│   └── __init__.py
│   └── Fit_ADT_sqrt_Arrhenius.ipynb
│   └── (additional ADT fitters)
│
├── alt_bayesian_fitters/            # ALT Bayesian modules and tests
│   └── __init__.py
│   └── ALT_Bayesian_fitters.py
│   └── test_ALT_Bayesian_fitters.py
│
└── modarres_ch5_validation/         # Reproduction and extension of Chapter 5
    ├── data/                        # All relevant CSV datasets
    ├── figures/                     # Exported plots, diagnostics, results
    └── notebooks/                   # Clean example notebooks (5.1–5.9)


This structure separates the core modelling code, Bayesian fitting tools, and validation notebooks, making the repo clean, modular, and easy to extend.

📈 Main Capabilities
1. ADT Fitters Module (adt_fitters/)

Includes implementations for accelerated degradation testing models, such as:

Square-root Arrhenius degradation model

LSQ, MLE, and Bayesian parameter estimation

Diagnostic plots & residual analysis

Noise modelling (additive and multiplicative)

Predictive degradation-time curves

2. ALT Bayesian Fitters Module (alt_bayesian_fitters/)

A growing library of Bayesian ALT tools:

PyMC-based Bayesian ALT models

Accelerated life model likelihood functions

Posterior predictive checks

MCMC sampling workflows

Unit tests demonstrating usage

3. Modarres et al. Chapter 5 Reproductions (modarres_ch5_validation/)

Fully reproducible Python implementations of:

Example 5.1: Basic degradation modelling

Example 5.2: Resistor degradation

Example 5.3: LED degradation

Example 5.4: Wear / weight-loss degradation

Example 5.6: LED luminosity degradation

Example 5.7: POD model

Example 5.8: Crack growth (Paris law-type models)

Example 5.9: Crack propagation simulation

These notebooks reproduce the textbook results and extend them with improved diagnostics, Bayesian estimation, and visualisation.

🧠 Technologies Used

Python 3.12+

NumPy / SciPy

Pandas

Matplotlib / Seaborn

PyMC (for Bayesian fitting)

Jupyter / IPython

Custom ADT and ALT modules built from scratch

🚀 How to Use This Repository
Clone the repo
git clone https://github.com/JRhino1991/ENRE641-PPoF_and_ALT.git
cd ENRE641-PPoF_and_ALT

Install dependencies

(Exact list may vary; adjust your environment as needed.)

pip install numpy scipy pandas matplotlib pymc seaborn

Run notebooks

Open any notebook:

jupyter notebook


All example notebooks and validation studies are located in:

modarres_ch5_validation/notebooks/

📚 Academic Context

This work supports:

Physics-of-Failure modelling

Accelerated degradation / accelerated life test design

Bayesian reliability estimation

Structural health monitoring

Probability of Detection (POD) modelling

Data-driven RUL prediction foundations

The repository is part of the course requirements and project work for ENRE641 – PPoF & ALT and aligns with ongoing research in reliability engineering and PHM.

🧪 Testing

Unit tests for the Bayesian ALT module are included in:

alt_bayesian_fitters/test_ALT_Bayesian_fitters.py


Run them with:

pytest


(Install with pip install pytest if needed.)

📄 License

This project is for academic and research purposes.
Feel free to fork or extend, but please cite appropriately if used in publications.

👤 Author

Justin Ryan
University of Maryland – Reliability Engineering
Australian Army (ESEP / ATEC placement)
