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


This layout separates the core modelling code, Bayesian fitting tools, and validation notebooks, making the repo clean, modular, and easy to extend.

📈 Main Capabilities
ADT Fitters (adt_fitters/)

Square-root Arrhenius degradation model

LSQ, MLE, and Bayesian estimation

Diagnostic and residual plots

Additive and multiplicative noise models

Predictive degradation-time curves

ALT Bayesian Fitters (alt_bayesian_fitters/)

PyMC-based Bayesian ALT models

Accelerated life likelihood functions

Posterior predictive checks

Full MCMC workflows

Included unit tests

Modarres Chapter 5 Validation (modarres_ch5_validation/)

Fully reproducible implementations of textbook examples:

Example 5.1 – Basic degradation

Example 5.2 – Resistor degradation

Example 5.3 – LED degradation

Example 5.4 – Wear / weight-loss

Example 5.6 – LED luminosity decay

Example 5.7 – POD modelling

Example 5.8 – Crack growth

Example 5.9 – Crack propagation simulation

These notebooks extend the original analyses with improved diagnostics, visualisation, and Bayesian estimation.

🚀 Usage
Clone
git clone https://github.com/JRhino1991/ENRE641-PPoF_and_ALT.git
cd ENRE641-PPoF_and_ALT

Install Dependencies
pip install numpy scipy pandas matplotlib seaborn pymc

Open Notebooks
jupyter notebook


Then open:

modarres_ch5_validation/notebooks/

🧠 Technologies Used

Python 3.12+

NumPy, SciPy, Pandas

PyMC

Matplotlib, Seaborn

Jupyter / IPython

🧪 Testing

The Bayesian ALT module includes unit tests:

alt_bayesian_fitters/test_ALT_Bayesian_fitters.py


Run with:

pytest

📄 License

This project is intended for academic and research use.
Please cite appropriately if used in publications.

👤 Author

Justin Ryan
University of Maryland – Reliability Engineering
Australian Army – ESEP / ATEC