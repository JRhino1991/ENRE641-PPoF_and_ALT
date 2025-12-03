# 📘 ENRE641 – Physics of Failure (PPoF) and Accelerated Life Testing (ALT)

This repository contains Python modules, Bayesian fitters, and validation notebooks developed as part of **ENRE641 – Physics of Failure and Accelerated Life Testing (University of Maryland)**. The project focuses on **accelerated degradation testing (ADT)**, **accelerated life testing (ALT)**, and **Bayesian reliability modelling**, including full reproduction and extension of **Modarres et al., 2020 – Chapter 5** examples.

---

## 🔧 Repository Structure

```
ENRE641-PPoF_and_ALT/
│
├── adt_fitters/                     # ADT fitting models (LSQ, MLE, Bayesian, POD and Meas Error)
│   └── ADT_fitters.py
│   └── adt_utils.py
│
├── alt_bayesian_fitters/            # ALT Bayesian modules and tests
│   └── ALT_Bayesian_fitters.py
│   └── Utils.py
│   └── test_ALT_Bayesian_fitters.py
│
└── modarres_ch5_validation/         # Reproduction and extension of Chapter 5
    ├── data/                        # All relevant CSV datasets
    └── notebooks/                   # Clean example notebooks (5.1–5.9)
```

---

## 📈 Main Capabilities

### 1. ADT Fitters (`adt_fitters/`)

* LSQ, MLE, and Bayesian parameter estimation
* Diagnostic and residual plots
* Additive & multiplicative noise models
* Predictive degradation-time curves
* Probability of Detection models
* Measurement error models

### 2. ALT Bayesian Fitters (`alt_bayesian_fitters/`)

* emcee-based Bayesian ALT models
* Accelerated life likelihood functions
* Posterior predictive checks
* Full MCMC workflows
* Unit tests included

### 3. Modarres Chapter 5 Validation (`modarres_ch5_validation/`)

Fully reproducible implementations of textbook examples:

* Example 5.1 – Basic degradation
* Example 5.2 – Resistor degradation
* Example 5.3 – LED degradation
* Example 5.4 – Wear/weight-loss
* Example 5.6 – LED luminosity
* Example 5.7 – POD modelling
* Example 5.8 – Crack growth
* Example 5.9 – Crack propagation

---

## 🚀 Usage

### Clone the repo

```
git clone https://github.com/JRhino1991/ENRE641-PPoF_and_ALT.git
cd ENRE641-PPoF_and_ALT
```

### Install dependencies

```
pip install numpy scipy pandas matplotlib seaborn pymc
```

### Open notebooks

```
jupyter notebook
```

Then open:

```
modarres_ch5_validation/notebooks/
```

---

## 🧠 Technologies Used

* Python 3.12+
* NumPy / SciPy
* Pandas
* Matplotlib / Seaborn
* emcee
* arviz
* Jupyter

---

## 🧪 Testing

```
pytest
```

---

## 📄 License

This project is intended for academic and research use.
Please cite appropriately if used in publications.

---

## 👤 Author

**Justin Ryan**
University of Maryland – Reliability Engineering
Australian Army - Corps of Royal Australian Electrical and Mechanical Engineers
