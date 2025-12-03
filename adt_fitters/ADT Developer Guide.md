# ADT Developer Guide

## Introduction

This Developer Guide is intended for contributors who want to extend or maintain the Accelerated Degradation Testing (ADT) model library. While the main README and API Reference document the user-facing interface, this guide explains **how the system works internally**, how to safely add new models, and how to validate contributions.

---

## Library Architecture Overview

The ADT library is structured around a shared base class, `Base_ADT_Model`, which provides all common functionality:

* Data handling (y, t, stress, unit)
* LSQ fitting
* MLE fitting (with additive or multiplicative noise)
* Bayesian inference via `emcee`
* Automatic plotting infrastructure
* TTF evaluation (posterior and MLE plug-in)
* Consistent parameter naming and result storage

Every concrete ADT model class only needs to supply its **model-specific math**, while reusing the common fitting and plotting machinery.

At a high level, each ADT model includes:

1. `_mu()` — the deterministic mean degradation model
2. `_negloglik()` — the model-specific likelihood
3. `_init_guess()` — LSQ starting values
4. `_get_LS_bounds()` — parameter bounds
5. `_model_description()` — Markdown/LaTeX documentation
6. Plotting and TTF helpers (optional)

---

## Base_ADT_Model Deep Dive

`Base_ADT_Model` handles the full workflow:

### **Model Initialization**

On instantiation, the class:

1. Stores all input arrays (y, t, stress, unit)
2. Normalizes shapes and checks validity
3. Sets confidence interval levels
4. Dispatches to the chosen method (`LS`, `MLE`, `Bayesian`)

### **LSQ Fitting**

* Uses SciPy `least_squares`
* Minimizes `(y - mu(theta))`
* Enforces parameter bounds supplied by the subclass
* Stores results in `theta_LS` and `sigma_LS`

### **MLE Fitting**

* Optimizes full likelihood, not squared error
* Supports two noise structures:

  * additive: `D ~ Normal(mu, sigma)`
  * multiplicative: `log(D) ~ Normal(log(mu), sigma)`
* Stores results in `theta_MLE`, `sigma_MLE`, `res_MLE`

### **Bayesian Inference**

* Uses `emcee` with automatically constructed priors
* Can use:

  * implicit priors from MLE/LS
  * explicit user priors
* Returns posterior samples via `*_s` arrays

### **Plotting & Diagnostics**

Automatically available:

* `plot_data()`
* `_plot_fit_LS()`
* `_plot_fit_MLE()`
* `_plot_fit_Bayesian()`
* Residual diagnostics
* Posterior predictive bands

### **TTF Evaluation**

All models can compute time-to-failure distributions using numerical root-finding applied to samples of model parameters and noise.

---

## How Fitting Pipelines Work

All ADT models follow a consistent three-stage pipeline:

### **1. LSQ (deterministic start)**

* Fast
* Provides stable starting point for MLE/Bayes
* Performs basic curve shape fitting

### **2. MLE (statistically correct)**

* Uses noise model
* Supports additive or multiplicative errors
* Provides parameter covariance (approximate)
* Used to centre Bayesian priors

### **3. Bayesian (full uncertainty)**

* Produces posterior distributions for all parameters
* Supports prior injection
* Enables posterior-predictive bands
* Needed for full TTF posterior distributions

---

## Noise Models

The library supports two canonical observational noise structures:

### **Additive Noise**

Used for models where D increases unbounded:

```
D_obs = D_true + epsilon
```

### **Multiplicative (Lognormal) Noise**

Used when the spread scales with D:

```
log(D_obs) = log(D_true) + epsilon
```

Multiplicative noise is preferred for:

* strictly positive degradation values
* exponential growth processes

The Mitsuom model (performance decreasing toward 0) uses its own lognormal-like structure on `(1 - D)`.

---

## Parameter Handling

All models expose parameters consistently:

* Natural scale for deterministic evaluation
* Log-transform for internal optimizations where needed
* Stored under meaningful names (`theta_LS`, `theta_MLE`, `a_s`, `Ea_s`, etc.)

The library enforces:

* positivity where required
* upper/lower clamps when appropriate
* clipping of values to avoid overflow/underflow

---

## Adding a New ADT Model (Complete Tutorial)

Below is a step-by-step guide to extending the library.

---

### **Required Methods**

Each ADT class must implement:

#### **1. `_mu(self, theta, t)`**

Returns the deterministic mean degradation model.

#### **2. `_negloglik(self, x)`**

Returns the negative log-likelihood for the chosen noise model.

#### **3. `_init_guess(self)`**

Returns reasonable LSQ starting values.

#### **4. `_get_LS_bounds(self, p)`**

Returns parameter bounds for LSQ.

#### **5. `_model_description(self)`**

Returns a Markdown+LaTeX block describing the model.

---

### **Optional Methods**

You may also implement:

* `_plot_fit_LS()`
* `_plot_fit_MLE()`
* `_plot_fit_Bayesian()`
* `_plot_use_TTF_distribution()`
* custom stress/time transformations
* custom data scaling or plotting logic

---

### **Model Description Block**

Each class should include a docstring block such as:

```
### Model

The Power–Power damage model is:

$$ D(t, S) = b S^n t^m $$

Parameters:
- b: scale
- n: stress exponent
- m: time exponent
```

This is displayed automatically when `print_results=True`.

---

## Full Template Class

Below is a minimal template for adding a new ADT model:

```python
class Fit_ADT_NewModel(Base_ADT_Model):

    def __init__(..., method="MLE", ...):
        super().__init__(...)
        # method dispatch
        if method == "LS":
            self._fit_LS()
            self._plot_fit_LS()
        elif method == "MLE":
            self._fit_LS(suppress_print=True)
            self._fit_MLE()
            self._plot_fit_MLE()
        elif method.startswith("bayes"):
            self._fit_LS(suppress_print=True)
            self._fit_MLE(suppress_print=True)
            self._fit_Bayesian()
            self._plot_fit_Bayesian()

    def _mu(self, theta, t):
        # return the model’s deterministic mean
        return ...

    def _negloglik(self, x):
        # negative log-likelihood
        return ...

    def _init_guess(self):
        return np.array([...])

    def _get_LS_bounds(self, p):
        return [...]

    def _model_description(self):
        return r"""
### Model
$$ D(t, S) = ... $$
"""
```

---

## Naming Conventions

Consistent naming helps readability:

* `a`, `b`, `n`, `m` → model parameters
* `Ea` → activation energy
* `S`, `T`, `t` → stress, temperature, time
* `theta_*` → parameter vectors
* `_s` suffix → posterior samples
* `Df` → failure threshold
* `*_plot_*` → plotting helpers

---

## Testing & Validation Checklist

Before adding a new model:

### **1. LSQ Fit**

* Does the model reproduce the main curve shape?
* Do residuals look random?

### **2. MLE Fit**

* Does likelihood improve over LSQ?
* Are sigma and parameters reasonable?
* Do noise bands bracket the data?

### **3. Bayesian Fit**

* Do posteriors converge?
* Do predictive bands look plausible?

### **4. TTF Evaluation**

* Does `sample_ttf_from_posterior()` behave correctly?
* Do distributions change sensibly with stress/temperature?

### **5. Numerical Stability**

* Are exponentials clipped where needed?
* Does `_mu()` handle extreme inputs?

### **6. Documentation**

* Does `_model_description()` describe the math?
* Are parameters defined clearly?

---

This developer guide provides the core structure needed for adding, maintaining, and validating new ADT degradation models. 
