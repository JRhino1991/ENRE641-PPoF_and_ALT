# ADT Model API Reference

This section documents the Accelerated Degradation Testing (ADT) model classes built on top of `Base_ADT_Model`.

All ADT classes share a **common interface**:

* You instantiate a model class with your degradation data and options.
* The constructor immediately runs the requested fitting procedure (`method="LS"`, `"MLE"`, `"Bayesian"`, …).
* Results are stored as attributes (e.g. `theta_LS`, `theta_MLE`, `a_s`, …), and standard plotting / TTF methods are available.

---

## 1. Common Interface

All ADT model classes follow the same basic signature:

```python
model = Some_ADT_Model_Class(
    degradation,
    stress,
    time,
    unit,
    stress_use,
    Df,
    CI=0.95,
    method="MLE",
    noise="additive",
    show_data_plot=False,
    show_LSQ_diagnostics=False,
    show_noise_bounds=True,
    show_use_TTF_dist=False,
    print_results=True,
    data_scale="linear",
    priors=None,
    **kwargs,
)
```

### 1.1. Core Arguments

* **`degradation`** (`array-like`): Observed response for each measurement.
* **`stress`** (`array-like`): Stress level (e.g. °C or load).
* **`time`** (`array-like`): Time or cycles for each observation.
* **`unit`** (`array-like of int`): Unit IDs.
* **`stress_use`** (`float`): Use-level stress.
* **`Df`** (`float`): Failure threshold.
* **`CI`** (`float`): Confidence/credible interval.
* **`method`** (`{"LS", "MLE", "MLE_hierarchical", "Bayesian"}`): Fitting method.
* **`noise`** (`{"additive", "multiplicative"}`): Observation noise model.
* **`show_data_plot`** (`bool`): Plot raw data.
* **`show_LSQ_diagnostics`** (`bool`): Show LS diagnostics.
* **`show_noise_bounds`** (`bool`): Show uncertainty bands.
* **`show_use_TTF_dist`** (`bool`): Plot TTF distribution.
* **`print_results`** (`bool`): Print Markdown result tables.
* **`data_scale`** (`{"linear", "ylog", "xlog", "loglog"}`): Axis scaling.
* **`priors`** (`dict or None`): Bayesian priors.

### 1.2. Common Attributes

* **LSQ fit**: `theta_LS`, `sigma_LS`
* **MLE fit**: `theta_MLE`, `sigma_MLE`, `res_MLE`
* **Bayesian fit**: Posterior samples (`a_s`, `b_s`, `n_s`, etc.)

### 1.3. Common Methods

* `plot_data()` – Plot raw data.
* `_plot_fit_LS()` – LS mean curve plots.
* `_plot_fit_MLE()` – MLE mean curve + noise bands.
* `_plot_fit_Bayesian()` – Bayesian posterior/predictive curves.
* `_plot_use_TTF_distribution()` – Bayesian TTF at use.
* `_plot_use_TTF_distribution_MLE()` – MLE plug-in TTF.

---

## 2. `Fit_ADT_Power_Arrhenius`

### 2.1. Model

Damage model (increasing) with temperature acceleration:

[ D(t, T) = a , t^{n} \exp\left( \frac{E_a}{k_B}\left(\frac{1}{T_{use}} - \frac{1}{T}\right) \right) ]

### 2.2. Constructor

```python
Fit_ADT_Power_Arrhenius(...)
```

---

## 3. `Fit_ADT_Power_Exponential`

### 3.1. Model

Damage model with power law in time + exponential in stress:

[ D(t, S) = b \exp(a S) t^{n} ]

### 3.2. Constructor

```python
Fit_ADT_Power_Exponential(...)
```

---

## 4. `Fit_ADT_Power_Power`

### 4.1. Model

Generic power–power damage model:

[ D(t, S) = b , S^{n} , t^{m} ]

### 4.2. Constructor

```python
Fit_ADT_Power_Power(...)
```

---

## 5. `Fit_ADT_Mitsuom_Arrhenius`

### 5.1. Model

Performance model (decreasing) with Arrhenius acceleration:

[ D(t, T) = [1 + a (t e^{E_a/k_B(1/T_{use} - 1/T)})^{b}]^{-1} ]

### 5.2. Constructor

```python
Fit_ADT_Mitsuom_Arrhenius(...)
```

---

## 6. Example Usage

```python
from adt_fitters import Fit_ADT_Power_Power

model = Fit_ADT_Power_Power(
    degradation=D,
    stress=S,
    time=t,
    unit=unit,
    stress_use=S_use,
    Df=Df,
    method="MLE",
    noise="multiplicative",
    show_data_plot=True,
)
```

---

## 7. Extending the Library

1. Subclass `Base_ADT_Model`.
2. Implement `_mu`, `_model_description`, `_init_guess`, `_negloglik`.
3. (Optional) Implement Bayesian + TTF + plotting helpers.
