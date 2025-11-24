import numpy as np
import pandas as pd
import emcee
from scipy import stats as ss
from scipy.special import gamma

from reliability.Utils import (
    ALT_fitters_input_checking,
    life_stress_plot,
    colorprint,
    round_and_string,
    log_prior_vector,
    get_param_domain,
)

from reliability.ALT_fitters import (
    Fit_Exponential_Dual_Exponential,
    Fit_Exponential_Dual_Power,
    Fit_Exponential_Exponential,
    Fit_Exponential_Eyring,
    Fit_Exponential_Power,
    Fit_Exponential_Power_Exponential,
    Fit_Lognormal_Dual_Exponential,
    Fit_Lognormal_Dual_Power,
    Fit_Lognormal_Exponential,
    Fit_Lognormal_Eyring,
    Fit_Lognormal_Power,
    Fit_Lognormal_Power_Exponential,
    Fit_Normal_Dual_Exponential,
    Fit_Normal_Dual_Power,
    Fit_Normal_Exponential,
    Fit_Normal_Eyring,
    Fit_Normal_Power,
    Fit_Normal_Power_Exponential,
    Fit_Weibull_Power,
    Fit_Weibull_Exponential,
    Fit_Weibull_Eyring,
    Fit_Weibull_Dual_Exponential,
    Fit_Weibull_Dual_Power,
    Fit_Weibull_Power_Exponential,
)

"""
This module fits life–stress models to accelerated life testing (ALT) data using
maximum likelihood estimation (MLE) via `reliability.ALT_fitters` and, optionally,
Bayesian parameter estimation with MCMC (emcee). For each supported combination of
distribution and life–stress model (e.g., Weibull–Power, Lognormal–Eyring,
Exponential–Dual_Power), the corresponding wrapper:

1) runs the matching MLE fitter from `reliability.ALT_fitters`,
2) constructs the same likelihood for the selected model,
3) samples the posterior with emcee using user-specified priors, and
4) reports posterior summaries and (optionally) posterior plots.

These models are appropriate for common thermal and non-thermal acceleration
relationships used in ALT (e.g., Power, Exponential, Eyring, Dual-Power,
Dual-Exponential, Power-Exponential). Ensure stresses are provided in consistent,
valid units (e.g., temperature in Kelvin if using Eyring; humidity as 0–1 if used).

Parameters
----------
failures : array, list
    Failure times.
failure_stress : array, list, optional
    Single-stress values corresponding to each failure (for single-stress models).
failure_stress_1 : array, list, optional
    First stress for each failure (for dual-stress models).
failure_stress_2 : array, list, optional
    Second stress for each failure (for dual-stress models).
right_censored : array, list, optional
    Right-censored times.
right_censored_stress : array, list, optional
    Single-stress values for each right-censored observation (single-stress models).
right_censored_stress_1 : array, list, optional
    First stress for each right-censored observation (dual-stress models).
right_censored_stress_2 : array, list, optional
    Second stress for each right-censored observation (dual-stress models).
use_level_stress : int, float, array-like, optional
    The use-level stress (or pair of stresses) at which to report the mean life
    and related quantities.
CI : float, optional
    Interval level for MLE confidence limits and Bayesian credible intervals.
    Must be in (0, 1). Default is 0.95.
optimizer : str, optional
    Optimization algorithm passed to the underlying MLE fitter. One of
    'TNC', 'L-BFGS-B', 'nelder-mead', 'powell', or 'best'. See the
    `reliability` documentation for details.

# --- Bayesian controls (optional) ---
priors : dict or mapping, optional
    Prior specifications by parameter name using `log_prior_vector`. Each prior
    may be one of:
      - ("Uniform", low, high)
      - ("Normal", mu, sigma)
      - ("Lognormal", mu_log, sigma_log)  # log(x) ~ N(mu_log, sigma_log^2)
      - ("LogUniform", low, high)         # uniform in log(x), for x>0
      - ("Triangular", low, mode, high)
    If omitted, weakly-informative defaults are used (appropriate positivity
    constraints are respected).
n_walkers : int, optional
    Number of emcee ensemble walkers. Default 32.
n_steps : int, optional
    Total MCMC steps per walker (including burn-in). Default 4000.
burn : int, optional
    Number of initial samples discarded as burn-in. Default 1000.
thin : int, optional
    Thinning factor applied after burn-in. Default 1.
random_seed : int, optional
    Random seed for reproducibility.
progress : bool, optional
    If True, emcee displays a progress bar. Default False.
show_bayes_plots : bool, optional
    If True and ArviZ/matplotlib are available, produces ArviZ trace/pair plots
    and a posterior life–stress curve evaluated at posterior-mean parameters.

Returns
-------
# MLE (from reliability.ALT_fitters)
<model parameters> : float
    Fitted MLE parameters for the chosen life–stress model (names depend on the class;
    e.g., a, b, c, n, m, beta, sigma).
<model parameters>_SE : float
    Standard errors (Wald) for the corresponding MLE parameters (when available).
results : pandas.DataFrame
    MLE point estimate, standard error, and lower/upper CI for each parameter.
goodness_of_fit : pandas.DataFrame
    MLE goodness-of-fit criteria (Log-likelihood, AICc, BIC, and −2*Log-likelihood where applicable).
change_of_parameters : pandas.DataFrame or None
    Per-stress parameter changes reported by the underlying MLE fitter (if applicable).
mean_life : float or None
    Mean life at `use_level_stress` from the MLE fit (if requested).
alpha_at_use_stress or mu_at_use_stress : float or None
    Equivalent Weibull scale (alpha) or Lognormal/Normal location (mu) at `use_level_stress`
    from the MLE fit (if requested).
distribution_at_use_stress : object or None
    Distribution object at the use level stress from the MLE fit (if provided by the base fitter).

# Bayesian outputs (after sampling)
bayes_samples : ndarray, shape (n_draws, P)
    Posterior draws for the model parameters (P depends on the class).
bayes_results : pandas.DataFrame
    Posterior summaries (mean, std, median, and equal-tail CrI at CI level).
<parameter>_post_mean : float
    Posterior means for each parameter (e.g., a_post_mean, beta_post_mean).
mean_life_posterior_mean : float or None
    Posterior-mean of the mean life at `use_level_stress`. For Weibull,
    this uses a plug-in E[T] = alpha_use * Γ(1 + 1/β_post_mean).
az_idata : arviz.InferenceData or None
    ArviZ container for the posterior (use for WAIC/LOO/PPC and plotting).
az_axes : dict
    Matplotlib axes from ArviZ plotting functions (when requested).
bayes_life_stress_plot : object or None
    Figure/axes from the posterior life–stress plot (when requested).

Notes
-----
- MLE is performed by the corresponding `reliability.ALT_fitters` class; the MLE
  outputs are mirrored on the wrapper for consistency.
- The posterior uses the same likelihood as the MLE fitter. Priors are applied
  via `log_prior_vector`, supporting Uniform, Normal, Lognormal, LogUniform,
  and Triangular forms (with domain checks for positive/scale parameters).
- For Bayesian predictive/model comparison, compute WAIC/LOO using `az.waic`
  and `az.loo` on `az_idata` after sampling.
- If your model involves temperature or humidity, ensure inputs follow the
  conventions expected by the underlying life–stress relationship (e.g., Kelvin,
  humidity in 0–1).
"""

# ArviZ is used when available and show_bayes_plots=True
try:
    import arviz as az
    _HAS_AZ = True
except Exception:
    _HAS_AZ = False

# ----------------------------- Likelihood helpers -----------------------------

SQRT2PI = np.sqrt(2.0 * np.pi)

def _logpdf_weibull(t, alpha, beta):
    """log f(t | alpha, beta) for t>0, alpha>0, beta>0"""
    z = t / alpha
    return (beta - 1.0) * np.log(z) + np.log(beta / alpha) - (z ** beta)


def _logsf_weibull(t, alpha, beta):
    """log S(t | alpha, beta)"""
    z = t / alpha
    return - (z ** beta)


def _logpdf_exponential_scale(t, scale):
    """Exponential with scale=scale (mean=scale). log f = -log(scale) - t/scale"""
    return -np.log(scale) - t / scale


def _logsf_exponential_scale(t, scale):
    """log S = -t/scale"""
    return - t / scale


def _logpdf_lognormal(t, mu, sigma):
    """log f for Lognormal(mu,sigma)"""
    z = (np.log(t) - mu) / sigma
    return -0.5 * (z ** 2) - np.log(t * sigma * SQRT2PI)


def _logsf_lognormal(t, mu, sigma):
    """log survival for Lognormal"""
    z = (np.log(t) - mu) / sigma
    return ss.norm.logsf(z)


def _logpdf_normal(t, mu, sigma):
    """log f for Normal(mu,sigma)"""
    z = (t - mu) / sigma
    return -0.5 * (z ** 2) - np.log(sigma * SQRT2PI)


def _logsf_normal(t, mu, sigma):
    """log survival for Normal(mu,sigma)"""
    z = (t - mu) / sigma
    return ss.norm.logsf(z)

# ------------------------- Utility: summarize MCMC posterior --------------------------

def _summarize_chain(samples, names, CI):
    rows = []
    for i, nm in enumerate(names):
        s = samples[:, i]
        mean = float(np.mean(s))
        std = float(np.std(s, ddof=1))
        med = float(np.median(s))
        lo = float(np.quantile(s, (1 - CI) / 2))
        hi = float(np.quantile(s, 1 - (1 - CI) / 2))
        rows.append([nm, mean, std, med, lo, hi])
    cols = ["Parameter", "Post Mean", "Post Std", "Post Median",
            f"Lower {int(100*CI)}% CrI", f"Upper {int(100*CI)}% CrI"]
    return pd.DataFrame(rows, columns=cols)


# Base classes
# For each distribution family, implement the life-stress mapping and likelihood pieces.
# --------------------------------------------------------------------------------------

# --------- Exponential distribution wrappers (scale = life(S)) ------------------------

class _BaseExponentialALTBayes:
    """Base with common run() for Exponential-dist ALT Bayes wrappers."""
    DIST_NAME = "Exponential"   # for prior dispatcher
    PARAM_NAMES = ()            # to be set by subclass
    MLE_CLASS = None            # underlying reliability fitter
    MODEL_NAME = None           # for prior dispatcher
    USE_LIFE_FUNC_POST = None   # function(S...) -> scale (mean life)

    def __init__(
        self,
        *,
        failures,
        failure_stress=None,
        failure_stress_1=None,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        priors=None,
        n_walkers=32,
        n_steps=4000,
        burn=1000,
        thin=1,
        random_seed=None,
        progress=False,
        show_bayes_plots=False,
    ):
        # Input checking
        inputs = ALT_fitters_input_checking(
            dist=self.DIST_NAME,
            life_stress_model=self.MODEL_NAME,
            failures=failures,
            failure_stress_1=(failure_stress_1 if failure_stress_1 is not None else failure_stress),
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=(right_censored_stress_1 if right_censored_stress_1 is not None else right_censored_stress),
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        self.CI = inputs.CI
        self.use_level_stress = inputs.use_level_stress

        # MLE via reliability
        _s2 = getattr(inputs, "failure_stress_2", None)
        has_s2 = _s2 is not None and np.size(_s2) > 0
        
        if has_s2:
            fit = self.MLE_CLASS(
                failures=inputs.failures,
                failure_stress_1=getattr(inputs, 'failure_stress_1', None),
                failure_stress_2=getattr(inputs, 'failure_stress_2', None),
                right_censored=inputs.right_censored,
                right_censored_stress_1=getattr(inputs, 'right_censored_stress_1', None),
                right_censored_stress_2=getattr(inputs, 'right_censored_stress_2', None),
                use_level_stress=self.use_level_stress,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
                optimizer=optimizer,
                CI=self.CI,
            )
        else:
            fit = self.MLE_CLASS(
                failures=inputs.failures,
                failure_stress=getattr(inputs, 'failure_stress_1', None),
                right_censored=inputs.right_censored,
                right_censored_stress=getattr(inputs, 'right_censored_stress_1', None),
                use_level_stress=self.use_level_stress,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
                optimizer=optimizer,
                CI=self.CI,
            )
        # copy attributes
        self.results = fit.results
        self.goodness_of_fit = fit.goodness_of_fit
        self.change_of_parameters = getattr(fit, "change_of_parameters", None)
        self.mean_life = getattr(fit, "mean_life", None)
        self.alpha_at_use_stress = getattr(fit, "alpha_at_use_stress", None)
        self.success = getattr(fit, "success", True)
        self.optimizer = getattr(fit, "optimizer", optimizer)

        # parameters (order must match PARAM_NAMES)
        for p in self.PARAM_NAMES:
            setattr(self, p, float(getattr(fit, p)))

        # SE if present
        for p in self.PARAM_NAMES:
            se_attr = f"{p}_SE"
            setattr(self, se_attr, getattr(fit, se_attr, np.nan))

        # store data for likelihood
        self._fail = np.asarray(inputs.failures, dtype=float)
        self._rc = np.asarray([] if inputs.right_censored is None else inputs.right_censored, dtype=float)
        # stresses
        self._S1f = np.asarray(inputs.failure_stress_1, dtype=float) if hasattr(inputs, "failure_stress_1") else None
        self._S2f = np.asarray(inputs.failure_stress_2, dtype=float) if hasattr(inputs, "failure_stress_2") else None
        self._S1rc = np.asarray([] if inputs.right_censored_stress_1 is None else inputs.right_censored_stress_1, dtype=float)
        self._S2rc = np.asarray([] if inputs.right_censored_stress_2 is None else inputs.right_censored_stress_2, dtype=float)

        # priors
        self.priors = {} if priors is None else priors

        # MCMC
        if random_seed is not None:
            np.random.seed(int(random_seed))

        theta_mle = np.array([getattr(self, p) for p in self.PARAM_NAMES], dtype=float)
        # Initialize proposal around MLE using SEs (10% perturbation fallback).
        se = []
        for p in self.PARAM_NAMES:
            s = getattr(self, f"{p}_SE", np.nan)
            if not np.isfinite(s) or s <= 0:
                s = 0.1 * max(1e-6, 1.0 + abs(getattr(self, p)))
            se.append(s)
        se = np.array(se, dtype=float)
        p0 = theta_mle + se * np.random.randn(n_walkers, len(self.PARAM_NAMES))

        for j, p in enumerate(self.PARAM_NAMES):
            dom = get_param_domain(self.MODEL_NAME, self.DIST_NAME, p)
            if dom == "positive":
                p0[:, j] = np.clip(p0[:, j], 1e-12, None)

        def _loglike(theta):
            # Implemented per-subclass
            raise NotImplementedError

        def _logpost(theta):
            param_dict = {nm: theta[i] for i, nm in enumerate(self.PARAM_NAMES)}
            lp = log_prior_vector(self.MODEL_NAME, self.DIST_NAME, param_dict, self.priors)
            if not np.isfinite(lp):
                return -np.inf
            return lp + _loglike(theta)

        # Bind subclass _loglike via closure
        self._closure_loglike = lambda th: self._loglike_impl(th)

        def _loglike(theta):
            return self._closure_loglike(theta)

        sampler = emcee.EnsembleSampler(n_walkers, len(self.PARAM_NAMES), _logpost)
        sampler.run_mcmc(p0, n_steps, progress=progress)

        chain_raw = sampler.get_chain()  # (steps, walkers, P)
        chain_flat = chain_raw[burn::thin, :, :].reshape(-1, len(self.PARAM_NAMES))
        self.bayes_samples = chain_flat
        self.bayes_results = _summarize_chain(chain_flat, list(self.PARAM_NAMES), self.CI)

        # posterior means
        post_means = np.mean(chain_flat, axis=0)
        for i, p in enumerate(self.PARAM_NAMES):
            setattr(self, f"{p}_post_mean", float(post_means[i]))

        # use-level posterior summary (Exponential: mean = life(scale))
        if self.use_level_stress is not None:
            try:
                L = (self.USE_LIFE_FUNC_POST)(self, self.use_level_stress)
                # For Exponential distribution, mean life = scale L
                self.mean_life_posterior_mean = float(np.mean(L)) if np.ndim(L) else float(L)
            except Exception:
                self.mean_life_posterior_mean = None

        # Print summary
        # RC count
        rc_list = [] if inputs.right_censored is None else list(inputs.right_censored)
        n = len(inputs.failures) + len(rc_list)
        CI_pct = int(round(100 * self.CI))
        frac_cens = (len(rc_list) / n * 100.0) if n > 0 else 0.0
        colorprint(f"Results from {self.__class__.__name__} ({CI_pct}% CrI):", bold=True, underline=True)
        print("Analysis method: Bayesian Estimation")
        print("Failures / Right censored:", f"{len(inputs.failures)}/{len(rc_list)}",
              f"({round_and_string(frac_cens)}% right censored)\n")
        print(self.bayes_results.to_string(index=False), "\n")

        if self.use_level_stress is not None:
            if getattr(self, "mean_life_posterior_mean", None) is not None:
                if isinstance(self.use_level_stress, (list, tuple, np.ndarray)):
                    use_str = "[" + ", ".join(round_and_string(s) for s in self.use_level_stress) + "]"
                else:
                    use_str = round_and_string(self.use_level_stress)
                print(f"At the use level stress(es) of {use_str}, the mean life (posterior) is "
                      f"{self.mean_life_posterior_mean:.5g}\n")

        # optional plots
        if show_bayes_plots:
            try:
                def _life_post_singleS(S):
                    return self.USE_LIFE_FUNC_POST(self, S)

                # for plotting helper, pass the correct signature
                if self.MODEL_NAME in ("Dual_Exponential", "Dual_Power", "Power_Exponential"):
                    life_func = lambda S1, S2: self.USE_LIFE_FUNC_POST(self, [S1, S2])
                else:
                    life_func = lambda S1: self.USE_LIFE_FUNC_POST(self, S1)

                self.bayes_life_stress_plot = life_stress_plot(
                    dist=self.DIST_NAME,
                    model=self.MODEL_NAME,
                    life_func=life_func,
                    failure_groups=getattr(fit, f"_{self.MLE_CLASS.__name__}__failure_groups", None),
                    stresses_for_groups=getattr(fit, f"_{self.MLE_CLASS.__name__}__stresses_for_groups", None),
                    use_level_stress=self.use_level_stress,
                    ax=True,
                )

                if _HAS_AZ:
                    # Reconstruct chains per parameter
                    steps, walkers, P = chain_raw.shape
                    chain_struc = np.swapaxes(chain_raw[burn::thin, :, :], 0, 1)
                    post_dict = {self.PARAM_NAMES[i]: chain_struc[:, :, i] for i in range(P)}
                    self.az_idata = az.from_dict(posterior=post_dict)
                    self.az_axes = {}
                    self.az_axes["trace"] = az.plot_trace(self.az_idata)
                    self.az_axes["pair"] = az.plot_pair(self.az_idata, var_names=list(self.PARAM_NAMES), kind="kde", marginals=True)
            except Exception:
                pass


# Specific Exponential ALT models

class Fit_Exponential_Exponential_Bayesian(_BaseExponentialALTBayes):
    """
    Bayesian wrapper for Fit_Exponential_Exponential.
    L(S) = b * exp(a / S) is the mean life (scale); T ~ Exponential(scale=L).
    Parameters: a, b
    """
    DIST_NAME = "Exponential"
    MODEL_NAME = "Exponential"
    PARAM_NAMES = ("a", "b")
    MLE_CLASS = Fit_Exponential_Exponential

    def _loglike_impl(self, theta):
        a, b = theta
        if b <= 0:
            return -np.inf
        Lf = b * np.exp(a / self._S1f)
        ll_f = _logpdf_exponential_scale(self._fail, Lf).sum() if len(self._fail) else 0.0
        Lrc = b * np.exp(a / self._S1rc) if len(self._S1rc) else np.array([])
        ll_rc = _logsf_exponential_scale(self._rc, Lrc).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, b = self.a_post_mean, self.b_post_mean
        return b * np.exp(a / np.asarray(S))


class Fit_Exponential_Power_Bayesian(_BaseExponentialALTBayes):
    """
    Bayesian wrapper for Fit_Exponential_Power.
    L(S) = a * (S ** n) is the mean life (scale); T ~ Exponential(scale=L).
    Parameters: a, n
    """
    DIST_NAME = "Exponential"
    MODEL_NAME = "Power"
    PARAM_NAMES = ("a", "n")
    MLE_CLASS = Fit_Exponential_Power

    def _loglike_impl(self, theta):
        a, n = theta
        if a <= 0:
            return -np.inf
        Lf = a * (self._S1f ** n)
        ll_f = _logpdf_exponential_scale(self._fail, Lf).sum() if len(self._fail) else 0.0
        Lrc = a * (self._S1rc ** n) if len(self._S1rc) else np.array([])
        ll_rc = _logsf_exponential_scale(self._rc, Lrc).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, n = self.a_post_mean, self.n_post_mean
        S = np.asarray(S)
        return a * (S ** n)


class Fit_Exponential_Eyring_Bayesian(_BaseExponentialALTBayes):
    """
    Bayesian wrapper for Fit_Exponential_Eyring.
    L(S) = (1/S) * exp(-(c - a/S)) is the mean life (scale); T ~ Exponential(scale=L).
    Parameters: a, c
    """
    DIST_NAME = "Exponential"
    MODEL_NAME = "Eyring"
    PARAM_NAMES = ("a", "c")
    MLE_CLASS = Fit_Exponential_Eyring

    def _loglike_impl(self, theta):
        a, c = theta
        S = self._S1f
        Lf = (1.0 / S) * np.exp(-(c - a / S))
        ll_f = _logpdf_exponential_scale(self._fail, Lf).sum() if len(self._fail) else 0.0
        S = self._S1rc
        Lrc = (1.0 / S) * np.exp(-(c - a / S)) if len(S) else np.array([])
        ll_rc = _logsf_exponential_scale(self._rc, Lrc).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, c = self.a_post_mean, self.c_post_mean
        S = np.asarray(S)
        return (1.0 / S) * np.exp(-(c - a / S))


class Fit_Exponential_Dual_Exponential_Bayesian(_BaseExponentialALTBayes):
    """
    Bayesian wrapper for Fit_Exponential_Dual_Exponential.
    L(S1,S2) = c * exp(a / S1 + b / S2) is the mean life (scale); T ~ Exponential(scale=L).
    Parameters: a, b, c
    """
    DIST_NAME = "Exponential"
    MODEL_NAME = "Dual_Exponential"
    PARAM_NAMES = ("a", "b", "c")
    MLE_CLASS = Fit_Exponential_Dual_Exponential

    def _loglike_impl(self, theta):
        a, b, c = theta
        if c <= 0:
            return -np.inf
        Lf = c * np.exp(a / self._S1f + b / self._S2f)
        ll_f = _logpdf_exponential_scale(self._fail, Lf).sum() if len(self._fail) else 0.0
        Lrc = c * np.exp(a / self._S1rc + b / self._S2rc) if len(self._rc) else np.array([])
        ll_rc = _logsf_exponential_scale(self._rc, Lrc).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, b, c = self.a_post_mean, self.b_post_mean, self.c_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return c * np.exp(a / S1 + b / S2)


class Fit_Exponential_Dual_Power_Bayesian(_BaseExponentialALTBayes):
    """
    Bayesian wrapper for Fit_Exponential_Dual_Power.
    L(S1,S2) = c * (S1 ** m) * (S2 ** n) is the mean life (scale); T ~ Exponential(scale=L).
    Parameters: c, m, n
    """
    DIST_NAME = "Exponential"
    MODEL_NAME = "Dual_Power"
    PARAM_NAMES = ("c", "m", "n")
    MLE_CLASS = Fit_Exponential_Dual_Power

    def _loglike_impl(self, theta):
        c, m, n = theta
        if c <= 0:
            return -np.inf
        Lf = c * (self._S1f ** m) * (self._S2f ** n)
        ll_f = _logpdf_exponential_scale(self._fail, Lf).sum() if len(self._fail) else 0.0
        Lrc = c * (self._S1rc ** m) * (self._S2rc ** n) if len(self._rc) else np.array([])
        ll_rc = _logsf_exponential_scale(self._rc, Lrc).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        c, m, n = self.c_post_mean, self.m_post_mean, self.n_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return c * (S1 ** m) * (S2 ** n)


class Fit_Exponential_Power_Exponential_Bayesian(_BaseExponentialALTBayes):
    """
    Bayesian wrapper for Fit_Exponential_Power_Exponential.
    L(S1,S2) = c * exp(a / S1) * (S2 ** n) is the mean life (scale); T ~ Exponential(scale=L).
    Parameters: a, c, n
    """
    DIST_NAME = "Exponential"
    MODEL_NAME = "Power_Exponential"
    PARAM_NAMES = ("a", "c", "n")
    MLE_CLASS = Fit_Exponential_Power_Exponential

    def _loglike_impl(self, theta):
        a, c, n = theta
        if c <= 0:
            return -np.inf
        Lf = c * np.exp(a / self._S1f) * (self._S2f ** n)
        ll_f = _logpdf_exponential_scale(self._fail, Lf).sum() if len(self._fail) else 0.0
        Lrc = c * np.exp(a / self._S1rc) * (self._S2rc ** n) if len(self._rc) else np.array([])
        ll_rc = _logsf_exponential_scale(self._rc, Lrc).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, c, n = self.a_post_mean, self.c_post_mean, self.n_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return c * np.exp(a / S1) * (S2 ** n)


# -------------------------- Lognormal distribution wrappers ---------------------------
# mu(S...) is life-stress mapping; sigma is global.

class _BaseLognormalALTBayes:
    DIST_NAME = "Lognormal"
    PARAM_NAMES = ()
    MLE_CLASS = None
    MODEL_NAME = None
    USE_LIFE_FUNC_POST = None  # returns mu(S...)

    def __init__(
        self,
        *,
        failures,
        failure_stress=None,
        failure_stress_1=None,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        priors=None,
        n_walkers=32,
        n_steps=4000,
        burn=1000,
        thin=1,
        random_seed=None,
        progress=False,
        show_bayes_plots=False,
    ):
        inputs = ALT_fitters_input_checking(
            dist=self.DIST_NAME,
            life_stress_model=self.MODEL_NAME,
            failures=failures,
            failure_stress_1=(failure_stress_1 if failure_stress_1 is not None else failure_stress),
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=(right_censored_stress_1 if right_censored_stress_1 is not None else right_censored_stress),
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        self.CI = inputs.CI
        self.use_level_stress = inputs.use_level_stress
        
        # MLE via reliability
        _s2 = getattr(inputs, "failure_stress_2", None)
        has_s2 = _s2 is not None and np.size(_s2) > 0
        
        if has_s2:
            fit = self.MLE_CLASS(
                failures=inputs.failures,
                failure_stress_1=getattr(inputs, 'failure_stress_1', None),
                failure_stress_2=getattr(inputs, 'failure_stress_2', None),
                right_censored=inputs.right_censored,
                right_censored_stress_1=getattr(inputs, 'right_censored_stress_1', None),
                right_censored_stress_2=getattr(inputs, 'right_censored_stress_2', None),
                use_level_stress=self.use_level_stress,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
                optimizer=optimizer,
                CI=self.CI,
            )
        else:
            fit = self.MLE_CLASS(
                failures=inputs.failures,
                failure_stress=getattr(inputs, 'failure_stress_1', None),
                right_censored=inputs.right_censored,
                right_censored_stress=getattr(inputs, 'right_censored_stress_1', None),
                use_level_stress=self.use_level_stress,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
                optimizer=optimizer,
                CI=self.CI,
            )

        self.results = fit.results
        self.goodness_of_fit = fit.goodness_of_fit
        self.change_of_parameters = getattr(fit, "change_of_parameters", None)
        self.mean_life = getattr(fit, "mean_life", None)
        self.mu_at_use_stress = getattr(fit, "mu_at_use_stress", None)
        self.success = getattr(fit, "success", True)
        self.optimizer = getattr(fit, "optimizer", optimizer)

        for p in self.PARAM_NAMES:
            setattr(self, p, float(getattr(fit, p)))

        for p in self.PARAM_NAMES:
            se_attr = f"{p}_SE"
            setattr(self, se_attr, getattr(fit, se_attr, np.nan))

        self._fail = np.asarray(inputs.failures, dtype=float)
        self._rc = np.asarray([] if inputs.right_censored is None else inputs.right_censored, dtype=float)
        self._S1f = np.asarray(inputs.failure_stress_1, dtype=float) if hasattr(inputs, "failure_stress_1") else None
        self._S2f = np.asarray(inputs.failure_stress_2, dtype=float) if hasattr(inputs, "failure_stress_2") else None
        self._S1rc = np.asarray([] if inputs.right_censored_stress_1 is None else inputs.right_censored_stress_1, dtype=float)
        self._S2rc = np.asarray([] if inputs.right_censored_stress_2 is None else inputs.right_censored_stress_2, dtype=float)

        self.priors = {} if priors is None else priors

        if random_seed is not None:
            np.random.seed(int(random_seed))

        theta_mle = np.array([getattr(self, p) for p in self.PARAM_NAMES], dtype=float)
        se = []
        for p in self.PARAM_NAMES:
            s = getattr(self, f"{p}_SE", np.nan)
            if not np.isfinite(s) or s <= 0:
                s = 0.1 * max(1e-6, 1.0 + abs(getattr(self, p)))
            se.append(s)
        se = np.array(se, dtype=float)
        p0 = theta_mle + se * np.random.randn(n_walkers, len(self.PARAM_NAMES))

        # enforce positivity for distribution shape parameters
        for j, p in enumerate(self.PARAM_NAMES):
            dom = get_param_domain(self.MODEL_NAME, self.DIST_NAME, p)
            if dom == "positive":
                p0[:, j] = np.clip(p0[:, j], 1e-12, None)

        def _loglike(theta):
            raise NotImplementedError

        def _logpost(theta):
            param_dict = {nm: theta[i] for i, nm in enumerate(self.PARAM_NAMES)}
            lp = log_prior_vector(self.MODEL_NAME, self.DIST_NAME, param_dict, self.priors)
            if not np.isfinite(lp):
                return -np.inf
            return lp + _loglike(theta)

        self._closure_loglike = lambda th: self._loglike_impl(th)

        def _loglike(theta):
            return self._closure_loglike(theta)

        sampler = emcee.EnsembleSampler(n_walkers, len(self.PARAM_NAMES), _logpost)
        sampler.run_mcmc(p0, n_steps, progress=progress)

        chain_raw = sampler.get_chain()
        chain_flat = chain_raw[burn::thin, :, :].reshape(-1, len(self.PARAM_NAMES))
        self.bayes_samples = chain_flat
        self.bayes_results = _summarize_chain(chain_flat, list(self.PARAM_NAMES), self.CI)

        post_means = np.mean(chain_flat, axis=0)
        for i, p in enumerate(self.PARAM_NAMES):
            setattr(self, f"{p}_post_mean", float(post_means[i]))

        # use-level posterior mean life: Lognormal mean = exp(mu + 0.5*sigma^2)
        if self.use_level_stress is not None:
            try:
                # derive mu(S_use)
                mu_use = (self.USE_LIFE_FUNC_POST)(self, self.use_level_stress)
                sigma = getattr(self, "sigma_post_mean", None)
                if sigma is None:
                    sigma = self.sigma
                self.mean_life_posterior_mean = float(np.exp(mu_use + 0.5 * sigma**2))
            except Exception:
                self.mean_life_posterior_mean = None

        rc_list = [] if inputs.right_censored is None else list(inputs.right_censored)
        n = len(inputs.failures) + len(rc_list)
        CI_pct = int(round(100 * self.CI))
        frac_cens = (len(rc_list) / n * 100.0) if n > 0 else 0.0
        colorprint(f"Results from {self.__class__.__name__} ({CI_pct}% CrI):", bold=True, underline=True)
        print("Analysis method: Bayesian Estimation")
        print("Failures / Right censored:", f"{len(inputs.failures)}/{len(rc_list)}",
              f"({round_and_string(frac_cens)}% right censored)\n")
        print(self.bayes_results.to_string(index=False), "\n")

        if self.use_level_stress is not None and getattr(self, "mean_life_posterior_mean", None) is not None:
            if isinstance(self.use_level_stress, (list, tuple, np.ndarray)):
                use_str = "[" + ", ".join(round_and_string(s) for s in self.use_level_stress) + "]"
            else:
                use_str = round_and_string(self.use_level_stress)
            print(f"At the use level stress(es) of {use_str}, the mean life (posterior) is "
                  f"{self.mean_life_posterior_mean:.5g}\n")

        if show_bayes_plots:
            try:
                if self.MODEL_NAME in ("Dual_Exponential", "Dual_Power", "Power_Exponential"):
                    life_func = lambda S1, S2: np.exp(self.USE_LIFE_FUNC_POST(self, [S1, S2]))
                else:
                    life_func = lambda S1: np.exp(self.USE_LIFE_FUNC_POST(self, S1))

                self.bayes_life_stress_plot = life_stress_plot(
                    dist=self.DIST_NAME,
                    model=self.MODEL_NAME,
                    life_func=life_func,
                    failure_groups=getattr(fit, f"_{self.MLE_CLASS.__name__}__failure_groups", None),
                    stresses_for_groups=getattr(fit, f"_{self.MLE_CLASS.__name__}__stresses_for_groups", None),
                    use_level_stress=self.use_level_stress,
                    ax=True,
                )

                if _HAS_AZ:
                    steps, walkers, P = chain_raw.shape
                    chain_struc = np.swapaxes(chain_raw[burn::thin, :, :], 0, 1)
                    post_dict = {self.PARAM_NAMES[i]: chain_struc[:, :, i] for i in range(P)}
                    self.az_idata = az.from_dict(posterior=post_dict)
                    self.az_axes = {}
                    self.az_axes["trace"] = az.plot_trace(self.az_idata)
                    self.az_axes["pair"] = az.plot_pair(self.az_idata, var_names=list(self.PARAM_NAMES), kind="kde", marginals=True)
            except Exception:
                pass


# Specific Lognormal ALT classes

class Fit_Lognormal_Exponential_Bayesian(_BaseLognormalALTBayes):
    """
    LNMU(S) = ln-mean mu(S) = ln b + a/S.  Parameters: a, b, sigma.
    """
    MODEL_NAME = "Exponential"
    PARAM_NAMES = ("a", "b", "sigma")
    MLE_CLASS = Fit_Lognormal_Exponential

    def _loglike_impl(self, theta):
        a, b, sigma = theta
        if sigma <= 0:
            return -np.inf
        mu_f = np.log(b) + a / self._S1f
        ll_f = _logpdf_lognormal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        mu_rc = np.log(b) + a / self._S1rc if len(self._S1rc) else np.array([])
        ll_rc = _logsf_lognormal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, b = self.a_post_mean, self.b_post_mean
        return np.log(b) + a / np.asarray(S)


class Fit_Lognormal_Power_Bayesian(_BaseLognormalALTBayes):
    """
    mu(S) = ln a + n ln S.  Parameters: a, n, sigma.
    """
    MODEL_NAME = "Power"
    PARAM_NAMES = ("a", "n", "sigma")
    MLE_CLASS = Fit_Lognormal_Power

    def _loglike_impl(self, theta):
        a, n, sigma = theta
        if sigma <= 0 or a <= 0:
            return -np.inf
        mu_f = np.log(a) + n * np.log(self._S1f)
        ll_f = _logpdf_lognormal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        mu_rc = np.log(a) + n * np.log(self._S1rc) if len(self._S1rc) else np.array([])
        ll_rc = _logsf_lognormal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, n = self.a_post_mean, self.n_post_mean
        S = np.asarray(S)
        return np.log(a) + n * np.log(S)

class Fit_Lognormal_Eyring_Bayesian(_BaseLognormalALTBayes):
    """
    mu(S) = -ln S - (c - a/S).  Parameters: a, c, sigma.
    """
    MODEL_NAME = "Eyring"
    PARAM_NAMES = ("a", "c", "sigma")
    MLE_CLASS = Fit_Lognormal_Eyring

    def _loglike_impl(self, theta):
        a, c, sigma = theta
        if sigma <= 0:
            return -np.inf
        S = self._S1f
        mu_f = -np.log(S) - (c - a / S)
        ll_f = _logpdf_lognormal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        S = self._S1rc
        mu_rc = -np.log(S) - (c - a / S) if len(S) else np.array([])
        ll_rc = _logsf_lognormal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, c = self.a_post_mean, self.c_post_mean
        S = np.asarray(S)
        return -np.log(S) - (c - a / S)


class Fit_Lognormal_Dual_Exponential_Bayesian(_BaseLognormalALTBayes):
    """
    mu(S1,S2) = ln c + a/S1 + b/S2.  Parameters: a, b, c, sigma.
    """
    MODEL_NAME = "Dual_Exponential"
    PARAM_NAMES = ("a", "b", "c", "sigma")
    MLE_CLASS = Fit_Lognormal_Dual_Exponential

    def _loglike_impl(self, theta):
        a, b, c, sigma = theta
        if sigma <= 0 or c <= 0:
            return -np.inf
        mu_f = np.log(c) + a / self._S1f + b / self._S2f
        ll_f = _logpdf_lognormal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        mu_rc = np.log(c) + a / self._S1rc + b / self._S2rc if len(self._rc) else np.array([])
        ll_rc = _logsf_lognormal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, b, c = self.a_post_mean, self.b_post_mean, self.c_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return np.log(c) + a / S1 + b / S2


class Fit_Lognormal_Dual_Power_Bayesian(_BaseLognormalALTBayes):
    """
    mu(S1,S2) = ln c + m ln S1 + n ln S2.  Parameters: c, m, n, sigma.
    """
    MODEL_NAME = "Dual_Power"
    PARAM_NAMES = ("c", "m", "n", "sigma")
    MLE_CLASS = Fit_Lognormal_Dual_Power

    def _loglike_impl(self, theta):
        c, m, n, sigma = theta
        if sigma <= 0 or c <= 0:
            return -np.inf
        mu_f = np.log(c) + m * np.log(self._S1f) + n * np.log(self._S2f)
        ll_f = _logpdf_lognormal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        mu_rc = np.log(c) + m * np.log(self._S1rc) + n * np.log(self._S2rc) if len(self._rc) else np.array([])
        ll_rc = _logsf_lognormal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        c, m, n = self.c_post_mean, self.m_post_mean, self.n_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return np.log(c) + m * np.log(S1) + n * np.log(S2)


class Fit_Lognormal_Power_Exponential_Bayesian(_BaseLognormalALTBayes):
    """
    mu(S1,S2) = ln c + a/S1 + n ln S2.  Parameters: a, c, n, sigma.
    """
    MODEL_NAME = "Power_Exponential"
    PARAM_NAMES = ("a", "c", "n", "sigma")
    MLE_CLASS = Fit_Lognormal_Power_Exponential

    def _loglike_impl(self, theta):
        a, c, n, sigma = theta
        if sigma <= 0 or c <= 0:
            return -np.inf
        mu_f = np.log(c) + a / self._S1f + n * np.log(self._S2f)
        ll_f = _logpdf_lognormal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        mu_rc = np.log(c) + a / self._S1rc + n * np.log(self._S2rc) if len(self._rc) else np.array([])
        ll_rc = _logsf_lognormal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, c, n = self.a_post_mean, self.c_post_mean, self.n_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return np.log(c) + a / S1 + n * np.log(S2)


# --------------------------- Normal distribution wrappers -----------------------------
# mu(S...) mapping; sigma global. Likelihood uses Normal pdf/sf.

class _BaseNormalALTBayes:
    DIST_NAME = "Normal"
    PARAM_NAMES = ()
    MLE_CLASS = None
    MODEL_NAME = None
    USE_LIFE_FUNC_POST = None

    def __init__(
        self,
        *,
        failures,
        failure_stress=None,
        failure_stress_1=None,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        priors=None,
        n_walkers=32,
        n_steps=4000,
        burn=1000,
        thin=1,
        random_seed=None,
        progress=False,
        show_bayes_plots=False,
    ):
        inputs = ALT_fitters_input_checking(
            dist=self.DIST_NAME,
            life_stress_model=self.MODEL_NAME,
            failures=failures,
            failure_stress_1=(failure_stress_1 if failure_stress_1 is not None else failure_stress),
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=(right_censored_stress_1 if right_censored_stress_1 is not None else right_censored_stress),
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )

        self.CI = inputs.CI
        self.use_level_stress = inputs.use_level_stress
        
        # MLE via reliability
        _s2 = getattr(inputs, "failure_stress_2", None)
        has_s2 = _s2 is not None and np.size(_s2) > 0
        
        if has_s2:
            fit = self.MLE_CLASS(
                failures=inputs.failures,
                failure_stress_1=getattr(inputs, 'failure_stress_1', None),
                failure_stress_2=getattr(inputs, 'failure_stress_2', None),
                right_censored=inputs.right_censored,
                right_censored_stress_1=getattr(inputs, 'right_censored_stress_1', None),
                right_censored_stress_2=getattr(inputs, 'right_censored_stress_2', None),
                use_level_stress=self.use_level_stress,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
                optimizer=optimizer,
                CI=self.CI,
            )
        else:
            fit = self.MLE_CLASS(
                failures=inputs.failures,
                failure_stress=getattr(inputs, 'failure_stress_1', None),
                right_censored=inputs.right_censored,
                right_censored_stress=getattr(inputs, 'right_censored_stress_1', None),
                use_level_stress=self.use_level_stress,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
                optimizer=optimizer,
                CI=self.CI,
            )


        self.results = fit.results
        self.goodness_of_fit = fit.goodness_of_fit
        self.change_of_parameters = getattr(fit, "change_of_parameters", None)
        self.mean_life = getattr(fit, "mean_life", None)
        self.mu_at_use_stress = getattr(fit, "mu_at_use_stress", None)
        self.success = getattr(fit, "success", True)
        self.optimizer = getattr(fit, "optimizer", optimizer)

        for p in self.PARAM_NAMES:
            setattr(self, p, float(getattr(fit, p)))

        for p in self.PARAM_NAMES:
            se_attr = f"{p}_SE"
            setattr(self, se_attr, getattr(fit, se_attr, np.nan))

        self._fail = np.asarray(inputs.failures, dtype=float)
        self._rc = np.asarray([] if inputs.right_censored is None else inputs.right_censored, dtype=float)
        self._S1f = np.asarray(inputs.failure_stress_1, dtype=float) if hasattr(inputs, "failure_stress_1") else None
        self._S2f = np.asarray(inputs.failure_stress_2, dtype=float) if hasattr(inputs, "failure_stress_2") else None
        self._S1rc = np.asarray([] if inputs.right_censored_stress_1 is None else inputs.right_censored_stress_1, dtype=float)
        self._S2rc = np.asarray([] if inputs.right_censored_stress_2 is None else inputs.right_censored_stress_2, dtype=float)

        self.priors = {} if priors is None else priors

        if random_seed is not None:
            np.random.seed(int(random_seed))

        theta_mle = np.array([getattr(self, p) for p in self.PARAM_NAMES], dtype=float)
        se = []
        for p in self.PARAM_NAMES:
            s = getattr(self, f"{p}_SE", np.nan)
            if not np.isfinite(s) or s <= 0:
                s = 0.1 * max(1e-6, 1.0 + abs(getattr(self, p)))
            se.append(s)
        se = np.array(se, dtype=float)
        p0 = theta_mle + se * np.random.randn(n_walkers, len(self.PARAM_NAMES))

        # enforce positivity for distribution shape parameters
        for j, p in enumerate(self.PARAM_NAMES):
            dom = get_param_domain(self.MODEL_NAME, self.DIST_NAME, p)
            if dom == "positive":
                p0[:, j] = np.clip(p0[:, j], 1e-12, None)

        def _loglike(theta):
            raise NotImplementedError

        def _logpost(theta):
            param_dict = {nm: theta[i] for i, nm in enumerate(self.PARAM_NAMES)}
            lp = log_prior_vector(self.MODEL_NAME, self.DIST_NAME, param_dict, self.priors)
            if not np.isfinite(lp):
                return -np.inf
            return lp + _loglike(theta)

        self._closure_loglike = lambda th: self._loglike_impl(th)

        def _loglike(theta):
            return self._closure_loglike(theta)

        sampler = emcee.EnsembleSampler(n_walkers, len(self.PARAM_NAMES), _logpost)
        sampler.run_mcmc(p0, n_steps, progress=progress)

        chain_raw = sampler.get_chain()
        chain_flat = chain_raw[burn::thin, :, :].reshape(-1, len(self.PARAM_NAMES))
        self.bayes_samples = chain_flat
        self.bayes_results = _summarize_chain(chain_flat, list(self.PARAM_NAMES), self.CI)

        post_means = np.mean(chain_flat, axis=0)
        for i, p in enumerate(self.PARAM_NAMES):
            setattr(self, f"{p}_post_mean", float(post_means[i]))

        # use-level posterior mean life for Normal is the location mu(S_use)
        if self.use_level_stress is not None:
            try:
                mu_use = (self.USE_LIFE_FUNC_POST)(self, self.use_level_stress)
                self.mean_life_posterior_mean = float(mu_use)
            except Exception:
                self.mean_life_posterior_mean = None

        rc_list = [] if inputs.right_censored is None else list(inputs.right_censored)
        n = len(inputs.failures) + len(rc_list)
        CI_pct = int(round(100 * self.CI))
        frac_cens = (len(rc_list) / n * 100.0) if n > 0 else 0.0
        colorprint(f"Results from {self.__class__.__name__} ({CI_pct}% CrI):", bold=True, underline=True)
        print("Analysis method: Bayesian Estimation")
        print("Failures / Right censored:", f"{len(inputs.failures)}/{len(rc_list)}",
              f"({round_and_string(frac_cens)}% right censored)\n")
        print(self.bayes_results.to_string(index=False), "\n")

        if self.use_level_stress is not None and getattr(self, "mean_life_posterior_mean", None) is not None:
            if isinstance(self.use_level_stress, (list, tuple, np.ndarray)):
                use_str = "[" + ", ".join(round_and_string(s) for s in self.use_level_stress) + "]"
            else:
                use_str = round_and_string(self.use_level_stress)
            print(f"At the use level stress(es) of {use_str}, the mean life (posterior) is "
                  f"{self.mean_life_posterior_mean:.5g}\n")

        if show_bayes_plots:
            try:
                if self.MODEL_NAME in ("Dual_Exponential", "Dual_Power", "Power_Exponential"):
                    life_func = lambda S1, S2: self.USE_LIFE_FUNC_POST(self, [S1, S2])
                else:
                    life_func = lambda S1: self.USE_LIFE_FUNC_POST(self, S1)

                self.bayes_life_stress_plot = life_stress_plot(
                    dist=self.DIST_NAME,
                    model=self.MODEL_NAME,
                    life_func=life_func,
                    failure_groups=getattr(fit, f"_{self.MLE_CLASS.__name__}__failure_groups", None),
                    stresses_for_groups=getattr(fit, f"_{self.MLE_CLASS.__name__}__stresses_for_groups", None),
                    use_level_stress=self.use_level_stress,
                    ax=True,
                )

                if _HAS_AZ:
                    steps, walkers, P = chain_raw.shape
                    chain_struc = np.swapaxes(chain_raw[burn::thin, :, :], 0, 1)
                    post_dict = {self.PARAM_NAMES[i]: chain_struc[:, :, i] for i in range(P)}
                    self.az_idata = az.from_dict(posterior=post_dict)
                    self.az_axes = {}
                    self.az_axes["trace"] = az.plot_trace(self.az_idata)
                    self.az_axes["pair"] = az.plot_pair(self.az_idata, var_names=list(self.PARAM_NAMES), kind="kde", marginals=True)
            except Exception:
                pass


# Specific Normal ALT classes

class Fit_Normal_Exponential_Bayesian(_BaseNormalALTBayes):
    """
    mu(S) = b * exp(a / S).  Parameters: a, b, sigma.
    """
    MODEL_NAME = "Exponential"
    PARAM_NAMES = ("a", "b", "sigma")
    MLE_CLASS = Fit_Normal_Exponential

    def _loglike_impl(self, theta):
        a, b, sigma = theta
        if sigma <= 0:
            return -np.inf
        mu_f = b * np.exp(a / self._S1f)
        ll_f = _logpdf_normal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        mu_rc = b * np.exp(a / self._S1rc) if len(self._S1rc) else np.array([])
        ll_rc = _logsf_normal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, b = self.a_post_mean, self.b_post_mean
        return b * np.exp(a / np.asarray(S))


class Fit_Normal_Power_Bayesian(_BaseNormalALTBayes):
    """
    mu(S) = a * S**n.  Parameters: a, n, sigma.
    """
    MODEL_NAME = "Power"
    PARAM_NAMES = ("a", "n", "sigma")
    MLE_CLASS = Fit_Normal_Power

    def _loglike_impl(self, theta):
        a, n, sigma = theta
        if sigma <= 0 or a <= 0:
            return -np.inf
        mu_f = a * (self._S1f ** n)
        ll_f = _logpdf_normal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        mu_rc = a * (self._S1rc ** n) if len(self._S1rc) else np.array([])
        ll_rc = _logsf_normal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, n = self.a_post_mean, self.n_post_mean
        S = np.asarray(S)
        return a * (S ** n)


class Fit_Normal_Eyring_Bayesian(_BaseNormalALTBayes):
    """
    mu(S) = (1/S) * exp(-(c - a/S)).  Parameters: a, c, sigma.
    """
    MODEL_NAME = "Eyring"
    PARAM_NAMES = ("a", "c", "sigma")
    MLE_CLASS = Fit_Normal_Eyring

    def _loglike_impl(self, theta):
        a, c, sigma = theta
        if sigma <= 0:
            return -np.inf
        S = self._S1f
        mu_f = (1.0 / S) * np.exp(-(c - a / S))
        ll_f = _logpdf_normal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        S = self._S1rc
        mu_rc = (1.0 / S) * np.exp(-(c - a / S)) if len(S) else np.array([])
        ll_rc = _logsf_normal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, c = self.a_post_mean, self.c_post_mean
        S = np.asarray(S)
        return (1.0 / S) * np.exp(-(c - a / S))


class Fit_Normal_Dual_Exponential_Bayesian(_BaseNormalALTBayes):
    """
    mu(S1,S2) = c * exp(a/S1 + b/S2).  Parameters: a, b, c, sigma.
    """
    MODEL_NAME = "Dual_Exponential"
    PARAM_NAMES = ("a", "b", "c", "sigma")
    MLE_CLASS = Fit_Normal_Dual_Exponential

    def _loglike_impl(self, theta):
        a, b, c, sigma = theta
        if sigma <= 0 or c <= 0:
            return -np.inf
        mu_f = c * np.exp(a / self._S1f + b / self._S2f)
        ll_f = _logpdf_normal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        mu_rc = c * np.exp(a / self._S1rc + b / self._S2rc) if len(self._rc) else np.array([])
        ll_rc = _logsf_normal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, b, c = self.a_post_mean, self.b_post_mean, self.c_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return c * np.exp(a / S1 + b / S2)


class Fit_Normal_Dual_Power_Bayesian(_BaseNormalALTBayes):
    """
    mu(S1,S2) = c * S1**m * S2**n.  Parameters: c, m, n, sigma.
    """
    MODEL_NAME = "Dual_Power"
    PARAM_NAMES = ("c", "m", "n", "sigma")
    MLE_CLASS = Fit_Normal_Dual_Power

    def _loglike_impl(self, theta):
        c, m, n, sigma = theta
        if sigma <= 0 or c <= 0:
            return -np.inf
        mu_f = c * (self._S1f ** m) * (self._S2f ** n)
        ll_f = _logpdf_normal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        mu_rc = c * (self._S1rc ** m) * (self._S2rc ** n) if len(self._rc) else np.array([])
        ll_rc = _logsf_normal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        c, m, n = self.c_post_mean, self.m_post_mean, self.n_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return c * (S1 ** m) * (S2 ** n)


class Fit_Normal_Power_Exponential_Bayesian(_BaseNormalALTBayes):
    """
    mu(S1,S2) = c * exp(a/S1) * S2**n.  Parameters: a, c, n, sigma.
    """
    MODEL_NAME = "Power_Exponential"
    PARAM_NAMES = ("a", "c", "n", "sigma")
    MLE_CLASS = Fit_Normal_Power_Exponential

    def _loglike_impl(self, theta):
        a, c, n, sigma = theta
        if sigma <= 0 or c <= 0:
            return -np.inf
        mu_f = c * np.exp(a / self._S1f) * (self._S2f ** n)
        ll_f = _logpdf_normal(self._fail, mu_f, sigma).sum() if len(self._fail) else 0.0
        mu_rc = c * np.exp(a / self._S1rc) * (self._S2rc ** n) if len(self._rc) else np.array([])
        ll_rc = _logsf_normal(self._rc, mu_rc, sigma).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, c, n = self.a_post_mean, self.c_post_mean, self.n_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return c * np.exp(a / S1) * (S2 ** n)


# -------------------------- Weibull distribution wrappers (scale = life(S)) ---------------------------

class _BaseWeibullALTBayes:
    """
    Base class with common MLE → MCMC workflow for Weibull ALT wrappers.
    Subclasses must define:
      - DIST_NAME = "Weibull"
      - MODEL_NAME (e.g., "Power", "Exponential", ...)
      - PARAM_NAMES (tuple) in order matching the underlying MLE_CLASS attributes
      - MLE_CLASS (reliability.ALT_fitters class)
      - USE_LIFE_FUNC_POST(self, S_or_pair) to compute alpha(S) using posterior-mean params
    """
    DIST_NAME = "Weibull"
    MODEL_NAME = None
    PARAM_NAMES = ()
    MLE_CLASS = None

    def __init__(
        self,
        *,
        failures,
        failure_stress=None,
        failure_stress_1=None,
        failure_stress_2=None,
        right_censored=None,
        right_censored_stress=None,
        right_censored_stress_1=None,
        right_censored_stress_2=None,
        use_level_stress=None,
        CI=0.95,
        optimizer=None,
        priors=None,
        n_walkers=32,
        n_steps=4000,
        burn=1000,
        thin=1,
        random_seed=None,
        progress=False,
        show_bayes_plots=False,
    ):
        # Standardized input checking
        inputs = ALT_fitters_input_checking(
            dist=self.DIST_NAME,
            life_stress_model=self.MODEL_NAME,
            failures=failures,
            failure_stress_1=(failure_stress_1 if failure_stress_1 is not None else failure_stress),
            failure_stress_2=failure_stress_2,
            right_censored=right_censored,
            right_censored_stress_1=(right_censored_stress_1 if right_censored_stress_1 is not None else right_censored_stress),
            right_censored_stress_2=right_censored_stress_2,
            CI=CI,
            use_level_stress=use_level_stress,
            optimizer=optimizer,
        )
        self.CI = inputs.CI
        self.use_level_stress = inputs.use_level_stress

        # MLE via reliability
        _s2 = getattr(inputs, "failure_stress_2", None)
        has_s2 = _s2 is not None and np.size(_s2) > 0

        if has_s2:
            fit = self.MLE_CLASS(
                failures=inputs.failures,
                failure_stress_1=getattr(inputs, 'failure_stress_1', None),
                failure_stress_2=getattr(inputs, 'failure_stress_2', None),
                right_censored=inputs.right_censored,
                right_censored_stress_1=getattr(inputs, 'right_censored_stress_1', None),
                right_censored_stress_2=getattr(inputs, 'right_censored_stress_2', None),
                use_level_stress=self.use_level_stress,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
                optimizer=optimizer,
                CI=self.CI,
            )
        else:
            fit = self.MLE_CLASS(
                failures=inputs.failures,
                failure_stress=getattr(inputs, 'failure_stress_1', None),
                right_censored=inputs.right_censored,
                right_censored_stress=getattr(inputs, 'right_censored_stress_1', None),
                use_level_stress=self.use_level_stress,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
                optimizer=optimizer,
                CI=self.CI,
            )

        # Copy core attributes
        self.results = fit.results
        self.goodness_of_fit = fit.goodness_of_fit
        self.change_of_parameters = getattr(fit, "change_of_parameters", None)
        self.mean_life = getattr(fit, "mean_life", None)
        self.alpha_at_use_stress = getattr(fit, "alpha_at_use_stress", None)
        self.success = getattr(fit, "success", True)
        self.optimizer = getattr(fit, "optimizer", optimizer)

        # Parameter values, SEs (names must align with underlying fitter)
        for p in self.PARAM_NAMES:
            setattr(self, p, float(getattr(fit, p)))
            se_attr = f"{p}_SE"
            setattr(self, se_attr, getattr(fit, se_attr, np.nan))

        # Data for likelihood
        self._fail = np.asarray(inputs.failures, dtype=float)
        self._rc = np.asarray([] if inputs.right_censored is None else inputs.right_censored, dtype=float)
        # Single/dual stress handling
        self._S1f = np.asarray(getattr(inputs, 'failure_stress_1', None), dtype=float) if getattr(inputs, 'failure_stress_1', None) is not None else None
        self._S2f = np.asarray(getattr(inputs, 'failure_stress_2', None), dtype=float) if getattr(inputs, 'failure_stress_2', None) is not None else None
        self._S1rc = np.asarray([] if getattr(inputs, 'right_censored_stress_1', None) is None else inputs.right_censored_stress_1, dtype=float)
        self._S2rc = np.asarray([] if getattr(inputs, 'right_censored_stress_2', None) is None else inputs.right_censored_stress_2, dtype=float)

        # Priors
        self.priors = {} if priors is None else priors

        # Initialize walkers around MLE
        if random_seed is not None:
            np.random.seed(int(random_seed))

        theta_mle = np.array([getattr(self, p) for p in self.PARAM_NAMES], dtype=float)
        se = []
        for p in self.PARAM_NAMES:
            s = getattr(self, f"{p}_SE", np.nan)
            if not np.isfinite(s) or s <= 0:
                s = 0.1 * max(1e-6, 1.0 + abs(getattr(self, p)))
            se.append(s)
        se = np.array(se, dtype=float)
        p0 = theta_mle + se * np.random.randn(32 if n_walkers is None else n_walkers, len(self.PARAM_NAMES))

        # enforce positivity for distribution shape parameters
        for j, p in enumerate(self.PARAM_NAMES):
            dom = get_param_domain(self.MODEL_NAME, self.DIST_NAME, p)
            if dom == "positive":
                p0[:, j] = np.clip(p0[:, j], 1e-12, None)

        def _loglike(theta):
            return self._loglike_impl(theta)

        def _logpost(theta):
            param_dict = {nm: theta[i] for i, nm in enumerate(self.PARAM_NAMES)}
            lp = log_prior_vector(self.MODEL_NAME, self.DIST_NAME, param_dict, self.priors)
            if not np.isfinite(lp):
                return -np.inf
            return lp + _loglike(theta)

        sampler = emcee.EnsembleSampler(32 if n_walkers is None else n_walkers, len(self.PARAM_NAMES), _logpost)
        sampler.run_mcmc(p0, n_steps, progress=progress)

        chain_raw = sampler.get_chain()  # (steps, walkers, P)
        chain_flat = chain_raw[burn::thin, :, :].reshape(-1, len(self.PARAM_NAMES))
        self.bayes_samples = chain_flat
        self.bayes_results = _summarize_chain(chain_flat, list(self.PARAM_NAMES), self.CI)

        # Posterior means
        post_means = np.mean(chain_flat, axis=0)
        for i, p in enumerate(self.PARAM_NAMES):
            setattr(self, f"{p}_post_mean", float(post_means[i]))

        # Use-level posterior mean life: E[T] = alpha_use * Gamma(1 + 1/beta_post_mean) (plug-in)
        if self.use_level_stress is not None:
            try:
                alpha_use = self.USE_LIFE_FUNC_POST(self, self.use_level_stress)
                beta_pm = getattr(self, "beta_post_mean", None)
                if beta_pm is None:
                    beta_pm = getattr(self, "beta", None)
                if np.ndim(alpha_use):
                    alpha_use = np.asarray(alpha_use, dtype=float)
                    self.mean_life_posterior_mean = float(np.mean(alpha_use * gamma(1.0 + 1.0 / beta_pm)))
                else:
                    self.mean_life_posterior_mean = float(alpha_use * gamma(1.0 + 1.0 / beta_pm))
            except Exception:
                self.mean_life_posterior_mean = None

        # Print summary
        rc_list = [] if inputs.right_censored is None else list(inputs.right_censored)
        n = len(inputs.failures) + len(rc_list)
        CI_pct = int(round(100 * self.CI))
        frac_cens = (len(rc_list) / n * 100.0) if n > 0 else 0.0
        colorprint(f"Results from {self.__class__.__name__} ({CI_pct}% CrI):", bold=True, underline=True)
        print("Analysis method: Bayesian Estimation")
        print("Failures / Right censored:", f"{len(inputs.failures)}/{len(rc_list)}",
              f"({round_and_string(frac_cens)}% right censored)\n")
        print(self.bayes_results.to_string(index=False), "\n")

        if self.use_level_stress is not None and getattr(self, "mean_life_posterior_mean", None) is not None:
            if isinstance(self.use_level_stress, (list, tuple, np.ndarray)):
                use_str = "[" + ", ".join(round_and_string(s) for s in self.use_level_stress) + "]"
            else:
                use_str = round_and_string(self.use_level_stress)
            print(f"At the use level stress(es) of {use_str}, the mean life (posterior) is "
                  f"{self.mean_life_posterior_mean:.5g}\n")

        # Posterior plots (optional)
        if show_bayes_plots:
            try:
                # Build a life_func that returns alpha(S) (scale)
                if self.MODEL_NAME in ("Dual_Exponential", "Dual_Power", "Power_Exponential"):
                    life_func = lambda S1, S2: self.USE_LIFE_FUNC_POST(self, [S1, S2])
                else:
                    life_func = lambda S1: self.USE_LIFE_FUNC_POST(self, S1)

                self.bayes_life_stress_plot = life_stress_plot(
                    dist=self.DIST_NAME,
                    model=self.MODEL_NAME,
                    life_func=life_func,
                    failure_groups=getattr(fit, f"_{self.MLE_CLASS.__name__}__failure_groups", None),
                    stresses_for_groups=getattr(fit, f"_{self.MLE_CLASS.__name__}__stresses_for_groups", None),
                    use_level_stress=self.use_level_stress,
                    ax=True,
                )

                if _HAS_AZ:
                    steps, walkers, P = chain_raw.shape
                    chain_struc = np.swapaxes(chain_raw[burn::thin, :, :], 0, 1)
                    post_dict = {self.PARAM_NAMES[i]: chain_struc[:, :, i] for i in range(P)}
                    self.az_idata = az.from_dict(posterior=post_dict)
                    self.az_axes = {}
                    self.az_axes["trace"] = az.plot_trace(self.az_idata)
                    self.az_axes["pair"] = az.plot_pair(self.az_idata, var_names=list(self.PARAM_NAMES), kind="kde", marginals=True)
            except Exception:
                pass

    # Subclasses must implement:
    # def _loglike_impl(self, theta): ...
    # def USE_LIFE_FUNC_POST(self, S): ...


# Specific Weibull ALT classes

class Fit_Weibull_Power_Bayesian(_BaseWeibullALTBayes):
    """
    Weibull + Power life–stress.
    alpha(S) = a * S**n,  T ~ Weibull(beta, alpha).
    Parameters: a, n, beta
    """
    MODEL_NAME = "Power"
    PARAM_NAMES = ("a", "n", "beta")
    MLE_CLASS = Fit_Weibull_Power

    def _loglike_impl(self, theta):
        a, n, beta = theta
        if a <= 0 or beta <= 0:
            return -np.inf
        alpha_f = a * (self._S1f ** n)
        ll_f = _logpdf_weibull(self._fail, alpha_f, beta).sum() if len(self._fail) else 0.0
        alpha_rc = a * (self._S1rc ** n) if len(self._S1rc) else np.array([])
        ll_rc = _logsf_weibull(self._rc, alpha_rc, beta).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, n = self.a_post_mean, self.n_post_mean
        S = np.asarray(S)
        return a * (S ** n)


class Fit_Weibull_Exponential_Bayesian(_BaseWeibullALTBayes):
    """
    Weibull + Exponential life–stress.
    alpha(S) = b * exp(a / S),  T ~ Weibull(beta, alpha).
    Parameters: a, b, beta
    """
    MODEL_NAME = "Exponential"
    PARAM_NAMES = ("a", "b", "beta")
    MLE_CLASS = Fit_Weibull_Exponential

    def _loglike_impl(self, theta):
        a, b, beta = theta
        if b <= 0 or beta <= 0:
            return -np.inf
        alpha_f = b * np.exp(a / self._S1f)
        ll_f = _logpdf_weibull(self._fail, alpha_f, beta).sum() if len(self._fail) else 0.0
        alpha_rc = b * np.exp(a / self._S1rc) if len(self._S1rc) else np.array([])
        ll_rc = _logsf_weibull(self._rc, alpha_rc, beta).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, b = self.a_post_mean, self.b_post_mean
        return b * np.exp(a / np.asarray(S))


class Fit_Weibull_Eyring_Bayesian(_BaseWeibullALTBayes):
    """
    Weibull + Eyring life–stress.
    alpha(S) = (1/S) * exp(-(c - a/S)),  T ~ Weibull(beta, alpha).
    Parameters: a, c, beta
    """
    MODEL_NAME = "Eyring"
    PARAM_NAMES = ("a", "c", "beta")
    MLE_CLASS = Fit_Weibull_Eyring

    def _loglike_impl(self, theta):
        a, c, beta = theta
        if beta <= 0:
            return -np.inf
        S = self._S1f
        alpha_f = (1.0 / S) * np.exp(-(c - a / S))
        ll_f = _logpdf_weibull(self._fail, alpha_f, beta).sum() if len(self._fail) else 0.0
        S = self._S1rc
        alpha_rc = (1.0 / S) * np.exp(-(c - a / S)) if len(S) else np.array([])
        ll_rc = _logsf_weibull(self._rc, alpha_rc, beta).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, c = self.a_post_mean, self.c_post_mean
        S = np.asarray(S)
        return (1.0 / S) * np.exp(-(c - a / S))


class Fit_Weibull_Dual_Exponential_Bayesian(_BaseWeibullALTBayes):
    """
    Weibull + Dual-Exponential life–stress.
    alpha(S1,S2) = c * exp(a/S1 + b/S2),  T ~ Weibull(beta, alpha).
    Parameters: a, b, c, beta
    """
    MODEL_NAME = "Dual_Exponential"
    PARAM_NAMES = ("a", "b", "c", "beta")
    MLE_CLASS = Fit_Weibull_Dual_Exponential

    def _loglike_impl(self, theta):
        a, b, c, beta = theta
        if c <= 0 or beta <= 0:
            return -np.inf
        alpha_f = c * np.exp(a / self._S1f + b / self._S2f)
        ll_f = _logpdf_weibull(self._fail, alpha_f, beta).sum() if len(self._fail) else 0.0
        alpha_rc = c * np.exp(a / self._S1rc + b / self._S2rc) if len(self._rc) else np.array([])
        ll_rc = _logsf_weibull(self._rc, alpha_rc, beta).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, b, c = self.a_post_mean, self.b_post_mean, self.c_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return c * np.exp(a / S1 + b / S2)


class Fit_Weibull_Dual_Power_Bayesian(_BaseWeibullALTBayes):
    """
    Weibull + Dual-Power life–stress.
    alpha(S1,S2) = c * S1**m * S2**n,  T ~ Weibull(beta, alpha).
    Parameters: c, m, n, beta
    """
    MODEL_NAME = "Dual_Power"
    PARAM_NAMES = ("c", "m", "n", "beta")
    MLE_CLASS = Fit_Weibull_Dual_Power

    def _loglike_impl(self, theta):
        c, m, n, beta = theta
        if c <= 0 or beta <= 0:
            return -np.inf
        alpha_f = c * (self._S1f ** m) * (self._S2f ** n)
        ll_f = _logpdf_weibull(self._fail, alpha_f, beta).sum() if len(self._fail) else 0.0
        alpha_rc = c * (self._S1rc ** m) * (self._S2rc ** n) if len(self._rc) else np.array([])
        ll_rc = _logsf_weibull(self._rc, alpha_rc, beta).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        c, m, n = self.c_post_mean, self.m_post_mean, self.n_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return c * (S1 ** m) * (S2 ** n)


class Fit_Weibull_Power_Exponential_Bayesian(_BaseWeibullALTBayes):
    """
    Weibull + Power-Exponential life–stress.
    alpha(S1,S2) = c * exp(a/S1) * S2**n,  T ~ Weibull(beta, alpha).
    Parameters: a, c, n, beta
    """
    MODEL_NAME = "Power_Exponential"
    PARAM_NAMES = ("a", "c", "n", "beta")
    MLE_CLASS = Fit_Weibull_Power_Exponential

    def _loglike_impl(self, theta):
        a, c, n, beta = theta
        if c <= 0 or beta <= 0:
            return -np.inf
        alpha_f = c * np.exp(a / self._S1f) * (self._S2f ** n)
        ll_f = _logpdf_weibull(self._fail, alpha_f, beta).sum() if len(self._fail) else 0.0
        alpha_rc = c * np.exp(a / self._S1rc) * (self._S2rc ** n) if len(self._rc) else np.array([])
        ll_rc = _logsf_weibull(self._rc, alpha_rc, beta).sum() if len(self._rc) else 0.0
        return ll_f + ll_rc

    @staticmethod
    def USE_LIFE_FUNC_POST(self, S):
        a, c, n = self.a_post_mean, self.c_post_mean, self.n_post_mean
        S1, S2 = np.asarray(S[0]), np.asarray(S[1])
        return c * np.exp(a / S1) * (S2 ** n)
