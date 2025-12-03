import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import minimize, least_squares
from scipy.stats import norm, probplot, t
from scipy.special import expit
from IPython.display import Markdown, display

from adt_utils import (
    loglik_normal,
    loglik_lognormal,
    loglik_trunc_normal,
    md_param_table_freq,
    plot_degradation_by_stress,
    plot_residual_diagnostics,
    md_param_table_bayes,
    sample_ttf_from_posterior,
    summarize_ttf,
    plot_ttf_hist_with_fits,
    POSITIVE,
    REAL,
    get_param_domain,
    log_prior_single,
    log_prior_vector,
)

class Base_ADT_Model:
    """
    Base class for ADT degradation/performance models.

    This class handles:
      - Data housekeeping (y, t, units)
      - LSQ fit (with model-specific mu(theta, t, ...))
      - MLE fit with additive or multiplicative noise
      - Markdown parameter tables for LS and MLE

    Subclasses must at minimum implement:
      - self.param_names  (list of parameter names, same length as theta)
      - _mu(self, theta, t)  -> model mean at time t
    and typically override:
      - _init_guess(self)
      - _get_LS_bounds(self)
      - _get_MLE_bounds(self, p)
      - optional plot methods (_plot_fit_LS, _plot_fit_MLE, etc.)
    """

    # ------------------------------------------------------------------
    # Constructor & data housekeeping
    # ------------------------------------------------------------------
    def __init__(
        self,
        y,
        time,
        unit=None,
        CI=0.95,
        method="MLE",
        noise="additive",
        scale_type="damage",          # "damage" or "performance" (used mainly for TTF helpers)
        Df=None,                      # failure threshold (damage or performance)
        show_LSQ_diagnostics=False,
        show_noise_bounds=True,
        show_use_TTF_dist=False,      # subclasses can use this if they implement TTF plots
        print_results=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        y : array-like
            Observed response (damage or performance).
        time : array-like
            Time values.
        unit : array-like or None
            Unit IDs; if None, all observations are treated as unit 0.
        CI : float
            Confidence level (e.g. 0.95).
        method : {"LS", "MLE", "MLE_hierarchical", "Bayesian"}
            Main fitting approach. This base class implements LS and MLE.
        noise : {"additive", "multiplicative"}
            Observation noise model applied to y.
        scale_type : {"damage", "performance"}
            Conceptual scale of y (increasing damage vs decreasing performance).
            Mainly relevant for TTF helpers, which live in adt_utils.
        Df : float or None
            Failure threshold on y-scale (e.g. damage >= Df or performance <= Df).
        show_LSQ_diagnostics : bool
            If True, show residual diagnostics after LS fit (subclass must provide _plot_LS_diagnostics or call utils).
        show_noise_bounds : bool
            If True, MLE plots may include noise bands (subclass responsibility).
        show_use_TTF_dist : bool
            If True and subclass implements TTF plotting, they can call the relevant helpers.
        print_results : bool
            If True, LS and MLE tables will be printed in Markdown.
        kwargs : dict
            Free-form; subclasses can store/use these as needed.
        """
        # --- Data ---
        self.y = np.asarray(y, float).ravel()
        self.t = np.asarray(time, float).ravel()

        if unit is None:
            self.unit = np.zeros_like(self.y, dtype=int)
        else:
            self.unit = np.asarray(unit, int).ravel()

        if not (self.y.shape == self.t.shape == self.unit.shape):
            raise ValueError("y, time, and unit must have the same shape.")

        self.unique_units, self.unit_index = np.unique(self.unit, return_inverse=True)
        self.J = len(self.unique_units)

        # --- Inference settings ---
        self.CI = float(CI)
        self.alpha = 1.0 - self.CI
        self.z_CI = norm.ppf(0.5 + self.CI / 2.0)

        self.method = method
        self.noise = noise.lower()
        if self.noise not in ("additive", "multiplicative"):
            raise ValueError("noise must be 'additive' or 'multiplicative'.")

        self.scale_type = scale_type.lower()
        if self.scale_type not in ("damage", "performance"):
            raise ValueError("scale_type must be 'damage' or 'performance'.")

        self.Df_use = None if Df is None else float(Df)

        self.show_LSQ_diagnostics = show_LSQ_diagnostics
        self.show_noise_bounds = show_noise_bounds
        self.show_use_TTF_dist = show_use_TTF_dist
        self.print_results = print_results
        self.kwargs = kwargs

        # Storage for fits
        self.theta_LS = None
        self.sigma_LS = None
        self.cov_LS = None

        self.theta_MLE = None
        self.sigma_MLE = None
        self.cov_MLE = None

        self.stress = kwargs.get("stress", None)          # 1D array or None
        self.stress_label = kwargs.get("stress_label", "Stress")
        self.legend_title = kwargs.get("legend_title", "Stress")
        self.data_scale = kwargs.get("data_scale", "linear")
        
        # Subclasses MUST set this before calling _run_fit_pipeline
        self.param_names = getattr(self, "param_names", None)

    # ------------------------------------------------------------------
    # Model description hook
    # ------------------------------------------------------------------
    def _model_description(self):
        """
        Return a Markdown string describing the degradation / performance model,
        including the equation and a short explanation of each parameter.

        Subclasses should override this; default is None (print nothing).
        """
        return None

    def _show_model_description(self):
        """
        Print the model description (if provided) just before parameter tables.
        """
        if not getattr(self, "print_results", False):
            return
        desc = self._model_description()
        if not desc:
            return

        # If the subclass returned plain text, give it a heading
        text = desc.strip()
        if not text.startswith("#"):
            text = "### Degradation / performance model\n\n" + text

        display(Markdown(text))
    
    # ------------------------------------------------------------------
    # Abstract / hook methods to be implemented by subclasses
    # ------------------------------------------------------------------
    def plot_data(self, scale=None):
        """
        Generic degradation-vs-time plot grouped by stress.
        Subclasses just need to set self.stress (or skip if N/A).
        """
        if self.stress is None:
            raise RuntimeError("No stress array available; set self.stress in subclass.")
    
        if scale is None:
            scale = getattr(self, "data_scale", "linear")
    
        plot_degradation_by_stress(
            t=self.t,
            D=self.y,
            stress=self.stress,
            unit=self.unit,
            title="Degradation vs Time",
            stress_label=self.stress_label,
            legend_title=self.legend_title,
            scale=scale,
            show_unit_lines=True,
        )

    def _mu(self, theta, t):
        """
        Deterministic mean model mu(t) for the subclass.

        Parameters
        ----------
        theta : array-like
            Model parameters (same order as self.param_names).
        t : array-like
            Time array.

        Returns
        -------
        mu : np.ndarray
            Mean response at times t.
        """
        raise NotImplementedError("Subclasses must implement _mu(theta, t).")

    def _init_guess(self):
        """
        Initial guess for LSQ fit.

        Default: crude generic guess based on percentiles of y and sqrt(t).
        Subclasses are encouraged to override with model-specific heuristics.
        """
        # avoid degenerate span
        g0_init = float(np.percentile(self.y, 10))
        span = max(np.percentile(self.y, 90) - g0_init, 1e-6)
        g1_init = span / max(np.sqrt(self.t.max()), 1.0)

        # as a generic 3-parameter template: [g0, g1, Ea]-like
        # subclasses with different dimensionality should override.
        theta0 = np.array([g0_init, g1_init, 0.7], dtype=float)
        return theta0
        
    def _get_LS_bounds(self, p):
        """
        Bounds for LSQ fit.

        Parameters
        ----------
        p : int
            Number of parameters in theta.

        Returns
        -------
        bounds : list of (low, high) or None
            If None, unconstrained LSQ is used for all parameters.

        Subclasses should override to enforce positivity / physical ranges.
        """
        return None  # unconstrained by default

    def _get_MLE_bounds(self, p):
        """
        Bounds for MLE parameters [theta, log_sigma].

        Parameters
        ----------
        p : int
            Number of parameters in theta.

        Returns
        -------
        bounds : list of (low, high)
        """
        # Default: use LS bounds (or None) for theta, and generic for log_sigma
        ls_bounds = self._get_LS_bounds(p)
        if ls_bounds is None:
            ls_bounds = [(None, None)] * p

        # log_sigma in [log(1e-6), log(1e3)] by default
        bounds = list(ls_bounds) + [(np.log(1e-6), np.log(1e3))]
        return bounds

    # Optional hooks for subclass-specific plotting
    def _plot_fit_LS(self):
        """Subclasses can implement LS fit plots if desired."""
        pass

    def _plot_fit_MLE(self):
        """Subclasses can implement MLE fit plots if desired."""
        pass

    def _plot_LS_diagnostics(self):
        """Generic LS residual plots using self.stress if available."""
        if self.theta_LS is None:
            return
        theta = self.theta_LS
        mu = self._mu(theta, self.t)
        resid = self.y - mu
    
        if self.stress is None:
            # fall back to time-only diagnostics? or just skip
            return
    
        plot_residual_diagnostics(
            t=self.t,
            stress=self.stress,
            resid=resid,
            mu=mu,
            title_prefix=f"LSQ diagnostics ({self.__class__.__name__})",
            stress_label=self.stress_label,
        )
  
    # ------------------------------------------------------------------
    # LSQ fit
    # ------------------------------------------------------------------
    def _ls_objective(self, theta):
        mu = self._mu(theta, self.t)
        resid = self.y - mu
        return np.sum(resid ** 2)

    def _fit_LS(self, suppress_print=False):
        theta0 = self._init_guess()
        theta0 = np.asarray(theta0, float)
        p = len(theta0)

        bounds = self._get_LS_bounds(p)

        res = minimize(
            self._ls_objective,
            theta0,
            method="L-BFGS-B" if bounds is not None else "BFGS",
            bounds=bounds,
        )

        if not res.success:
            print("Warning: LS fit did not converge:", res.message)

        self.theta_LS = res.x

        mu = self._mu(self.theta_LS, self.t)
        resid = self.y - mu
        dof = max(len(self.y) - len(self.theta_LS), 1)
        self.sigma_LS = np.sqrt(np.sum(resid ** 2) / dof)

        # Try to extract covariance
        hess_inv = getattr(res, "hess_inv", None)
        if hasattr(hess_inv, "todense"):
            self.cov_LS = np.array(hess_inv.todense())
        elif isinstance(hess_inv, np.ndarray) and hess_inv.ndim == 2:
            self.cov_LS = hess_inv
        else:
            self.cov_LS = None

        if self.print_results and not suppress_print and self.param_names is not None:
            noise_label = "D" if self.noise == "additive" else "log D"
            self._show_model_description()    
            md_param_table_freq(
                theta=self.theta_LS,
                cov=self.cov_LS,
                names=self.param_names,
                z_CI=self.z_CI,
                CI=self.CI,
                sigma=self.sigma_LS,
                noise_label=noise_label,
                label="LSQ",
            )
            

    # ------------------------------------------------------------------
    # MLE fit
    # ------------------------------------------------------------------
    def _negloglik_additive(self, params):
        params = np.asarray(params, float)
        theta = params[:-1]
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)

        mu = self._mu(theta, self.t)
        ll = loglik_normal(self.y, mu, sigma)
        return -ll

    def _negloglik_multiplicative(self, params):
        params = np.asarray(params, float)
        theta = params[:-1]
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)

        mu = self._mu(theta, self.t)
        ll = loglik_lognormal(self.y, mu, sigma)
        return -ll

    def _negloglik(self, params):
        """
        Dispatch to additive / multiplicative negative log-likelihood
        based on self.noise. Subclasses can override this entirely if
        they need a custom likelihood (e.g. sqrt-Arrhenius).
        """
        if self.noise == "additive":
            return self._negloglik_additive(params)
        elif self.noise == "multiplicative":
            return self._negloglik_multiplicative(params)
        else:
            raise ValueError(f"Unknown noise model: {self.noise}")

    def _fit_MLE(self, suppress_print: bool = False):
        """
        Generic MLE fit for additive / multiplicative noise.
    
        If suppress_print=True, do NOT print parameter table (used when
        called just to initialise Bayesian priors).
        """
        if self.theta_LS is None:
            raise RuntimeError("Run LS before MLE to get initial guesses.")
    
        theta0 = self.theta_LS
        sigma0 = self.sigma_LS if self.sigma_LS is not None else np.std(self.y)
        p = len(theta0)
    
        # [your existing bounds builder, e.g. _get_MLE_bounds(p)]
        bounds = self._get_MLE_bounds(p)
    
        # Pack params = [theta..., log_sigma]
        x0 = np.concatenate([theta0, [np.log(sigma0 + 1e-6)]])
    
        res = minimize(
            self._negloglik,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
        )
    
        if not res.success:
            print("Warning: MLE did not converge:", res.message)
    
        x_hat = res.x
        self.theta_MLE = x_hat[:-1]
        self.sigma_MLE = float(np.exp(x_hat[-1]))
    
        # Covariance from inverse Hessian if available
        hess_inv = getattr(res, "hess_inv", None)
        if hasattr(hess_inv, "todense"):
            self.cov_MLE = np.array(hess_inv.todense())
        elif isinstance(hess_inv, np.ndarray) and hess_inv.ndim == 2:
            self.cov_MLE = hess_inv
        else:
            self.cov_MLE = None
    
        # ONLY print if we are not suppressing and user wants results
        if self.print_results and not suppress_print:
            noise_label = "D" if self.noise == "additive" else "log D"
            self._show_model_description()    
            md_param_table_freq(
                theta=self.theta_MLE,
                cov=self.cov_MLE,
                names=self.param_names,
                z_CI=self.z_CI,
                CI=self.CI,
                sigma=self.sigma_MLE,
                noise_label=noise_label,
                label=f"MLE ({self.noise})",
            )
            
    # ------------------------------------------------------------------
    # Bayesian fit
    # ------------------------------------------------------------------
    def _run_emcee(self, log_prob, init, nwalkers=32, nburn=1000, nsamp=2000):
        ndim = len(init)
        rng = np.random.default_rng(123)
        p0 = init + rng.normal(scale=0.2, size=(nwalkers, ndim))
    
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        state = sampler.run_mcmc(p0, nburn, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, nsamp, progress=True)
        chain = sampler.get_chain(flat=True)
        return sampler, chain

    # ------------------------------------------------------------------
    # Generic Bayesian fitter with optional custom priors
    # ------------------------------------------------------------------
    def _fit_Bayesian_generic(
        self,
        param_names,
        model_name,
        default_centers,
        default_sds,
        loglike_func,
        priors=None,
        dist_name="NormalError",
        nwalkers=32,
        nburn=1000,
        nsamp=2000,
    ):
        """
        Generic Bayesian engine for ADT models.

        - param_names: list of parameter names, e.g. ["g0", "g1", "Ea", "sigma"]
        - model_name : string key used in _MODEL_PARAM_DOMAIN for domains
        - default_centers: dict name -> centre on NATURAL scale (usually MLE)
        - default_sds: dict name -> prior scale on TRANSFORM space (eta)
        - loglike_func: function(theta_dict) -> log-likelihood
                        where theta_dict maps name -> natural value
        - priors: None or dict of priors (passed to log_prior_vector)
        - dist_name: for prior domain lookup; "NormalError" for ADT

        Parameterisation:
        - For POSITIVE domain: sample in eta = log(x)
        - For REAL domain:     sample in eta = x
        - If priors is None: independent Normal priors on eta
        - If priors is dict:  use log_prior_vector on NATURAL params
        """
        param_names = list(param_names)
        P = len(param_names)

        # Determine domains for each parameter
        domains = []
        for name in param_names:
            dom = get_param_domain(model_name, dist_name, name)
            if dom not in (POSITIVE, REAL):
                dom = REAL
            domains.append(dom)

        # Build initial eta0 and prior means in transform space
        eta0 = np.zeros(P, float)
        sd_eta = np.zeros(P, float)

        for j, name in enumerate(param_names):
            c = float(default_centers[name])
            s = float(default_sds.get(name, 0.5))

            if domains[j] == POSITIVE:
                c = max(c, 1e-12)
                eta0[j] = np.log(c)
            else:  # REAL
                eta0[j] = c

            sd_eta[j] = max(s, 1e-8)

        priors_dict = priors  # might be None

        # ---------- PRIOR ON eta ----------
        def log_prior_eta(eta):
            eta = np.asarray(eta, float)
            if eta.shape != eta0.shape:
                return -np.inf

            # Case 1: no custom priors -> independent Normal on eta
            if priors_dict is None:
                z = (eta - eta0) / sd_eta
                lp = -0.5 * np.sum(z**2) - np.sum(np.log(sd_eta * np.sqrt(2.0 * np.pi)))
                return lp

            # Case 2: custom priors on NATURAL scale via log_prior_vector
            theta_nat = {}
            for j, name in enumerate(param_names):
                if domains[j] == POSITIVE:
                    x = np.exp(eta[j])
                else:
                    x = eta[j]
                theta_nat[name] = x

            lp = log_prior_vector(
                model=model_name,
                dist=dist_name,
                theta_dict=theta_nat,
                priors=priors_dict,
            )
            return lp

        # ---------- POSTERIOR ON eta ----------
        def log_prob_eta(eta):
            lp = log_prior_eta(eta)
            if not np.isfinite(lp):
                return -np.inf

            # Build natural param dict
            theta_nat = {}
            for j, name in enumerate(param_names):
                if domains[j] == POSITIVE:
                    x = np.exp(eta[j])
                else:
                    x = eta[j]
                theta_nat[name] = x

            ll = loglike_func(theta_nat)
            if not np.isfinite(ll):
                return -np.inf

            return lp + ll

        # Run MCMC (self._run_emcee should return flattened chain)
        sampler, chain = self._run_emcee(
            log_prob_eta,
            init=eta0,
            nwalkers=nwalkers,
            nburn=nburn,
            nsamp=nsamp,
        )

        # Convert chain (eta) to NATURAL parameter samples
        chain = np.asarray(chain, float)  # (N, P)
        nat_samples = {}
        for j, name in enumerate(param_names):
            eta_j = chain[:, j]
            if domains[j] == POSITIVE:
                nat_samples[name] = np.exp(eta_j)
            else:
                nat_samples[name] = eta_j

        # Optional ArviZ wrapper
        try:
            import arviz as az
            self.idata_bayes = az.from_emcee(
                sampler,
                var_names=param_names,
            )
        except Exception:
            self.idata_bayes = None

        return sampler, chain, nat_samples

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _run_fit_pipeline(self):
        """
        Run the main fitting pipeline according to self.method.

        Subclasses typically call this at the end of their __init__ after:
          - setting any model-specific attributes (e.g. stresses, T_use, etc.)
          - defining self.param_names
        """
        if self.param_names is None:
            raise RuntimeError("Subclasses must set self.param_names before calling _run_fit_pipeline().")

        method_norm = self.method.lower()

        if method_norm == "ls":
            self._fit_LS(suppress_print=False)
            if self.show_LSQ_diagnostics and hasattr(self, "_plot_LS_diagnostics"):
                # subclass or utils-based
                self._plot_LS_diagnostics()
            if hasattr(self, "_plot_fit_LS"):
                self._plot_fit_LS()

        elif method_norm == "mle":
            # LS as initialisation; quiet table
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics and hasattr(self, "_plot_LS_diagnostics"):
                self._plot_LS_diagnostics()
            self._fit_MLE()
            if hasattr(self, "_plot_fit_MLE"):
                self._plot_fit_MLE()

        elif method_norm in ("mle_hierarchical", "mle_hierarchical".lower()):
            # Base class does not implement hierarchical MLE
            raise NotImplementedError("Hierarchical MLE is model-specific; implement in subclass.")

        elif method_norm.startswith("bayes"):
            # Base class does not implement Bayesian; subclasses (e.g. sqrt-Arrhenius) can.
            raise NotImplementedError("Bayesian fitting is not implemented in Base_ADT_Model.")

        else:
            raise ValueError(f"Unknown method: {self.method}")

K_BOLTZ_eV = 8.617333262e-5  # eV/K

class Fit_ADT_sqrt_Arrhenius(Base_ADT_Model):
    r"""
    Sqrt-time + Arrhenius *damage* model:

        D(t, T) = g0 + g1 * sqrt(t) * exp(Ea / k * (1/T_use - 1/T))

    where:
        - D(t, T) is damage (increasing over time)
        - T is stress temperature
        - T_use is use temperature
        - Ea is activation energy in eV

    This class supports:
        - LSQ fit
        - MLE (additive or multiplicative noise)
        - Bayesian fit via emcee (with log-parameter priors)
        - Optional TTF-at-use histogram from Bayesian posterior
    """

    def __init__(
        self,
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
    ):
        """
        Parameters
        ----------
        degradation : array-like
            Observed damage D_ij.
        stress : array-like
            Temperature in °C for each observation.
        time : array-like
            Time values (e.g. hours).
        unit : array-like
            Unit IDs (integers).
        stress_use : float
            Use temperature in °C.
        Df : float
            Failure threshold on *damage* scale (fail when D >= Df).
        CI : float
            Confidence/credible level (default 0.95).
        method : {"LS", "MLE", "MLE_hierarchical", "Bayesian"}
            Fit method. This class implements LS, MLE, and Bayesian.
        noise : {"additive", "multiplicative"}
            Observation noise model on D.
        show_data_plot : bool
            If True, scatter plot of D vs t by temperature.
        show_LSQ_diagnostics : bool
            If True, show LS residual diagnostics.
        show_noise_bounds : bool
            If True, show noise bands on MLE & Bayes plots.
        show_use_TTF_dist : bool
            If True (and method is Bayesian), compute and plot TTF at use.
        print_results : bool
            If True, print parameter tables in Markdown.
        kwargs :
            Additional options (currently passed through to Base_ADT_Model).
        """
        method_norm = method.lower()

        # 1) Call base constructor 
        super().__init__(
            y=degradation,
            time=time,
            unit=unit,
            CI=CI,
            method=method,
            noise=noise,
            scale_type="damage",
            Df=Df,
            show_LSQ_diagnostics=show_LSQ_diagnostics,
            show_noise_bounds=show_noise_bounds,
            show_use_TTF_dist=show_use_TTF_dist,
            print_results=print_results,
            stress=stress,
            stress_label="Temperature (°C)",
            legend_title="Temperature",
            data_scale=data_scale,
            **kwargs,
        )

        # 2) Model-specific fields
        self.T_C = np.asarray(stress, float).ravel()
        if self.T_C.shape != self.y.shape:
            raise ValueError("stress (T_C) must have same shape as degradation (y).")
        self.T_K = self.T_C + 273.15
        self.T_use_C = float(stress_use)
        self.T_use_K = self.T_use_C + 273.15

        self.data_scale = data_scale          # <--- store preferred scale
        self.param_names = ["g0", "g1", "Ea (eV)"]
        self.priors = priors  # may be None or a dict

        # 3) Optional raw data plot
        if show_data_plot:
            self.plot_data()

        # 4) Run method-specific pipeline (same as you had, but MLE is silent for Bayes)
        if method_norm == "ls":
            self._fit_LS(suppress_print=False)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._plot_fit_LS()

        elif method_norm == "mle":
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE()
            self._plot_fit_MLE()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution_MLE()

        elif method_norm.startswith("bayes"):
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            # use MLE only to set priors => suppress_print=True
            self._fit_MLE(suppress_print=True)
            self._fit_Bayesian()
            self._plot_fit_Bayesian()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution()

        elif method_norm in ("mle_hierarchical", "mle_heirachical"):
            raise NotImplementedError("Hierarchical MLE not yet implemented for sqrt-Arrhenius.")

        else:
            raise ValueError(f"Unknown method: {method}")
            
    # ------------------------------------------------------------------
    # Model Description
    # ------------------------------------------------------------------                
    def _model_description(self):
        return r"""
### Degradation / damage model

**Model form (damage scale)**

We model accumulated damage as a square-root function of (Arrhenius-scaled) time:

$$
\mathrm{AF}_T(T)
= \exp\!\left(
  \frac{E_a}{k_B}
  \left(
    \frac{1}{T_{\text{use}}} - \frac{1}{T}
  \right)
\right),
$$

$$
D(t, T)
= \gamma_{0} + \gamma_{1} \,\sqrt{\mathrm{AF}_T(T)\, t}.
$$

**Parameters:**

- **$\gamma_{0}$** – baseline damage at \(t=0\) (often fixed or close to 0)  
- **$\gamma_{1}$** – scale factor on the square-root damage term  
- **$E_{a}$** – Arrhenius activation energy (eV) for temperature acceleration  
- **$T_{\text{use}}$** – use-level temperature (fixed input, not fitted)  
"""
        
    # ------------------------------------------------------------------
    # Model mean: data level
    # ------------------------------------------------------------------
    def _mu(self, theta, t):
        """
        Mean damage model at the *data* temperatures (self.T_K):

            D_i = g0 + g1 * sqrt(t_i) * exp(Ea / k * (1/T_use - 1/T_i))

        This is used internally by Base_ADT_Model for LS/MLE on the observed data.
        """
        g0, g1, Ea = theta
        t = np.asarray(t, float)

        if t.shape != self.t.shape:
            # For safety, enforce data-shape usage; plotting uses separate helpers.
            raise ValueError(
                "_mu is intended for data-fitting only (t must match self.t shape). "
                "Use model-specific helpers for grids/other temperatures."
            )

        accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / self.T_K))
        return g0 + g1 * np.sqrt(t) * accel
        
    # --------------------------------------------------------------
    # Negative log-likelihood (called by Base_ADT_Model._fit_MLE)
    # --------------------------------------------------------------
    def _negloglik(self, x):
        """
        x = [g0, g1, Ea, log_sigma]
        Likelihood depends on additive or multiplicative noise on damage.
        """
        g0, g1, Ea = x[:-1]
        log_sigma = x[-1]
        sigma = np.exp(log_sigma)

        # No negative parameters allowed
        if sigma <= 0 or g1 <= 0 or Ea <= 0:
            return np.inf

        theta = np.array([g0, g1, Ea], float)
        mu = self._mu(theta, self.t)

        if self.noise == "additive":
            # D ~ Normal(mu, sigma)
            if not np.all(np.isfinite(mu)):
                return np.inf
            return -np.sum(norm.logpdf(self.y, loc=mu, scale=sigma))

        elif self.noise == "multiplicative":
            # log D ~ Normal(log mu, sigma)
            D_pos = np.clip(self.y, 1e-12, None)
            mu_pos = np.clip(mu, 1e-12, None)
            logD = np.log(D_pos)
            logmu = np.log(mu_pos)
            z = (logD - logmu) / sigma

            if not np.all(np.isfinite(z)):
                return np.inf

            return (
                np.sum(np.log(D_pos)) +   # jacobian for lognormal
                len(D_pos) * np.log(sigma) -
                np.sum(norm.logpdf(z))
            )

        else:
            raise ValueError(f"Unknown noise model: {self.noise}")

    # ------------------------------------------------------------------
    # Initial guess & bounds
    # ------------------------------------------------------------------
    def _init_guess(self):
        """
        Model-specific initial guess for [g0, g1, Ea].
        """
        g0_init = float(np.percentile(self.y, 10))
        span = max(np.percentile(self.y, 90) - g0_init, 1e-6)
        g1_init = span / max(np.sqrt(self.t.max()), 1.0)
        Ea_init = 0.7  # eV – generic starter
        return np.array([g0_init, g1_init, Ea_init], dtype=float)

    def _get_LS_bounds(self, p):
        # p == 3 for [g0, g1, Ea]
        return [
            (None, None),   # g0
            (1e-10, None),  # g1 > 0
            (1e-3, 5.0),    # Ea in [0.001, 5] eV
        ]

    # MLE bounds: reuse LS bounds + log_sigma
    # (Base_ADT_Model._get_MLE_bounds already does this, so we don't override)

    # ------------------------------------------------------------------
    # LS diagnostics & plots
    # ------------------------------------------------------------------

    def _plot_fit_LS(self):
        if self.theta_LS is None:
            return

        theta = self.theta_LS
        g0, g1, Ea = theta

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        cmap = plt.get_cmap("viridis")
        colors = {T: cmap(i / max(len(temps) - 1, 1)) for i, T in enumerate(temps)}

        # Scatter data
        for T in temps:
            mask = (self.T_C == T)
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=25,
                alpha=0.7,
                color=colors[T],
                label=f"{T:.0f} °C data",
            )

        # Mean curves for each temperature
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        for T in temps:
            T_K = T + 273.15
            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K))
            mu_grid = g0 + g1 * np.sqrt(t_grid) * accel
            ax.plot(
                t_grid,
                mu_grid,
                color=colors[T],
                lw=2,
                label=f"{T:.0f} °C LS mean",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title("LSQ Sqrt-Arrhenius Degradation Fit")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=2)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # MLE plot
    # ------------------------------------------------------------------
    def _plot_fit_MLE(self):
        if self.theta_MLE is None:
            return

        theta = self.theta_MLE
        g0, g1, Ea = theta
        sigma = self.sigma_MLE

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, T in enumerate(temps):
            idx = min(i, len(base_colors) - 1)
            styles[T] = dict(color=base_colors[idx], marker=base_markers[idx])

        # Scatter data
        for T in temps:
            mask = (self.T_C == T)
            st = styles[T]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"{T:.0f} °C data",
            )

        # Mean curves + noise bands
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)

        for T in temps:
            T_K = T + 273.15
            st = styles[T]

            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K))
            mu_grid = g0 + g1 * np.sqrt(t_grid) * accel

            ax.plot(
                t_grid,
                mu_grid,
                color=st["color"],
                lw=2,
                label=f"{T:.0f} °C MLE mean",
            )

            if self.show_noise_bounds and sigma is not None:
                if self.noise == "additive":
                    lower = mu_grid - self.z_CI * sigma
                    upper = mu_grid + self.z_CI * sigma
                else:
                    # multiplicative/lognormal on D
                    factor = np.exp(self.z_CI * sigma)
                    lower = mu_grid / factor
                    upper = mu_grid * factor

                ax.fill_between(
                    t_grid,
                    lower,
                    upper,
                    color=st["color"],
                    alpha=0.2,
                    label=f"{T:.0f} °C {int(100*self.CI)}% band",
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"MLE Sqrt-Arrhenius Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Bayesian fit (emcee)
    # ------------------------------------------------------------------
    def _fit_Bayesian(self, nwalkers=32, nburn=1000, nsamp=2000):
        """
        Bayesian fit for sqrt-Arrhenius using Base_ADT_Model._fit_Bayesian_generic.

        - If self.priors is None: independent Normal priors on transform coords
          (log g0, log g1, log Ea, log sigma) centred at MLE/LS.
        - If self.priors is a dict: priors on NATURAL scale for
          { "g0", "g1", "Ea", "sigma" } dispatched via log_prior_vector
          with model="Sqrt_Arrhenius", dist="NormalError".
        """
        # Get prior centres from MLE if available, else LS
        if self.theta_MLE is not None and self.sigma_MLE is not None:
            g0_hat, g1_hat, Ea_hat = self.theta_MLE
            sig_hat = self.sigma_MLE
        elif self.theta_LS is not None and self.sigma_LS is not None:
            g0_hat, g1_hat, Ea_hat = self.theta_LS
            sig_hat = self.sigma_LS
        else:
            raise RuntimeError("Run LS or MLE before Bayesian (no initial estimates).")

        # Centres on NATURAL scale
        default_centers = {
            "g0": float(g0_hat),
            "g1": float(g1_hat),
            "Ea": float(Ea_hat),
            "sigma": float(sig_hat),
        }

        # Prior sd on TRANSFORM space (eta). 0.5 is what you used before.
        default_sds = {
            "g0": 0.5,
            "g1": 0.5,
            "Ea": 0.5,
            "sigma": 0.5,
        }

        t_time = self.t
        D_obs = self.y

        # Likelihood in terms of NATURAL parameters
        def loglike(theta_dict):
            g0 = theta_dict["g0"]
            g1 = theta_dict["g1"]
            Ea = theta_dict["Ea"]
            sigma = theta_dict["sigma"]

            if sigma <= 0 or g1 <= 0 or Ea <= 0:
                return -np.inf

            params = np.array([g0, g1, Ea], float)
            mu = self._mu(params, t_time)

            if self.noise == "additive":
                if not np.all(np.isfinite(mu)):
                    return -np.inf
                return np.sum(norm.logpdf(D_obs, loc=mu, scale=sigma))

            elif self.noise == "multiplicative":
                D_pos = np.clip(D_obs, 1e-12, None)
                mu_pos = np.clip(mu,   1e-12, None)
                logD = np.log(D_pos)
                logmu = np.log(mu_pos)
                if not np.all(np.isfinite(logmu)):
                    return -np.inf
                z = (logD - logmu) / sigma
                return (
                    -np.sum(np.log(D_pos))
                    - len(D_pos)*np.log(sigma)
                    + np.sum(norm.logpdf(z))
                )
            else:
                return -np.inf

        param_names = ["g0", "g1", "Ea", "sigma"]

        sampler, chain, nat_samples = self._fit_Bayesian_generic(
            param_names=param_names,
            model_name="Sqrt_Arrhenius_ADT",
            default_centers=default_centers,
            default_sds=default_sds,
            loglike_func=loglike,
            priors=getattr(self, "priors", None),
            dist_name="NormalError",
            nwalkers=nwalkers,
            nburn=nburn,
            nsamp=nsamp,
        )

        # Store posterior draws
        self.g0_s    = nat_samples["g0"]
        self.g1_s    = nat_samples["g1"]
        self.Ea_s    = nat_samples["Ea"]
        self.sigma_s = nat_samples["sigma"]

        # Markdown parameter table
        if self.print_results:
            self._show_model_description() 
            md_param_table_bayes(
                {
                    "g0": self.g0_s,
                    "g1": self.g1_s,
                    "Ea (eV)": self.Ea_s,
                    r"$\sigma$": self.sigma_s,
                },
                cred_mass=self.CI,
                label=f"Bayesian ({self.noise})",
            )

    # ------------------------------------------------------------------
    # Bayesian plot
    # ------------------------------------------------------------------
    def _plot_fit_Bayesian(self, nsamples=1000, show_predictive=True):
        if not hasattr(self, "g0_s"):
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, T in enumerate(temps):
            idx = min(i, len(base_colors) - 1)
            styles[T] = dict(color=base_colors[idx], marker=base_markers[idx])

        # Scatter data
        for T in temps:
            mask = (self.T_C == T)
            st = styles[T]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"{T:.0f} °C data",
            )

        # Thin posterior for plotting
        rng = np.random.default_rng(7)
        n_post = len(self.g0_s)
        ns = min(nsamples, n_post)
        sel = rng.choice(n_post, size=ns, replace=False)

        _g0  = self.g0_s[sel]
        _g1  = self.g1_s[sel]
        _Ea  = self.Ea_s[sel]
        _sig = self.sigma_s[sel]

        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        alpha = self.alpha
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        for T in temps:
            T_K = T + 273.15
            st = styles[T]

            accel = np.exp(
                _Ea[:, None] / K_BOLTZ_eV
                * (1.0 / self.T_use_K - 1.0 / T_K)
            )
            sqrt_t = np.sqrt(t_grid[None, :])
            mu_draws = _g0[:, None] + _g1[:, None] * sqrt_t * accel

            mu_mean = mu_draws.mean(axis=0)
            lo_mean, hi_mean = np.quantile(mu_draws, [q_lo, q_hi], axis=0)

            ax.plot(
                t_grid,
                mu_mean,
                color=st["color"],
                lw=2,
                label=f"{T:.0f} °C Bayes mean",
            )

            if show_predictive:
                eps = rng.standard_normal(size=mu_draws.shape)

                if self.noise == "additive":
                    y_draws = mu_draws + _sig[:, None] * eps
                else:
                    mu_clip = np.clip(mu_draws, 1e-12, None)
                    log_y = np.log(mu_clip) + _sig[:, None] * eps
                    y_draws = np.exp(log_y)

                lo_pred, hi_pred = np.quantile(y_draws, [q_lo, q_hi], axis=0)
                ax.fill_between(
                    t_grid,
                    lo_pred,
                    hi_pred,
                    alpha=0.15,
                    color=st["color"],
                    label=None,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"Bayesian Sqrt-Arrhenius Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # TTF at use (Bayesian posterior -> histogram + fitted curves)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution(
        self,
        Df=None,
        T_eval_C=None,
        n_samps=8000,
        unit_label="time units",
    ):
        """
        Draw posterior TTF samples at use (or given) conditions,
        print summary, and plot histogram + Weibull / Gamma / Lognormal fits.

        Currently uses **Bayesian posterior** only
        (i.e. self.g0_s, self.g1_s, self.Ea_s, self.sigma_s).
        """
        if not hasattr(self, "g0_s"):
            print("Bayesian posterior not available; cannot compute TTF distribution.")
            return

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
        else:
            T_eval_K = float(T_eval_C) + 273.15

        theta_samples = np.column_stack([self.g0_s, self.g1_s, self.Ea_s])
        sigma_samples = np.asarray(self.sigma_s, float)

        def mu_fun_factory(theta_vec):
            g0, g1, Ea = theta_vec
            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K))

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                return g0 + g1 * np.sqrt(t_arr) * accel

            return mu_fun

        # 1) Sample TTF posterior
        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        # 2) Summarise
        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at use (Df={Df})",
            unit_label=unit_label,
            as_markdown=True,
        )

        # 3) Plot histogram + fitted distributions
        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title="TTF posterior and best distribution fits",
            bins=60,
        )
    # ------------------------------------------------------------------
    # TTF at use (MLE plug-in version)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution_MLE(
        self,
        Df=None,
        T_eval_C=None,
        n_samps=8000,
        unit_label="time units",
    ):
        """
        Draw TTF samples at use temperature (or T_eval_C) using *MLE* parameters.

        This treats (g0, g1, Ea, sigma) as known (plug-in),
        so variability is only from the stochastic noise model.
        """
        if (self.theta_MLE is None) or (self.sigma_MLE is None):
            raise RuntimeError("MLE fit not available – run method='MLE' first.")

        # Failure threshold
        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        # Evaluation temperature
        if T_eval_C is None:
            T_eval_K = self.T_use_K
            label_T = f"T={self.T_use_C:.0f} °C"
        else:
            T_eval_K = float(T_eval_C) + 273.15
            label_T = f"T={float(T_eval_C):.0f} °C"

        g0_hat, g1_hat, Ea_hat = self.theta_MLE
        sigma_hat = float(self.sigma_MLE)

        # Pseudo "posterior" samples: repeat MLE
        theta_samples = np.tile(
            np.array([g0_hat, g1_hat, Ea_hat], float)[None, :],
            (n_samps, 1),
        )
        sigma_samples = np.full(n_samps, sigma_hat, float)

        def mu_fun_factory(theta_vec):
            g0, g1, Ea = theta_vec

            accel = np.exp(
                Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K)
            )

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 0.0, None)
                return g0 + g1 * np.sqrt(t_pos) * accel

            return mu_fun

        # 1) Sample TTFs
        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        # 2) Summarise
        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at {label_T} (Df={Df}, MLE plug-in)",
            unit_label=unit_label,
            as_markdown=True,
        )

        # 3) Histogram + fitted distributions
        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF (MLE plug-in) at {label_T}",
            bins=60,
        )

class Fit_ADT_Mitsuom_Arrhenius(Base_ADT_Model):
    r"""
    Mitsuom + Arrhenius *performance* model:

        D(t, T) = [ 1 + a * ( t * exp(Ea / k_B * (1/T_use - 1/T)) )^{b} ]^{-1}

    where:
        - D(t, T) is performance in (0, 1], decreasing over time
        - t is time
        - T is stress temperature (°C input, converted to K internally)
        - T_use is use temperature
        - a > 0, b > 0 control the degradation rate/curvature
        - Ea > 0 is activation energy in eV

    Supports:
        - LSQ fit
        - MLE (additive or multiplicative noise)
        - Bayesian fit via emcee (through _fit_Bayesian_generic with priors)
        - Optional TTF-at-use histogram (Bayesian or MLE plug-in)
    """

    def __init__(
        self,
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
    ):
        method_norm = method.lower()

        # 1) Base constructor
        super().__init__(
            y=degradation,
            time=time,
            unit=unit,
            CI=CI,
            method=method,
            noise=noise,
            # Mitsuom here is a performance model (starts near 1, decays):
            scale_type="performance",
            Df=Df,
            show_LSQ_diagnostics=show_LSQ_diagnostics,
            show_noise_bounds=show_noise_bounds,
            show_use_TTF_dist=show_use_TTF_dist,
            print_results=print_results,
            stress=stress,
            stress_label="Temperature (°C)",
            legend_title="Temperature",
            data_scale=data_scale,
            **kwargs,
        )

        # 2) Model-specific fields
        self.T_C = np.asarray(stress, float).ravel()
        if self.T_C.shape != self.y.shape:
            raise ValueError("stress (T_C) must have same shape as degradation/performance (y).")
        self.T_K = self.T_C + 273.15
        self.T_use_C = float(stress_use)
        self.T_use_K = self.T_use_C + 273.15
        self.priors = priors
        self.data_scale = data_scale
        self.param_names = ["a", "b", "Ea (eV)"]

        # 3) Optional raw data plot
        if show_data_plot:
            self.plot_data()

        # 4) Method-specific pipeline
        if method_norm == "ls":
            self._fit_LS(suppress_print=False)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._plot_fit_LS()

        elif method_norm == "mle":
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE()
            self._plot_fit_MLE()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution_MLE()

        elif method_norm.startswith("bayes"):
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE(suppress_print=True)   # centres priors
            self._fit_Bayesian()
            self._plot_fit_Bayesian()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution()

        elif method_norm in ("mle_hierarchical", "mle_heirachical"):
            raise NotImplementedError("Hierarchical MLE not yet implemented for Mitsuom–Arrhenius.")

        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Model Description
    # ------------------------------------------------------------------   
    def _model_description(self):
        return r"""
### Degradation / performance model

**Model form (performance scale)**

A Mitsuom-type performance model with Arrhenius temperature acceleration:

$$
D(t, T)
= \left[
  1 + a \left(
    \exp\!\left(
      \frac{E_a}{k_B}
      \left(
        \frac{1}{T_{\text{use}}}
        - \frac{1}{T}
      \right)
    \right)
    \, t
  \right)^{b}
\right]^{-1}.
$$

Performance $D(t,T)$ typically starts near 1 and decreases toward 0 as damage accumulates.

**Parameters:**

- **$a$** – scale factor on the accumulated damage term  
- **$b$** – time/curvature exponent (how sharply degradation accelerates)  
- **$E_{a}$** – Arrhenius activation energy (eV) controlling temperature acceleration  
- **$T_{\text{use}}$** – fixed use-level reference temperature (input, not fitted)  
"""

    # ------------------------------------------------------------------
    # Model mean: data level
    # ------------------------------------------------------------------
    def _mu(self, theta, t):
        """
        Mean performance model at the *data* temperatures (self.T_K):

            D_i = [ 1 + a * ( t_i * exp(Ea / k_B * (1/T_use - 1/T_i)) )^{b} ]^{-1}

        Used internally by Base_ADT_Model for LS/MLE on observed data.
        """
        a, b, Ea = theta
        t = np.asarray(t, float)

        if t.shape != self.t.shape:
            raise ValueError(
                "_mu is intended for data-fitting only (t must match self.t shape). "
                "Use model-specific helpers for grids/other temperatures."
            )

        t_pos = np.clip(t, 1e-12, None)
        # Arrhenius acceleration factor at each test T
        expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / self.T_K)
        expo = np.clip(expo, -50.0, 50.0)  # guard overflow
        accel = np.exp(expo)

        L = t_pos * accel  # "effective" accelerated time
        return 1.0 / (1.0 + a * (L ** b))

    # --------------------------------------------------------------
    # Negative log-likelihood
    # --------------------------------------------------------------
    def _negloglik(self, x):
        """
        x = [a, b, Ea, log_sigma]
    
        - noise == "additive":
            plain homoscedastic Normal errors on D
                D_i ~ N(mu_i, sigma^2)
    
        - noise == "multiplicative":
            lognormal errors on D (same form as the
            MLE multiplicative version you liked).
        """
        a, b, Ea = x[:-1]
        log_sigma = x[-1]
        sigma = np.exp(log_sigma)
    
        # basic constraints
        if sigma <= 0 or a <= 0 or b <= 0 or Ea <= 0:
            return np.inf
    
        theta = np.array([a, b, Ea], float)
        mu = self._mu(theta, self.t)
    
        if self.noise == "additive":
            # ----- plain Normal noise on D -----
            if not np.all(np.isfinite(mu)):
                return np.inf
    
            # equivalent to -loglik_normal(self.y, mu, sigma)
            return -np.sum(norm.logpdf(self.y, loc=mu, scale=sigma))
    
        elif self.noise == "multiplicative":
            # ----- lognormal noise on D (your “good” multiplicative model) -----
            D_pos  = np.clip(self.y, 1e-12, 1.0)
            mu_pos = np.clip(mu,   1e-12, 1.0)
    
            logD  = np.log(D_pos)
            logmu = np.log(mu_pos)
            z     = (logD - logmu) / sigma
    
            if not np.all(np.isfinite(z)):
                return np.inf
    
            # neg log-lik for lognormal model
            return (
                np.sum(np.log(D_pos)) +
                len(D_pos) * np.log(sigma) -
                np.sum(norm.logpdf(z))
            )
            
        else:
            raise ValueError(f"Unknown noise model: {self.noise}")

    # ------------------------------------------------------------------
    # Initial guess & bounds
    # ------------------------------------------------------------------
    def _init_guess(self):
        return np.array([9e-4, 0.63, 0.30], dtype=float)
    
    def _get_LS_bounds(self, p):
        return [
            (1e-8, 1e2),   # a
            (1e-2, 10.0),  # b
            (1e-3, 2.0),   # Ea
        ]

    # ------------------------------------------------------------------
    # LS diagnostics & plots
    # ------------------------------------------------------------------
    def _plot_fit_LS(self):
        if self.theta_LS is None:
            return

        a, b, Ea = self.theta_LS

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        cmap = plt.get_cmap("viridis")
        colors = {T: cmap(i / max(len(temps) - 1, 1)) for i, T in enumerate(temps)}

        # Scatter data
        for T in temps:
            mask = (self.T_C == T)
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=25,
                alpha=0.7,
                color=colors[T],
                label=f"{T:.0f} °C data",
            )

        # Mean curves
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        for T in temps:
            T_K = T + 273.15
            expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K)
            expo = np.clip(expo, -50.0, 50.0)
            accel = np.exp(expo)
            L = t_pos * accel
            mu_grid = 1.0 / (1.0 + a * (L ** b))

            ax.plot(
                t_grid,
                mu_grid,
                color=colors[T],
                lw=2,
                label=f"{T:.0f} °C LS mean",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Performance D")
        ax.set_title("LSQ Mitsuom–Arrhenius Degradation Fit")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=2)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # MLE plot
    # ------------------------------------------------------------------
    def _plot_fit_MLE(self):
        if self.theta_MLE is None:
            return

        a, b, Ea = self.theta_MLE
        sigma = self.sigma_MLE

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, T in enumerate(temps):
            idx = min(i, len(base_colors) - 1)
            styles[T] = dict(color=base_colors[idx], marker=base_markers[idx])

        # Scatter data
        for T in temps:
            mask = (self.T_C == T)
            st = styles[T]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"{T:.0f} °C data",
            )

        # Mean curves + noise bands
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        for T in temps:
            T_K = T + 273.15
            st = styles[T]

            expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K)
            expo = np.clip(expo, -50.0, 50.0)
            accel = np.exp(expo)
            L = t_pos * accel
            mu_grid = 1.0 / (1.0 + a * (L ** b))

            ax.plot(
                t_grid,
                mu_grid,
                color=st["color"],
                lw=2,
                label=f"{T:.0f} °C MLE mean",
            )

            if self.show_noise_bounds and sigma is not None:
                if self.noise == "additive":
                    lower = mu_grid - self.z_CI * sigma
                    upper = mu_grid + self.z_CI * sigma
                else:
                    # same heteroscedastic rule used in the likelihood:
                    mu_clip = np.clip(mu_grid, 1e-6, 1.0 - 1e-6)
                    sigma_grid = sigma * (1.0 - mu_clip)
                    sigma_grid = np.clip(sigma_grid, 1e-6, None)

                    lower = mu_grid - self.z_CI * sigma_grid
                    upper = mu_grid + self.z_CI * sigma_grid

                # keep bands in [0, 1] for performance
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)

                ax.fill_between(
                    t_grid,
                    lower,
                    upper,
                    color=st["color"],
                    alpha=0.2,
                    label=f"{T:.0f} °C {int(100*self.CI)}% band",
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Performance D")
        ax.set_title(f"MLE Mitsuom–Arrhenius Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Bayesian fit (via generic helper)
    # ------------------------------------------------------------------
    def _fit_Bayesian(self, nwalkers=32, nburn=1000, nsamp=2000):
        """
        Bayesian fit for Mitsuom–Arrhenius ADT:
    
            D(t, T) = [1 + a * (t * accel)^b]^{-1}
    
        Natural parameters: {a, b, Ea, sigma}
    
        - noise == "additive": plain Normal errors, same as MLE
        - noise == "multiplicative": lognormal errors, same form as MLE
        """
        # --- prior centres from MLE (preferred) or LS ---
        if self.theta_MLE is not None and self.sigma_MLE is not None:
            a_hat, b_hat, Ea_hat = self.theta_MLE
            sig_hat = self.sigma_MLE
        elif self.theta_LS is not None and self.sigma_LS is not None:
            a_hat, b_hat, Ea_hat = self.theta_LS
            sig_hat = self.sigma_LS
        else:
            raise RuntimeError("Run LS or MLE before Bayesian (no initial estimates).")
    
        default_centers = {
            "a":     float(a_hat),
            "b":     float(b_hat),
            "Ea":    float(Ea_hat),
            "sigma": float(sig_hat),
        }
    
        default_sds = {
            "a":     0.5,
            "b":     0.5,
            "Ea":    0.5,
            "sigma": 0.5,
        }
    
        t_time = self.t
        D_obs  = self.y
    
        def loglike(theta_dict):
            a     = theta_dict["a"]
            b     = theta_dict["b"]
            Ea    = theta_dict["Ea"]
            sigma = theta_dict["sigma"]
    
            if sigma <= 0 or a <= 0 or b <= 0 or Ea <= 0:
                return -np.inf
    
            theta = np.array([a, b, Ea], float)
            mu = self._mu(theta, t_time)
    
            if self.noise == "additive":
                # ----- same model as MLE additive: D ~ N(mu, sigma^2) -----
                if not np.all(np.isfinite(mu)):
                    return -np.inf
                return np.sum(norm.logpdf(D_obs, loc=mu, scale=sigma))
    
            elif self.noise == "multiplicative":
                # ----- same lognormal model as MLE multiplicative -----
                D_pos  = np.clip(D_obs, 1e-12, 1.0)
                mu_pos = np.clip(mu,    1e-12, 1.0)
                logD   = np.log(D_pos)
                logmu  = np.log(mu_pos)
                if not np.all(np.isfinite(logmu)):
                    return -np.inf
                z = (logD - logmu) / sigma
                return (
                    -np.sum(np.log(D_pos))
                    - len(D_pos) * np.log(sigma)
                    + np.sum(norm.logpdf(z))
                )
    
            else:
                return -np.inf
    
        param_names = ["a", "b", "Ea", "sigma"]
    
        sampler, chain, nat_samples = self._fit_Bayesian_generic(
            param_names=param_names,
            model_name="Mitsuom_Arrhenius_ADT",
            default_centers=default_centers,
            default_sds=default_sds,
            loglike_func=loglike,
            priors=getattr(self, "priors", None),
            dist_name="NormalError",
            nwalkers=nwalkers,
            nburn=nburn,
            nsamp=nsamp,
        )
    
        # Store posterior draws
        self.a_s     = nat_samples["a"]
        self.b_s     = nat_samples["b"]
        self.Ea_s    = nat_samples["Ea"]
        self.sigma_s = nat_samples["sigma"]
    
        if self.print_results:
            self._show_model_description() 
            md_param_table_bayes(
                {
                    "a": self.a_s,
                    "b": self.b_s,
                    "Ea (eV)": self.Ea_s,
                    r"$\sigma$": self.sigma_s,
                },
                cred_mass=self.CI,
                label=f"Bayesian ({self.noise})",
            )

    # ------------------------------------------------------------------
    # Bayesian plot
    # ------------------------------------------------------------------
    def _plot_fit_Bayesian(self, nsamples=1000, show_predictive=True):
        if not hasattr(self, "a_s"):
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, T in enumerate(temps):
            idx = min(i, len(base_colors) - 1)
            styles[T] = dict(color=base_colors[idx], marker=base_markers[idx])

        # Scatter data
        for T in temps:
            mask = (self.T_C == T)
            st = styles[T]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"{T:.0f} °C data",
            )

        rng = np.random.default_rng(7)
        n_post = len(self.a_s)
        ns = min(nsamples, n_post)
        sel = rng.choice(n_post, size=ns, replace=False)

        _a   = self.a_s[sel]
        _b   = self.b_s[sel]
        _Ea  = self.Ea_s[sel]
        _sig = self.sigma_s[sel]

        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)
        alpha = self.alpha
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        for T in temps:
            T_K = T + 273.15
            st = styles[T]

            expo = _Ea[:, None] / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K)
            expo = np.clip(expo, -50.0, 50.0)
            accel = np.exp(expo)
            L = t_pos[None, :] * accel
            mu_draws = 1.0 / (1.0 + _a[:, None] * (L ** _b[:, None]))

            mu_med = np.median(mu_draws, axis=0)
            lo_mean, hi_mean = np.quantile(mu_draws, [q_lo, q_hi], axis=0)

            ax.plot(
                t_grid,
                mu_med,
                color=st["color"],
                lw=2,
                label=f"{T:.0f} °C Bayes Median",
            )

            if show_predictive:
                eps = rng.standard_normal(size=mu_draws.shape)

                if self.noise == "additive":
                    sigma_draws = _sig[:, None] * eps
                    y_draws = mu_med + sigma_draws
                else:
                    # heteroscedastic additive: sigma_i = sigma * (1 - mu)
                    sigma_draws = _sig[:, None] * (1.0 - mu_draws)
                    sigma_draws = np.clip(sigma_draws, 1e-12, None)
                    y_draws = mu_med + sigma_draws * eps

                lo_pred, hi_pred = np.quantile(y_draws, [q_lo, q_hi], axis=0)
                ax.fill_between(
                    t_grid,
                    lo_pred,
                    hi_pred,
                    alpha=0.15,
                    color=st["color"],
                    label=None,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Performance D")
        ax.set_title(f"Bayesian Mitsuom–Arrhenius Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # TTF at use (Bayesian posterior)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution(
        self,
        Df=None,
        T_eval_C=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if not hasattr(self, "a_s"):
            print("Bayesian posterior not available; cannot compute TTF distribution.")
            return

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
        else:
            T_eval_K = float(T_eval_C) + 273.15

        theta_samples = np.column_stack([self.a_s, self.b_s, self.Ea_s])
        sigma_samples = np.asarray(self.sigma_s, float)

        def mu_fun_factory(theta_vec):
            a, b, Ea = theta_vec
            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 1e-12, None)
                expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K)
                expo = np.clip(expo, -50.0, 50.0)
                accel = np.exp(expo)
                L = t_pos * accel
                return 1.0 / (1.0 + a * (L ** b))
            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="performance",   # failure when performance <= Df
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at use (Df={Df})",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title="TTF posterior and best distribution fits (Mitsuom–Arrhenius)",
            bins=60,
        )

    # ------------------------------------------------------------------
    # TTF at use (MLE plug-in)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution_MLE(
        self,
        Df=None,
        T_eval_C=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if (self.theta_MLE is None) or (self.sigma_MLE is None):
            raise RuntimeError("MLE fit not available – run method='MLE' first.")

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
            label_T = f"T={self.T_use_C:.0f} °C"
        else:
            T_eval_K = float(T_eval_C) + 273.15
            label_T = f"T={float(T_eval_C):.0f} °C"

        a_hat, b_hat, Ea_hat = self.theta_MLE
        sigma_hat = float(self.sigma_MLE)

        theta_samples = np.tile(
            np.array([a_hat, b_hat, Ea_hat], float)[None, :],
            (n_samps, 1),
        )
        sigma_samples = np.full(n_samps, sigma_hat, float)

        def mu_fun_factory(theta_vec):
            a, b, Ea = theta_vec
            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 1e-12, None)
                expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K)
                expo = np.clip(expo, -50.0, 50.0)
                accel = np.exp(expo)
                L = t_pos * accel
                return 1.0 / (1.0 + a * (L ** b))
            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="performance",
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at {label_T} (Df={Df}, MLE plug-in, Mitsuom–Arrhenius)",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF (MLE plug-in) at {label_T} – Mitsuom–Arrhenius",
            bins=60,
        )

class Fit_ADT_Mitsuom_Arrhenius_Power(Base_ADT_Model):
    r"""
    Dual-stress Mitsuom + Arrhenius + Power *performance* model:

        D(t, T, S) = [ 1 + a * ( exp(Ea/k_B * (1/T_use - 1/T)) * S^n * t )^b ]^{-1}

    where:
        - D(t, T, S) is performance in (0, 1], decreasing over time
        - t is time
        - T is stress temperature (°C input, converted to K internally)
        - S is second stress (e.g., current in mA)
        - T_use, S_use are use-level stress conditions
        - a > 0, b > 0 control the degradation rate/curvature
        - Ea > 0 is activation energy in eV
        - n > 0 is power exponent for the second stress

    Supports:
        - LSQ fit (natural parameters a, b, Ea, n)
        - MLE:
            * noise="additive"  -> Gaussian errors on D (log-param MLE, like prototype)
            * noise="multiplicative" -> lognormal errors on D (via Base_ADT_Model._fit_MLE)
        - Bayesian fit (additive noise ONLY) via emcee

    NOTE:
        The Bayesian additive case is coded to mirror the prototype notebook
        (log-parameterisation, Gaussian errors on D) but wrapped in the
        ADT_fitters interface.
    """

    def __init__(
        self,
        degradation,
        temp_C,
        stress_S,
        time,
        unit,
        stress_use_T,
        stress_use_S,
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
    ):
        method_norm = method.lower()
        noise_norm = noise.lower()

        # --- Base constructor ------------------------------------------------
        super().__init__(
            y=degradation,
            time=time,
            unit=unit,
            CI=CI,
            method=method,
            noise=noise,
            scale_type="performance",
            Df=Df,
            show_LSQ_diagnostics=show_LSQ_diagnostics,
            show_noise_bounds=show_noise_bounds,
            show_use_TTF_dist=show_use_TTF_dist,
            print_results=print_results,
            stress=temp_C,
            stress_label="Temperature (°C)",
            legend_title="Condition",
            data_scale=data_scale,
            **kwargs,
        )

        # --- Model-specific fields ------------------------------------------
        self.T_C = np.asarray(temp_C, float).ravel()
        self.S_S = np.asarray(stress_S, float).ravel()  # second stress (e.g., current)
        if self.T_C.shape != self.y.shape or self.S_S.shape != self.y.shape:
            raise ValueError("temp_C, stress_S must match degradation shape.")

        self.T_K = self.T_C + 273.15
        self.T_use_C = float(stress_use_T)
        self.T_use_K = self.T_use_C + 273.15
        self.S_use = float(stress_use_S)

        self.priors = priors
        self.data_scale = data_scale
        self.param_names = ["a", "b", "Ea (eV)", "n"]

        # --- Optional raw data plot ------------------------------------------
        if show_data_plot:
            self.plot_data_dual()

        # --- Method pipeline -------------------------------------------------
        if method_norm == "ls":
            self._fit_LS(suppress_print=False)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics_dual()
            self._plot_fit_LS_dual()

        elif method_norm == "mle":
            # Always start from LS
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics_dual()

            if noise_norm == "additive":
                # Use prototype-style log-param MLE (Gaussian on D)
                self._fit_MLE_dual()
            else:
                # Use generic base MLE with self._negloglik + lognormal noise
                self._fit_MLE()

            self._plot_fit_MLE_dual()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution_MLE_dual()

        elif method_norm.startswith("bayes"):
            # Now supports additive *and* multiplicative noise
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics_dual()

            # Centre priors using an MLE fit that matches the noise type
            if noise_norm == "additive":
                # prototype-style log-param MLE on D
                self._fit_MLE_dual(suppress_print=True)
            else:
                # generic MLE with lognormal errors on D
                self._fit_MLE(suppress_print=True)

            self._fit_Bayesian_dual()
            self._plot_fit_Bayesian_dual()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution_dual()


        else:
            raise ValueError(f"Unknown method: {method}")
            
    # ------------------------------------------------------------------
    # Model Description
    # ------------------------------------------------------------------   
    def _model_description(self):
        return r"""
### Degradation / performance model

**Model form (performance scale)**

A Mitsuom-type *performance* model with Arrhenius temperature and
power-law stress acceleration:

$$
D(t, T, S)
= \left[
  1 + a \left(
    \exp\!\left(\frac{E_a}{k_B}
      \left(\frac{1}{T_{\text{use}}} - \frac{1}{T}\right)
    \right)
    \, S^{n} \, t
  \right)^{b}
\right]^{-1}.
$$

Here performance $D(t,T,S)$ typically starts near 1 and decreases
towards 0 as damage accumulates.

**Parameters:**

- **$a$** – scale factor on the accumulated damage term  
- **$b$** – time/curvature exponent (how sharply degradation accelerates)  
- **$E_{a}$** – Arrhenius activation energy (eV)  
- **$n$** – power exponent on the second stress \(S\) (e.g. current in mA)  
- **$T_{\text{use}}, S_{\text{use}}$** – fixed use-level stresses (inputs, not fitted)  
"""

    # ------------------------------------------------------------------
    # Mean model: data level
    # ------------------------------------------------------------------
    def _mu(self, theta, t):
        """
        Mean performance model at the *data* stresses:

            D_i = [ 1 + a * ( exp(Ea/k_B * (1/T_use - 1/T_i)) * S_i^n * t_i )^b ]^{-1}

        Used internally for LS/MLE/Bayes on observed data.
        """
        a, b, Ea, n = theta
        t = np.asarray(t, float)

        if t.shape != self.t.shape:
            raise ValueError(
                "_mu is intended for data-fitting only "
                "(t must match self.t shape)."
            )

        t_pos = np.clip(t, 1e-12, None)

        # Arrhenius acceleration factor at each test T
        expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / self.T_K)
        expo = np.clip(expo, -50.0, 50.0)
        accel = np.exp(expo)

        base = accel * (self.S_S ** n) * t_pos   # dual-stress "effective time"
        return 1.0 / (1.0 + a * (base ** b))

    # ------------------------------------------------------------------
    # LS initial guess & bounds
    # ------------------------------------------------------------------
    def _init_guess(self):
        # Same spirit as prototype p0
        return np.array([9e-4, 0.63, 0.30, 0.5], dtype=float)

    def _get_LS_bounds(self, p):
        # a, b, Ea, n
        return [
            (1e-20, 1e2),   # a
            (1e-2, 10.0),   # b
            (1e-3, 2.0),    # Ea [eV]
            (1e-3, 10.0),   # n
        ]

    # ------------------------------------------------------------------
    # NEGATIVE LOG-LIKELIHOOD (for generic MLE, esp. multiplicative)
    # ------------------------------------------------------------------
    def _negloglik(self, x):
        """
        x = [a, b, Ea, n, log_sigma]

        - noise == "additive":
            D_i ~ N(mu_i, sigma^2)  (homoscedastic)

        - noise == "multiplicative":
            log D ~ N(log mu, sigma^2)  (lognormal errors on D)
        """
        a, b, Ea, n = x[:-1]
        log_sigma = x[-1]
        sigma = np.exp(log_sigma)

        if sigma <= 0 or a <= 0 or b <= 0 or Ea <= 0 or n <= 0:
            return np.inf

        theta = np.array([a, b, Ea, n], float)
        mu = self._mu(theta, self.t)

        if self.noise == "additive":
            if not np.all(np.isfinite(mu)):
                return np.inf
            # plain Gaussian on D
            return -np.sum(norm.logpdf(self.y, loc=mu, scale=sigma))

        elif self.noise == "multiplicative":
            # lognormal noise on D
            D_pos  = np.clip(self.y, 1e-12, 1.0)
            mu_pos = np.clip(mu,   1e-12, 1.0)

            logD  = np.log(D_pos)
            logmu = np.log(mu_pos)
            z     = (logD - logmu) / sigma

            if not np.all(np.isfinite(z)):
                return np.inf

            # neg log-likelihood for lognormal model
            return (
                np.sum(np.log(D_pos)) +
                len(D_pos) * np.log(sigma) -
                np.sum(norm.logpdf(z))
            )

        else:
            raise ValueError(f"Unknown noise model: {self.noise}")

    # ------------------------------------------------------------------
    # Data plot (dual-stress: group by (T_C, S_S))
    # ------------------------------------------------------------------
    def plot_data_dual(self):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Assign styles per unique (T, S) condition
        conds = sorted(set(zip(self.T_C.astype(int), self.S_S.astype(int))))
        base_colors = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        styles = {}
        for i, cond in enumerate(conds):
            idx = min(i, len(base_colors) - 1)
            styles[cond] = dict(color=base_colors[idx],
                                marker=base_markers[idx])

        # Group by condition, then by unit
        df_plot = pd.DataFrame({
            "Unit": self.unit,
            "T_C": self.T_C,
            "S_S": self.S_S,
            "t": self.t,
            "D": self.y,
        })
        for (temp, S), gcond in df_plot.groupby(["T_C", "S_S"]):
            st_style = styles[(int(temp), int(S))]
            for unit, gunit in gcond.groupby("Unit"):
                gunit = gunit.sort_values("t")
                ax.plot(gunit["t"], gunit["D"], color=st_style["color"], linewidth=1.5)
                ax.scatter(gunit["t"], gunit["D"], s=25,
                           color=st_style["color"], marker=st_style["marker"], edgecolors="none")

        # Legend (one per condition)
        handles, labels = [], []
        for (temp, S), st_style in styles.items():
            h = ax.plot([], [], color=st_style["color"], marker=st_style["marker"],
                        linestyle="-", linewidth=1.5)[0]
            handles.append(h)
            labels.append(f"{temp:.0f} °C, {S:.0f} mA")
        ax.legend(handles, labels, title="Condition", frameon=False, ncol=1, loc="best")

        ax.set_xlabel("Time")
        ax.set_ylabel("Performance D")
        ax.set_title("Dual-stress Mitsuom–Arrhenius–Power: Data")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # LS diagnostics (dual-stress)
    # ------------------------------------------------------------------
    def _plot_LS_diagnostics_dual(self):
        if self.theta_LS is None:
            return

        a, b, Ea, n = self.theta_LS
        mu_hat = self._mu(self.theta_LS, self.t)
        resid = self.y - mu_hat

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Styles per condition
        conds = sorted(set(zip(self.T_C.astype(int), self.S_S.astype(int))))
        base_colors = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        def _style_for(temp_c, S):
            idx = conds.index((temp_c, S))
            idx = min(idx, len(base_colors) - 1)
            return base_colors[idx], base_markers[idx]

        # a) Residuals vs Fitted
        for (temp_c, S) in conds:
            m = (self.T_C.astype(int) == temp_c) & (self.S_S.astype(int) == S)
            color, marker = _style_for(temp_c, S)
            axes[0].scatter(
                mu_hat[m], resid[m], s=15, color=color, marker=marker, alpha=0.85,
                label=f"{temp_c} °C, {S} mA",
            )
        axes[0].axhline(0, color="k", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Fitted $\hat{D}$")
        axes[0].set_ylabel("Residual (D - $\hat{D}$)")
        axes[0].set_title("Residuals vs Fitted")
        axes[0].legend(frameon=False, ncol=2)

        # b) Residuals vs Time
        for (temp_c, S) in conds:
            m = (self.T_C.astype(int) == temp_c) & (self.S_S.astype(int) == S)
            color, marker = _style_for(temp_c, S)
            axes[1].scatter(
                self.t[m], resid[m], s=15, color=color, marker=marker, alpha=0.85,
            )
        axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residuals vs Time")

        # c) QQ plot (overall)
        probplot(resid, dist=norm, plot=axes[2])
        axes[2].set_title("QQ plot of residuals")

        plt.tight_layout()
        plt.show()
        
    # ------------------------------------------------------------------
    # LS fit plot (dual-stress)
    # ------------------------------------------------------------------
    def _plot_fit_LS_dual(self):
        """
        Plot LSQ fitted curves for each (T, S) condition, using theta_LS.
        """
        if self.theta_LS is None:
            return

        a, b, Ea, n = self.theta_LS

        fig, ax = plt.subplots(figsize=(8, 6))

        # Unique temperature–stress conditions
        conds = sorted(set(zip(self.T_C.astype(int), self.S_S.astype(int))))
        base_colors = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        styles = {}
        for i, cond in enumerate(conds):
            idx = min(i, len(base_colors) - 1)
            styles[cond] = dict(color=base_colors[idx],
                                marker=base_markers[idx])

        # Scatter data
        for (temp, S) in conds:
            m = (self.T_C.astype(int) == temp) & (self.S_S.astype(int) == S)
            st_style = styles[(temp, S)]
            ax.scatter(
                self.t[m],
                self.y[m],
                s=35,
                alpha=0.9,
                color=st_style["color"],
                marker=st_style["marker"],
                label=f"{temp} °C, {S} mA data",
            )

        # LS mean curves
        t_grid = np.linspace(0.0, float(self.t.max()) * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        for (temp, S) in conds:
            st_style = styles[(temp, S)]
            T_K = temp + 273.15

            expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K)
            expo = np.clip(expo, -50.0, 50.0)
            accel = np.exp(expo)
            base = accel * (float(S) ** n) * t_pos
            mu_grid = 1.0 / (1.0 + a * (base ** b))

            ax.plot(
                t_grid,
                mu_grid,
                color=st_style["color"],
                lw=2,
                label=f"{temp} °C, {S} mA LS mean",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Performance D")
        ax.set_title("LS Mitsuom–Arrhenius–Power Fit")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        # De-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(2, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # MLE (Gaussian errors on D; log-parameterisation like prototype)
    #   Used only when noise == "additive"
    # ------------------------------------------------------------------
    def _fit_MLE_dual(self, suppress_print=False):
        """
        MLE for additive Gaussian noise on D, matching the prototype:

            log-parameters: (log a, log b, log Ea, log n, log sigma)

        This is *only* used when noise == "additive".  For multiplicative
        noise we fall back to Base_ADT_Model._fit_MLE with self._negloglik.
        """
        if self.theta_LS is None:
            raise RuntimeError("Run LS before _fit_MLE_dual.")

        a_hat, b_hat, Ea_hat, n_hat = self.theta_LS

        # crude sigma from LS residuals
        mu_ls = self._mu(self.theta_LS, self.t)
        resid_ls = self.y - mu_ls
        sigma0 = float(np.std(resid_ls, ddof=3))
        sigma0 = max(sigma0, 1e-6)

        def nll_logparams(theta_log):
            log_a, log_b, log_Ea, log_n, log_sig = theta_log
            a = np.exp(log_a)
            b = np.exp(log_b)
            Ea = np.exp(log_Ea)
            n = np.exp(log_n)
            sig = np.exp(log_sig)

            if not np.isfinite(a * b * Ea * n * sig) or sig <= 0:
                return np.inf

            mu = self._mu([a, b, Ea, n], self.t)
            res = self.y - mu
            return 0.5 * np.sum((res / sig) ** 2 + 2.0 * np.log(sig) + np.log(2 * np.pi))

        init_log = np.log([
            max(a_hat, 1e-12),
            max(b_hat, 1e-3),
            max(Ea_hat, 1e-6),
            max(n_hat, 1e-6),
            sigma0,
        ])

        res = minimize(nll_logparams, init_log, method="L-BFGS-B")

        if not res.success:
            print("Warning: MLE (dual, additive) did not converge:", res.message)

        log_a, log_b, log_Ea, log_n, log_sig = res.x
        a_mle, b_mle, Ea_mle, n_mle, sig_mle = np.exp([log_a, log_b, log_Ea, log_n, log_sig])

        self.theta_MLE = np.array([a_mle, b_mle, Ea_mle, n_mle], float)
        self.sigma_MLE = float(sig_mle)

        # store residuals for diagnostics
        mu_mle = self._mu(self.theta_MLE, self.t)
        self.resid_MLE = self.y - mu_mle

        # Covariance from inverse Hessian if available
        hess_inv = getattr(res, "hess_inv", None)
        if hasattr(hess_inv, "todense"):
            self.cov_MLE = np.array(hess_inv.todense())
        elif isinstance(hess_inv, np.ndarray) and hess_inv.ndim == 2:
            self.cov_MLE = hess_inv
        else:
            self.cov_MLE = None
        
        if (not suppress_print) and self.print_results:
            # Parameters (exclude sigma from theta; pass it separately)
            self._show_model_description()            
            theta = np.array([a_mle, b_mle, Ea_mle, n_mle], float)
            names = ["a", "b", "Ea (eV)", "n"]

            md_param_table_freq(
                theta=theta,
                cov=self.cov_MLE,                  
                names=names,
                z_CI=self.z_CI,                    # already defined in Base_ADT_Model
                CI=self.CI,
                sigma=sig_mle,                     # noise scale
                noise_label="D",                   # errors on D
                label=f"MLE ({self.noise})",
            )

    # ------------------------------------------------------------------
    # MLE plot (dual-stress)
    # ------------------------------------------------------------------
    def _plot_fit_MLE_dual(self):
        if self.theta_MLE is None:
            return

        a, b, Ea, n = self.theta_MLE
        sigma = self.sigma_MLE

        fig, ax = plt.subplots(figsize=(8, 6))

        conds = sorted(set(zip(self.T_C.astype(int), self.S_S.astype(int))))
        base_colors = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        styles = {}
        for i, cond in enumerate(conds):
            idx = min(i, len(base_colors) - 1)
            styles[cond] = dict(color=base_colors[idx],
                                marker=base_markers[idx])

        # Scatter data
        for (temp, S) in conds:
            m = (self.T_C.astype(int) == temp) & (self.S_S.astype(int) == S)
            st_style = styles[(temp, S)]
            ax.scatter(
                self.t[m],
                self.y[m],
                s=35,
                alpha=0.9,
                color=st_style["color"],
                marker=st_style["marker"],
                label=f"{temp} °C, {S} mA data",
            )

        # Mean curves + bands
        t_grid = np.linspace(0.0, float(self.t.max()) * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        for (temp, S) in conds:
            st_style = styles[(temp, S)]
            T_K = temp + 273.15

            expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K)
            expo = np.clip(expo, -50.0, 50.0)
            accel = np.exp(expo)
            base = accel * (float(S) ** n) * t_pos
            mu_grid = 1.0 / (1.0 + a * (base ** b))

            ax.plot(
                t_grid,
                mu_grid,
                color=st_style["color"],
                lw=2,
                label=f"{temp} °C, {S} mA MLE mean",
            )

            if self.show_noise_bounds and sigma is not None:
                if self.noise == "additive":
                    lower = mu_grid - self.z_CI * sigma
                    upper = mu_grid + self.z_CI * sigma
                else:
                    # heteroscedastic: sigma_i = sigma * (1 - mu)
                    mu_clip = np.clip(mu_grid, 1e-6, 1.0 - 1e-6)
                    sigma_grid = sigma * (1.0 - mu_clip)
                    sigma_grid = np.clip(sigma_grid, 1e-6, None)
                    lower = mu_grid - self.z_CI * sigma_grid
                    upper = mu_grid + self.z_CI * sigma_grid

                lower = np.clip(lower, 0.0, 1.2)
                upper = np.clip(upper, 0.0, 1.2)

                ax.fill_between(
                    t_grid,
                    lower,
                    upper,
                    color=st_style["color"],
                    alpha=0.2,
                    label=f"{temp} °C, {S} mA {int(100*self.CI)}% band",
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Performance D")
        ax.set_title(f"MLE Mitsuom–Arrhenius–Power Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(2, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Bayesian fit (additive Gaussian errors, matches prototype)
    # ------------------------------------------------------------------
    def _fit_Bayesian_dual(self, nwalkers=32, nburn=1000, nsamp=2000):
        """
        Bayesian fit for Mitsuom–Arrhenius–Power (dual stress).

        Parameterisation (both noise models):
            theta = (log a, log b, log Ea, log n, log sigma)

        - noise == "additive":
            D ~ N(mu, sigma^2)

        - noise == "multiplicative":
            log D ~ N(log mu, sigma^2)  (lognormal errors on D;
            mu is the *median* curve)
        """
        if self.theta_MLE is None or self.sigma_MLE is None:
            raise RuntimeError("Run an MLE fit before _fit_Bayesian_dual.")

        # LSQ for centre of a,b,Ea,n; MLE for sigma
        a_hat, b_hat, Ea_hat, n_hat = self.theta_LS
        sig_mle = float(self.sigma_MLE)

        # Priors on logs (same as before)
        mu_log_a,  sd_log_a  = np.log(a_hat), 0.5
        mu_log_b,  sd_log_b  = np.log(b_hat), 0.5
        mu_log_Ea, sd_log_Ea = np.log(Ea_hat), 0.5
        mu_log_n,  sd_log_n  = np.log(n_hat), 0.5
        mu_log_s,  sd_log_s  = np.log(sig_mle), 0.5

        def log_prior(theta_log):
            log_a, log_b, log_Ea, log_n, log_sig = theta_log
            lp  = -0.5 * ((log_a  - mu_log_a)  / sd_log_a)  ** 2 - np.log(sd_log_a  * np.sqrt(2*np.pi))
            lp += -0.5 * ((log_b  - mu_log_b)  / sd_log_b)  ** 2 - np.log(sd_log_b  * np.sqrt(2*np.pi))
            lp += -0.5 * ((log_Ea - mu_log_Ea) / sd_log_Ea) ** 2 - np.log(sd_log_Ea * np.sqrt(2*np.pi))
            lp += -0.5 * ((log_n  - mu_log_n)  / sd_log_n)  ** 2 - np.log(sd_log_n  * np.sqrt(2*np.pi))
            lp += -0.5 * ((log_sig - mu_log_s) / sd_log_s)  ** 2 - np.log(sd_log_s  * np.sqrt(2*np.pi))
            return lp

        def log_likelihood(theta_log):
            log_a, log_b, log_Ea, log_n, log_sig = theta_log
            a   = np.exp(log_a)
            b   = np.exp(log_b)
            Ea  = np.exp(log_Ea)
            n   = np.exp(log_n)
            sig = np.exp(log_sig)

            if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(Ea)
                    and np.isfinite(n) and np.isfinite(sig)) or sig <= 0 or Ea <= 0:
                return -np.inf

            mu = self._mu([a, b, Ea, n], self.t)

            if self.noise == "additive":
                # D ~ N(mu, sigma^2)
                res = self.y - mu
                return -0.5 * np.sum((res / sig) ** 2 + 2.0 * np.log(sig) + np.log(2 * np.pi))

            elif self.noise == "multiplicative":
                # log D ~ N(log mu, sigma^2)  (lognormal)
                D_pos  = np.clip(self.y,  1e-12, 1.0)
                mu_pos = np.clip(mu,      1e-12, 1.0)
                logD   = np.log(D_pos)
                logmu  = np.log(mu_pos)

                # use normal logpdf on logD, subtract Jacobian term sum(logD)
                ll = np.sum(norm.logpdf(logD, loc=logmu, scale=sig)) - np.sum(np.log(D_pos))
                return ll

            else:
                return -np.inf

        def log_prob(theta_log):
            lp = log_prior(theta_log)
            if not np.isfinite(lp):
                return -np.inf
            ll = log_likelihood(theta_log)
            return lp + ll

        # Init walkers near the MLE solution we already computed
        a_mle, b_mle, Ea_mle, n_mle = self.theta_MLE
        sig_mle = self.sigma_MLE

        init_log_a  = np.log(max(a_mle, 1e-15))
        init_log_b  = np.log(max(b_mle, 1e-3))
        init_log_Ea = np.log(max(Ea_mle, 1e-3))
        init_log_n  = np.log(max(n_mle,  1e-3))
        init_log_s  = np.log(max(sig_mle, 1e-3))

        ndim = 5
        rng = np.random.default_rng(123)
        p0 = np.array([init_log_a, init_log_b, init_log_Ea, init_log_n, init_log_s]) + \
             rng.normal(scale=[0.2, 0.1, 0.2, 0.2, 0.2], size=(nwalkers, ndim))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

        # Burn-in + sampling
        sampler.run_mcmc(p0, nburn, progress=True)
        sampler.reset()
        sampler.run_mcmc(None, nsamp, progress=True)

        chain = sampler.get_chain(flat=True)
        log_a_s, log_b_s, log_Ea_s, log_n_s, log_sig_s = [chain[:, i] for i in range(ndim)]
        a_s   = np.exp(log_a_s)
        b_s   = np.exp(log_b_s)
        Ea_s  = np.exp(log_Ea_s)
        n_s   = np.exp(log_n_s)
        sig_s = np.exp(log_sig_s)

        self.a_s     = a_s
        self.b_s     = b_s
        self.Ea_s    = Ea_s
        self.n_s     = n_s
        self.sigma_s = sig_s

        if self.print_results:
            self._show_model_description() 
            md_param_table_bayes(
                {
                    "a": a_s,
                    "b": b_s,
                    "Ea (eV)": Ea_s,
                    "n": n_s,
                    r"$\sigma$": sig_s,
                },
                cred_mass=self.CI,
                label=f"Bayesian ({self.noise})",
            )

    # ------------------------------------------------------------------
    # Bayesian plot (dual-stress, additive, MEDIAN curve)
    # ------------------------------------------------------------------
    def _plot_fit_Bayesian_dual(self, nsamples=2000, show_predictive=True):
        if not hasattr(self, "a_s"):
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        # Conditions & styles
        conds = sorted(set(zip(self.T_C.astype(int), self.S_S.astype(int))))
        base_colors = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        styles = {}
        for i, cond in enumerate(conds):
            idx = min(i, len(base_colors) - 1)
            styles[cond] = dict(color=base_colors[idx],
                                marker=base_markers[idx])

        # Scatter data
        for (temp, S) in conds:
            m = (self.T_C.astype(int) == temp) & (self.S_S.astype(int) == S)
            st_style = styles[(temp, S)]
            ax.scatter(
                self.t[m],
                self.y[m],
                s=35,
                alpha=0.9,
                color=st_style["color"],
                marker=st_style["marker"],
                label=f"{temp} °C, {S} mA data",
            )

        # Subsample posterior
        rng = np.random.default_rng(7)
        n_post = len(self.a_s)
        ns = min(nsamples, n_post)
        sel = rng.choice(n_post, size=ns, replace=False)

        _a   = self.a_s[sel]
        _b   = self.b_s[sel]
        _Ea  = self.Ea_s[sel]
        _n   = self.n_s[sel]
        _sig = self.sigma_s[sel]

        t_grid = np.linspace(0.0, float(self.t.max()) * 1.05, 400)
        alpha = self.alpha
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        for (temp, S) in conds:
            T_eval = temp + 273.15
            st_style = styles[(temp, S)]

            # Acceleration + stress factor for each draw
            accel = np.exp(_Ea[:, None] / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval))
            s_fac = (float(S) ** _n)[:, None]

            base = t_grid[None, :] * accel * s_fac
            mu_draws = 1.0 / (1.0 + _a[:, None] * (base ** _b[:, None]))

            # Median curve (more robust than mean)
            mu_med = np.median(mu_draws, axis=0)
            ax.plot(
                t_grid,
                mu_med,
                color=st_style["color"],
                lw=2,
                label=f"{temp} °C, {S} mA Bayes median",
            )

            if show_predictive:
                if self.noise == "additive":
                    # D = mu + eps, eps ~ N(0, sigma)
                    eps = rng.standard_normal(size=mu_draws.shape)
                    y_draws = mu_draws + _sig[:, None] * eps
                elif self.noise == "multiplicative":
                    # log D = log(mu) + eps, eps ~ N(0, sigma)
                    mu_clip = np.clip(mu_draws, 1e-12, 1.0)
                    eps = rng.standard_normal(size=mu_draws.shape)
                    logY = np.log(mu_clip) + _sig[:, None] * eps
                    y_draws = np.exp(logY)
                else:
                    continue

                lo_pred, hi_pred = np.quantile(y_draws, [q_lo, q_hi], axis=0)
                ax.fill_between(
                    t_grid,
                    lo_pred,
                    hi_pred,
                    alpha=0.15,
                    color=st_style["color"],
                    label=None,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Performance D")
        ax.set_title(f"Bayesian Mitsuom–Arrhenius–Power Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(2, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # TTF at use (Bayesian posterior, dual stress)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution_dual(
        self,
        Df=None,
        T_eval_C=None,
        S_eval=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if not hasattr(self, "a_s"):
            print("Bayesian posterior not available; cannot compute TTF distribution.")
            return

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
            label_T = f"{self.T_use_C:.0f} °C"
        else:
            T_eval_K = float(T_eval_C) + 273.15
            label_T = f"{float(T_eval_C):.0f} °C"

        if S_eval is None:
            S_eval = self.S_use
        label_S = f"{S_eval:.0f} mA"

        theta_samples = np.column_stack([self.a_s, self.b_s, self.Ea_s, self.n_s])
        sigma_samples = np.asarray(self.sigma_s, float)

        def mu_fun_factory(theta_vec):
            a, b, Ea, n = theta_vec

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 1e-12, None)
                expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K)
                expo = np.clip(expo, -50.0, 50.0)
                accel = np.exp(expo)
                base = t_pos * accel * (S_eval ** n)
                return 1.0 / (1.0 + a * (base ** b))

            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="performance",   # failure when performance <= Df
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at use ({label_T}, {label_S}, Df={Df}) – Bayes",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF posterior and best fits (Mitsuom–Arrhenius–Power, {label_T}, {label_S})",
            bins=60,
        )

    # ------------------------------------------------------------------
    # TTF at use (MLE plug-in, dual stress)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution_MLE_dual(
        self,
        Df=None,
        T_eval_C=None,
        S_eval=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if (self.theta_MLE is None) or (self.sigma_MLE is None):
            raise RuntimeError("MLE fit not available – run method='MLE' first.")

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
            label_T = f"{self.T_use_C:.0f} °C"
        else:
            T_eval_K = float(T_eval_C) + 273.15
            label_T = f"{float(T_eval_C):.0f} °C"

        if S_eval is None:
            S_eval = self.S_use
        label_S = f"{S_eval:.0f} mA"

        a_hat, b_hat, Ea_hat, n_hat = self.theta_MLE
        sigma_hat = float(self.sigma_MLE)

        theta_samples = np.tile(
            np.array([a_hat, b_hat, Ea_hat, n_hat], float)[None, :],
            (n_samps, 1),
        )
        sigma_samples = np.full(n_samps, sigma_hat, float)

        def mu_fun_factory(theta_vec):
            a, b, Ea, n = theta_vec

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 1e-12, None)
                expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K)
                expo = np.clip(expo, -50.0, 50.0)
                accel = np.exp(expo)
                base = t_pos * accel * (S_eval ** n)
                return 1.0 / (1.0 + a * (base ** b))

            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="performance",
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at use ({label_T}, {label_S}, Df={Df}, MLE plug-in)",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF (MLE plug-in) at {label_T}, {label_S} – Mitsuom–Arrhenius–Power",
            bins=60,
        )


class Fit_ADT_Mitsuom_Arrhenius_Power1(Base_ADT_Model):
    r"""
    Dual-stress Mitsuom + Arrhenius + Power *performance* model:

        D(t, T, S) = [ 1 + c + a * ( exp(Ea/k_B * (1/T_use - 1/T)) * S^n * t )^b ]^{-1}

    where:
        - D(t, T, S) is performance in (0, 1], decreasing over time
        - t is time
        - T is stress temperature (°C input, converted to K internally)
        - S is second stress (e.g., current in mA)
        - T_use, S_use are use-level stress conditions
        - c > 0 controls the degradation vertical height
        - a > 0, b > 0 control the degradation rate/curvature
        - Ea > 0 is activation energy in eV
        - n > 0 is power exponent for the second stress

    Supports:
        - LSQ fit (natural parameters a, b, Ea, n)
        - MLE:
            * noise="additive"  -> Gaussian errors on D (log-param MLE, like prototype)
            * noise="multiplicative" -> lognormal errors on D (via Base_ADT_Model._fit_MLE)
        - Bayesian fit (additive noise ONLY) via emcee

    NOTE:
        The Bayesian additive case is coded to mirror the prototype notebook
        (log-parameterisation, Gaussian errors on D) but wrapped in the
        ADT_fitters interface.
    """

    def __init__(
        self,
        degradation,
        temp_C,
        stress_S,
        time,
        unit,
        stress_use_T,
        stress_use_S,
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
    ):
        method_norm = method.lower()
        noise_norm = noise.lower()

        # --- Base constructor ------------------------------------------------
        super().__init__(
            y=degradation,
            time=time,
            unit=unit,
            CI=CI,
            method=method,
            noise=noise,
            scale_type="performance",
            Df=Df,
            show_LSQ_diagnostics=show_LSQ_diagnostics,
            show_noise_bounds=show_noise_bounds,
            show_use_TTF_dist=show_use_TTF_dist,
            print_results=print_results,
            stress=temp_C,
            stress_label="Temperature (°C)",
            legend_title="Condition",
            data_scale=data_scale,
            **kwargs,
        )

        # --- Model-specific fields ------------------------------------------
        self.T_C = np.asarray(temp_C, float).ravel()
        self.S_S = np.asarray(stress_S, float).ravel()  # second stress (e.g., current)
        if self.T_C.shape != self.y.shape or self.S_S.shape != self.y.shape:
            raise ValueError("temp_C, stress_S must match degradation shape.")

        self.T_K = self.T_C + 273.15
        self.T_use_C = float(stress_use_T)
        self.T_use_K = self.T_use_C + 273.15
        self.S_use = float(stress_use_S)

        self.priors = priors
        self.data_scale = data_scale
        self.param_names = ["a", "b", "Ea (eV)", "n", "c"]

        # --- Optional raw data plot ------------------------------------------
        if show_data_plot:
            self.plot_data_dual()

        # --- Method pipeline -------------------------------------------------
        if method_norm == "ls":
            self._fit_LS(suppress_print=False)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics_dual()
            if self.print_results:
                self._print_LS_use_life()
            self._plot_fit_LS_dual()

        elif method_norm == "mle":
            # Always start from LS
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics_dual()

            if noise_norm == "additive":
                # Use prototype-style log-param MLE (Gaussian on D)
                self._fit_MLE_dual()
            else:
                # Use generic base MLE with self._negloglik + lognormal noise
                self._fit_MLE()

            self._plot_fit_MLE_dual()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution_MLE_dual()

        elif method_norm.startswith("bayes"):
            # Now supports additive *and* multiplicative noise
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics_dual()

            # Centre priors using an MLE fit that matches the noise type
            if noise_norm == "additive":
                # prototype-style log-param MLE on D
                self._fit_MLE_dual(suppress_print=True)
            else:
                # generic MLE with lognormal errors on D
                self._fit_MLE(suppress_print=True)

            self._fit_Bayesian_dual()
            self._plot_fit_Bayesian_dual()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution_dual()


        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Model Description
    # ------------------------------------------------------------------   
    def _model_description(self):
        return r"""
### Degradation / performance model (with offset)

**Model form (performance scale)**

This variant adds a vertical-offset parameter \(c\) to the Mitsuom model:

$$
D(t, T, S)
= \left[
  1 + c
  + a \left(
    \exp\!\left(\frac{E_a}{k_B}
      \left(\frac{1}{T_{\text{use}}} - \frac{1}{T}\right)
    \right)
    \, S^{n} \, t
  \right)^{b}
\right]^{-1}.
$$

The additional \(c\) term allows the starting performance level or
overall “height” of the degradation curve to shift slightly away from
exactly 1 while still maintaining \(1 + c > 0\).

**Parameters:**

- **$c$** – offset on the damage term (controls vertical position of the curve)  
- **$a$** – scale factor on the accumulated damage term  
- **$b$** – time/curvature exponent (how sharply degradation accelerates)  
- **$E_{a}$** – Arrhenius activation energy (eV)  
- **$n$** – power exponent on the second stress \(S\) (e.g. current in mA)  
- **$T_{\text{use}}, S_{\text{use}}$** – fixed use-level stresses (inputs, not fitted)  
"""

    # ------------------------------------------------------------------
    # Model Description
    # ------------------------------------------------------------------   
    def _mu(self, theta, t):
        """
        Mean performance at the data stresses:
    
            D_i = [1 + c + a * (AF_i * S_i^n * t_i)^b]^{-1}
        """
        a, b, Ea, n, c = theta
        t = np.asarray(t, float)
    
        if t.shape != self.t.shape:
            raise ValueError("_mu is intended for data-fitting only (t must match self.t).")
    
        t_pos = np.clip(t, 1e-12, None)
    
        # Arrhenius factor
        expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / self.T_K)
        expo = np.clip(expo, -50.0, 50.0)
        accel = np.exp(expo)
    
        base = accel * (self.S_S ** n) * t_pos
        denom = 1.0 + c + a * (base ** b)
        return 1.0 / denom

    # ------------------------------------------------------------------
    # LS initial guess & bounds
    # ------------------------------------------------------------------
    def _init_guess(self):
        # (a, b, Ea, n, c)
        return np.array([9e-4, 0.63, 0.30, 0.5, 0.0], dtype=float)
    
    def _get_LS_bounds(self, p):
        # a, b, Ea, n, c
        return [
            (1e-20, 1e2),    # a
            (1e-2, 10.0),    # b
            (1e-3, 2.0),     # Ea
            (1e-3, 10.0),    # n
            (-0.1,  0.3),   # c (keep 1+c>0)
        ]
    # ------------------------------------------------------------------
    # Deterministic "median" life at use (LS curve)
    # ------------------------------------------------------------------
    def _life_at_use_LS(self, Df=None):
        """
        Compute the deterministic life at the use condition
        (T_use_C, S_use) for the LS fit, defined as the time t such that

            D(t, T_use, S_use) = Df

        Returns
        -------
        t_life : float or None
            Life at use in the same time units as self.t, or None if the
            LS curve never reaches Df.
        """
        if self.theta_LS is None:
            return None

        if Df is None:
            Df = getattr(self, "Df_use", None)
        if Df is None:
            return None

        a, b, Ea, n, c = self.theta_LS

        # Use-condition Arrhenius term is 1 (T = T_use)
        # Model at use: D(t) = 1 / (1 + c + a * (S_use^n * t)^b)
        # Solve 1 / (1 + c + a*(S_use^n * t)^b) = Df
        rhs = 1.0 / Df - 1.0 - c   # = a * (S_use^n * t)^b

        # If rhs <= 0, curve never drops to Df
        if rhs <= 0 or a <= 0 or b <= 0 or n <= 0 or self.S_use <= 0:
            return None

        t_life = (rhs / a) ** (1.0 / b) / (self.S_use ** n)
        if not np.isfinite(t_life) or t_life <= 0:
            return None
        return float(t_life)

    def _print_LS_use_life(self, unit_label="time units"):
        """
        Print a one-line summary of LS-based life at use, immediately under
        the LS parameter table.
        """
        Df = getattr(self, "Df_use", None)
        t_life = self._life_at_use_LS(Df=Df)

        T_lab = f"{self.T_use_C:.0f} °C"
        S_lab = f"{self.S_use:g}"

        if t_life is None:
            print(
                f"\nEstimated life at use ({T_lab}, {S_lab} mA) "
                f"for Df={Df!r}: not defined (LS curve does not reach Df)."
            )
        else:
            print(
                f"\nEstimated life at use ({T_lab}, {S_lab} mA) "
                f"for Df={Df:.3g}: t_med ≈ {t_life:,.3g} {unit_label}"
            )

    # ------------------------------------------------------------------
    # NEGATIVE LOG-LIKELIHOOD (for generic MLE, esp. multiplicative)
    # ------------------------------------------------------------------
    def _negloglik(self, x):
        """
        x = [a, b, Ea, n, c, log_sigma]

        - noise == "additive":
            D_i ~ N(mu_i, sigma^2)  (homoscedastic)

        - noise == "multiplicative":
            log D ~ N(log mu, sigma^2)  (lognormal errors on D)
        """
        a, b, Ea, n, c = x[:-1]
        log_sigma = x[-1]
        sigma = np.exp(log_sigma)

        if sigma <= 0 or a <= 0 or b <= 0 or Ea <= 0 or n <= 0 or (1.0 + c) <= 1e-8:
            return np.inf

        theta = np.array([a, b, Ea, n, c], float)
        mu = self._mu(theta, self.t)

        if self.noise == "additive":
            if not np.all(np.isfinite(mu)):
                return np.inf
            # plain Gaussian on D
            return -np.sum(norm.logpdf(self.y, loc=mu, scale=sigma))

        elif self.noise == "multiplicative":
            # lognormal noise on D
            D_pos  = np.clip(self.y, 1e-12, 1.0)
            mu_pos = np.clip(mu,   1e-12, 1.0)

            logD  = np.log(D_pos)
            logmu = np.log(mu_pos)
            z     = (logD - logmu) / sigma

            if not np.all(np.isfinite(z)):
                return np.inf

            # neg log-likelihood for lognormal model
            return (
                np.sum(np.log(D_pos)) +
                len(D_pos) * np.log(sigma) -
                np.sum(norm.logpdf(z))
            )

        else:
            raise ValueError(f"Unknown noise model: {self.noise}")

    # ------------------------------------------------------------------
    # Data plot (dual-stress: group by (T_C, S_S))
    # ------------------------------------------------------------------
    def plot_data_dual(self):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Assign styles per unique (T, S) condition
        conds = sorted(set(zip(self.T_C.astype(int), self.S_S.astype(int))))
        base_colors = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        styles = {}
        for i, cond in enumerate(conds):
            idx = min(i, len(base_colors) - 1)
            styles[cond] = dict(color=base_colors[idx],
                                marker=base_markers[idx])

        # Group by condition, then by unit
        df_plot = pd.DataFrame({
            "Unit": self.unit,
            "T_C": self.T_C,
            "S_S": self.S_S,
            "t": self.t,
            "D": self.y,
        })
        for (temp, S), gcond in df_plot.groupby(["T_C", "S_S"]):
            st_style = styles[(int(temp), int(S))]
            for unit, gunit in gcond.groupby("Unit"):
                gunit = gunit.sort_values("t")
                ax.plot(gunit["t"], gunit["D"], color=st_style["color"], linewidth=1.5)
                ax.scatter(gunit["t"], gunit["D"], s=25,
                           color=st_style["color"], marker=st_style["marker"], edgecolors="none")

        # Legend (one per condition)
        handles, labels = [], []
        for (temp, S), st_style in styles.items():
            h = ax.plot([], [], color=st_style["color"], marker=st_style["marker"],
                        linestyle="-", linewidth=1.5)[0]
            handles.append(h)
            labels.append(f"{temp:.0f} °C, {S:.0f} mA")
        ax.legend(handles, labels, title="Condition", frameon=False, ncol=1, loc="best")

        ax.set_xlabel("Time")
        ax.set_ylabel("Performance D")
        ax.set_title("Dual-stress Mitsuom–Arrhenius–Power: Data")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # LS diagnostics (dual-stress)
    # ------------------------------------------------------------------
    def _plot_LS_diagnostics_dual(self):
        if self.theta_LS is None:
            return

        a, b, Ea, n, c = self.theta_LS
        mu_hat = self._mu(self.theta_LS, self.t)
        resid = self.y - mu_hat

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Styles per condition
        conds = sorted(set(zip(self.T_C.astype(int), self.S_S.astype(int))))
        base_colors = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        def _style_for(temp_c, S):
            idx = conds.index((temp_c, S))
            idx = min(idx, len(base_colors) - 1)
            return base_colors[idx], base_markers[idx]

        # a) Residuals vs Fitted
        for (temp_c, S) in conds:
            m = (self.T_C.astype(int) == temp_c) & (self.S_S.astype(int) == S)
            color, marker = _style_for(temp_c, S)
            axes[0].scatter(
                mu_hat[m], resid[m], s=15, color=color, marker=marker, alpha=0.85,
                label=f"{temp_c} °C, {S} mA",
            )
        axes[0].axhline(0, color="k", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Fitted $\hat{D}$")
        axes[0].set_ylabel("Residual (D - $\hat{D}$)")
        axes[0].set_title("Residuals vs Fitted")
        axes[0].legend(frameon=False, ncol=2)

        # b) Residuals vs Time
        for (temp_c, S) in conds:
            m = (self.T_C.astype(int) == temp_c) & (self.S_S.astype(int) == S)
            color, marker = _style_for(temp_c, S)
            axes[1].scatter(
                self.t[m], resid[m], s=15, color=color, marker=marker, alpha=0.85,
            )
        axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residuals vs Time")

        # c) QQ plot (overall)
        probplot(resid, dist=norm, plot=axes[2])
        axes[2].set_title("QQ plot of residuals")

        plt.tight_layout()
        plt.show()
        
    # ------------------------------------------------------------------
    # LS fit plot (dual-stress, offset model)
    # ------------------------------------------------------------------
    def _plot_fit_LS_dual(self):
        """
        Plot LSQ fitted curves for each (T, S) condition, using theta_LS,
        including the offset parameter c, PLUS the use-condition curve.
        """
        if self.theta_LS is None:
            return

        a, b, Ea, n, c = self.theta_LS

        fig, ax = plt.subplots(figsize=(8, 6))

        # Unique temperature–stress conditions
        conds = sorted(set(zip(self.T_C.astype(int), self.S_S.astype(int))))
        base_colors = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        styles = {}
        for i, cond in enumerate(conds):
            idx = min(i, len(base_colors) - 1)
            styles[cond] = dict(color=base_colors[idx],
                                marker=base_markers[idx])

        # Scatter data
        for (temp, S) in conds:
            m = (self.T_C.astype(int) == temp) & (self.S_S.astype(int) == S)
            st_style = styles[(temp, S)]
            ax.scatter(
                self.t[m],
                self.y[m],
                s=35,
                alpha=0.9,
                color=st_style["color"],
                marker=st_style["marker"],
                label=f"{temp} °C, {S} mA data",
            )

        # LS mean curves (test conditions)
        t_grid = np.linspace(0.0, float(self.t.max()) * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        for (temp, S) in conds:
            st_style = styles[(temp, S)]
            T_K = temp + 273.15

            expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K)
            expo = np.clip(expo, -50.0, 50.0)
            accel = np.exp(expo)
            base = accel * (float(S) ** n) * t_pos
            denom = 1.0 + c + a * (base ** b)
            mu_grid = 1.0 / denom

            ax.plot(
                t_grid,
                mu_grid,
                color=st_style["color"],
                lw=2,
                label=f"{temp} °C, {S} mA LS mean",
            )

        # NEW: LS mean curve at USE stress (T_use_C, S_use)
        # At use: AF = exp(Ea/k_B * (1/T_use - 1/T_use)) = 1
        base_use = (self.S_use ** n) * t_pos
        denom_use = 1.0 + c + a * (base_use ** b)
        mu_use = 1.0 / denom_use

        ax.plot(
            t_grid,
            mu_use,
            "k--",
            lw=2.5,
            label=f"Use LS mean ({self.T_use_C:.0f} °C, {self.S_use:g} mA)",
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Performance D")
        ax.set_title("LS Mitsuom–Arrhenius–Power (offset) Fit")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        # De-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(2, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # MLE plot (dual-stress)
    # ------------------------------------------------------------------
    def _fit_MLE_dual(self, suppress_print=False):
        if self.theta_LS is None:
            raise RuntimeError("Run LS before MLE_dual.")
    
        a_hat, b_hat, Ea_hat, n_hat, c_hat = self.theta_LS
    
        mu_ls = self._mu(self.theta_LS, self.t)
        resid_ls = self.y - mu_ls
        sigma0 = max(float(np.std(resid_ls, ddof=4)), 1e-6)
    
        def nll_logparams(theta_log):
            log_a, log_b, log_Ea, log_n, log1pc, log_sig = theta_log
            a   = np.exp(log_a)
            b   = np.exp(log_b)
            Ea  = np.exp(log_Ea)
            n   = np.exp(log_n)
            sig = np.exp(log_sig)
            c   = np.exp(log1pc) - 1.0   # ensure 1+c>0
    
            if (not np.isfinite(a*b*Ea*n*sig) or sig <= 0 or Ea <= 0
                    or (1.0 + c) <= 1e-8):
                return np.inf
    
            mu = self._mu([a, b, Ea, n, c], self.t)
            res = self.y - mu
            return 0.5 * np.sum((res / sig) ** 2 + 2.0 * np.log(sig) + np.log(2*np.pi))
    
        init_log = np.log([
            max(a_hat, 1e-12),
            max(b_hat, 1e-3),
            max(Ea_hat, 1e-6),
            max(n_hat, 1e-6),
            1.0 + max(c_hat, -0.9),  # for log1pc
            sigma0,
        ])
    
        res = minimize(nll_logparams, init_log, method="L-BFGS-B")
    
        if not res.success:
            print("Warning: MLE (dual) did not converge:", res.message)
    
        log_a, log_b, log_Ea, log_n, log1pc, log_sig = res.x
        a_mle   = np.exp(log_a)
        b_mle   = np.exp(log_b)
        Ea_mle  = np.exp(log_Ea)
        n_mle   = np.exp(log_n)
        c_mle   = np.exp(log1pc) - 1.0
        sig_mle = np.exp(log_sig)
    
        self.theta_MLE = np.array([a_mle, b_mle, Ea_mle, n_mle, c_mle], float)
        self.sigma_MLE = float(sig_mle)
    
        mu_mle = self._mu(self.theta_MLE, self.t)
        self.resid_MLE = self.y - mu_mle

        # Covariance from inverse Hessian if available
        hess_inv = getattr(res, "hess_inv", None)
        if hasattr(hess_inv, "todense"):
            self.cov_MLE = np.array(hess_inv.todense())
        elif isinstance(hess_inv, np.ndarray) and hess_inv.ndim == 2:
            self.cov_MLE = hess_inv
        else:
            self.cov_MLE = None
        
        if (not suppress_print) and self.print_results:
            # Parameters (exclude sigma from theta; pass it separately)
            self._show_model_description()    
            theta = np.array([a_mle, b_mle, Ea_mle, n_mle, c_mle], float)
            names = ["a", "b", "Ea (eV)", "n", "c"]
        
            md_param_table_freq(
                theta=theta,
                cov=self.cov_MLE,         
                names=names,
                z_CI=self.z_CI,                    # already defined in Base_ADT_Model
                CI=self.CI,
                sigma=sig_mle,                     # noise scale
                noise_label="D",                   # errors on D
                label=f"MLE ({self.noise})",
            )
            
    # ------------------------------------------------------------------
    # MLE plot (dual-stress)
    # ------------------------------------------------------------------            
    def _plot_fit_MLE_dual(self):
        if self.theta_MLE is None:
            return

        a, b, Ea, n, c = self.theta_MLE
        sigma = self.sigma_MLE

        fig, ax = plt.subplots(figsize=(8, 6))

        conds = sorted(set(zip(self.T_C.astype(int), self.S_S.astype(int))))
        base_colors = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        styles = {}
        for i, cond in enumerate(conds):
            idx = min(i, len(base_colors) - 1)
            styles[cond] = dict(color=base_colors[idx],
                                marker=base_markers[idx])

        # Scatter data
        for (temp, S) in conds:
            m = (self.T_C.astype(int) == temp) & (self.S_S.astype(int) == S)
            st_style = styles[(temp, S)]
            ax.scatter(
                self.t[m],
                self.y[m],
                s=35,
                alpha=0.9,
                color=st_style["color"],
                marker=st_style["marker"],
                label=f"{temp} °C, {S} mA data",
            )

        # Mean curves + bands
        t_grid = np.linspace(0.0, float(self.t.max()) * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        for (temp, S) in conds:
            st_style = styles[(temp, S)]
            T_K = temp + 273.15

            expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K)
            expo = np.clip(expo, -50.0, 50.0)
            accel = np.exp(expo)
            base = accel * (float(S) ** n) * t_pos
            mu_grid = 1.0 / (1.0 + c+ a * (base ** b))

            ax.plot(
                t_grid,
                mu_grid,
                color=st_style["color"],
                lw=2,
                label=f"{temp} °C, {S} mA MLE mean",
            )

            if self.show_noise_bounds and sigma is not None:
                if self.noise == "additive":
                    lower = mu_grid - self.z_CI * sigma
                    upper = mu_grid + self.z_CI * sigma
                else:
                    # heteroscedastic: sigma_i = sigma * (1 - mu)
                    mu_clip = np.clip(mu_grid, 1e-6, 1.0 - 1e-6)
                    sigma_grid = sigma * (1.0 - mu_clip)
                    sigma_grid = np.clip(sigma_grid, 1e-6, None)
                    lower = mu_grid - self.z_CI * sigma_grid
                    upper = mu_grid + self.z_CI * sigma_grid

            
                # for plotting
                lower = np.clip(lower, 0.0, 1.2)
                upper = np.clip(upper, 0.0, 1.2)
            
                ax.fill_between(
                    t_grid,
                    lower,
                    upper,
                    color=st_style["color"],
                    alpha=0.2,
                    label=f"{temp} °C, {S} mA {int(100*self.CI)}% band",
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Performance D")
        ax.set_title(f"MLE Mitsuom–Arrhenius–Power Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(2, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Bayesian fit (additive Gaussian errors, matches prototype)
    # ------------------------------------------------------------------
    def _fit_Bayesian_dual(
        self,
        nwalkers=32,
        nburn=1000,
        nsamp=2000,
    ):
        """
        Bayesian fit for Mitsuom–Arrhenius–Power1 model using the generic
        ADT Bayesian engine, but preserving the *original* specialised 
        prior for c:  eta_c = log(1+c).
        """

        if (self.theta_MLE is None) or (self.sigma_MLE is None):
            raise RuntimeError("Run MLE_dual before Bayesian_dual.")

        # --- Extract MLE for default prior centres (NATURAL scale)
        a_hat, b_hat, Ea_hat, n_hat, c_hat = self.theta_MLE
        sig_hat = float(self.sigma_MLE)

        default_centers = {
            "a":     float(a_hat),
            "b":     float(b_hat),
            "Ea":    float(Ea_hat),
            "n":     float(n_hat),
            "c":     float(c_hat),
            "sigma": float(sig_hat),
        }

        # Default SDs in *transform* space
        default_sds = {
            "a":     0.5,
            "b":     0.5,
            "Ea":    0.5,
            "n":     0.5,
            "sigma": 0.5,
        }

        # Special case: c uses eta_c = log1p(c)
        mu_log1pc_def = np.log(1.0 + max(c_hat, -0.9))
        default_sds["c"] = 0.5

        param_names = ["a", "b", "Ea", "n", "c", "sigma"]
        model_name = "Mitsuom_Arrhenius_Power1"

        # ---------- Custom transform handling for c ----------
        def transform_to_eta(name, x):
            """Natural → transform."""
            if name == "c":
                return np.log(1.0 + x)         # ensures c > -1
            return np.log(x) if name != "sigma" and name != "c" else np.log(x)

        def inverse_transform(name, eta):
            """Transform → natural parameter."""
            if name == "c":
                return np.exp(eta) - 1.0       # invert log1p
            return np.exp(eta)

        # ---------- Log-likelihood using natural parameters ----------
        def loglike_func(theta):
            a     = theta["a"]
            b     = theta["b"]
            Ea    = theta["Ea"]
            n     = theta["n"]
            c     = theta["c"]
            sigma = theta["sigma"]

            if (
                sigma <= 0 or Ea <= 0 or
                (1 + c) <= 1e-8 or
                not np.all(np.isfinite([a, b, Ea, n, c, sigma]))
            ):
                return -np.inf

            mu = self._mu([a, b, Ea, n, c], self.t)

            if self.noise == "additive":
                return np.sum(norm.logpdf(self.y, loc=mu, scale=sigma))

            Dp  = np.clip(self.y,  1e-12, 1.0)
            mup = np.clip(mu,      1e-12, 1.0)
            return np.sum(norm.logpdf(np.log(Dp), loc=np.log(mup), scale=sigma)) - np.sum(np.log(Dp))

        # ---------- Use generic engine but override transform for c ----------
        def _custom_init_eta():
            """Build eta0 from default centers with special rule for c."""
            eta0 = []
            for name in param_names:
                if name == "c":
                    eta0.append(mu_log1pc_def)
                else:
                    x = default_centers[name]
                    eta0.append(np.log(max(x, 1e-12)))
            return np.array(eta0, float)

        # Hook into generic runner:
        eta0 = _custom_init_eta()

        # need sd per param on transform space
        sd_eta = np.array([default_sds.get(nm, 0.5) for nm in param_names], float)

        # ---- PRIOR ON eta ----
        def log_prior_eta(eta):
            if self.priors is None:
                z = (eta - eta0) / sd_eta
                return -0.5 * np.sum(z*z) - np.sum(np.log(sd_eta*np.sqrt(2*np.pi)))

            # custom priors: convert to NATURAL scale
            theta_nat = {nm: inverse_transform(nm, eta[j]) for j, nm in enumerate(param_names)}
            return log_prior_vector(
                model=model_name,
                dist="NormalError",
                theta_dict=theta_nat,
                priors=self.priors,
            )

        # ---- POSTERIOR ----
        def log_prob_eta(eta):
            lp = log_prior_eta(eta)
            if not np.isfinite(lp):
                return -np.inf

            theta_nat = {nm: inverse_transform(nm, eta[j]) for j, nm in enumerate(param_names)}
            ll = loglike_func(theta_nat)
            if not np.isfinite(ll):
                return -np.inf
            return lp + ll

        # Run emcee
        sampler, chain = self._run_emcee(
            log_prob_eta,
            init=eta0,
            nwalkers=nwalkers,
            nburn=nburn,
            nsamp=nsamp,
        )

        # Convert η → natural parameters
        chain = np.asarray(chain)
        nat_samples = {
            nm: np.asarray([inverse_transform(nm, eta_j) for eta_j in chain[:, j]])
            for j, nm in enumerate(param_names)
        }

        # Store posterior
        self.a_s     = nat_samples["a"]
        self.b_s     = nat_samples["b"]
        self.Ea_s    = nat_samples["Ea"]
        self.n_s     = nat_samples["n"]
        self.c_s     = nat_samples["c"]
        self.sigma_s = nat_samples["sigma"]

        if self.print_results:
            self._show_model_description()
            md_param_table_bayes(
                {
                    "a": self.a_s,
                    "b": self.b_s,
                    "Ea (eV)": self.Ea_s,
                    "n": self.n_s,
                    "c": self.c_s,
                    r"$\sigma$": self.sigma_s,
                },
                cred_mass=self.CI,
                label=f"Bayesian ({self.noise})",
            )

    # ------------------------------------------------------------------
    # Bayesian plot (dual-stress, additive, MEDIAN curve)
    # ------------------------------------------------------------------
    def _plot_fit_Bayesian_dual(self, nsamples=2000, show_predictive=True):
        if not hasattr(self, "a_s"):
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        # Conditions & styles
        conds = sorted(set(zip(self.T_C.astype(int), self.S_S.astype(int))))
        base_colors = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        styles = {}
        for i, cond in enumerate(conds):
            idx = min(i, len(base_colors) - 1)
            styles[cond] = dict(color=base_colors[idx],
                                marker=base_markers[idx])

        # Scatter data
        for (temp, S) in conds:
            m = (self.T_C.astype(int) == temp) & (self.S_S.astype(int) == S)
            st_style = styles[(temp, S)]
            ax.scatter(
                self.t[m],
                self.y[m],
                s=35,
                alpha=0.9,
                color=st_style["color"],
                marker=st_style["marker"],
                label=f"{temp} °C, {S} mA data",
            )

        # Subsample posterior
        rng = np.random.default_rng(7)
        n_post = len(self.a_s)
        ns = min(nsamples, n_post)
        sel = rng.choice(n_post, size=ns, replace=False)
        _a   = self.a_s[sel]
        _b   = self.b_s[sel]
        _Ea  = self.Ea_s[sel]
        _n   = self.n_s[sel]
        _c   = self.c_s[sel]
        _sig = self.sigma_s[sel]

        t_grid = np.linspace(0.0, float(self.t.max()) * 1.05, 400)
        alpha = self.alpha
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        for (temp, S) in conds:
            T_eval = temp + 273.15
            st_style = styles[(temp, S)]

            # Acceleration + stress factor for each draw
            accel = np.exp(_Ea[:, None] / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval))
            s_fac = (float(S) ** _n)[:, None]

            base = t_grid[None, :] * accel * s_fac
            denom = 1.0 + _c[:, None] + _a[:, None] * (base ** _b[:, None])
            mu_draws = 1.0 / denom

            # Median curve (more robust than mean)
            mu_med = np.median(mu_draws, axis=0)
            ax.plot(
                t_grid,
                mu_med,
                color=st_style["color"],
                lw=2,
                label=f"{temp} °C, {S} mA Bayes median",
            )

            if show_predictive:
                if self.noise == "additive":
                    # D = mu + eps, eps ~ N(0, sigma)
                    eps = rng.standard_normal(size=mu_draws.shape)
                    y_draws = mu_draws + _sig[:, None] * eps
                    y_draws = np.clip(y_draws, 0.0, 1.1)
                elif self.noise == "multiplicative":
                    # log D = log(mu) + eps, eps ~ N(0, sigma)
                    mu_clip = np.clip(mu_draws, 1e-12, 1.0)
                    eps = rng.standard_normal(size=mu_draws.shape)
                    logY = np.log(mu_clip) + _sig[:, None] * eps
                    y_draws = np.exp(logY)
                    y_draws = np.clip(y_draws, 0.0, 1.1)
                else:
                    continue

                lo_pred, hi_pred = np.quantile(y_draws, [q_lo, q_hi], axis=0)
                ax.fill_between(
                    t_grid,
                    lo_pred,
                    hi_pred,
                    alpha=0.15,
                    color=st_style["color"],
                    label=None,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Performance D")
        ax.set_title(f"Bayesian Mitsuom–Arrhenius–Power Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(2, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # TTF at use (Bayesian posterior, dual stress)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution_dual(
        self,
        Df=None,
        T_eval_C=None,
        S_eval=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if not hasattr(self, "a_s"):
            print("Bayesian posterior not available; cannot compute TTF distribution.")
            return

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
            label_T = f"{self.T_use_C:.0f} °C"
        else:
            T_eval_K = float(T_eval_C) + 273.15
            label_T = f"{float(T_eval_C):.0f} °C"

        if S_eval is None:
            S_eval = self.S_use
        label_S = f"{S_eval:.0f} mA"

        theta_samples = np.column_stack([self.a_s, self.b_s, self.Ea_s, self.n_s, self.c_s])
        sigma_samples = np.asarray(self.sigma_s, float)

        def mu_fun_factory(theta_vec):
            a, b, Ea, n, c = theta_vec

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 1e-12, None)
                expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K)
                expo = np.clip(expo, -50.0, 50.0)
                accel = np.exp(expo)
                base = t_pos * accel * (S_eval ** n)
                return 1.0 / (1.0 + c + a * (base ** b))

            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="performance",   # failure when performance <= Df
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at use ({label_T}, {label_S}, Df={Df}) – Bayes",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF posterior and best fits (Mitsuom–Arrhenius–Power, {label_T}, {label_S})",
            bins=60,
        )

    # ------------------------------------------------------------------
    # TTF at use (MLE plug-in, dual stress)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution_MLE_dual(
        self,
        Df=None,
        T_eval_C=None,
        S_eval=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if (self.theta_MLE is None) or (self.sigma_MLE is None):
            raise RuntimeError("MLE fit not available – run method='MLE' first.")

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
            label_T = f"{self.T_use_C:.0f} °C"
        else:
            T_eval_K = float(T_eval_C) + 273.15
            label_T = f"{float(T_eval_C):.0f} °C"

        if S_eval is None:
            S_eval = self.S_use
        label_S = f"{S_eval:.0f} mA"

        a_hat, b_hat, Ea_hat, n_hat, c_hat = self.theta_MLE
        sigma_hat = float(self.sigma_MLE)

        theta_samples = np.tile(
            np.array([a_hat, b_hat, Ea_hat, n_hat, c_hat], float)[None, :],
            (n_samps, 1),
        )
        sigma_samples = np.full(n_samps, sigma_hat, float)

        def mu_fun_factory(theta_vec):
            a, b, Ea, n, c = theta_vec

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 1e-12, None)
                expo = Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K)
                expo = np.clip(expo, -50.0, 50.0)
                accel = np.exp(expo)
                base = t_pos * accel * (S_eval ** n)
                return 1.0 / (1.0 + c + a * (base ** b))

            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="performance",
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at use ({label_T}, {label_S}, Df={Df}, MLE plug-in)",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF (MLE plug-in) at {label_T}, {label_S} – Mitsuom–Arrhenius–Power",
            bins=60,
        )

class Fit_ADT_Exponential_Arrhenius(Base_ADT_Model):
    r"""
    Exponential-time + Arrhenius *damage* model:

        D(t, T) = b * exp(a * t) * exp(Ea / k * (1/T_use - 1/T))

    where:
        - D(t, T) is damage (increasing over time)
        - t is time
        - T is stress temperature (°C input, converted to K)
        - T_use is use temperature
        - b > 0 is a scale parameter
        - a > 0 is the time exponential rate
        - Ea > 0 is activation energy in eV

    Supports:
        - LSQ fit
        - MLE (additive or multiplicative noise)
        - Bayesian fit via emcee (with log-parameter priors)
        - Optional TTF-at-use histogram (Bayesian or MLE plug-in)
    """

    def __init__(
        self,
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
    ):
        method_norm = method.lower()

        # Base constructor
        super().__init__(
            y=degradation,
            time=time,
            unit=unit,
            CI=CI,
            method=method,
            noise=noise,
            scale_type="damage",
            Df=Df,
            show_LSQ_diagnostics=show_LSQ_diagnostics,
            show_noise_bounds=show_noise_bounds,
            show_use_TTF_dist=show_use_TTF_dist,
            print_results=print_results,
            stress=stress,
            stress_label="Temperature (°C)",
            legend_title="Temperature",
            data_scale=data_scale,
            **kwargs,
        )

        # Temperatures
        self.T_C = np.asarray(stress, float).ravel()
        if self.T_C.shape != self.y.shape:
            raise ValueError("stress (T_C) must have same shape as degradation (y).")
        self.T_K = self.T_C + 273.15
        self.T_use_C = float(stress_use)
        self.T_use_K = self.T_use_C + 273.15
        self.priors = priors  # may be None or a dict
        self.data_scale = data_scale
        self.param_names = ["b", "a", "Ea (eV)"]

        if show_data_plot:
            self.plot_data()

        # Pipeline (same pattern as sqrt-Arrhenius)
        if method_norm == "ls":
            self._fit_LS(suppress_print=False)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._plot_fit_LS()

        elif method_norm == "mle":
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE()
            self._plot_fit_MLE()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution_MLE()

        elif method_norm.startswith("bayes"):
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE(suppress_print=True)   # to set priors
            self._fit_Bayesian()
            self._plot_fit_Bayesian()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution()

        elif method_norm in ("mle_hierarchical", "mle_heirachical"):
            raise NotImplementedError("Hierarchical MLE not yet implemented for Exponential-Arrhenius.")

        else:
            raise ValueError(f"Unknown method: {method}")
            
    # ------------------------------------------------------------------
    # Model Description
    # ------------------------------------------------------------------   
    def _model_description(self):
        return r"""
### Degradation / Damage Model

**Model form**

The degradation at time $t$ under stress temperature $T$ is assumed to follow an 
exponential-in-time law, accelerated by an Arrhenius temperature factor:

$$
D(t, T)
= b\,\exp(a\,t)\,
  \exp\!\left(
      \frac{E_a}{k_B}
      \left(
        \frac{1}{T_{\text{use}}} - \frac{1}{T}
      \right)
  \right),
$$


### **Parameters**

- **$b$** – initial damage scale at \(t=0\)  
- **$a$** – time growth rate (1/time)  
- **$E_{a}$** – Arrhenius activation energy (eV)  
- **$T_{\text{use}}$** – use-level temperature (fixed, not fitted)  

"""

    # ------------------------------------------------------------------
    # Mean model
    # ------------------------------------------------------------------
    def _mu(self, theta, t):
        """
        Mean damage at the data temperatures (self.T_K):

            D_i = b * exp(a * t_i) * exp(Ea / k * (1/T_use - 1/T_i))
        """
        b, a, Ea = theta
        t = np.asarray(t, float)

        if t.shape != self.t.shape:
            raise ValueError(
                "_mu is intended for data-fitting only (t must match self.t shape). "
                "Use model-specific helpers for grids/other temperatures."
            )

        t_pos = np.clip(t, 0.0, None)
        accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / self.T_K))
        return b * np.exp(a * t_pos) * accel

    # ------------------------------------------------------------------
    # Negative log-likelihood
    # ------------------------------------------------------------------
    def _negloglik(self, x):
        """
        x = [b, a, Ea, log_sigma]
        """
        b, a, Ea = x[:-1]
        log_sigma = x[-1]
        sigma = np.exp(log_sigma)

        # Require positive parameters
        if sigma <= 0 or b <= 0 or a <= 0 or Ea <= 0:
            return np.inf

        theta = np.array([b, a, Ea], float)
        mu = self._mu(theta, self.t)

        if self.noise == "additive":
            if not np.all(np.isfinite(mu)):
                return np.inf
            return -np.sum(norm.logpdf(self.y, loc=mu, scale=sigma))

        elif self.noise == "multiplicative":
            D_pos = np.clip(self.y, 1e-12, None)
            mu_pos = np.clip(mu, 1e-12, None)
            logD = np.log(D_pos)
            logmu = np.log(mu_pos)
            z = (logD - logmu) / sigma

            if not np.all(np.isfinite(z)):
                return np.inf

            return (
                np.sum(np.log(D_pos)) +
                len(D_pos) * np.log(sigma) -
                np.sum(norm.logpdf(z))
            )
        else:
            raise ValueError(f"Unknown noise model: {self.noise}")

    # ------------------------------------------------------------------
    # Initial guess & bounds
    # ------------------------------------------------------------------
    def _init_guess(self):
        """
        Initial guess for [b, a, Ea].
        """
        y_pos = self.y[self.y > 0]
        if y_pos.size == 0:
            b_init = 1.0
        else:
            b_init = max(np.percentile(y_pos, 10), 1e-6)

        # small positive rate; exponential in time
        a_init = 1e-3
        Ea_init = 0.7  # eV

        return np.array([b_init, a_init, Ea_init], dtype=float)

    def _get_LS_bounds(self, p):
        # p == 3 for [b, a, Ea]
        return [
            (1e-10, None),  # b > 0
            (1e-8, None),   # a > 0
            (1e-3, 5.0),    # Ea in [0.001, 5] eV
        ]

    # ------------------------------------------------------------------
    # LS plots
    # ------------------------------------------------------------------
    def _plot_fit_LS(self):
        if self.theta_LS is None:
            return

        b, a, Ea = self.theta_LS

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        cmap = plt.get_cmap("viridis")
        colors = {T: cmap(i / max(len(temps) - 1, 1)) for i, T in enumerate(temps)}

        for T in temps:
            mask = (self.T_C == T)
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=25,
                alpha=0.7,
                color=colors[T],
                label=f"{T:.0f} °C data",
            )

        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 0.0, None)

        for T in temps:
            T_K = T + 273.15
            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K))
            mu_grid = b * np.exp(a * t_pos) * accel
            ax.plot(
                t_grid,
                mu_grid,
                color=colors[T],
                lw=2,
                label=f"{T:.0f} °C LS mean",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title("LSQ Exponential–Arrhenius Degradation Fit")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=2)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # MLE plots
    # ------------------------------------------------------------------
    def _plot_fit_MLE(self):
        if self.theta_MLE is None:
            return

        b, a, Ea = self.theta_MLE
        sigma = self.sigma_MLE

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, T in enumerate(temps):
            idx = min(i, len(base_colors) - 1)
            styles[T] = dict(color=base_colors[idx], marker=base_markers[idx])

        for T in temps:
            mask = (self.T_C == T)
            st = styles[T]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"{T:.0f} °C data",
            )

        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 0.0, None)

        for T in temps:
            T_K = T + 273.15
            st = styles[T]

            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K))
            mu_grid = b * np.exp(a * t_pos) * accel

            ax.plot(
                t_grid,
                mu_grid,
                color=st["color"],
                lw=2,
                label=f"{T:.0f} °C MLE mean",
            )

            if self.show_noise_bounds and sigma is not None:
                if self.noise == "additive":
                    lower = mu_grid - self.z_CI * sigma
                    upper = mu_grid + self.z_CI * sigma
                else:
                    factor = np.exp(self.z_CI * sigma)
                    lower = mu_grid / factor
                    upper = mu_grid * factor

                ax.fill_between(
                    t_grid,
                    lower,
                    upper,
                    color=st["color"],
                    alpha=0.2,
                    label=f"{T:.0f} °C {int(100*self.CI)}% band",
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"MLE Exponential–Arrhenius Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Bayesian fit
    # ------------------------------------------------------------------
    def _fit_Bayesian(self, nwalkers=32, nburn=1000, nsamp=2000):
        """
        Bayesian fit for Exponential–Arrhenius ADT:

            D(t, T) = b * exp(a * t) * exp(Ea / k_B * (1/T_use - 1/T))

        Natural parameters: {b, a, Ea, sigma}.
        """
        if self.theta_MLE is not None and self.sigma_MLE is not None:
            b_hat, a_hat, Ea_hat = self.theta_MLE
            sig_hat = self.sigma_MLE
        elif self.theta_LS is not None and self.sigma_LS is not None:
            b_hat, a_hat, Ea_hat = self.theta_LS
            sig_hat = self.sigma_LS
        else:
            raise RuntimeError("Run LS or MLE before Bayesian (no initial estimates).")

        default_centers = {
            "b":     float(b_hat),
            "a":     float(a_hat),
            "Ea":    float(Ea_hat),
            "sigma": float(sig_hat),
        }

        default_sds = {
            "b":     0.5,
            "a":     0.5,
            "Ea":    0.5,
            "sigma": 0.5,
        }

        t_time = self.t
        D_obs = self.y

        def loglike(theta_dict):
            b     = theta_dict["b"]
            a     = theta_dict["a"]
            Ea    = theta_dict["Ea"]
            sigma = theta_dict["sigma"]

            if sigma <= 0 or b <= 0 or Ea <= 0:
                return -np.inf

            theta = np.array([b, a, Ea], float)
            mu = self._mu(theta, t_time)

            if self.noise == "additive":
                if not np.all(np.isfinite(mu)):
                    return -np.inf
                return np.sum(norm.logpdf(D_obs, loc=mu, scale=sigma))

            elif self.noise == "multiplicative":
                D_pos  = np.clip(D_obs, 1e-12, None)
                mu_pos = np.clip(mu,   1e-12, None)
                logD   = np.log(D_pos)
                logmu  = np.log(mu_pos)
                if not np.all(np.isfinite(logmu)):
                    return -np.inf
                z = (logD - logmu) / sigma
                return (
                    -np.sum(np.log(D_pos))
                    - len(D_pos) * np.log(sigma)
                    + np.sum(norm.logpdf(z))
                )
            else:
                return -np.inf

        param_names = ["b", "a", "Ea", "sigma"]

        sampler, chain, nat_samples = self._fit_Bayesian_generic(
            param_names=param_names,
            model_name="Exponential_Arrhenius_ADT",
            default_centers=default_centers,
            default_sds=default_sds,
            loglike_func=loglike,
            priors=getattr(self, "priors", None),
            dist_name="NormalError",
            nwalkers=nwalkers,
            nburn=nburn,
            nsamp=nsamp,
        )

        self.b_s     = nat_samples["b"]
        self.a_s     = nat_samples["a"]
        self.Ea_s    = nat_samples["Ea"]
        self.sigma_s = nat_samples["sigma"]

        if self.print_results:
            self._show_model_description() 
            md_param_table_bayes(
                {
                    "b": self.b_s,
                    "a": self.a_s,
                    "Ea (eV)": self.Ea_s,
                    r"$\sigma$": self.sigma_s,
                },
                cred_mass=self.CI,
                label=f"Bayesian ({self.noise})",
            )


    # ------------------------------------------------------------------
    # Bayesian plot
    # ------------------------------------------------------------------
    def _plot_fit_Bayesian(self, nsamples=1000, show_predictive=True):
        if not hasattr(self, "b_s"):
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, T in enumerate(temps):
            idx = min(i, len(base_colors) - 1)
            styles[T] = dict(color=base_colors[idx], marker=base_markers[idx])

        for T in temps:
            mask = (self.T_C == T)
            st = styles[T]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"{T:.0f} °C data",
            )

        rng = np.random.default_rng(7)
        n_post = len(self.b_s)
        ns = min(nsamples, n_post)
        sel = rng.choice(n_post, size=ns, replace=False)

        _b  = self.b_s[sel]
        _a  = self.a_s[sel]
        _Ea = self.Ea_s[sel]
        _sig = self.sigma_s[sel]

        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 0.0, None)
        alpha = self.alpha
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        for T in temps:
            T_K = T + 273.15
            st = styles[T]

            accel = np.exp(
                _Ea[:, None] / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K)
            )
            mu_draws = _b[:, None] * np.exp(_a[:, None] * t_pos[None, :]) * accel

            mu_mean = mu_draws.mean(axis=0)
            lo_mean, hi_mean = np.quantile(mu_draws, [q_lo, q_hi], axis=0)

            ax.plot(
                t_grid,
                mu_mean,
                color=st["color"],
                lw=2,
                label=f"{T:.0f} °C Bayes mean",
            )

            if show_predictive:
                eps = rng.standard_normal(size=mu_draws.shape)

                if self.noise == "additive":
                    y_draws = mu_draws + _sig[:, None] * eps
                else:
                    mu_clip = np.clip(mu_draws, 1e-12, None)
                    log_y = np.log(mu_clip) + _sig[:, None] * eps
                    y_draws = np.exp(log_y)

                lo_pred, hi_pred = np.quantile(y_draws, [q_lo, q_hi], axis=0)
                ax.fill_between(
                    t_grid,
                    lo_pred,
                    hi_pred,
                    alpha=0.15,
                    color=st["color"],
                    label=None,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"Bayesian Exponential–Arrhenius Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # TTF at use (Bayesian posterior)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution(
        self,
        Df=None,
        T_eval_C=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if not hasattr(self, "b_s"):
            print("Bayesian posterior not available; cannot compute TTF distribution.")
            return

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
        else:
            T_eval_K = float(T_eval_C) + 273.15

        theta_samples = np.column_stack([self.b_s, self.a_s, self.Ea_s])
        sigma_samples = np.asarray(self.sigma_s, float)

        def mu_fun_factory(theta_vec):
            b, a, Ea = theta_vec
            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K))

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 0.0, None)
                return b * np.exp(a * t_pos) * accel

            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at use (Df={Df})",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title="TTF posterior and best distribution fits",
            bins=60,
        )

    # ------------------------------------------------------------------
    # TTF at use (MLE plug-in)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution_MLE(
        self,
        Df=None,
        T_eval_C=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if (self.theta_MLE is None) or (self.sigma_MLE is None):
            raise RuntimeError("MLE fit not available – run method='MLE' first.")

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
            label_T = f"T={self.T_use_C:.0f} °C"
        else:
            T_eval_K = float(T_eval_C) + 273.15
            label_T = f"T={float(T_eval_C):.0f} °C"

        b_hat, a_hat, Ea_hat = self.theta_MLE
        sigma_hat = float(self.sigma_MLE)

        theta_samples = np.tile(
            np.array([b_hat, a_hat, Ea_hat], float)[None, :],
            (n_samps, 1),
        )
        sigma_samples = np.full(n_samps, sigma_hat, float)

        def mu_fun_factory(theta_vec):
            b, a, Ea = theta_vec
            accel = np.exp(
                Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K)
            )

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 0.0, None)
                return b * np.exp(a * t_pos) * accel

            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at {label_T} (Df={Df}, MLE plug-in)",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF (MLE plug-in) at {label_T}",
            bins=60,
        )

class Fit_ADT_Linear_Arrhenius(Base_ADT_Model):
    r"""
    Linear-time + Arrhenius *damage* model:

        D(t, T) = a + b * t * exp(Ea / k * (1/T_use - 1/T))

    where:
        - D(t, T) is damage (increasing over time)
        - t is time
        - T is stress temperature (°C input, converted to K)
        - T_use is use temperature
        - a is an intercept (baseline damage)
        - b > 0 is the time slope (per unit time, at reference accel)
        - Ea > 0 is activation energy in eV

    Supports:
        - LSQ fit
        - MLE (additive or multiplicative noise)
        - Bayesian fit via emcee (with log-parameter priors)
        - Optional TTF-at-use histogram (Bayesian or MLE plug-in)
    """

    def __init__(
        self,
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
    ):
        method_norm = method.lower()

        # Base constructor
        super().__init__(
            y=degradation,
            time=time,
            unit=unit,
            CI=CI,
            method=method,
            noise=noise,
            scale_type="damage",
            Df=Df,
            show_LSQ_diagnostics=show_LSQ_diagnostics,
            show_noise_bounds=show_noise_bounds,
            show_use_TTF_dist=show_use_TTF_dist,
            print_results=print_results,
            stress=stress,
            stress_label="Temperature (°C)",
            legend_title="Temperature",
            data_scale=data_scale,
            **kwargs,
        )

        # Temperatures
        self.T_C = np.asarray(stress, float).ravel()
        if self.T_C.shape != self.y.shape:
            raise ValueError("stress (T_C) must have same shape as degradation (y).")
        self.T_K = self.T_C + 273.15
        self.T_use_C = float(stress_use)
        self.T_use_K = self.T_use_C + 273.15
        self.priors = priors  # may be None or a dict
        self.data_scale = data_scale
        self.param_names = ["a", "b", "Ea (eV)"]

        if show_data_plot:
            self.plot_data()

        # Pipeline (mirrors your other Arrhenius classes)
        if method_norm == "ls":
            self._fit_LS(suppress_print=False)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._plot_fit_LS()

        elif method_norm == "mle":
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE()
            self._plot_fit_MLE()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution_MLE()

        elif method_norm.startswith("bayes"):
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE(suppress_print=True)   # centre priors
            self._fit_Bayesian()
            self._plot_fit_Bayesian()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution()

        elif method_norm in ("mle_hierarchical", "mle_heirachical"):
            raise NotImplementedError("Hierarchical MLE not yet implemented for Linear-Arrhenius.")

        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Model Description
    # ------------------------------------------------------------------   
    def _model_description(self):
        return r"""
### Degradation / damage model

**Model form (damage scale)**

Damage is linear in (Arrhenius-scaled) time:

$$
\mathrm{AF}_T(T)
= \exp\!\left(
  \frac{E_a}{k_B}
  \left(
    \frac{1}{T_{\text{use}}} - \frac{1}{T}
  \right)
\right),
$$

$$
D(t, T)
= \gamma_{0} + \gamma_{1} \,\mathrm{AF}_T(T)\, t.
$$

**Parameters:**

- **$\gamma_{0}$** – baseline damage at \(t=0\)  
- **$\gamma_{1}$** – linear degradation rate (per unit Arrhenius-scaled time)  
- **$E_a$** – Arrhenius activation energy (eV)  
- $T_{\text{use}}$ – use-level temperature (fixed input, not fitted)  
"""

    # ------------------------------------------------------------------
    # Mean model
    # ------------------------------------------------------------------
    def _mu(self, theta, t):
        """
        Mean damage at the *data* temperatures (self.T_K):

            D_i = a + b * t_i * exp(Ea / k * (1/T_use - 1/T_i))
        """
        a, b, Ea = theta
        t = np.asarray(t, float)

        if t.shape != self.t.shape:
            raise ValueError(
                "_mu is intended for data-fitting only (t must match self.t shape). "
                "Use model-specific helpers for grids/other temperatures."
            )

        t_pos = np.clip(t, 0.0, None)
        accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / self.T_K))
        return a + b * t_pos * accel

    # ------------------------------------------------------------------
    # Negative log-likelihood
    # ------------------------------------------------------------------
    def _negloglik(self, x):
        """
        x = [a, b, Ea, log_sigma]
        """
        a, b, Ea = x[:-1]
        log_sigma = x[-1]
        sigma = np.exp(log_sigma)

        # Require positive slope and Ea, sigma; intercept a can be signed
        if sigma <= 0 or b <= 0 or Ea <= 0:
            return np.inf

        theta = np.array([a, b, Ea], float)
        mu = self._mu(theta, self.t)

        if self.noise == "additive":
            if not np.all(np.isfinite(mu)):
                return np.inf
            return -np.sum(norm.logpdf(self.y, loc=mu, scale=sigma))

        elif self.noise == "multiplicative":
            D_pos = np.clip(self.y, 1e-12, None)
            mu_pos = np.clip(mu, 1e-12, None)
            logD = np.log(D_pos)
            logmu = np.log(mu_pos)
            z = (logD - logmu) / sigma

            if not np.all(np.isfinite(z)):
                return np.inf

            return (
                np.sum(np.log(D_pos)) +
                len(D_pos) * np.log(sigma) -
                np.sum(norm.logpdf(z))
            )
        else:
            raise ValueError(f"Unknown noise model: {self.noise}")

    # ------------------------------------------------------------------
    # Initial guess & bounds
    # ------------------------------------------------------------------
    def _init_guess(self):
        """
        Initial guess for [a, b, Ea].
        """
        y_pos = self.y[self.y > 0]
        if y_pos.size == 0:
            a_init = 0.0
        else:
            a_init = float(np.percentile(y_pos, 10))

        span = max(float(np.percentile(self.y, 90) - a_init), 1e-6)
        t_max = max(self.t.max(), 1.0)
        b_init = span / t_max
        Ea_init = 0.7  # eV

        return np.array([a_init, b_init, Ea_init], dtype=float)

    def _get_LS_bounds(self, p):
        # p == 3 for [a, b, Ea]
        return [
            (None, None),   # a (intercept) can be signed
            (1e-10, None),  # b > 0
            (1e-3, 5.0),    # Ea in [0.001, 5] eV
        ]

    # ------------------------------------------------------------------
    # LS plots
    # ------------------------------------------------------------------
    def _plot_fit_LS(self):
        if self.theta_LS is None:
            return

        a, b, Ea = self.theta_LS

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        cmap = plt.get_cmap("viridis")
        colors = {T: cmap(i / max(len(temps) - 1, 1)) for i, T in enumerate(temps)}

        # Scatter data
        for T in temps:
            mask = (self.T_C == T)
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=25,
                alpha=0.7,
                color=colors[T],
                label=f"{T:.0f} °C data",
            )

        # Mean curves
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 0.0, None)

        for T in temps:
            T_K = T + 273.15
            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K))
            mu_grid = a + b * t_pos * accel
            ax.plot(
                t_grid,
                mu_grid,
                color=colors[T],
                lw=2,
                label=f"{T:.0f} °C LS mean",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title("LSQ Linear–Arrhenius Degradation Fit")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=2)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # MLE plots
    # ------------------------------------------------------------------
    def _plot_fit_MLE(self):
        if self.theta_MLE is None:
            return

        a, b, Ea = self.theta_MLE
        sigma = self.sigma_MLE

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, T in enumerate(temps):
            idx = min(i, len(base_colors) - 1)
            styles[T] = dict(color=base_colors[idx], marker=base_markers[idx])

        # Data
        for T in temps:
            mask = (self.T_C == T)
            st = styles[T]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"{T:.0f} °C data",
            )

        # Curves + noise bands
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 0.0, None)

        for T in temps:
            T_K = T + 273.15
            st = styles[T]

            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K))
            mu_grid = a + b * t_pos * accel

            ax.plot(
                t_grid,
                mu_grid,
                color=st["color"],
                lw=2,
                label=f"{T:.0f} °C MLE mean",
            )

            if self.show_noise_bounds and sigma is not None:
                if self.noise == "additive":
                    lower = mu_grid - self.z_CI * sigma
                    upper = mu_grid + self.z_CI * sigma
                else:
                    factor = np.exp(self.z_CI * sigma)
                    lower = mu_grid / factor
                    upper = mu_grid * factor

                ax.fill_between(
                    t_grid,
                    lower,
                    upper,
                    color=st["color"],
                    alpha=0.2,
                    label=f"{T:.0f} °C {int(100*self.CI)}% band",
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"MLE Linear–Arrhenius Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Bayesian fit
    # ------------------------------------------------------------------
    def _fit_Bayesian(self, nwalkers=32, nburn=1000, nsamp=2000):
        """
        Bayesian fit for Linear–Arrhenius ADT:

            D(t, T) = a + b * t * exp(Ea / k_B * (1/T_use - 1/T))

        Natural parameters: {a, b, Ea, sigma}.
        """
        if self.theta_MLE is not None and self.sigma_MLE is not None:
            a_hat, b_hat, Ea_hat = self.theta_MLE
            sig_hat = self.sigma_MLE
        elif self.theta_LS is not None and self.sigma_LS is not None:
            a_hat, b_hat, Ea_hat = self.theta_LS
            sig_hat = self.sigma_LS
        else:
            raise RuntimeError("Run LS or MLE before Bayesian (no initial estimates).")

        default_centers = {
            "a":     float(a_hat),
            "b":     float(b_hat),
            "Ea":    float(Ea_hat),
            "sigma": float(sig_hat),
        }

        default_sds = {
            "a":     0.5,
            "b":     0.5,
            "Ea":    0.5,
            "sigma": 0.5,
        }

        t_time = self.t
        D_obs = self.y

        def loglike(theta_dict):
            a     = theta_dict["a"]
            b     = theta_dict["b"]
            Ea    = theta_dict["Ea"]
            sigma = theta_dict["sigma"]

            if sigma <= 0 or Ea <= 0:
                return -np.inf

            theta = np.array([a, b, Ea], float)
            mu = self._mu(theta, t_time)

            if self.noise == "additive":
                if not np.all(np.isfinite(mu)):
                    return -np.inf
                return np.sum(norm.logpdf(D_obs, loc=mu, scale=sigma))

            elif self.noise == "multiplicative":
                D_pos  = np.clip(D_obs, 1e-12, None)
                mu_pos = np.clip(mu,   1e-12, None)
                logD   = np.log(D_pos)
                logmu  = np.log(mu_pos)
                if not np.all(np.isfinite(logmu)):
                    return -np.inf
                z = (logD - logmu) / sigma
                return (
                    -np.sum(np.log(D_pos))
                    - len(D_pos) * np.log(sigma)
                    + np.sum(norm.logpdf(z))
                )
            else:
                return -np.inf

        param_names = ["a", "b", "Ea", "sigma"]

        sampler, chain, nat_samples = self._fit_Bayesian_generic(
            param_names=param_names,
            model_name="Linear_Arrhenius_ADT",
            default_centers=default_centers,
            default_sds=default_sds,
            loglike_func=loglike,
            priors=getattr(self, "priors", None),
            dist_name="NormalError",
            nwalkers=nwalkers,
            nburn=nburn,
            nsamp=nsamp,
        )

        self.a_s     = nat_samples["a"]
        self.b_s     = nat_samples["b"]
        self.Ea_s    = nat_samples["Ea"]
        self.sigma_s = nat_samples["sigma"]

        if self.print_results:
            self._show_model_description() 
            md_param_table_bayes(
                {
                    "a": self.a_s,
                    "b": self.b_s,
                    "Ea (eV)": self.Ea_s,
                    r"$\sigma$": self.sigma_s,
                },
                cred_mass=self.CI,
                label=f"Bayesian ({self.noise})",
            )

    # ------------------------------------------------------------------
    # Bayesian plot
    # ------------------------------------------------------------------
    def _plot_fit_Bayesian(self, nsamples=1000, show_predictive=True):
        if not hasattr(self, "a_s"):
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, T in enumerate(temps):
            idx = min(i, len(base_colors) - 1)
            styles[T] = dict(color=base_colors[idx], marker=base_markers[idx])

        # data
        for T in temps:
            mask = (self.T_C == T)
            st = styles[T]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"{T:.0f} °C data",
            )

        rng = np.random.default_rng(7)
        n_post = len(self.a_s)
        ns = min(nsamples, n_post)
        sel = rng.choice(n_post, size=ns, replace=False)

        _a  = self.a_s[sel]
        _b  = self.b_s[sel]
        _Ea = self.Ea_s[sel]
        _sig = self.sigma_s[sel]

        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 0.0, None)
        alpha = self.alpha
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        for T in temps:
            T_K = T + 273.15
            st = styles[T]

            accel = np.exp(
                _Ea[:, None] / K_BOLTZ_eV
                * (1.0 / self.T_use_K - 1.0 / T_K)
            )
            mu_draws = _a[:, None] + _b[:, None] * t_pos[None, :] * accel

            mu_mean = mu_draws.mean(axis=0)
            lo_mean, hi_mean = np.quantile(mu_draws, [q_lo, q_hi], axis=0)

            ax.plot(
                t_grid,
                mu_mean,
                color=st["color"],
                lw=2,
                label=f"{T:.0f} °C Bayes mean",
            )

            if show_predictive:
                eps = rng.standard_normal(size=mu_draws.shape)

                if self.noise == "additive":
                    y_draws = mu_draws + _sig[:, None] * eps
                else:
                    mu_clip = np.clip(mu_draws, 1e-12, None)
                    log_y = np.log(mu_clip) + _sig[:, None] * eps
                    y_draws = np.exp(log_y)

                lo_pred, hi_pred = np.quantile(y_draws, [q_lo, q_hi], axis=0)
                ax.fill_between(
                    t_grid,
                    lo_pred,
                    hi_pred,
                    alpha=0.15,
                    color=st["color"],
                    label=None,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"Bayesian Linear–Arrhenius Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # TTF at use (Bayesian posterior)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution(
        self,
        Df=None,
        T_eval_C=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if not hasattr(self, "a_s"):
            print("Bayesian posterior not available; cannot compute TTF distribution.")
            return

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
        else:
            T_eval_K = float(T_eval_C) + 273.15

        theta_samples = np.column_stack([self.a_s, self.b_s, self.Ea_s])
        sigma_samples = np.asarray(self.sigma_s, float)

        def mu_fun_factory(theta_vec):
            a, b, Ea = theta_vec
            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K))

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 0.0, None)
                return a + b * t_pos * accel

            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at use (Df={Df})",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title="TTF posterior and best distribution fits",
            bins=60,
        )

    # ------------------------------------------------------------------
    # TTF at use (MLE plug-in)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution_MLE(
        self,
        Df=None,
        T_eval_C=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if (self.theta_MLE is None) or (self.sigma_MLE is None):
            raise RuntimeError("MLE fit not available – run method='MLE' first.")

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
            label_T = f"T={self.T_use_C:.0f} °C"
        else:
            T_eval_K = float(T_eval_C) + 273.15
            label_T = f"T={float(T_eval_C):.0f} °C"

        a_hat, b_hat, Ea_hat = self.theta_MLE
        sigma_hat = float(self.sigma_MLE)

        theta_samples = np.tile(
            np.array([a_hat, b_hat, Ea_hat], float)[None, :],
            (n_samps, 1),
        )
        sigma_samples = np.full(n_samps, sigma_hat, float)

        def mu_fun_factory(theta_vec):
            a, b, Ea = theta_vec
            accel = np.exp(
                Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K)
            )

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 0.0, None)
                return a + b * t_pos * accel

            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at {label_T} (Df={Df}, MLE plug-in)",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF (MLE plug-in) at {label_T}",
            bins=60,
        )


class Fit_ADT_Power_Arrhenius(Base_ADT_Model):
    r"""
    Power-time + Arrhenius *damage* model:

        D(t, T) = a * t^n * exp(Ea / k * (1/T_use - 1/T))

    where:
        - D(t, T) is damage (increasing over time)
        - t is time
        - T is stress temperature
        - T_use is use temperature
        - a > 0 is a scale parameter
        - n > 0 is the time exponent
        - Ea > 0 is activation energy in eV

    This class supports:
        - LSQ fit
        - MLE (additive or multiplicative noise)
        - Bayesian fit via emcee (with log-parameter priors)
        - Optional TTF-at-use histogram (Bayesian or MLE plug-in)
    """

    def __init__(
        self,
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
    ):
        """
        Parameters
        ----------
        degradation : array-like
            Observed damage D_ij.
        stress : array-like
            Temperature in °C for each observation.
        time : array-like
            Time values (e.g. hours).
        unit : array-like
            Unit IDs (integers).
        stress_use : float
            Use temperature in °C.
        Df : float
            Failure threshold on *damage* scale (fail when D >= Df).
        CI : float
            Confidence/credible level (default 0.95).
        method : {"LS", "MLE", "MLE_hierarchical", "Bayesian"}
            Fit method. This class implements LS, MLE, and Bayesian.
        noise : {"additive", "multiplicative"}
            Observation noise model on D.
        show_data_plot : bool
            If True, scatter plot of D vs t by temperature.
        show_LSQ_diagnostics : bool
            If True, show LS residual diagnostics.
        show_noise_bounds : bool
            If True, show noise bands on MLE & Bayes plots.
        show_use_TTF_dist : bool
            If True, compute and plot TTF at use (Bayes or MLE plug-in).
        print_results : bool
            If True, print parameter tables in Markdown.
        """
        method_norm = method.lower()

        # 1) Base constructor
        super().__init__(
            y=degradation,
            time=time,
            unit=unit,
            CI=CI,
            method=method,
            noise=noise,
            scale_type="damage",
            Df=Df,
            show_LSQ_diagnostics=show_LSQ_diagnostics,
            show_noise_bounds=show_noise_bounds,
            show_use_TTF_dist=show_use_TTF_dist,
            print_results=print_results,
            stress=stress,
            stress_label="Temperature (°C)",
            legend_title="Temperature",
            data_scale=data_scale,
            **kwargs,
        )

        # 2) Model-specific fields
        self.T_C = np.asarray(stress, float).ravel()
        if self.T_C.shape != self.y.shape:
            raise ValueError("stress (T_C) must have same shape as degradation (y).")
        self.T_K = self.T_C + 273.15
        self.T_use_C = float(stress_use)
        self.T_use_K = self.T_use_C + 273.15
        self.priors = priors  # may be None or a dict
        self.data_scale = data_scale
        self.param_names = ["a", "n", "Ea (eV)"]

        # 3) Optional raw data plot
        if show_data_plot:
            self.plot_data()

        # 4) Method-specific pipeline
        if method_norm == "ls":
            self._fit_LS(suppress_print=False)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._plot_fit_LS()

        elif method_norm == "mle":
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE()
            self._plot_fit_MLE()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution_MLE()

        elif method_norm.startswith("bayes"):
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE(suppress_print=True)   # for priors
            self._fit_Bayesian()
            self._plot_fit_Bayesian()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution()

        elif method_norm in ("mle_hierarchical", "mle_heirachical"):
            raise NotImplementedError("Hierarchical MLE not yet implemented for Power-Arrhenius.")

        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Model Description
    # ------------------------------------------------------------------   
    def _model_description(self):
        return r"""
### Degradation / damage model

**Model form (damage scale)**

We combine a power law in the second stress \(S\) with an Arrhenius
temperature acceleration:

$$
\mathrm{AF}_T(T)
= \exp\!\left(
  \frac{E_a}{k_B}
  \left(
    \frac{1}{T_{\text{use}}} - \frac{1}{T}
  \right)
\right),
$$

$$
D(t, T, S)
= b \,\mathrm{AF}_T(T)\, S^{n} \, t^{m}.
$$

**Parameters:**

- **$b$** – scale factor on the accumulated damage  
- **$n$** – stress exponent for the second stress \(S\) (e.g. current)  
- **$m$** – time exponent (damage growth with time)  
- **$E_{a}$** – Arrhenius activation energy (eV)  
- $T_{\text{use}}$ – use-level temperature (fixed input, not fitted)  
- $S_{\text{use}}$ – use-level second stress (fixed input, not fitted)  
"""

    # ------------------------------------------------------------------
    # Model mean: data level
    # ------------------------------------------------------------------
    def _mu(self, theta, t):
        """
        Mean damage model at the *data* temperatures (self.T_K):

            D_i = a * t_i^n * exp(Ea / k * (1/T_use - 1/T_i))

        Used internally by Base_ADT_Model for LS/MLE on observed data.
        """
        a, n, Ea = theta
        t = np.asarray(t, float)

        if t.shape != self.t.shape:
            raise ValueError(
                "_mu is intended for data-fitting only (t must match self.t shape). "
                "Use model-specific helpers for grids/other temperatures."
            )

        t_pos = np.clip(t, 1e-12, None)
        accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / self.T_K))
        return a * (t_pos ** n) * accel

    # --------------------------------------------------------------
    # Negative log-likelihood
    # --------------------------------------------------------------
    def _negloglik(self, x):
        """
        x = [a, n, Ea, log_sigma]
        Likelihood depends on additive or multiplicative noise on damage.
        """
        a, n, Ea = x[:-1]
        log_sigma = x[-1]
        sigma = np.exp(log_sigma)

        # constrain positives
        if sigma <= 0 or a <= 0 or n <= 0 or Ea <= 0:
            return np.inf

        theta = np.array([a, n, Ea], float)
        mu = self._mu(theta, self.t)

        if self.noise == "additive":
            if not np.all(np.isfinite(mu)):
                return np.inf
            return -np.sum(norm.logpdf(self.y, loc=mu, scale=sigma))

        elif self.noise == "multiplicative":
            D_pos = np.clip(self.y, 1e-12, None)
            mu_pos = np.clip(mu, 1e-12, None)
            logD = np.log(D_pos)
            logmu = np.log(mu_pos)
            z = (logD - logmu) / sigma

            if not np.all(np.isfinite(z)):
                return np.inf

            return (
                np.sum(np.log(D_pos)) +
                len(D_pos) * np.log(sigma) -
                np.sum(norm.logpdf(z))
            )

        else:
            raise ValueError(f"Unknown noise model: {self.noise}")

    # ------------------------------------------------------------------
    # Initial guess & bounds
    # ------------------------------------------------------------------
    def _init_guess(self):
        """
        Model-specific initial guess for [a, n, Ea].
        """
        # crude heuristics: use scale from upper tail and mild exponent
        y_med = float(np.percentile(self.y[self.y > 0], 50)) if np.any(self.y > 0) else 1.0
        a_init = max(y_med, 1e-6)
        n_init = 0.5
        Ea_init = 0.7  # eV – generic starter
        return np.array([a_init, n_init, Ea_init], dtype=float)

    def _get_LS_bounds(self, p):
        # p == 3 for [a, n, Ea]
        return [
            (1e-10, None),  # a > 0
            (1e-6, None),   # n > 0
            (1e-3, 5.0),    # Ea in [0.001, 5] eV
        ]

    # ------------------------------------------------------------------
    # LS diagnostics & plots
    # ------------------------------------------------------------------
    def _plot_fit_LS(self):
        if self.theta_LS is None:
            return

        a, n, Ea = self.theta_LS

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        cmap = plt.get_cmap("viridis")
        colors = {T: cmap(i / max(len(temps) - 1, 1)) for i, T in enumerate(temps)}

        # Scatter data
        for T in temps:
            mask = (self.T_C == T)
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=25,
                alpha=0.7,
                color=colors[T],
                label=f"{T:.0f} °C data",
            )

        # Mean curves for each temperature
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        for T in temps:
            T_K = T + 273.15
            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K))
            mu_grid = a * (t_pos ** n) * accel
            ax.plot(
                t_grid,
                mu_grid,
                color=colors[T],
                lw=2,
                label=f"{T:.0f} °C LS mean",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title("LSQ Power–Arrhenius Degradation Fit")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=2)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # MLE plot
    # ------------------------------------------------------------------
    def _plot_fit_MLE(self):
        if self.theta_MLE is None:
            return

        a, n, Ea = self.theta_MLE
        sigma = self.sigma_MLE

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, T in enumerate(temps):
            idx = min(i, len(base_colors) - 1)
            styles[T] = dict(color=base_colors[idx], marker=base_markers[idx])

        # Scatter data
        for T in temps:
            mask = (self.T_C == T)
            st = styles[T]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"{T:.0f} °C data",
            )

        # Mean curves + noise bands
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        for T in temps:
            T_K = T + 273.15
            st = styles[T]

            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K))
            mu_grid = a * (t_pos ** n) * accel

            ax.plot(
                t_grid,
                mu_grid,
                color=st["color"],
                lw=2,
                label=f"{T:.0f} °C MLE mean",
            )

            if self.show_noise_bounds and sigma is not None:
                if self.noise == "additive":
                    lower = mu_grid - self.z_CI * sigma
                    upper = mu_grid + self.z_CI * sigma
                else:
                    factor = np.exp(self.z_CI * sigma)
                    lower = mu_grid / factor
                    upper = mu_grid * factor

                ax.fill_between(
                    t_grid,
                    lower,
                    upper,
                    color=st["color"],
                    alpha=0.2,
                    label=f"{T:.0f} °C {int(100*self.CI)}% band",
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"MLE Power–Arrhenius Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Bayesian fit (emcee)
    # ------------------------------------------------------------------
    def _fit_Bayesian(self, nwalkers=32, nburn=1000, nsamp=2000):
        """
        Bayesian fit for Power–Arrhenius ADT:

            D(t, T) = a * t^n * exp(Ea / k_B * (1/T_use - 1/T))

        Natural parameters: {a, n, Ea, sigma}.
        """
        if self.theta_MLE is not None and self.sigma_MLE is not None:
            a_hat, n_hat, Ea_hat = self.theta_MLE
            sig_hat = self.sigma_MLE
        elif self.theta_LS is not None and self.sigma_LS is not None:
            a_hat, n_hat, Ea_hat = self.theta_LS
            sig_hat = self.sigma_LS
        else:
            raise RuntimeError("Run LS or MLE before Bayesian (no initial estimates).")

        default_centers = {
            "a":     float(a_hat),
            "n":     float(n_hat),
            "Ea":    float(Ea_hat),
            "sigma": float(sig_hat),
        }

        default_sds = {
            "a":     0.5,
            "n":     0.5,
            "Ea":    0.5,
            "sigma": 0.5,
        }

        t_time = self.t
        D_obs = self.y

        def loglike(theta_dict):
            a     = theta_dict["a"]
            n     = theta_dict["n"]
            Ea    = theta_dict["Ea"]
            sigma = theta_dict["sigma"]

            if sigma <= 0 or a <= 0 or Ea <= 0:
                return -np.inf

            theta = np.array([a, n, Ea], float)
            mu = self._mu(theta, t_time)  # uses self.T_K + T_use_K internally

            if self.noise == "additive":
                if not np.all(np.isfinite(mu)):
                    return -np.inf
                return np.sum(norm.logpdf(D_obs, loc=mu, scale=sigma))

            elif self.noise == "multiplicative":
                D_pos  = np.clip(D_obs, 1e-12, None)
                mu_pos = np.clip(mu,   1e-12, None)
                logD   = np.log(D_pos)
                logmu  = np.log(mu_pos)
                if not np.all(np.isfinite(logmu)):
                    return -np.inf
                z = (logD - logmu) / sigma
                return (
                    -np.sum(np.log(D_pos))
                    - len(D_pos) * np.log(sigma)
                    + np.sum(norm.logpdf(z))
                )
            else:
                return -np.inf

        param_names = ["a", "n", "Ea", "sigma"]

        sampler, chain, nat_samples = self._fit_Bayesian_generic(
            param_names=param_names,
            model_name="Power_Arrhenius_ADT",
            default_centers=default_centers,
            default_sds=default_sds,
            loglike_func=loglike,
            priors=getattr(self, "priors", None),
            dist_name="NormalError",
            nwalkers=nwalkers,
            nburn=nburn,
            nsamp=nsamp,
        )

        self.a_s     = nat_samples["a"]
        self.n_s     = nat_samples["n"]
        self.Ea_s    = nat_samples["Ea"]
        self.sigma_s = nat_samples["sigma"]

        if self.print_results:
            self._show_model_description() 
            md_param_table_bayes(
                {
                    "a": self.a_s,
                    "n": self.n_s,
                    "Ea (eV)": self.Ea_s,
                    r"$\sigma$": self.sigma_s,
                },
                cred_mass=self.CI,
                label=f"Bayesian ({self.noise})",
            )


    # ------------------------------------------------------------------
    # Bayesian plot
    # ------------------------------------------------------------------
    def _plot_fit_Bayesian(self, nsamples=1000, show_predictive=True):
        if not hasattr(self, "a_s"):
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        temps = np.unique(self.T_C)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, T in enumerate(temps):
            idx = min(i, len(base_colors) - 1)
            styles[T] = dict(color=base_colors[idx], marker=base_markers[idx])

        # Scatter data
        for T in temps:
            mask = (self.T_C == T)
            st = styles[T]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"{T:.0f} °C data",
            )

        # Thin posterior
        rng = np.random.default_rng(7)
        n_post = len(self.a_s)
        ns = min(nsamples, n_post)
        sel = rng.choice(n_post, size=ns, replace=False)

        _a  = self.a_s[sel]
        _n  = self.n_s[sel]
        _Ea = self.Ea_s[sel]
        _sig = self.sigma_s[sel]

        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)
        alpha = self.alpha
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        for T in temps:
            T_K = T + 273.15
            st = styles[T]

            accel = np.exp(
                _Ea[:, None] / K_BOLTZ_eV
                * (1.0 / self.T_use_K - 1.0 / T_K)
            )
            t_pow = t_pos[None, :] ** _n[:, None]
            mu_draws = _a[:, None] * t_pow * accel

            mu_mean = mu_draws.mean(axis=0)
            lo_mean, hi_mean = np.quantile(mu_draws, [q_lo, q_hi], axis=0)

            ax.plot(
                t_grid,
                mu_mean,
                color=st["color"],
                lw=2,
                label=f"{T:.0f} °C Bayes mean",
            )

            if show_predictive:
                eps = rng.standard_normal(size=mu_draws.shape)

                if self.noise == "additive":
                    y_draws = mu_draws + _sig[:, None] * eps
                else:
                    mu_clip = np.clip(mu_draws, 1e-12, None)
                    log_y = np.log(mu_clip) + _sig[:, None] * eps
                    y_draws = np.exp(log_y)

                lo_pred, hi_pred = np.quantile(y_draws, [q_lo, q_hi], axis=0)
                ax.fill_between(
                    t_grid,
                    lo_pred,
                    hi_pred,
                    alpha=0.15,
                    color=st["color"],
                    label=None,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"Bayesian Power–Arrhenius Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # TTF at use (Bayesian posterior)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution(
        self,
        Df=None,
        T_eval_C=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if not hasattr(self, "a_s"):
            print("Bayesian posterior not available; cannot compute TTF distribution.")
            return

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
        else:
            T_eval_K = float(T_eval_C) + 273.15

        theta_samples = np.column_stack([self.a_s, self.n_s, self.Ea_s])
        sigma_samples = np.asarray(self.sigma_s, float)

        def mu_fun_factory(theta_vec):
            a, n, Ea = theta_vec
            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K))

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 1e-12, None)
                return a * (t_pos ** n) * accel

            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at use (Df={Df})",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title="TTF posterior and best distribution fits",
            bins=60,
        )

    # ------------------------------------------------------------------
    # TTF at use (MLE plug-in)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution_MLE(
        self,
        Df=None,
        T_eval_C=None,
        n_samps=8000,
        unit_label="time units",
    ):
        if (self.theta_MLE is None) or (self.sigma_MLE is None):
            raise RuntimeError("MLE fit not available – run method='MLE' first.")

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if T_eval_C is None:
            T_eval_K = self.T_use_K
            label_T = f"T={self.T_use_C:.0f} °C"
        else:
            T_eval_K = float(T_eval_C) + 273.15
            label_T = f"T={float(T_eval_C):.0f} °C"

        a_hat, n_hat, Ea_hat = self.theta_MLE
        sigma_hat = float(self.sigma_MLE)

        theta_samples = np.tile(
            np.array([a_hat, n_hat, Ea_hat], float)[None, :],
            (n_samps, 1),
        )
        sigma_samples = np.full(n_samps, sigma_hat, float)

        def mu_fun_factory(theta_vec):
            a, n, Ea = theta_vec
            accel = np.exp(
                Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_eval_K)
            )

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 1e-12, None)
                return a * (t_pos ** n) * accel

            return mu_fun

        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at {label_T} (Df={Df}, MLE plug-in)",
            unit_label=unit_label,
            as_markdown=True,
        )

        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF (MLE plug-in) at {label_T}",
            bins=60,
        )


class Fit_ADT_Power_Exponential(Base_ADT_Model):
    r"""
    Power–Exponential *damage* model:

        D(t, S) = b * exp(a * S ) * t^n

    where
        - D(t, S)  : damage (increasing over time)
        - S        : stress (e.g. applied weight in grams)
        - b > 0    : scale parameter
        - a        : stress-sensitivity parameter
        - n >= 0   : time exponent

    Supports:
        - LSQ fit
        - MLE (additive or multiplicative noise on D)
        - Bayesian fit via emcee
        - Optional TTF-at-use histogram from Bayesian posterior
    """

    def __init__(
        self,
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
    ):
        """
        Parameters
        ----------
        degradation : array-like
            Observed damage D_ij.
        stress : array-like
            Stress level S_ij for each observation (same shape as degradation).
        time : array-like
            Time values (e.g. hours).
        unit : array-like
            Unit IDs (integers).
        stress_use : float
            Use-stress value S_use at which TTF is to be evaluated.
        Df : float
            Failure threshold on *damage* scale (fail when D >= Df).
        CI : float
            Confidence/credible level (default 0.95).
        method : {"LS", "MLE", "Bayesian"}
            Fit method. (No hierarchical version here yet.)
        noise : {"additive", "multiplicative"}
            Observation noise model on D.
        show_data_plot : bool
            If True, scatter/line plot of D vs t by stress.
        show_LSQ_diagnostics : bool
            If True, show LS residual diagnostics.
        show_noise_bounds : bool
            If True, show noise bands on MLE & Bayes plots.
        show_use_TTF_dist : bool
            If True (and method is Bayesian), compute and plot TTF at use.
        print_results : bool
            If True, print parameter tables in Markdown.
        data_scale : {"linear","ylog","xlog","loglog"}
            Axis scaling for the data plot.
        kwargs :
            Additional options passed through to Base_ADT_Model.
        """
        method_norm = method.lower()

        # 1) Base class: y, t, unit, noise, etc.
        super().__init__(
            y=degradation,
            time=time,
            unit=unit,
            CI=CI,
            method=method,
            noise=noise,
            scale_type="damage",
            Df=Df,
            show_LSQ_diagnostics=show_LSQ_diagnostics,
            show_noise_bounds=show_noise_bounds,
            show_use_TTF_dist=show_use_TTF_dist,
            print_results=print_results,
            stress=stress,
            stress_label="Stress",
            legend_title="Stress",
            data_scale=data_scale,
            **kwargs,
        )

        # 2) Model-specific: stress
        self.S = np.asarray(stress, float).ravel()
        if self.S.shape != self.y.shape:
            raise ValueError("stress must have the same shape as degradation.")
        self.S_use = float(stress_use)
        self.data_scale = data_scale
        self.priors = priors  # may be None or a dict
        self.param_names = ["b", "a", "n"]

        # 3) Optional raw data plot
        if show_data_plot:
            self.plot_data()

        # 4) Method-specific pipeline
        if method_norm == "ls":
            self._fit_LS(suppress_print=False)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._plot_fit_LS()

        elif method_norm == "mle":
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE()
            self._plot_fit_MLE()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution_MLE()  

        elif method_norm.startswith("bayes"):
            # LS (quiet) → MLE (quiet) → Bayes
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE(suppress_print=True)
            self._fit_Bayesian()
            self._plot_fit_Bayesian()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution()

        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Model Description
    # ------------------------------------------------------------------             
    def _model_description(self):
        return r"""
### Degradation / damage model

**Model form (damage scale)**

The Power–Exponential model uses a power law in stress and an exponential in time
(e.g. saturating or rapidly increasing damage):

$$
D(t, S)
= b \, S^{n_S} \left( 1 - e^{-\lambda_t t} \right)
$$

or equivalently (depending on sign convention in the code),

$$
D(t, S)
= b \, S^{n_S} \big( e^{\lambda_t t} - 1 \big).
$$

**Parameters:**

- **$b$** – overall scale factor on damage  
- **$n$** – stress exponent (sensitivity to stress \(S\))  
- **$\lambda_{t}$** – time rate parameter (how quickly damage grows with time)  
- **$S$** – accelerating stress (e.g. load, current)  
"""

    # ------------------------------------------------------------------
    # Model mean on data scale
    # ------------------------------------------------------------------
    def _mu(self, theta, t):
        b, a, n = theta
        S = self.S
        t_pos = np.clip(t, 1e-12, None)
        exp_term = np.exp(np.clip(a * S, -700, 700))
        return b * exp_term * (t_pos**n)

    # ------------------------------------------------------------------
    # LSQ init & bounds
    # ------------------------------------------------------------------
    def _init_guess(self):
        """
        Heuristic initial guess for [b, a, n].
        """
        # rough b from lower tail of y
        b_init = max(float(np.percentile(self.y, 10)), 1e-6)
        # rough n from how much y grows with t
        if np.all(self.t > 0):
            span_y = max(np.percentile(self.y, 90) - np.percentile(self.y, 10), 1e-6)
            span_t = max(np.log(self.t.max() / max(self.t.min(), 1e-6)), 1e-6)
            n_init = max(span_y / (10.0 * span_t), 0.1)
        else:
            n_init = 0.5
        # a could be mild; start at 0.0
        a_init = 0.0
        return np.array([b_init, a_init, n_init], dtype=float)

    def _get_LS_bounds(self, p):
        # [b, a, n]: b>0, n>0, a free
        return [
            (1e-10, None),  # b > 0
            (None, None),   # a ∈ R
            (1e-6, None),   # n > 0
        ]

    # ------------------------------------------------------------------
    # Negative log-likelihood for MLE (Base calls self._negloglik)
    # ------------------------------------------------------------------
    def _negloglik(self, x):
        """
        x = [b, a, n, log_sigma]
        """
        b, a, n = x[:-1]
        log_sigma = x[-1]
        sigma = np.exp(log_sigma)

        # basic validity checks
        if sigma <= 0 or b <= 0 or n <= 0:
            return np.inf

        theta = np.array([b, a, n], float)
        mu = self._mu(theta, self.t)

        if self.noise == "additive":
            if not np.all(np.isfinite(mu)):
                return np.inf
            return -np.sum(norm.logpdf(self.y, loc=mu, scale=sigma))

        elif self.noise == "multiplicative":
            D_pos = np.clip(self.y, 1e-12, None)
            mu_pos = np.clip(mu, 1e-12, None)
            logD = np.log(D_pos)
            logmu = np.log(mu_pos)
            z = (logD - logmu) / sigma

            if not np.all(np.isfinite(z)):
                return np.inf

            # lognormal jacobian + normal on log-scale
            return (
                np.sum(np.log(D_pos)) +
                len(D_pos) * np.log(sigma) -
                np.sum(norm.logpdf(z))
            )

        else:
            raise ValueError(f"Unknown noise model: {self.noise}")

    # ------------------------------------------------------------------
    # LS diagnostics & plots
    # ------------------------------------------------------------------

    def _plot_fit_LS(self):
        if self.theta_LS is None:
            return

        b, a, n = self.theta_LS
        fig, ax = plt.subplots(figsize=(8, 6))

        stresses = np.unique(self.S)
        # Fixed palette + markers (shared with MLE for consistency)
        base_colors  = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        styles = {}
        for i, S in enumerate(stresses):
            idx = min(i, len(base_colors) - 1)
            styles[S] = dict(color=base_colors[idx],
                             marker=base_markers[idx])

        # Data
        for S in stresses:
            mask = (self.S == S)
            st = styles[S]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=25,
                alpha=0.8,
                color=st["color"],
                marker=st["marker"],
                label=f"S={S:g} data",
            )

        # model curves
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 0.0, None)

        for S in stresses:
            st = styles[S]
            factor = np.exp(a * S)
            mu_grid = b * factor * (t_pos ** n)
            ax.plot(
                t_grid,
                mu_grid,
                color=st["color"],
                lw=2,
                label=f"S={S:g} LS mean",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title("LSQ Power–Exponential Degradation Fit")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        # de-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=2)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # MLE plot
    # ------------------------------------------------------------------
    def _plot_fit_MLE(self):
        if self.theta_MLE is None:
            return

        b, a, n = self.theta_MLE
        sigma = self.sigma_MLE

        fig, ax = plt.subplots(figsize=(8, 6))

        stresses = np.unique(self.S)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, S in enumerate(stresses):
            idx = min(i, len(base_colors) - 1)
            styles[S] = dict(color=base_colors[idx], marker=base_markers[idx])

        # data
        for S in stresses:
            mask = (self.S == S)
            st = styles[S]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"S={S:g} data",
            )

        # curves + noise bands
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 0.0, None)

        for S in stresses:
            st = styles[S]
            factor = np.exp(a * S)
            mu_grid = b * factor * (t_pos ** n)

            ax.plot(
                t_grid,
                mu_grid,
                color=st["color"],
                lw=2,
                label=f"S={S:g} MLE mean",
            )

            if self.show_noise_bounds and sigma is not None:
                if self.noise == "additive":
                    lower = mu_grid - self.z_CI * sigma
                    upper = mu_grid + self.z_CI * sigma
                else:
                    factor_noise = np.exp(self.z_CI * sigma)
                    lower = mu_grid / factor_noise
                    upper = mu_grid * factor_noise

                ax.fill_between(
                    t_grid,
                    lower,
                    upper,
                    color=st["color"],
                    alpha=0.2,
                    label=f"S={S:g} {int(100*self.CI)}% band",
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"MLE Power–Exponential Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Bayesian fit (note: a is allowed to be negative, so prior is Normal)
    # ------------------------------------------------------------------
    def _fit_Bayesian(self, nwalkers=32, nburn=1000, nsamp=2000):
        """
        Bayesian fit for Power–Exponential ADT using Base_ADT_Model._fit_Bayesian_generic.

        Natural parameters: {b, a, n, sigma}.
        """
        # 1) Prior centres from MLE, fallback to LS
        if self.theta_MLE is not None and self.sigma_MLE is not None:
            b_hat, a_hat, n_hat = self.theta_MLE
            sig_hat = self.sigma_MLE
        elif self.theta_LS is not None and self.sigma_LS is not None:
            b_hat, a_hat, n_hat = self.theta_LS
            sig_hat = self.sigma_LS
        else:
            raise RuntimeError("Run LS or MLE before Bayesian (no initial estimates).")

        default_centers = {
            "b":     float(b_hat),
            "a":     float(a_hat),
            "n":     float(n_hat),
            "sigma": float(sig_hat),
        }

        default_sds = {
            "b":     0.5,
            "a":     0.5,
            "n":     0.5,
            "sigma": 0.5,
        }

        t_time = self.t
        D_obs = self.y

        def loglike(theta_dict):
            b     = theta_dict["b"]
            a     = theta_dict["a"]
            n     = theta_dict["n"]
            sigma = theta_dict["sigma"]

            if sigma <= 0 or b <= 0:
                return -np.inf

            theta = np.array([b, a, n], float)
            mu = self._mu(theta, t_time)  # uses self.S internally

            if self.noise == "additive":
                if not np.all(np.isfinite(mu)):
                    return -np.inf
                return np.sum(norm.logpdf(D_obs, loc=mu, scale=sigma))

            elif self.noise == "multiplicative":
                D_pos  = np.clip(D_obs, 1e-12, None)
                mu_pos = np.clip(mu,   1e-12, None)
                logD   = np.log(D_pos)
                logmu  = np.log(mu_pos)
                if not np.all(np.isfinite(logmu)):
                    return -np.inf
                z = (logD - logmu) / sigma
                return (
                    -np.sum(np.log(D_pos))
                    - len(D_pos) * np.log(sigma)
                    + np.sum(norm.logpdf(z))
                )
            else:
                return -np.inf

        param_names = ["b", "a", "n", "sigma"]

        sampler, chain, nat_samples = self._fit_Bayesian_generic(
            param_names=param_names,
            model_name="Power_Exponential_ADT",
            default_centers=default_centers,
            default_sds=default_sds,
            loglike_func=loglike,
            priors=getattr(self, "priors", None),
            dist_name="NormalError",
            nwalkers=nwalkers,
            nburn=nburn,
            nsamp=nsamp,
        )

        self.b_s     = nat_samples["b"]
        self.a_s     = nat_samples["a"]
        self.n_s     = nat_samples["n"]
        self.sigma_s = nat_samples["sigma"]

        if self.print_results:
            self._show_model_description() 
            md_param_table_bayes(
                {
                    "b": self.b_s,
                    "a": self.a_s,
                    "n": self.n_s,
                    r"$\sigma$": self.sigma_s,
                },
                cred_mass=self.CI,
                label=f"Bayesian ({self.noise})",
            )


    # ------------------------------------------------------------------
    # Bayesian plot
    # ------------------------------------------------------------------
    def _plot_fit_Bayesian(self, nsamples=1000, show_predictive=True):
        if not hasattr(self, "b_s"):
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        stresses = np.unique(self.S)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, S in enumerate(stresses):
            idx = min(i, len(base_colors) - 1)
            styles[S] = dict(color=base_colors[idx], marker=base_markers[idx])

        # data
        for S in stresses:
            mask = (self.S == S)
            st = styles[S]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"S={S:g} data",
            )

        # thin posterior
        rng = np.random.default_rng(7)
        n_post = len(self.b_s)
        ns = min(nsamples, n_post)
        sel = rng.choice(n_post, size=ns, replace=False)

        _b  = self.b_s[sel]
        _a  = self.a_s[sel]
        _n  = self.n_s[sel]
        _sig = self.sigma_s[sel]

        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 0.0, None)

        alpha = self.alpha
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        for S in stresses:
            st = styles[S]
            # ns x Nt
            factor = np.exp(_a[:, None] * S)
            mu_draws = _b[:, None] * factor * (t_pos[None, :] ** _n[:, None])

            mu_mean = mu_draws.mean(axis=0)
            lo_mean, hi_mean = np.quantile(mu_draws, [q_lo, q_hi], axis=0)

            ax.plot(
                t_grid,
                mu_mean,
                color=st["color"],
                lw=2,
                label=f"S={S:g} Bayes mean",
            )

            if show_predictive:
                eps = rng.standard_normal(size=mu_draws.shape)

                if self.noise == "additive":
                    y_draws = mu_draws + _sig[:, None] * eps
                else:
                    mu_clip = np.clip(mu_draws, 1e-12, None)
                    log_y = np.log(mu_clip) + _sig[:, None] * eps
                    y_draws = np.exp(log_y)

                lo_pred, hi_pred = np.quantile(y_draws, [q_lo, q_hi], axis=0)
                ax.fill_between(
                    t_grid,
                    lo_pred,
                    hi_pred,
                    alpha=0.15,
                    color=st["color"],
                    label=None,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"Bayesian Power–Exponential Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        # de-dup legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # TTF at use (Bayesian posterior)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution(
        self,
        Df=None,
        S_eval=None,
        n_samps=8000,
        unit_label="time units",
    ):
        """
        Draw posterior TTF samples at use stress (or given S_eval),
        print summary, and plot histogram + Weibull/Gamma/Lognormal fits.

        Uses Bayesian posterior samples of (b, a, n, sigma).
        """
        if not hasattr(self, "b_s"):
            print("Bayesian posterior not available; cannot compute TTF distribution.")
            return

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if S_eval is None:
            S_eval = self.S_use

        theta_samples = np.column_stack([self.b_s, self.a_s, self.n_s])
        sigma_samples = np.asarray(self.sigma_s, float)

        def mu_fun_factory(theta_vec):
            b, a, n = theta_vec

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 0.0, None)
                return b * np.exp(a * S_eval) * (t_pos ** n)

            return mu_fun

        # 1) Sample TTF posterior
        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        # 2) Summarise
        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at S={S_eval:g} (Df={Df})",
            unit_label=unit_label,
            as_markdown=True,
        )

        # 3) Histogram + fitted distributions
        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF posterior and fits at S={S_eval:g}",
            bins=60,
        )

    # ------------------------------------------------------------------
    # TTF at use (MLE plug-in version)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution_MLE(
        self,
        Df=None,
        S_eval=None,
        n_samps=8000,
        unit_label="time units",
    ):
        """
        Draw TTF samples at use stress (or S_eval) using *MLE* parameters.

        This treats (b, a, n, sigma) as known (plug-in),
        so variability is only from the stochastic noise model.
        """
        if (self.theta_MLE is None) or (self.sigma_MLE is None):
            raise RuntimeError("MLE fit not available – run method='MLE' first.")

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if S_eval is None:
            S_eval = self.S_use

        b_hat, a_hat, n_hat = self.theta_MLE
        sigma_hat = float(self.sigma_MLE)

        # Build pseudo "posterior samples" by repeating the MLE
        theta_samples = np.tile(
            np.array([b_hat, a_hat, n_hat], float)[None, :],
            (n_samps, 1)
        )
        sigma_samples = np.full(n_samps, sigma_hat, float)

        def mu_fun_factory(theta_vec):
            b, a, n = theta_vec

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 0.0, None)
                expo = np.clip(a * S_eval, -50.0, 50.0)   # numerical guardrail
                return b * np.exp(expo) * (t_pos ** n)

            return mu_fun

        # 1) Sample TTFs
        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        # 2) Summarise
        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at S={S_eval:g} (Df={Df}, MLE plug-in)",
            unit_label=unit_label,
            as_markdown=True,
        )

        # 3) Histogram + fitted distributions
        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF (MLE plug-in) at S={S_eval:g}",
            bins=60,
        )


class Fit_ADT_Power_Power(Base_ADT_Model):
    r"""
    Power–Power *damage* model:

        D(S, t) = b * S^n * t^m

    where
        - D(S, t) : damage (increasing over time)
        - S       : stress (e.g. applied weight in grams)
        - t       : time / cycles
        - b > 0   : scale parameter
        - n > 0   : stress exponent
        - m > 0   : time exponent

    Supports:
        - LSQ fit
        - MLE (additive or multiplicative noise on D)
        - Bayesian fit via emcee
        - Optional TTF-at-use histogram from Bayesian posterior
    """

    def __init__(
        self,
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
    ):
        """
        Parameters
        ----------
        degradation : array-like
            Observed damage D_ij.
        stress : array-like
            Stress level S_ij for each observation (same shape as degradation).
        time : array-like
            Time values (e.g. cycles).
        unit : array-like
            Unit IDs (integers).
        stress_use : float
            Use-stress value S_use at which TTF is to be evaluated (for TTF plots).
        Df : float
            Failure threshold on *damage* scale (fail when D >= Df).
        CI : float
            Confidence/credible level (default 0.95).
        method : {"LS", "MLE", "Bayesian"}
            Fit method. (No hierarchical version here yet.)
        noise : {"additive", "multiplicative"}
            Observation noise model on D.
        show_data_plot : bool
            If True, scatter/line plot of D vs t by stress.
        show_LSQ_diagnostics : bool
            If True, show LS residual diagnostics.
        show_noise_bounds : bool
            If True, show noise bands on MLE & Bayes plots.
        show_use_TTF_dist : bool
            If True (and method is Bayesian), compute and plot TTF at use.
        print_results : bool
            If True, print parameter tables in Markdown.
        data_scale : {"linear","ylog","xlog","loglog"}
            Axis scaling for the data plot.
        kwargs :
            Additional options passed through to Base_ADT_Model.
        """
        method_norm = method.lower()

        # 1) Base: y, t, unit, noise, etc.
        super().__init__(
            y=degradation,
            time=time,
            unit=unit,
            CI=CI,
            method=method,
            noise=noise,
            scale_type="damage",
            Df=Df,
            show_LSQ_diagnostics=show_LSQ_diagnostics,
            show_noise_bounds=show_noise_bounds,
            show_use_TTF_dist=show_use_TTF_dist,
            print_results=print_results,
            **kwargs,
        )

        # 2) Model-specific: stress
        self.S = np.asarray(stress, float).ravel()
        if self.S.shape != self.y.shape:
            raise ValueError("stress must have the same shape as degradation.")
        self.S_use = float(stress_use)   # only used for TTF-at-use
        self.data_scale = data_scale
        self.priors = priors  # may be None or a dict
        self.param_names = ["b", "n (stress)", "m (time)"]

        # 3) Optional raw data plot
        if show_data_plot:
            self.plot_data(scale=self.data_scale)

        # 4) Method-specific pipeline
        if method_norm == "ls":
            self._fit_LS(suppress_print=False)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._plot_fit_LS()

        elif method_norm == "mle":
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE()
            self._plot_fit_MLE()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution_MLE()            

        elif method_norm.startswith("bayes"):
            # LS (quiet) → MLE (quiet) → Bayes
            self._fit_LS(suppress_print=True)
            if self.show_LSQ_diagnostics:
                self._plot_LS_diagnostics()
            self._fit_MLE(suppress_print=True)
            self._fit_Bayesian()
            self._plot_fit_Bayesian()
            if self.show_use_TTF_dist:
                self._plot_use_TTF_distribution()

        else:
            raise ValueError(f"Unknown method: {method}")
            
    # ------------------------------------------------------------------
    # Model Description
    # ------------------------------------------------------------------         
    def _model_description(self):
        return r"""
### Degradation / damage model

**Model form (damage scale)**

The Power–Power model uses power laws in both stress and time:

$$
D(t, S)
= b \, S^{n} \, t^{m}.
$$

**Parameters:**

- **$b$** – scale factor on the overall damage level  
- **$n$** – stress exponent (how strongly damage scales with stress \(S\))  
- **$m$** – time exponent (how quickly damage accelerates with time)  
- **$S$** – accelerating stress (e.g. load, current, pressure)  
"""

    # ------------------------------------------------------------------
    # Data plot wrapper
    # ------------------------------------------------------------------
    def plot_data(self, scale=None):
        if scale is None:
            scale = getattr(self, "data_scale", "linear")

        plot_degradation_by_stress(
            t=self.t,
            D=self.y,
            stress=self.S,
            unit=self.unit,
            title="Degradation vs Time",
            stress_label="Stress",
            legend_title="Stress",
            scale=scale,
            show_unit_lines=True,
        )

    # ------------------------------------------------------------------
    # Model mean on data scale: D = b * S^n * t^m
    # ------------------------------------------------------------------
    def _mu(self, theta, t):
        b, n_S, m_t = theta
        t_pos = np.clip(t, 1e-12, None)
        S_pos = np.clip(self.S, 1e-12, None)
        return b * (S_pos ** n_S) * (t_pos ** m_t)

    # ------------------------------------------------------------------
    # LSQ init & bounds
    # ------------------------------------------------------------------
    def _init_guess(self):
        """
        Heuristic initial guess for [b, n_S, m_t].
        """
        b_init = max(float(np.percentile(self.y, 10)), 1e-6)

        # crude slope-based guesses on log-log for time exponent
        if np.all(self.t > 0):
            log_t = np.log(self.t)
            log_y = np.log(np.clip(self.y, 1e-12, None))
            m_t_init = max(
                (np.percentile(log_y, 90) - np.percentile(log_y, 10))
                / max(np.percentile(log_t, 90) - np.percentile(log_t, 10), 1e-6),
                0.1,
            )
        else:
            m_t_init = 0.5

        # stress exponent start at 1
        n_S_init = 1.0

        return np.array([b_init, n_S_init, m_t_init], dtype=float)

    def _get_LS_bounds(self, p):
        # [b, n_S, m_t]: all > 0
        return [
            (1e-10, None),  # b > 0
            (1e-6, None),   # n_S > 0
            (1e-6, None),   # m_t > 0
        ]

    # ------------------------------------------------------------------
    # Negative log-likelihood for MLE (Base calls self._negloglik)
    # ------------------------------------------------------------------
    def _negloglik(self, x):
        """
        x = [b, n_S, m_t, log_sigma]
        """
        b, n_S, m_t = x[:-1]
        log_sigma = x[-1]
        sigma = np.exp(log_sigma)

        # basic validity checks
        if sigma <= 0 or b <= 0 or n_S <= 0 or m_t <= 0:
            return np.inf

        theta = np.array([b, n_S, m_t], float)
        mu = self._mu(theta, self.t)

        if self.noise == "additive":
            if not np.all(np.isfinite(mu)):
                return np.inf
            return -np.sum(norm.logpdf(self.y, loc=mu, scale=sigma))

        elif self.noise == "multiplicative":
            D_pos = np.clip(self.y, 1e-12, None)
            mu_pos = np.clip(mu, 1e-12, None)
            logD = np.log(D_pos)
            logmu = np.log(mu_pos)
            z = (logD - logmu) / sigma

            if not np.all(np.isfinite(z)):
                return np.inf

            # lognormal jacobian + normal on log-scale
            return (
                np.sum(np.log(D_pos)) +
                len(D_pos) * np.log(sigma) -
                np.sum(norm.logpdf(z))
            )

        else:
            raise ValueError(f"Unknown noise model: {self.noise}")

    # ------------------------------------------------------------------
    # LS diagnostics & plots
    # ------------------------------------------------------------------
    def _plot_LS_diagnostics(self):
        if self.theta_LS is None:
            return
        theta = self.theta_LS
        mu = self._mu(theta, self.t)
        resid = self.y - mu

        plot_residual_diagnostics(
            t=self.t,
            stress=self.S,
            resid=resid,
            mu=mu,
            title_prefix="LSQ diagnostics (Power–Power model)",
            stress_label="Stress",
        )

    def _plot_fit_LS(self):
        if self.theta_LS is None:
            return

        b, n_S, m_t = self.theta_LS
        fig, ax = plt.subplots(figsize=(8, 6))

        stresses = np.unique(self.S)

        # Fixed palette + markers (shared with MLE for consistency)
        base_colors  = ["blue", "orange", "k", "green", "red", "purple"]
        base_markers = ["o", "s", "^", "D", "v", "P"]

        styles = {}
        for i, S in enumerate(stresses):
            idx = min(i, len(base_colors) - 1)
            styles[S] = dict(color=base_colors[idx],
                             marker=base_markers[idx])

        # Data
        for S in stresses:
            mask = (self.S == S)
            st = styles[S]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=25,
                alpha=0.8,
                color=st["color"],
                marker=st["marker"],
                label=f"S={S:g} data",
            )

        # Model curves
        t_grid = np.linspace(0.0, float(self.t.max()) * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        for S in stresses:
            st = styles[S]
            S_pos = max(S, 1e-12)
            mu_grid = b * (S_pos ** n_S) * (t_pos ** m_t)

            ax.plot(
                t_grid,
                mu_grid,
                color=st["color"],
                lw=2,
                label=f"S={S:g} LS mean",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title("LSQ Power–Power Degradation Fit")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        # De-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=2)

        plt.tight_layout()
        plt.show()


    # ------------------------------------------------------------------
    # MLE plot
    # ------------------------------------------------------------------
    def _plot_fit_MLE(self):
        if self.theta_MLE is None:
            return

        b, n_S, m_t = self.theta_MLE
        sigma = self.sigma_MLE

        fig, ax = plt.subplots(figsize=(8, 6))

        stresses = np.unique(self.S)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, S in enumerate(stresses):
            idx = min(i, len(base_colors) - 1)
            styles[S] = dict(color=base_colors[idx], marker=base_markers[idx])

        # data
        for S in stresses:
            mask = (self.S == S)
            st = styles[S]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"S={S:g} data",
            )

        # curves + noise bands
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        for S in stresses:
            st = styles[S]
            S_pos = max(S, 1e-12)
            mu_grid = b * (S_pos ** n_S) * (t_pos ** m_t)

            ax.plot(
                t_grid,
                mu_grid,
                color=st["color"],
                lw=2,
                label=f"S={S:g} MLE mean",
            )

            if self.show_noise_bounds and sigma is not None:
                if self.noise == "additive":
                    lower = mu_grid - self.z_CI * sigma
                    upper = mu_grid + self.z_CI * sigma
                else:
                    factor_noise = np.exp(self.z_CI * sigma)
                    lower = mu_grid / factor_noise
                    upper = mu_grid * factor_noise

                ax.fill_between(
                    t_grid,
                    lower,
                    upper,
                    color=st["color"],
                    alpha=0.2,
                    label=f"S={S:g} {int(100*self.CI)}% band",
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"MLE Power–Power Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(3, len(uniq)))

        plt.tight_layout()
        plt.show()
        
    # ------------------------------------------------------------------
    # Bayesian fit: log b, log n_S, log m_t, log σ ~ Normal
    # with optional custom priors on natural parameters.
    # ------------------------------------------------------------------
    def _fit_Bayesian(self, nwalkers=32, nburn=1000, nsamp=2000):
        """
        Bayesian fit for Power–Power ADT using Base_ADT_Model._fit_Bayesian_generic.

        - If self.priors is None:
            independent Normal priors on transformed coords
            (log b, log n_S, log m_t, log sigma) centred at MLE/LS.
        - If self.priors is a dict:
            priors on NATURAL scale for
            {"b", "n_S", "m_t", "sigma"} dispatched via log_prior_vector
            with model="Power_Power_ADT", dist="NormalError".
        """
        # 1) Prior centres from MLE, fallback to LS
        if self.theta_MLE is not None and self.sigma_MLE is not None:
            b_hat, nS_hat, m_hat = self.theta_MLE
            sig_hat = self.sigma_MLE
        elif self.theta_LS is not None and self.sigma_LS is not None:
            b_hat, nS_hat, m_hat = self.theta_LS
            sig_hat = self.sigma_LS
        else:
            raise RuntimeError("Run LS or MLE before Bayesian (no initial estimates).")

        default_centers = {
            "b":     float(b_hat),
            "n_S":   float(nS_hat),
            "m_t":   float(m_hat),
            "sigma": float(sig_hat),
        }

        # Transform-space prior sd for default (log / identity coords)
        default_sds = {
            "b":     0.5,
            "n_S":   0.5,
            "m_t":   0.5,
            "sigma": 0.5,
        }

        t_time = self.t
        D_obs = self.y

        def loglike(theta_dict):
            b     = theta_dict["b"]
            n_S   = theta_dict["n_S"]
            m_t   = theta_dict["m_t"]
            sigma = theta_dict["sigma"]

            # basic validity
            if sigma <= 0 or b <= 0:
                return -np.inf

            theta = np.array([b, n_S, m_t], float)
            mu = self._mu(theta, t_time)  # uses self.S internally

            if self.noise == "additive":
                if not np.all(np.isfinite(mu)):
                    return -np.inf
                return np.sum(norm.logpdf(D_obs, loc=mu, scale=sigma))

            elif self.noise == "multiplicative":
                D_pos  = np.clip(D_obs, 1e-12, None)
                mu_pos = np.clip(mu,   1e-12, None)
                logD   = np.log(D_pos)
                logmu  = np.log(mu_pos)
                if not np.all(np.isfinite(logmu)):
                    return -np.inf
                z = (logD - logmu) / sigma
                return (
                    -np.sum(np.log(D_pos))
                    - len(D_pos) * np.log(sigma)
                    + np.sum(norm.logpdf(z))
                )
            else:
                return -np.inf

        param_names = ["b", "n_S", "m_t", "sigma"]

        sampler, chain, nat_samples = self._fit_Bayesian_generic(
            param_names=param_names,
            model_name="Power_Power_ADT",
            default_centers=default_centers,
            default_sds=default_sds,
            loglike_func=loglike,
            priors=getattr(self, "priors", None),
            dist_name="NormalError",
            nwalkers=nwalkers,
            nburn=nburn,
            nsamp=nsamp,
        )

        # Store posterior draws on NATURAL scale
        self.b_s     = nat_samples["b"]
        self.n_S_s   = nat_samples["n_S"]
        self.m_t_s   = nat_samples["m_t"]
        self.sigma_s = nat_samples["sigma"]

        # Markdown table
        if self.print_results:
            self._show_model_description() 
            md_param_table_bayes(
                {
                    "b": self.b_s,
                    "n (stress)": self.n_S_s,
                    "m (time)": self.m_t_s,
                    r"$\sigma$": self.sigma_s,
                },
                cred_mass=self.CI,
                label=f"Bayesian ({self.noise})",
            )



    # ------------------------------------------------------------------
    # Bayesian plot
    # ------------------------------------------------------------------
    def _plot_fit_Bayesian(self, nsamples=1000, show_predictive=True):
        if not hasattr(self, "b_s"):
            return

        fig, ax = plt.subplots(figsize=(6, 5))

        stresses = np.unique(self.S)
        base_colors = ["blue", "orange", "k", "green", "red"]
        base_markers = ["o", "s", "^", "D", "v"]

        styles = {}
        for i, S in enumerate(stresses):
            idx = min(i, len(base_colors) - 1)
            styles[S] = dict(color=base_colors[idx], marker=base_markers[idx])

        # data
        for S in stresses:
            mask = (self.S == S)
            st = styles[S]
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=35,
                alpha=0.9,
                color=st["color"],
                marker=st["marker"],
                label=f"S={S:g} data",
            )

        # thin posterior
        rng = np.random.default_rng(7)
        n_post = len(self.b_s)
        ns = min(nsamples, n_post)
        sel = rng.choice(n_post, size=ns, replace=False)

        _b   = self.b_s[sel]
        _nS  = self.n_S_s[sel]
        _m   = self.m_t_s[sel] 
        _sig = self.sigma_s[sel]

        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        alpha = self.alpha
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        for S in stresses:
            st = styles[S]
            S_pos = max(S, 1e-12)

            mu_draws = _b[:, None] * (S_pos ** _nS[:, None]) * (t_pos[None, :] ** _m[:, None])

            mu_mean = mu_draws.mean(axis=0)
            lo_mean, hi_mean = np.quantile(mu_draws, [q_lo, q_hi], axis=0)

            ax.plot(
                t_grid,
                mu_mean,
                color=st["color"],
                lw=2,
                label=f"S={S:g} Bayes mean",
            )

            if show_predictive:
                eps = rng.standard_normal(size=mu_draws.shape)

                if self.noise == "additive":
                    y_draws = mu_draws + _sig[:, None] * eps
                else:
                    mu_clip = np.clip(mu_draws, 1e-12, None)
                    log_y = np.log(mu_clip) + _sig[:, None] * eps
                    y_draws = np.exp(log_y)

                lo_pred, hi_pred = np.quantile(y_draws, [q_lo, q_hi], axis=0)
                ax.fill_between(
                    t_grid,
                    lo_pred,
                    hi_pred,
                    alpha=0.15,
                    color=st["color"],
                    label=None,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title(f"Bayesian Power–Power Fit ({self.noise} noise)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

        # de-dup legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h
        ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=min(2, len(uniq)))

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # TTF at use (Bayesian posterior)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution(
        self,
        Df=None,
        S_eval=None,
        n_samps=8000,
        unit_label="time units",
    ):
        """
        Draw posterior TTF samples at use stress (or given S_eval),
        print summary, and plot histogram + Weibull/Gamma/Lognormal fits.

        Uses Bayesian posterior samples of (b, n_S, m, sigma).
        """
        if not hasattr(self, "b_s"):
            print("Bayesian posterior not available; cannot compute TTF distribution.")
            return

        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        if S_eval is None:
            S_eval = self.S_use

        theta_samples = np.column_stack([self.b_s, self.n_S_s, self.m_t_s])
        sigma_samples = np.asarray(self.sigma_s, float)

        def mu_fun_factory(theta_vec):
            b, n_S, m_t = theta_vec

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 1e-12, None)
                S_pos = max(S_eval, 1e-12)
                return b * (S_pos ** n_S) * (t_pos ** m_t)

            return mu_fun

        # 1) Sample TTF posterior
        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        # 2) Summarise
        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at S={S_eval:g} (Df={Df})",
            unit_label=unit_label,
            as_markdown=True,
        )

        # 3) Histogram + fitted distributions
        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF posterior and fits at S={S_eval:g}",
            bins=60,
        )
    # ------------------------------------------------------------------
    # TTF at use (MLE plug-in version)
    # ------------------------------------------------------------------
    def _plot_use_TTF_distribution_MLE(
        self,
        Df=None,
        S_eval=None,
        n_samps=8000,
        unit_label="time units",
    ):
        """
        Draw TTF samples at use stress (or S_eval) using *MLE* parameters.

        This treats (b, n_S, m, sigma) as known (plug-in),
        so variability is only from the stochastic noise model.
        """
        if (self.theta_MLE is None) or (self.sigma_MLE is None):
            raise RuntimeError("MLE fit not available – run method='MLE' first.")

        # Failure threshold
        if Df is None:
            if self.Df_use is None:
                raise ValueError("Df (failure threshold) is not specified.")
            Df = self.Df_use

        # Stress at which to evaluate TTF
        if S_eval is None:
            S_eval = self.S_use

        b_hat, nS_hat, m_hat = self.theta_MLE
        sigma_hat = float(self.sigma_MLE)

        # Pseudo "posterior" samples from MLE
        theta_samples = np.tile(
            np.array([b_hat, nS_hat, m_hat], float)[None, :],
            (n_samps, 1),
        )
        sigma_samples = np.full(n_samps, sigma_hat, float)

        def mu_fun_factory(theta_vec):
            b, n_S, m_t = theta_vec

            def mu_fun(t):
                t_arr = np.asarray(t, float)
                t_pos = np.clip(t_arr, 1e-12, None)
                S_pos = max(S_eval, 1e-12)
                return b * (S_pos ** n_S) * (t_pos ** m_t)

            return mu_fun

        # 1) Sample TTFs
        ttf_samples = sample_ttf_from_posterior(
            mu_fun_factory=mu_fun_factory,
            theta_samples=theta_samples,
            sigma_samples=sigma_samples,
            threshold=Df,
            noise_type=self.noise,
            scale_type="damage",
            n_samps=n_samps,
            seed=24,
        )

        # 2) Summarise
        summarize_ttf(
            ttf_samples,
            hdi_prob=self.CI,
            label=f"TTF at S={S_eval:g} (Df={Df}, MLE plug-in)",
            unit_label=unit_label,
            as_markdown=True,
        )

        # 3) Histogram + fitted distributions
        plot_ttf_hist_with_fits(
            ttf_samples,
            unit_label=unit_label,
            title=f"TTF (MLE plug-in) at S={S_eval:g}",
            bins=60,
        )

class Fit_Logistic_POD:
    """
    Fit a (possibly left-truncated) log–logistic Probability of Detection (POD) model
    using LSQ, MLE, or Bayesian (via MCMC) methods.

    Model (plain log–logistic):
        POD_plain(x) = expit(beta0 + beta1 * ln(x))

    If a_lth is provided, a left-truncated POD is used:
        POD_trunc(x) = 0,                           x < a_lth
                     = [POD_plain(x) - POD_plain(a_lth)] / (1 - POD_plain(a_lth)), x >= a_lth

    Parameters
    ----------
    D_measured : array-like
        Measured indication from the NDE system. Any entry that is NaN, "N/A", "-", or "---"
        is interpreted as *no damage detected* (i.e. detection = 0). All other entries
        are treated as *detected* (detection = 1).
    D_true : array-like
        True flaw size (or other ground-truth damage metric). Must be convertible to float.
    a_lth : float or None, optional
        Left truncation threshold. If None, the plain (untruncated) log–logistic POD is used.
    method : {"LSQ","MLE","Bayesian"}
        Estimation method.
    CI : float, optional
        Confidence / credible level for intervals (e.g. 0.95).
    print_results : bool
        If True, print a markdown table summarizing parameter estimates.
    plot_results : bool
        If True, generate a POD plot with data and fitted curve(s).
    random_state : int or None
        Seed for Bayesian MCMC.
    """

    def __init__(
        self,
        D_measured,
        D_true,
        a_lth=None,
        method="MLE",
        CI=0.95,
        print_results=True,
        plot_results=True,
        random_state=None,
    ):
        self.method = str(method).lower()
        if self.method not in {"lsq", "mle", "bayesian"}:
            raise ValueError("method must be one of: 'LSQ', 'MLE', 'Bayesian'")

        self.CI = float(CI)
        if not (0.0 < self.CI < 1.0):
            raise ValueError("CI must be in (0,1)")

        self.a_lth = a_lth
        self.print_results = bool(print_results)
        self.plot_results = bool(plot_results)
        self.random_state = random_state

        # Pre-process data
        self._prepare_data(D_measured, D_true)

        # Storage for results
        self.result_lsq = None
        self.result_mle = None
        self.result_bayes = None

        # Fit
        if self.method == "lsq":
            self._fit_lsq()
        elif self.method == "mle":
            self._fit_mle()
        elif self.method == "bayesian":
            self._fit_bayesian()

        # Print & plot
        if self.print_results:
            self._print_summary_markdown()
        if self.plot_results:
            self._plot_pod()

    # ------------------------------------------------------------------
    # Data prep & POD helpers
    # ------------------------------------------------------------------
    def _prepare_data(self, D_measured, D_true):
        D_true = np.asarray(D_true, dtype=float)

        # Interpret measured indications
        measured_series = pd.Series(D_measured).astype(str).str.strip()

        # Values that mean "no detection"
        no_det_tokens = {"", "nan", "none", "n/a", "na", "-", "--", "---"}
        detected = ~measured_series.str.lower().isin(no_det_tokens)
        detected = detected.to_numpy().astype(int)

        # Guard: drop rows where D_true is NaN
        mask = np.isfinite(D_true)
        self.D_true = D_true[mask]
        self.detected = detected[mask]

        # Group by D_true for grouped binomial LSQ / MLE
        df = pd.DataFrame({"D_true": self.D_true, "detected": self.detected})
        agg = (
            df.groupby("D_true")["detected"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "n_detected", "count": "n_total"})
        )
        agg["p_obs"] = agg["n_detected"] / agg["n_total"]
        self.agg = agg

    def _pod_plain(self, x, b0, b1):
        x = np.asarray(x)
        x_safe = np.maximum(x, 1e-12)
        return expit(b0 + b1 * np.log(x_safe))

    def _pod(self, x, b0, b1):
        """
        Wrapper that applies left truncation if a_lth is not None.
        """
        x = np.asarray(x)
        if self.a_lth is None:
            return self._pod_plain(x, b0, b1)
        # Left-truncated version
        pod_plain = self._pod_plain(x, b0, b1)
        pod_alth = self._pod_plain(self.a_lth, b0, b1)
        # Avoid division by zero
        denom = np.clip(1.0 - pod_alth, 1e-12, 1.0)
        pod_trunc = (pod_plain - pod_alth) / denom
        pod_trunc = np.where(x < self.a_lth, 0.0, pod_trunc)
        return np.clip(pod_trunc, 1e-12, 1.0 - 1e-12)

    # ------------------------------------------------------------------
    # LSQ
    # ------------------------------------------------------------------
    def _fit_lsq(self):
        x = self.agg["D_true"].to_numpy()
        p_obs = self.agg["p_obs"].to_numpy()

        def residuals(params):
            b0, b1 = params
            p_pred = self._pod(x, b0, b1)
            return p_pred - p_obs

        # Simple starting guess: logistic on log(x)
        x_log = np.log(np.maximum(x, 1e-12))
        # Avoid perfect separation issues in tiny samples – use linear regression for init
        A = np.vstack([np.ones_like(x_log), x_log]).T
        # Clip p_obs to (0,1) for logit
        p_clip = np.clip(p_obs, 1e-4, 1 - 1e-4)
        y = np.log(p_clip / (1 - p_clip))
        try:
            beta_init, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            beta_init = np.array([0.0, 1.0])

        res = least_squares(residuals, x0=beta_init, method="trf")
        self.result_lsq = {
            "beta0": res.x[0],
            "beta1": res.x[1],
            "success": res.success,
            "cost": res.cost,
        }

    # ------------------------------------------------------------------
    # MLE
    # ------------------------------------------------------------------
    def _neg_log_likelihood(self, params):
        b0, b1 = params
        x = self.agg["D_true"].to_numpy()
        n_det = self.agg["n_detected"].to_numpy()
        n_tot = self.agg["n_total"].to_numpy()

        p = self._pod(x, b0, b1)
        # clamp for stability
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ll = n_det * np.log(p) + (n_tot - n_det) * np.log(1 - p)
        return -np.sum(ll)

    def _fit_mle(self):
        # Use LSQ estimate as starting point if available
        if self.result_lsq is None:
            self._fit_lsq()
        beta_init = np.array(
            [self.result_lsq["beta0"], self.result_lsq["beta1"]], dtype=float
        )

        res = minimize(
            self._neg_log_likelihood,
            x0=beta_init,
            method="BFGS",
        )

        if not res.success:
            # Still store best effort
            beta_hat = res.x
        else:
            beta_hat = res.x

        # Approximate covariance from inverse Hessian
        H_inv = res.hess_inv
        if hasattr(H_inv, "todense"):
            H_inv = np.array(H_inv.todense())
        cov = np.array(H_inv, dtype=float).reshape(2, 2)

        se = np.sqrt(np.diag(cov))
        z = norm.ppf(0.5 + self.CI / 2.0)
        ci_low = beta_hat - z * se
        ci_high = beta_hat + z * se

        self.result_mle = {
            "beta0": beta_hat[0],
            "beta1": beta_hat[1],
            "cov": cov,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "success": res.success,
            "fun": res.fun,
        }

    # ------------------------------------------------------------------
    # Bayesian (emcee)
    # ------------------------------------------------------------------
    def _log_prior(self, theta):
        b0, b1 = theta
        # Simple weakly-informative uniform priors
        if -20.0 < b0 < 20.0 and 0.01 < b1 < 20.0:
            return 0.0
        return -np.inf

    def _log_likelihood_bayes(self, theta):
        b0, b1 = theta
        x = self.agg["D_true"].to_numpy()
        n_det = self.agg["n_detected"].to_numpy()
        n_tot = self.agg["n_total"].to_numpy()
        p = self._pod(x, b0, b1)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ll = n_det * np.log(p) + (n_tot - n_det) * np.log(1 - p)
        return np.sum(ll)

    def _log_posterior(self, theta):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood_bayes(theta)

    def _fit_bayesian(self):
        try:
            import emcee
            import arviz as az
        except ImportError as e:
            raise ImportError(
                "Bayesian fitting requires 'emcee' and 'arviz'. "
                "Install them to use method='Bayesian'."
            ) from e

        # Make sure we have an MLE first (for a sensible starting point)
        if self.result_mle is None:
            self._fit_mle()

        # Centre walkers at the MLE (β0, β1)
        b0_hat = self.result_mle["beta0"]
        b1_hat = self.result_mle["beta1"]
        theta_center = np.array([b0_hat, b1_hat], dtype=float)

        rng = np.random.default_rng(self.random_state)
        ndim = 2          # (β0, β1)
        nwalkers = 32
        # Small Gaussian ball around the MLE
        p0 = theta_center + 1e-2 * rng.normal(size=(nwalkers, ndim))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_posterior)

        n_burn = 2000
        n_sample = 2000
        sampler.run_mcmc(p0, n_burn + n_sample, progress=True)

        # Flatten chains after burn-in: shape (n_samples, 2)
        chain = sampler.get_chain(discard=n_burn, flat=True)

        # Basic summaries
        mean = np.mean(chain, axis=0)
        median = np.median(chain, axis=0)

        # Equal-tail interval
        alpha_pct = (1.0 - self.CI) * 100.0 / 2.0  # e.g. 2.5 for 95% CI
        eti_lo, eti_hi = np.percentile(
            chain,
            [alpha_pct, 100.0 - alpha_pct],
            axis=0,
        )

        # HDI per parameter
        ndim = chain.shape[1]
        hdi_lo = np.empty(ndim)
        hdi_hi = np.empty(ndim)
        for j in range(ndim):
            hdi_j = az.hdi(chain[:, j], hdi_prob=self.CI)
            hdi_j = np.asarray(hdi_j)
            hdi_lo[j] = hdi_j[0]
            hdi_hi[j] = hdi_j[1]

        names = [r"$\beta_0$", r"$\beta_1$"]

        self.result_bayes = {
            "samples": chain,
            "names": names,
            "mean": mean,
            "median": median,
            "eti_low": eti_lo,
            "eti_high": eti_hi,
            "hdi_low": hdi_lo,
            "hdi_high": hdi_hi,
        }
    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _print_summary_markdown(self):
        from IPython.display import Markdown, display

        lines = []

        # --- NEW: POD model statement printed once at the top ---
        lines.append("### Probability of Detection (POD) Model: Logistic")
        lines.append("")
        lines.append(
            r"Plain log–logistic POD: "
            r"$POD_{\text{plain}}(x) = \mathrm{expit}\big(\beta_0 + \beta_1 \ln x\big)$."
        )

        if self.a_lth is not None:
            lines.append("")
            lines.append(
                r"Left–truncated at $a_{lth}$: "
                r"$POD(x) = 0,\ x < a_{lth};\quad "
                r"POD(x) = \dfrac{POD_{\text{plain}}(x) - POD_{\text{plain}}(a_{lth})}"
                r"{1 - POD_{\text{plain}}(a_{lth})},\ x \ge a_{lth}$."
            )

        lines.append("")

        # --- LSQ block (unchanged) ---
        if self.result_lsq is not None:
            lines.append("### LSQ estimates")
            lines.append("")
            lines.append("| Parameter | Estimate |")
            lines.append("|-----------|----------|")
            lines.append(f"| $\\beta_0$ | {self.result_lsq['beta0']:.4f} |")
            lines.append(f"| $\\beta_1$ | {self.result_lsq['beta1']:.4f} |")
            lines.append("")

        # --- MLE block (unchanged) ---
        if self.result_mle is not None:
            beta0 = self.result_mle["beta0"]
            beta1 = self.result_mle["beta1"]
            se0, se1 = self.result_mle["se"]
            ci0_lo, ci1_lo = self.result_mle["ci_low"]
            ci0_hi, ci1_hi = self.result_mle["ci_high"]

            lines.append("### MLE estimates")
            lines.append("")
            lines.append(
                f"| Parameter | Estimate | Std. Error | {int(self.CI*100)}% CI low | {int(self.CI*100)}% CI high |"
            )
            lines.append("|-----------|----------|------------|----------------|-----------------|")
            lines.append(
                f"| $\\beta_0$ | {beta0:.4f} | {se0:.4f} | {ci0_lo:.4f} | {ci0_hi:.4f} |"
            )
            lines.append(
                f"| $\\beta_1$ | {beta1:.4f} | {se1:.4f} | {ci1_lo:.4f} | {ci1_hi:.4f} |"
            )
            lines.append("")

        # --- Bayesian block (unchanged) ---
        if self.result_bayes is not None:
            mean = self.result_bayes["mean"]
            median = self.result_bayes["median"]
            eti_lo = self.result_bayes["eti_low"]
            eti_hi = self.result_bayes["eti_high"]
            hdi_lo = self.result_bayes["hdi_low"]
            hdi_hi = self.result_bayes["hdi_high"]

            lines.append("### Bayesian posterior summaries")
            lines.append("")
            lines.append(
                f"| Parameter | Post. mean | Post. median | {int(self.CI*100)}% ETI low | {int(self.CI*100)}% ETI high | {int(self.CI*100)}% HDI low | {int(self.CI*100)}% HDI high |"
            )
            lines.append(
                "|-----------|------------|--------------|-----------------|------------------|------------------|-------------------|"
            )
            lines.append(
                f"| $\\beta_0$ | {mean[0]:.4f} | {median[0]:.4f} | {eti_lo[0]:.4f} | {eti_hi[0]:.4f} | {hdi_lo[0]:.4f} | {hdi_hi[0]:.4f} |"
            )
            lines.append(
                f"| $\\beta_1$ | {mean[1]:.4f} | {median[1]:.4f} | {eti_lo[1]:.4f} | {eti_hi[1]:.4f} | {hdi_lo[1]:.4f} | {hdi_hi[1]:.4f} |"
            )
            lines.append("")

        if lines:
            md = "\n".join(lines)
            display(Markdown(md))

    def _plot_pod(self):
        x = self.agg["D_true"].to_numpy()
        p_obs = self.agg["p_obs"].to_numpy()
        n_tot = self.agg["n_total"].to_numpy()

        # Grid for smooth curve
        x_grid = np.linspace(np.min(x)*0.9, np.max(x)*1.05, 400)

        plt.figure(figsize=(7, 5))

        # Plot empirical detection probabilities
        # Marker size proportional to sample size
        s = 20 + 5 * n_tot
        plt.scatter(x, p_obs, s=s, alpha=0.7, label="Empirical POD")

        # LSQ curve
        if self.result_lsq is not None:
            y_lsq = self._pod(x_grid, self.result_lsq["beta0"], self.result_lsq["beta1"])
            plt.plot(x_grid, y_lsq, label="LSQ fit", linestyle="--")

        # MLE curve + (asymptotic) CI band
        if self.result_mle is not None:
            beta0_hat = self.result_mle["beta0"]
            beta1_hat = self.result_mle["beta1"]
            y_mle = self._pod(self._ensure_positive(x_grid), beta0_hat, beta1_hat)
            plt.plot(x_grid, y_mle, label="MLE fit", linewidth=2)

            cov = self.result_mle.get("cov", None)
            if cov is not None:
                try:
                    # Draw from N(theta_hat, cov) to get an MLE-based CI envelope
                    rng = np.random.default_rng(self.random_state)
                    n_draws = 3000
                    thetas = rng.multivariate_normal(
                        mean=np.array([beta0_hat, beta1_hat]),
                        cov=cov,
                        size=n_draws,
                    )

                    P_mle = []
                    for b0_j, b1_j in thetas:
                        P_mle.append(self._pod(x_grid, b0_j, b1_j))
                    P_mle = np.vstack(P_mle)

                    alpha_pct = (1 - self.CI) * 50
                    mle_lo = np.percentile(P_mle, alpha_pct, axis=0)
                    mle_hi = np.percentile(P_mle, 100 - alpha_pct, axis=0)

                    plt.fill_between(
                        x_grid,
                        mle_lo,
                        mle_hi,
                        alpha=0.15,
                        label=f"MLE {int(self.CI*100)}% CI",
                    )
                except np.linalg.LinAlgError:
                    # If covariance is not positive-definite, just skip the band
                    pass

        # Bayesian curve + credible band
        if self.result_bayes is not None:
            samples = self.result_bayes["samples"]
            # Thin if huge
            nsamp = min(3000, samples.shape[0])
            idx = np.random.default_rng(self.random_state).choice(
                samples.shape[0], size=nsamp, replace=False
            )
            P = []
            for j in idx:
                b0_j, b1_j = samples[j]
                P.append(self._pod(x_grid, b0_j, b1_j))
            P = np.vstack(P)
            p_med = np.median(P, axis=0)
            p_lo = np.percentile(P, (1 - self.CI) * 50, axis=0)
            p_hi = np.percentile(P, 100 - (1 - self.CI) * 50, axis=0)
            plt.plot(x_grid, p_med, label="Bayes median", linestyle="-.", alpha=0.9)
            plt.fill_between(
                x_grid,
                p_lo,
                p_hi,
                alpha=0.2,
                label=f"{int(self.CI*100)}% CrI",
            )

        # Optional: mark a_lth
        if self.a_lth is not None:
            plt.axvline(
                self.a_lth,
                color="gray",
                linestyle=":",
                label=f"$a_{{lth}}$ = {self.a_lth:g}",
            )

        plt.ylim(-0.05, 1.05)
        plt.xlabel("True flaw size")
        plt.ylabel("Probability of Detection")
        plt.title("Log–Logistic POD fit")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _ensure_positive(x, eps=1e-12):
        return np.maximum(x, eps)

class Fit_Error_Calibration:
    """
    Fit additive or multiplicative NDE calibration models with LSQ, MLE, or Bayesian methods.

    Models
    ------
    Additive:
        D_m = m * D_e + c + ε,     ε ~ N(0, σ_e²)

    Multiplicative:
        ln(D_m) = m_t * ln(D_e) + c_t + ε_t,   ε_t ~ N(0, σ_{e_t}²)

    Censoring:
        D_measured entries ending with '+' are treated as right-censored
        measurements in the *measured* domain. LSQ uses all points, but
        MLE/Bayesian use only uncensored data in the likelihood (as in the
        original example code).

    Parameters
    ----------
    D_measured : array-like
        Reported measurements (strings or numeric). Values ending with '+'
        denote censored measurements (e.g., '80+').
    D_true : array-like
        True (model) flaw size, numeric.
    model : {"additive", "multiplicative"}
        Calibration model to fit.
    method : {"LSQ","MLE","Bayesian"}
        Estimation method.
    CI : float
        Confidence / credible level for intervals (e.g. 0.90, 0.95).
    print_results : bool
        If True, print markdown parameter tables.
    plot_results : bool
        If True, generate calibration plots.
    random_state : int or None
        Seed for randomness (Bayesian sampling & envelopes).
    """

    def __init__(
        self,
        D_measured,
        D_true,
        model="additive",
        method="MLE",
        CI=0.90,
        print_results=True,
        plot_results=True,
        random_state=None,
    ):
        self.model = str(model).lower()
        if self.model not in {"additive", "multiplicative"}:
            raise ValueError("model must be 'additive' or 'multiplicative'")

        self.method = str(method).lower()
        if self.method not in {"lsq", "mle", "bayesian"}:
            raise ValueError("method must be one of: 'LSQ', 'MLE', 'Bayesian'")

        self.CI = float(CI)
        if not (0.0 < self.CI < 1.0):
            raise ValueError("CI must be in (0,1)")

        self.print_results = bool(print_results)
        self.plot_results = bool(plot_results)
        self.random_state = random_state

        # Storage for results
        self.result_lsq = None
        self.result_mle = None
        self.result_bayes = None

        # Prepare data (handles censoring)
        self._prepare_data(D_measured, D_true)

        # Fit according to method
        if self.method == "lsq":
            self._fit_lsq()
        elif self.method == "mle":
            self._fit_lsq()
            self._fit_mle()
        elif self.method == "bayesian":
            self._fit_lsq()
            self._fit_mle()
            self._fit_bayesian()

        # Output
        if self.print_results:
            self._print_summary_markdown()
        if self.plot_results:
            self._plot_calibration()

    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------
    def _prepare_data(self, D_measured, D_true):
        # Coerce to Series for easier string ops
        measured_raw = pd.Series(D_measured).astype(str)
        true_raw = pd.to_numeric(pd.Series(D_true), errors="coerce")

        # Censoring: entries ending in '+'
        is_censored = measured_raw.str.contains(r"\+")
        measured_clean = measured_raw.str.replace("+", "", regex=False)

        D_e = pd.to_numeric(measured_clean, errors="coerce")
        D_m = true_raw

        # Drop invalid
        mask_valid = (~D_e.isna()) & (~D_m.isna())
        self.D_e = D_e[mask_valid].reset_index(drop=True)
        self.D_m = D_m[mask_valid].reset_index(drop=True)
        self.is_censored = is_censored[mask_valid].reset_index(drop=True)

        # Uncensored mask
        self.mask_unc = ~self.is_censored

        # For multiplicative: positive-only subset
        mask_pos = (self.D_e > 0) & (self.D_m > 0)
        self.mask_pos = mask_pos
        self.mask_pos_unc = self.mask_pos & self.mask_unc

    # ------------------------------------------------------------------
    # LSQ fits
    # ------------------------------------------------------------------
    def _fit_lsq(self):
        if self.model == "additive":
            x = self.D_e.values
            y = self.D_m.values

            def residuals(params, x, y):
                m, c = params
                return y - (m * x + c)

            p0 = np.array([1.0, 0.0])
            res = least_squares(residuals, p0, args=(x, y))
            m_add, c_add = res.x

            # RMSE of residuals (sigma_e)
            resid = residuals(res.x, x, y)
            sigma_e = np.sqrt(np.mean(resid**2))

            self.result_lsq = {
                "params": np.array([m_add, c_add, sigma_e]),
                "names": ["m", "c", r"$\sigma_{\varepsilon}$"],
                "success": res.success,
                "cost": res.cost,
            }

        elif self.model == "multiplicative":
            mask_pos = self.mask_pos
            x = self.D_e[mask_pos].values
            y = self.D_m[mask_pos].values

            def residuals(params, x, y):
                mt, ct = params
                return np.log(y) - (mt * np.log(x) + ct)

            p0 = np.array([1.0, 0.0])
            res = least_squares(residuals, p0, args=(x, y))
            m_t, c_t = res.x

            resid = residuals(res.x, x, y)
            sigma_e_t = np.sqrt(np.mean(resid**2))

            self.result_lsq = {
                "params": np.array([m_t, c_t, sigma_e_t]),
                "names": [r"$m_{t}$", r"$c_t$", r"$\sigma_{\varepsilon_{t}}$"],
                "success": res.success,
                "cost": res.cost,
            }

    # ------------------------------------------------------------------
    # MLE
    # ------------------------------------------------------------------
    def _nll_additive(self, theta):
        m, c, log_sigma = theta
        sigma = np.exp(log_sigma)
        x = self.D_e[self.mask_unc].values
        y = self.D_m[self.mask_unc].values
        mu = m * x + c
        r = y - mu
        return 0.5 * np.sum((r / sigma) ** 2 + np.log(2 * np.pi) + 2 * log_sigma)

    def _nll_multiplicative(self, theta):
        mt, ct, log_sigma_t = theta
        sigma_t = np.exp(log_sigma_t)
        x = np.log(self.D_e[self.mask_pos_unc].values)
        y = np.log(self.D_m[self.mask_pos_unc].values)
        mu = mt * x + ct
        r = y - mu
        return 0.5 * np.sum((r / sigma_t) ** 2 + np.log(2 * np.pi) + 2 * log_sigma_t)

    def _fit_mle(self):
        if self.result_lsq is None:
            self._fit_lsq()

        # Initial from LSQ
        m0, c0, sigma0 = self.result_lsq["params"]
        log_sigma0 = np.log(max(sigma0, 1e-8))

        if self.model == "additive":
            p0 = np.array([m0, c0, log_sigma0])
            res = minimize(self._nll_additive, p0, method="L-BFGS-B")
        else:  # multiplicative
            p0 = np.array([m0, c0, log_sigma0])
            res = minimize(self._nll_multiplicative, p0, method="L-BFGS-B")

        theta_hat = res.x
        # Make sure sigma param is exponentiated for reporting
        theta_hat[2] = np.exp(theta_hat[2])

        # Approximate covariance matrix from Hessian inverse
        H_inv = res.hess_inv
        if hasattr(H_inv, "todense"):
            H_inv = np.array(H_inv.todense())
        cov = np.array(H_inv, dtype=float)

        # Transform covariance for sigma via delta-method
        cov_mc_log = cov
        m_var, c_var, log_var = np.diag(cov_mc_log)
        m_log_cov = cov_mc_log[0, 2]
        c_log_cov = cov_mc_log[1, 2]
        sigma = theta_hat[2]

        cov_mc_sigma = np.zeros_like(cov_mc_log)
        cov_mc_sigma[0, 0] = m_var
        cov_mc_sigma[1, 1] = c_var
        cov_mc_sigma[2, 2] = (sigma**2) * log_var
        cov_mc_sigma[0, 2] = cov_mc_sigma[2, 0] = sigma * m_log_cov
        cov_mc_sigma[1, 2] = cov_mc_sigma[2, 1] = sigma * c_log_cov

        theta_hat_mc_sigma = np.array(
            [theta_hat[0], theta_hat[1], sigma], dtype=float
        )

        se = np.sqrt(np.diag(cov_mc_sigma))
        z = norm.ppf(0.5 + self.CI / 2.0)
        ci_low = theta_hat_mc_sigma - z * se
        ci_high = theta_hat_mc_sigma + z * se

        if self.model == "additive":
            names = ["m", "c", r"$\sigma_{\varepsilon}$"]
        else:
            names = [r"$m_{t}$", r"$c_t$", r"$\sigma_{\varepsilon_{t}}$"]

        self.result_mle = {
            "params": theta_hat_mc_sigma,
            "names": names,
            "cov": cov_mc_sigma,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "success": res.success,
            "fun": res.fun,
        }

    # ------------------------------------------------------------------
    # Bayesian (emcee + arviz)
    # ------------------------------------------------------------------
    def _log_prior(self, theta):
        # Priors on (m, c, log_sigma) or (m_t, c_t, log_sigma_t)
        m, c, log_sigma = theta
        # Weakly informative: wide normal-ish via uniform window
        if -10 < m < 10 and -100 < c < 100 and -5 < log_sigma < 5:
            return 0.0
        return -np.inf

    def _log_likelihood_bayes_additive(self, theta):
        m, c, log_sigma = theta
        sigma = np.exp(log_sigma)
        x = self.D_e[self.mask_unc].values
        y = self.D_m[self.mask_unc].values
        mu = m * x + c
        r = y - mu
        return -0.5 * np.sum((r / sigma) ** 2 + np.log(2 * np.pi) + 2 * log_sigma)

    def _log_likelihood_bayes_multiplicative(self, theta):
        mt, ct, log_sigma_t = theta
        sigma_t = np.exp(log_sigma_t)
        x = np.log(self.D_e[self.mask_pos_unc].values)
        y = np.log(self.D_m[self.mask_pos_unc].values)
        mu = mt * x + ct
        r = y - mu
        return -0.5 * np.sum((r / sigma_t) ** 2 + np.log(2 * np.pi) + 2 * log_sigma_t)

    def _log_posterior(self, theta):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        if self.model == "additive":
            ll = self._log_likelihood_bayes_additive(theta)
        else:
            ll = self._log_likelihood_bayes_multiplicative(theta)
        return lp + ll

    def _fit_bayesian(self):
        try:
            import emcee
            import arviz as az
        except ImportError as e:
            raise ImportError(
                "Bayesian fitting requires 'emcee' and 'arviz'. "
                "Install them to use method='Bayesian'."
            ) from e

        if self.result_mle is None:
            self._fit_mle()

        # Start near MLE in (m, c, log_sigma) or (m_t, c_t, log_sigma_t)
        if self.model == "additive":
            m_hat, c_hat, sigma_hat = self.result_mle["params"]
        else:  # multiplicative
            m_hat, c_hat, sigma_hat = self.result_mle["params"]

        log_sigma_hat = np.log(sigma_hat)
        theta_center = np.array([m_hat, c_hat, log_sigma_hat])

        rng = np.random.default_rng(self.random_state)
        ndim = 3
        nwalkers = 32
        p0 = theta_center + 1e-2 * rng.normal(size=(nwalkers, ndim))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_posterior)
        n_burn = 2000
        n_sample = 2000
        sampler.run_mcmc(p0, n_burn + n_sample, progress=True)

        chain_raw = sampler.get_chain(discard=n_burn, flat=True)
        # Transform (m, c, log_sigma) → (m, c, sigma)
        chain = chain_raw.copy()
        chain[:, 2] = np.exp(chain_raw[:, 2])  # sigma

        # Basic summaries across draws
        mean = np.mean(chain, axis=0)
        median = np.median(chain, axis=0)

        alpha_pct = (1 - self.CI) * 100 / 2.0
        eti_lo, eti_hi = np.percentile(chain, [alpha_pct, 100 - alpha_pct], axis=0)

        # HDI per parameter using 1D chains – avoids all shape confusion
        n_params = chain.shape[1]
        hdi_lo = np.empty(n_params)
        hdi_hi = np.empty(n_params)
        for j in range(n_params):
            hdi_j = az.hdi(chain[:, j], hdi_prob=self.CI)  # 1D input
            hdi_j = np.asarray(hdi_j)
            # Typically [low, high]
            hdi_lo[j] = float(hdi_j[0])
            hdi_hi[j] = float(hdi_j[1])

        if self.model == "additive":
            names = ["m", "c", r"$\sigma_{\varepsilon}$"]
        else:
            names = [r"$m_{t}$", r"$c_t$", r"$\sigma_{\varepsilon_{t}}$"]

        self.result_bayes = {
            "samples": chain,
            "names": names,
            "mean": mean,
            "median": median,
            "eti_low": eti_lo,
            "eti_high": eti_hi,
            "hdi_low": hdi_lo,
            "hdi_high": hdi_hi,
        }

    # ------------------------------------------------------------------
    # Markdown summary
    # ------------------------------------------------------------------
    def _print_summary_markdown(self):
        from IPython.display import Markdown, display

        lines = []

        # --- NEW: print measurement error model once, before any parameters ---
        lines.append("### Measurement Error Model")
        lines.append("")
        if self.model == "additive":
            # y ≡ D_m, ŷ ≡ m D_e + c
            lines.append(
                r"Additive error: "
                r"$y \mid \hat{y} \sim \mathcal{N}\!\big(m\,\hat{y} + c,\ \sigma_{\varepsilon}^2\big)$"
            )
        else:
            # y ≡ D_m, ŷ ≡ exp(m_t ln D_e + c_t)
            lines.append(
                r"Multiplicative error (log–log): "
                r"$\ln y \mid \ln \hat{y} \sim \mathcal{N}\!\big(m_t \ln \hat{y} + c_t,\ \sigma_{\varepsilon_t}^2\big)$"
            )
        lines.append("")

        if self.result_lsq is not None:
            lines.append("### LSQ estimates")
            lines.append("")
            header = "| Parameter | Estimate |"
            lines.append(header)
            lines.append("|-----------|----------|")
            for name, val in zip(self.result_lsq["names"], self.result_lsq["params"]):
                lines.append(f"| {name} | {val:.4f} |")
            lines.append("")

        if self.result_mle is not None:
            names = self.result_mle["names"]
            params = self.result_mle["params"]
            se = self.result_mle["se"]
            ci_low = self.result_mle["ci_low"]
            ci_high = self.result_mle["ci_high"]

            lines.append("### MLE estimates")
            lines.append("")
            lines.append(
                f"| Parameter | Estimate | Std. Error | {int(self.CI*100)}% CI low | {int(self.CI*100)}% CI high |"
            )
            lines.append(
                "|-----------|----------|------------|----------------|-----------------|"
            )
            for name, val, se_i, lo, hi in zip(names, params, se, ci_low, ci_high):
                lines.append(
                    f"| {name} | {val:.4f} | {se_i:.4f} | {lo:.4f} | {hi:.4f} |"
                )
            lines.append("")

        if self.result_bayes is not None:
            names = self.result_bayes["names"]
            mean = self.result_bayes["mean"]
            median = self.result_bayes["median"]
            eti_lo = self.result_bayes["eti_low"]
            eti_hi = self.result_bayes["eti_high"]
            hdi_lo = self.result_bayes["hdi_low"]
            hdi_hi = self.result_bayes["hdi_high"]

            lines.append("### Bayesian posterior summaries")
            lines.append("")
            lines.append(
                f"| Parameter | Post. mean | Post. median | {int(self.CI*100)}% ETI low | {int(self.CI*100)}% ETI high | {int(self.CI*100)}% HDI low | {int(self.CI*100)}% HDI high |"
            )
            lines.append(
                "|-----------|------------|--------------|-----------------|------------------|------------------|-------------------|"
            )
            for i, name in enumerate(names):
                lines.append(
                    f"| {name} | {mean[i]:.4f} | {median[i]:.4f} | {eti_lo[i]:.4f} | {eti_hi[i]:.4f} | {hdi_lo[i]:.4f} | {hdi_hi[i]:.4f} |"
                )
            lines.append("")

        if lines:
            display(Markdown("\n".join(lines)))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _plot_calibration(self):
        rng = np.random.default_rng(self.random_state)

        if self.model == "additive":
            x_all = self.D_e.values
            y_all = self.D_m.values
            is_cens = self.is_censored.values

            x_unc = self.D_e[self.mask_unc].values
            y_unc = self.D_m[self.mask_unc].values
            n = len(x_unc)

            x_grid = np.linspace(x_all.min(), x_all.max(), 400)

            plt.figure(figsize=(8, 6))
            # Data
            plt.scatter(
                x_all[~is_cens],
                y_all[~is_cens],
                s=60,
                color="tab:blue",
                edgecolor="black",
                label="Uncensored data",
            )
            plt.scatter(
                x_all[is_cens],
                y_all[is_cens],
                s=70,
                facecolor="white",
                edgecolor="black",
                marker="^",
                label="Censored ('+')",
            )

            # LSQ center curve
            if self.result_lsq is not None:
                m_lsq, c_lsq, sigma_lsq = self.result_lsq["params"]
                y_lsq = m_lsq * x_grid + c_lsq
                plt.plot(
                    x_grid,
                    y_lsq,
                    color="black",
                    lw=2,
                    label=f"Additive LSQ fit (σₑ={sigma_lsq:.2f})",
                )

            # MLE center and prediction CI
            if self.result_mle is not None:
                m_hat, c_hat, sigma_hat = self.result_mle["params"]

                # Center prediction
                y_center = m_hat * x_grid + c_hat
                plt.plot(
                    x_grid,
                    y_center,
                    color="tab:orange",
                    lw=2,
                    ls="-.",
                    label="Additive MLE fit",
                )

                # Classical mean prediction CI in linear regression
                x_bar = x_unc.mean()
                Sxx = np.sum((x_unc - x_bar) ** 2)
                # Residual std (unbiased)
                resid = y_unc - (m_hat * x_unc + c_hat)
                s_resid = np.sqrt(np.sum(resid**2) / (n - 2))
                tval = t.ppf(0.5 + self.CI / 2.0, df=n - 2)
                se_mean = s_resid * np.sqrt(
                    1.0 / n + (x_grid - x_bar) ** 2 / max(Sxx, 1e-12)
                )
                ci_lo = y_center - tval * se_mean
                ci_hi = y_center + tval * se_mean

                plt.fill_between(
                    x_grid,
                    ci_lo,
                    ci_hi,
                    color="tab:orange",
                    alpha=0.2,
                    label=f"MLE {int(self.CI*100)}% mean CI",
                )

            # Bayesian predictive band for mean (optional)
            if self.result_bayes is not None:
                samples = self.result_bayes["samples"]
                nsamp = min(3000, samples.shape[0])
                idx = rng.choice(samples.shape[0], size=nsamp, replace=False)
                Y = []
                for j in idx:
                    m_j, c_j, sigma_j = samples[j]
                    Y.append(m_j * x_grid + c_j)
                Y = np.vstack(Y)
                y_med = np.median(Y, axis=0)
                alpha_pct = (1 - self.CI) * 50
                y_lo = np.percentile(Y, alpha_pct, axis=0)
                y_hi = np.percentile(Y, 100 - alpha_pct, axis=0)
                plt.plot(
                    x_grid,
                    y_med,
                    color="tab:green",
                    lw=2,
                    ls="--",
                    label="Bayes median fit",
                )
                plt.fill_between(
                    x_grid,
                    y_lo,
                    y_hi,
                    color="tab:green",
                    alpha=0.15,
                    label=f"{int(self.CI*100)}% Bayes CrI (mean)",
                )

            plt.xlabel("Measured flaw size $D_e$ (mm)")
            plt.ylabel("Model flaw size $D_m$ (mm)")
            plt.title("Additive Error Calibration")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # --------------------------------------------------------------
        # Multiplicative model: log-log plot
        # --------------------------------------------------------------
        else:
            x_all = self.D_e.values
            y_all = self.D_m.values
            is_cens = self.is_censored.values
            mask_pos = self.mask_pos.values
            mask_pos_unc = self.mask_pos_unc.values

            x_pos = x_all[mask_pos]
            y_pos = y_all[mask_pos]

            x_grid = np.linspace(x_pos.min(), x_pos.max(), 400)
            x_grid_pos = x_grid[x_grid > 0]
            xlog = np.log(self.D_e[mask_pos_unc].values)
            ylog = np.log(self.D_m[mask_pos_unc].values)
            n = len(xlog)

            plt.figure(figsize=(8, 6))
            # Data (only positive for multiplicative)
            plt.scatter(
                x_pos[~is_cens[mask_pos]],
                y_pos[~is_cens[mask_pos]],
                s=60,
                color="tab:blue",
                edgecolor="black",
                label="Uncensored data",
            )
            plt.scatter(
                x_pos[is_cens[mask_pos]],
                y_pos[is_cens[mask_pos]],
                s=70,
                facecolor="white",
                edgecolor="black",
                marker="^",
                label="Censored ('+')",
            )

            # LSQ curve
            if self.result_lsq is not None:
                m_t_lsq, c_t_lsq, sigma_e_t_lsq = self.result_lsq["params"]
                y_mult_lsq = np.exp(m_t_lsq * np.log(x_grid_pos) + c_t_lsq)
                plt.plot(
                    x_grid_pos,
                    y_mult_lsq,
                    color="tab:orange",
                    lw=2,
                    label=f"Multiplicative LSQ fit (σₑₜ={sigma_e_t_lsq:.3f})",
                )

            # MLE curve + CI in log space
            if self.result_mle is not None:
                m_t_hat, c_t_hat, sigma_t_hat = self.result_mle["params"]
                xglog = np.log(x_grid_pos)
                y_center_log = m_t_hat * xglog + c_t_hat
                y_center = np.exp(y_center_log)
                plt.plot(
                    x_grid_pos,
                    y_center,
                    color="black",
                    lw=2,
                    ls="-.",
                    label="Multiplicative MLE fit",
                )

                # Classical CI for mean log-prediction
                xlog_bar = xlog.mean()
                Sxx_log = np.sum((xlog - xlog_bar) ** 2)
                resid_log = ylog - (m_t_hat * xlog + c_t_hat)
                s_log = np.sqrt(np.sum(resid_log**2) / (n - 2))
                tval = t.ppf(0.5 + self.CI / 2.0, df=n - 2)
                se_mean_log = s_log * np.sqrt(
                    1.0 / n + (xglog - xlog_bar) ** 2 / max(Sxx_log, 1e-12)
                )
                log_lo = y_center_log - tval * se_mean_log
                log_hi = y_center_log + tval * se_mean_log

                mult_lo = np.exp(log_lo)
                mult_hi = np.exp(log_hi)
                plt.fill_between(
                    x_grid_pos,
                    mult_lo,
                    mult_hi,
                    color="black",
                    alpha=0.2,
                    label=f"MLE {int(self.CI*100)}% mean CI",
                )

            # Bayesian band (mean in log space → back-transform)
            if self.result_bayes is not None:
                samples = self.result_bayes["samples"]
                nsamp = min(3000, samples.shape[0])
                idx = rng.choice(samples.shape[0], size=nsamp, replace=False)
                Y = []
                xglog = np.log(x_grid_pos)
                for j in idx:
                    m_t_j, c_t_j, sigma_t_j = samples[j]
                    ylog_j = m_t_j * xglog + c_t_j
                    Y.append(np.exp(ylog_j))
                Y = np.vstack(Y)
                y_med = np.median(Y, axis=0)
                alpha_pct = (1 - self.CI) * 50
                y_lo = np.percentile(Y, alpha_pct, axis=0)
                y_hi = np.percentile(Y, 100 - alpha_pct, axis=0)
                plt.plot(
                    x_grid_pos,
                    y_med,
                    color="tab:green",
                    lw=2,
                    ls="--",
                    label="Bayes median fit",
                )
                plt.fill_between(
                    x_grid_pos,
                    y_lo,
                    y_hi,
                    color="tab:green",
                    alpha=0.15,
                    label=f"{int(self.CI*100)}% Bayes CrI (mean)",
                )

            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Measured flaw size $D_e$ (mm, log scale)")
            plt.ylabel("Model flaw size $D_m$ (mm, log scale)")
            plt.title("Multiplicative Error Calibration")
            plt.legend()
            plt.grid(True, which="both", alpha=0.3)
            plt.tight_layout()
            plt.show()