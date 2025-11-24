# base_adt_model.py
import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import minimize
from scipy.stats import norm

from adt_utils import (
    loglik_normal,
    loglik_lognormal,
    md_param_table_freq,
    plot_degradation_by_stress,
    plot_residual_diagnostics,
    md_param_table_bayes,
    sample_ttf_from_posterior,
    summarize_ttf,
    plot_ttf_hist_with_fits,
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


# fit_adt_sqrt_arrhenius.py

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
        data_scale="linear",      # <--- NEW: how you want data plotted
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
        Simple Bayesian fit using emcee with log-normal priors on (g0, g1, Ea, sigma).

        Uses:
            - MLE estimates (if available) to centre priors
            - LS estimates as fallback

        Stores:
            self.g0_s, self.g1_s, self.Ea_s, self.sigma_s
        """
        # Prior centres from MLE if available, else LS
        if self.theta_MLE is not None and self.sigma_MLE is not None:
            g0_hat, g1_hat, Ea_hat = self.theta_MLE
            sig_hat = self.sigma_MLE
        elif self.theta_LS is not None and self.sigma_LS is not None:
            g0_hat, g1_hat, Ea_hat = self.theta_LS
            sig_hat = self.sigma_LS
        else:
            raise RuntimeError("Run LS or MLE before Bayesian (no initial estimates).")

        mu_logg0 = np.log(max(g0_hat, 1e-10))
        mu_logg1 = np.log(max(g1_hat, 1e-10))
        mu_logEa = np.log(max(Ea_hat, 1e-3))
        mu_logs  = np.log(max(sig_hat, 1e-6))

        sd_logg0 = 0.5
        sd_logg1 = 0.5
        sd_logEa = 0.5
        sd_logs  = 0.5

        t_time = self.t
        D_obs = self.y

        def log_prior(theta):
            log_g0, log_g1, log_Ea, log_sig = theta

            lp  = -0.5*((log_g0 - mu_logg0)/sd_logg0)**2 - np.log(sd_logg0*np.sqrt(2*np.pi))
            lp += -0.5*((log_g1 - mu_logg1)/sd_logg1)**2 - np.log(sd_logg1*np.sqrt(2*np.pi))
            lp += -0.5*((log_Ea - mu_logEa)/sd_logEa)**2 - np.log(sd_logEa*np.sqrt(2*np.pi))
            lp += -0.5*((log_sig - mu_logs)/sd_logs)**2   - np.log(sd_logs*np.sqrt(2*np.pi))
            return lp

        def log_likelihood(theta):
            log_g0, log_g1, log_Ea, log_sig = theta
            g0    = np.exp(log_g0)
            g1    = np.exp(log_g1)
            Ea    = np.exp(log_Ea)
            sigma = np.exp(log_sig)

            params = np.array([g0, g1, Ea], float)
            mu = self._mu(params, t_time)

            if self.noise == "additive":
                if sigma <= 0 or not np.all(np.isfinite(mu)):
                    return -np.inf
                return np.sum(norm.logpdf(D_obs, loc=mu, scale=sigma))

            elif self.noise == "multiplicative":
                D_pos = np.clip(D_obs, 1e-12, None)
                mu_pos = np.clip(mu, 1e-12, None)
                logD = np.log(D_pos)
                logmu = np.log(mu_pos)
                if sigma <= 0 or not np.all(np.isfinite(logmu)):
                    return -np.inf
                z = (logD - logmu) / sigma
                return (
                    -np.sum(np.log(D_pos))
                    - len(D_pos)*np.log(sigma)
                    + np.sum(norm.logpdf(z))
                )
            else:
                return -np.inf

        def log_prob(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            ll = log_likelihood(theta)
            if not np.isfinite(ll):
                return -np.inf
            return lp + ll

        init = np.array([mu_logg0, mu_logg1, mu_logEa, mu_logs], float)
        sampler, chain = self._run_emcee(log_prob, init, nwalkers, nburn, nsamp)
        log_g0_s, log_g1_s, log_Ea_s, log_sig_s = chain.T

        self.g0_s    = np.exp(log_g0_s)
        self.g1_s    = np.exp(log_g1_s)
        self.Ea_s    = np.exp(log_Ea_s)
        self.sigma_s = np.exp(log_sig_s)

        # Optional: store ArviZ InferenceData
        try:
            import arviz as az
            self.idata_bayes = az.from_emcee(
                sampler,
                var_names=["g0", "g1", "Ea", "sigma"],
            )
        except ImportError:
            self.idata_bayes = None

        # Posterior table
        if self.print_results:
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

# fit_adt_power_exponential.py

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
        data_scale="linear",   # "linear", "ylog", "xlog", "loglog"
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

        # Parameter names in order used by _mu, _init_guess, priors, etc.
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
        cmap = plt.get_cmap("viridis")
        colors = {S: cmap(i / max(len(stresses) - 1, 1)) for i, S in enumerate(stresses)}

        # data
        for S in stresses:
            mask = (self.S == S)
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=25,
                alpha=0.7,
                color=colors[S],
                label=f"S={S:g} data",
            )

        # model curves
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 0.0, None)

        for S in stresses:
            factor = np.exp(a * S)
            mu_grid = b * factor * (t_pos ** n)
            ax.plot(
                t_grid,
                mu_grid,
                color=colors[S],
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
        Simple Bayesian fit using emcee with:

            log b ~ Normal(mu_logb, sd_logb^2)
            a     ~ Normal(mu_a,   sd_a^2)
            log n ~ Normal(mu_logn, sd_logn^2)
            log σ ~ Normal(mu_logs, sd_logs^2)
        """
        # Prior centres from MLE if available, else LS
        if self.theta_MLE is not None and self.sigma_MLE is not None:
            b_hat, a_hat, n_hat = self.theta_MLE
            sig_hat = self.sigma_MLE
        elif self.theta_LS is not None and self.sigma_LS is not None:
            b_hat, a_hat, n_hat = self.theta_LS
            sig_hat = self.sigma_LS
        else:
            raise RuntimeError("Run LS or MLE before Bayesian (no initial estimates).")

        mu_logb  = np.log(max(b_hat, 1e-10))
        mu_a     = float(a_hat)
        mu_logn  = np.log(max(n_hat, 1e-6))
        mu_logs  = np.log(max(sig_hat, 1e-6))

        sd_logb  = 0.5
        sd_a     = 0.5
        sd_logn  = 0.5
        sd_logs  = 0.5

        t_time = self.t
        S_stress = self.S
        D_obs = self.y

        def log_prior(theta):
            log_b, a, log_n, log_sig = theta

            lp  = -0.5*((log_b - mu_logb)/sd_logb)**2  - np.log(sd_logb*np.sqrt(2*np.pi))
            lp += -0.5*((a - mu_a)/sd_a)**2            - np.log(sd_a*np.sqrt(2*np.pi))
            lp += -0.5*((log_n - mu_logn)/sd_logn)**2  - np.log(sd_logn*np.sqrt(2*np.pi))
            lp += -0.5*((log_sig - mu_logs)/sd_logs)**2- np.log(sd_logs*np.sqrt(2*np.pi))
            return lp

        def log_likelihood(theta):
            log_b, a, log_n, log_sig = theta
            b     = np.exp(log_b)
            n     = np.exp(log_n)
            sigma = np.exp(log_sig)

            # Quick sanity checks
            if b <= 0 or n <= 0 or sigma <= 0:
                return -np.inf

            t_pos = np.clip(t_time, 1e-12, None)

            # ----- LOG-SCALE MEAN -----
            # log mu = log b + a*S + n*log t
            # clip a*S to avoid overflow in exp
            aS = a * S_stress
            aS = np.clip(aS, -50.0, 50.0)  # tighter than -700/700 is fine here

            log_mu = log_b + aS + n * np.log(t_pos)

            # Guard against insane means
            if not np.all(np.isfinite(log_mu)):
                return -np.inf

            if self.noise == "additive":
                # Need mu itself, but we can clip to avoid overflow
                log_mu_clipped = np.clip(log_mu, -50.0, 50.0)
                mu = np.exp(log_mu_clipped)

                if not np.all(np.isfinite(mu)):
                    return -np.inf

                return np.sum(norm.logpdf(D_obs, loc=mu, scale=sigma))

            elif self.noise == "multiplicative":
                # Lognormal: log D ~ N(log mu, sigma^2)
                D_pos  = np.clip(D_obs, 1e-12, None)
                logD   = np.log(D_pos)

                # If log_mu is huge, treat as invalid
                if not np.all(np.isfinite(log_mu)):
                    return -np.inf

                z = (logD - log_mu) / sigma

                if not np.all(np.isfinite(z)):
                    return -np.inf

                # lognormal Jacobian + normal on log scale
                return (
                    -np.sum(np.log(D_pos))
                    - len(D_pos) * np.log(sigma)
                    + np.sum(norm.logpdf(z))
                )

            else:
                return -np.inf


        def log_prob(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            ll = log_likelihood(theta)
            if not np.isfinite(ll):
                return -np.inf
            return lp + ll

        init = np.array([mu_logb, mu_a, mu_logn, mu_logs], float)
        sampler, chain = self._run_emcee(log_prob, init, nwalkers, nburn, nsamp)
        log_b_s, a_s, log_n_s, log_sig_s = chain.T

        self.b_s     = np.exp(log_b_s)
        self.a_s     = a_s
        self.n_s     = np.exp(log_n_s)
        self.sigma_s = np.exp(log_sig_s)

        # Optional ArviZ
        try:
            import arviz as az
            self.idata_bayes = az.from_emcee(
                sampler,
                var_names=["b", "a", "n", "sigma"],
            )
        except ImportError:
            self.idata_bayes = None

        # Posterior table
        if self.print_results:
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
        data_scale="linear",   # "linear", "ylog", "xlog", "loglog"
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

        # Parameter names in order used by _mu, _init_guess, priors, etc.
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
        cmap = plt.get_cmap("viridis")
        colors = {S: cmap(i / max(len(stresses) - 1, 1)) for i, S in enumerate(stresses)}

        # data
        for S in stresses:
            mask = (self.S == S)
            ax.scatter(
                self.t[mask],
                self.y[mask],
                s=25,
                alpha=0.7,
                color=colors[S],
                label=f"S={S:g} data",
            )

        # model curves
        t_grid = np.linspace(0.0, self.t.max() * 1.05, 200)
        t_pos = np.clip(t_grid, 1e-12, None)

        for S in stresses:
            S_pos = max(S, 1e-12)
            mu_grid = b * (S_pos ** n_S) * (t_pos ** m_t)
            ax.plot(
                t_grid,
                mu_grid,
                color=colors[S],
                lw=2,
                label=f"S={S:g} LS mean",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Damage D")
        ax.set_title("LSQ Power–Power Degradation Fit")
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
    # ------------------------------------------------------------------
    def _fit_Bayesian(self, nwalkers=32, nburn=1000, nsamp=2000):
        """
        Simple Bayesian fit using emcee with:

            log b   ~ Normal(mu_logb,   sd_logb^2)
            log n_S ~ Normal(mu_lognS,  sd_lognS^2)
            log m_t ~ Normal(mu_logm,   sd_logm^2)
            log σ   ~ Normal(mu_logs,   sd_logs^2)
        """
        # Prior centres from MLE if available, else LS
        if self.theta_MLE is not None and self.sigma_MLE is not None:
            b_hat, nS_hat, m_hat = self.theta_MLE
            sig_hat = self.sigma_MLE
        elif self.theta_LS is not None and self.sigma_LS is not None:
            b_hat, nS_hat, m_hat = self.theta_LS
            sig_hat = self.sigma_LS
        else:
            raise RuntimeError("Run LS or MLE before Bayesian (no initial estimates).")

        mu_logb   = np.log(max(b_hat,   1e-10))
        mu_lognS  = np.log(max(nS_hat,  1e-6))
        mu_logm   = np.log(max(m_hat,   1e-6))
        mu_logs   = np.log(max(sig_hat, 1e-6))

        sd_logb   = 0.5
        sd_lognS  = 0.5
        sd_logm   = 0.5
        sd_logs   = 0.5

        t_time = self.t
        S_stress = self.S
        D_obs = self.y

        def log_prior(theta):
            log_b, log_nS, log_m, log_sig = theta

            lp  = -0.5*((log_b  - mu_logb )/sd_logb )**2 - np.log(sd_logb *np.sqrt(2*np.pi))
            lp += -0.5*((log_nS - mu_lognS)/sd_lognS)**2 - np.log(sd_lognS*np.sqrt(2*np.pi))
            lp += -0.5*((log_m  - mu_logm )/sd_logm )**2 - np.log(sd_logm *np.sqrt(2*np.pi))
            lp += -0.5*((log_sig- mu_logs )/sd_logs )**2 - np.log(sd_logs *np.sqrt(2*np.pi))
            return lp

        def log_likelihood(theta):
            log_b, log_nS, log_m, log_sig = theta
            b     = np.exp(log_b)
            n_S   = np.exp(log_nS)
            m_t   = np.exp(log_m)
            sigma = np.exp(log_sig)

            if b <= 0 or n_S <= 0 or m_t <= 0 or sigma <= 0:
                return -np.inf

            S_pos = np.clip(S_stress, 1e-12, None)
            t_pos = np.clip(t_time, 1e-12, None)
            mu = b * (S_pos ** n_S) * (t_pos ** m_t)

            if self.noise == "additive":
                if not np.all(np.isfinite(mu)):
                    return -np.inf
                return np.sum(norm.logpdf(D_obs, loc=mu, scale=sigma))

            elif self.noise == "multiplicative":
                D_pos = np.clip(D_obs, 1e-12, None)
                mu_pos = np.clip(mu, 1e-12, None)
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

        def log_prob(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            ll = log_likelihood(theta)
            if not np.isfinite(ll):
                return -np.inf
            return lp + ll

        init = np.array([mu_logb, mu_lognS, mu_logm, mu_logs], float)
        ndim = 4

        rng = np.random.default_rng(123)
        p0 = init + rng.normal(scale=[0.2, 0.2, 0.2, 0.2], size=(nwalkers, ndim))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

        # burn-in
        state = sampler.run_mcmc(p0, nburn, progress=True)
        sampler.reset()

        # main run
        sampler.run_mcmc(state, nsamp, progress=True)

        chain = sampler.get_chain(flat=True)
        log_b_s, log_nS_s, log_m_s, log_sig_s = chain.T

        self.b_s     = np.exp(log_b_s)
        self.nS_s    = np.exp(log_nS_s)
        self.m_s     = np.exp(log_m_s)
        self.sigma_s = np.exp(log_sig_s)

        # Optional ArviZ
        try:
            import arviz as az
            self.idata_bayes = az.from_emcee(
                sampler,
                var_names=["b", "n", "m", "sigma"],
            )
        except ImportError:
            self.idata_bayes = None

        # Posterior table
        if self.print_results:
            md_param_table_bayes(
                {
                    "b": self.b_s,
                    "n": self.nS_s,
                    "m": self.m_s,
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
        _nS  = self.nS_s[sel]
        _m   = self.m_s[sel]
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
        plt.savefig("Power-Power Bayesian Fit")
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

        theta_samples = np.column_stack([self.b_s, self.nS_s, self.m_s])
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.special import expit
from scipy.stats import norm

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
        # Local import to avoid hard dependency if user only wants LSQ/MLE
        try:
            import emcee
            import arviz as az
        except ImportError as e:
            raise ImportError(
                "Bayesian fitting requires the 'emcee' and 'arviz' packages. "
                "Install them via pip or conda to use method='Bayesian'."
            ) from e

        # Use MLE as a sensible center for walkers
        if self.result_mle is None:
            self._fit_mle()
        beta_mle = np.array([self.result_mle["beta0"], self.result_mle["beta1"]])

        rng = np.random.default_rng(self.random_state)
        ndim = 2
        nwalkers = 32
        # Initialize walkers around MLE with small Gaussian noise
        p0 = beta_mle + 1e-2 * rng.normal(size=(nwalkers, ndim))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_posterior)
        n_burn = 2000
        n_sample = 2000
        sampler.run_mcmc(p0, n_burn + n_sample, progress=False)

        chain = sampler.get_chain(discard=n_burn, flat=True)
        b0_s = chain[:, 0]
        b1_s = chain[:, 1]

        # Posterior summaries
        mean = np.array([np.mean(b0_s), np.mean(b1_s)])
        median = np.array([np.median(b0_s), np.median(b1_s)])

        alpha = (1 - self.CI) * 100 / 2.0
        q_lo, q_hi = np.percentile(chain, [alpha, 100 - alpha], axis=0)

        # HDI via ArviZ
        hdi = az.hdi(chain, hdi_prob=self.CI)
        hdi_lo = hdi[0]
        hdi_hi = hdi[1]

        self.result_bayes = {
            "samples": chain,
            "mean": mean,
            "median": median,
            "eti_low": q_lo,
            "eti_high": q_hi,
            "hdi_low": hdi_lo,
            "hdi_high": hdi_hi,
        }

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _print_summary_markdown(self):
        from IPython.display import Markdown, display

        lines = []

        if self.result_lsq is not None:
            lines.append("### LSQ estimates")
            lines.append("")
            lines.append("| Parameter | Estimate |")
            lines.append("|-----------|----------|")
            lines.append(f"| $\\beta_0$ | {self.result_lsq['beta0']:.4f} |")
            lines.append(f"| $\\beta_1$ | {self.result_lsq['beta1']:.4f} |")
            lines.append("")

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
