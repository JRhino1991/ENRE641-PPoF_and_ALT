# adt_utils.py

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.stats import norm, probplot, weibull_min, gamma, lognorm
from IPython.display import Markdown, display
from scipy.optimize import brentq


# ---------------------------------------------------------------------
# 1. Basic statistical helpers
# ---------------------------------------------------------------------

def hdi(samples, cred_mass=0.95):
    """
    Highest Density Interval (HDI) for 1D samples.
    Returns (hdi_min, hdi_max).
    """
    x = np.sort(np.asarray(samples).ravel())
    n = len(x)
    if n == 0:
        return np.nan, np.nan

    if cred_mass <= 0.0 or cred_mass >= 1.0:
        raise ValueError("cred_mass must be in (0,1).")

    interval_idx_inc = int(np.floor(cred_mass * n))
    if interval_idx_inc < 1:
        return np.nan, np.nan

    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    j_min = np.argmin(interval_width)
    hdi_min = x[j_min]
    hdi_max = x[j_min + interval_idx_inc]

    return float(hdi_min), float(hdi_max)


def loglik_normal(y, mu, sigma):
    """
    Additive Normal log-likelihood:
        y = mu + eps, eps ~ N(0, sigma^2).
    """
    if sigma <= 0:
        return -np.inf
    if not np.all(np.isfinite(mu)):
        return -np.inf
    return np.sum(norm.logpdf(y, loc=mu, scale=sigma))


def loglik_lognormal(y, mu, sigma):
    """
    Lognormal log-likelihood on y with 'mean scale' mu:
        log y ~ Normal(log mu, sigma^2).

    Both y and mu must be positive.
    """
    if sigma <= 0:
        return -np.inf

    y = np.asarray(y, float)
    mu = np.asarray(mu, float)

    if np.any(y <= 0) or np.any(mu <= 0):
        return -np.inf

    log_y = np.log(y)
    log_mu = np.log(mu)

    z = (log_y - log_mu) / sigma
    # p(y|mu,sigma) = 1/(y*sigma) * phi((log y - log mu)/sigma)
    return -np.sum(np.log(y)) - len(y) * np.log(sigma) + np.sum(norm.logpdf(z))

def loglik_trunc_normal(y, mu, sigma, eps=1e-12):
    """
    Log-likelihood for y ~ Normal(mu, sigma^2) truncated to [0, 1].

    Works elementwise and returns scalar sum over all observations.
    """
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)

    # Clip into open interval to avoid log(0) / cdf issues
    y_clip  = np.clip(y,  0.0 + eps, 1.0 - eps)
    mu_clip = np.clip(mu, 0.0 - 10.0, 1.0 + 10.0)  # mu can be outside [0,1], but not crazy

    sigma = float(sigma)
    if sigma <= 0:
        return -np.inf

    z      = (y_clip  - mu_clip) / sigma
    a_tr   = (0.0     - mu_clip) / sigma
    b_tr   = (1.0     - mu_clip) / sigma
    denom  = norm.cdf(b_tr) - norm.cdf(a_tr)

    # If truncation mass is (numerically) 0 anywhere, bail out
    if np.any(denom <= eps) or not np.all(np.isfinite(denom)):
        return -np.inf

    # log φ(z) - log σ - log(Φ(b)-Φ(a))
    logpdf = norm.logpdf(z) - np.log(sigma) - np.log(denom)
    return np.sum(logpdf)


# ---------------------------------------------------------------------
# 2. Markdown parameter table printers
# ---------------------------------------------------------------------

def md_param_table_freq(theta, cov, names, z_CI, CI,
                        sigma=None, noise_label=None, label=""):
    """
    Markdown table for frequentist estimates.

    Parameters
    ----------
    theta : array-like
        Point estimates of parameters.
    cov : array-like or None
        Covariance matrix of theta (for SE and CI). If None, SE/CI omitted.
    names : list of str
        Parameter names (same length as theta).
    z_CI : float
        Normal quantile for two-sided CI, e.g. norm.ppf(0.975) for 95%.
    CI : float
        Confidence level, e.g. 0.95.
    sigma : float or None
        Noise scale estimate to print as a separate line.
    noise_label : str or None
        Label for noise type, e.g. "D" or "log D".
    label : str
        Title (prefix) for the table.
    """
    theta = np.asarray(theta, float)
    CI_pct = int(CI * 100)

    if cov is not None:
        cov = np.asarray(cov, float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            cov = None

    lines = []
    heading = label or "Parameter Estimates"
    lines.append(f"### {heading} ({CI_pct}% CI)\n")
    lines.append(f"| Parameter | Estimate | SE | {CI_pct}% CI |")
    lines.append("|-----------|----------|----|--------------|")

    if cov is None:
        for n, v in zip(names, theta):
            lines.append(f"| {n} | {v:.6g} | — | — |")
    else:
        se = np.sqrt(np.diag(cov))
        for n, v, s in zip(names, theta, se):
            lo = v - z_CI * s
            hi = v + z_CI * s
            lines.append(f"| {n} | {v:.6g} | {s:.4g} | [{lo:.4g}, {hi:.4g}] |")

    if sigma is not None:
        if noise_label is None:
            noise_label = "σ"
        lines.append(
            f"\n**Noise {noise_label}:** {sigma:.6g}\n"
        )

    display(Markdown("\n".join(lines)))


def md_param_table_bayes(samples_dict, cred_mass=0.95, label="Bayesian posterior"):
    """
    Markdown table for Bayesian posterior samples.

    Parameters
    ----------
    samples_dict : dict
        Keys are parameter names, values are 1D arrays of posterior samples.
    cred_mass : float
        Credible interval mass (CrI), e.g. 0.95.
    label : str
        Title for the table.
    """
    CrI_pct = int(cred_mass * 100)

    lines = []
    lines.append(f"### {label} Parameter Estimates ({CrI_pct}% CrI)\n")
    lines.append(f"| Parameter | Mean | Median | {CrI_pct}% CrI |")
    lines.append("|-----------|------|--------|----------------|")

    for name, s in samples_dict.items():
        s = np.asarray(s).ravel()
        mean = np.mean(s)
        median = np.median(s)
        lo, hi = hdi(s, cred_mass=cred_mass)
        lines.append(
            f"| {name} | {mean:.6g} | {median:.6g} | [{lo:.6g}, {hi:.6g}] |"
        )

    display(Markdown("\n".join(lines)))


# ---------------------------------------------------------------------
# 3. Plotting helpers
# ---------------------------------------------------------------------

# adt_utils.py

def plot_degradation_by_stress(
    t,
    D,
    stress,
    unit=None,
    title="Degradation vs Time by Stress",
    stress_label="Stress level",
    legend_title="Stress",
    scale="linear",           # "linear", "semilogx", "semilogy", "loglog"
    base_colors=None,
    base_markers=None,
    show_unit_lines=True,
):
    """
    Scatter/line plot of degradation vs time, grouped by stress level.

    Parameters
    ----------
    t : array-like
        Time.
    D : array-like
        Degradation / performance.
    stress : array-like
        Generic stress (temperature, weight, voltage, etc.).
    unit : array-like or None
        Unit IDs; if given and show_unit_lines=True, connect repeated
        measurements from the same unit at the same stress.
    stress_label : str
        Label for the stress axis / legend (e.g. "Temperature (°C)",
        "Applied weight S (g)", "Voltage (V)").
    legend_title : str
        Legend title (e.g. "Temperature", "Weight level").
    scale : {"linear","semilogx","semilogy","loglog"}
        Axis scaling.
    show_unit_lines : bool
        If True and unit is provided, draw line segments joining points
        from the same unit at each stress level.
    """

    t = np.asarray(t, float)
    D = np.asarray(D, float)
    stress = np.asarray(stress, float)

    if unit is None:
        unit = np.zeros_like(t, dtype=int)
    else:
        unit = np.asarray(unit, int)

    fig, ax = plt.subplots(figsize=(8, 6))

    levels = np.unique(stress)
    n_levels = len(levels)

    if base_colors is None:
        base_colors = ["blue", "orange", "k", "green", "red"]
    if base_markers is None:
        base_markers = ["o", "s", "^", "D", "v"]

    styles = {}
    for i, s in enumerate(levels):
        idx = min(i, len(base_colors) - 1)
        styles[s] = dict(color=base_colors[idx],
                         marker=base_markers[idx])

    for s in levels:
        mask_s = (stress == s)
        st = styles[s]

        # scatter
        ax.scatter(
            t[mask_s],
            D[mask_s],
            s=35,
            alpha=0.9,
            color=st["color"],
            marker=st["marker"],
            label=f"{s:g}",
        )

        # optional unit lines
        if show_unit_lines and unit is not None:
            for u in np.unique(unit[mask_s]):
                m_su = mask_s & (unit == u)
                if np.sum(m_su) > 1:
                    # sort by time for nice lines
                    idx_sort = np.argsort(t[m_su])
                    ax.plot(
                        t[m_su][idx_sort],
                        D[m_su][idx_sort],
                        color=st["color"],
                        alpha=0.6,
                        linewidth=1.2,
                    )

    # axis scaling
    if scale == "xlog":
        ax.set_xscale("log")
    elif scale == "ylog":
        ax.set_yscale("log")
    elif scale == "loglog":
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time")
    ax.set_ylabel("Degradation / Performance")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(title=legend_title, frameon=False, ncol=min(n_levels, 3))
    plt.tight_layout()
    plt.show()


def plot_residual_diagnostics(
    t,
    stress,
    resid,
    mu,
    title_prefix="",
    stress_label="Stress level",
    base_colors=None,
    base_markers=None,
):
    """
    Standard 3-panel residual diagnostics:

    1) Residuals vs fitted
    2) Residuals vs time
    3) Normal QQ plot of residuals
    """

    t = np.asarray(t, float)
    stress = np.asarray(stress, float)
    resid = np.asarray(resid, float)
    mu = np.asarray(mu, float)

    levels = np.unique(stress)
    n_levels = len(levels)

    if base_colors is None:
        base_colors = ["blue", "orange", "k", "green", "red"]
    if base_markers is None:
        base_markers = ["o", "s", "^", "D", "v"]

    styles = {}
    for i, s in enumerate(levels):
        idx = min(i, len(base_colors) - 1)
        styles[s] = dict(color=base_colors[idx],
                         marker=base_markers[idx])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Residuals vs Fitted
    ax = axes[0]
    for s in levels:
        mask = (stress == s)
        st = styles[s]
        ax.scatter(
            mu[mask],
            resid[mask],
            color=st["color"],
            marker=st["marker"],
            s=35,
            alpha=0.9,
            label=f"{s:g}",
        )
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"Fitted $\hat{D}$")
    ax.set_ylabel(r"Residual $(D - \hat{D})$")
    ax.set_title("Residuals vs Fitted")
    ax.legend(title=stress_label, frameon=False, ncol=min(n_levels, 3))

    # 2) Residuals vs Time
    ax = axes[1]
    for s in levels:
        mask = (stress == s)
        st = styles[s]
        ax.scatter(
            t[mask],
            resid[mask],
            color=st["color"],
            marker=st["marker"],
            s=35,
            alpha=0.9,
        )
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Time")

    # 3) QQ plot
    ax = axes[2]
    (osm, osr), (slope, intercept, r) = probplot(resid, dist="norm", plot=None)
    ax.scatter(osm, osr, s=20)
    ax.plot(osm, intercept + slope * osm, "r-", lw=2)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered values")
    ax.set_title("Normal QQ plot of residuals")

    suptitle = title_prefix.strip()
    if suptitle:
        fig.suptitle(suptitle, y=1.02)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 4. TTF helpers (for both damage↑ and performance↓ models)
# ---------------------------------------------------------------------

def failure_cdf(mu, sigma, threshold, noise_type="additive",
                scale_type="damage"):
    """
    Compute G(t) = P{failure by time t | mu(t), sigma} at a given time.

    Parameters
    ----------
    mu : array-like or scalar
        Deterministic model mean at time t, on the same scale as the
        underlying process (damage or performance).
    sigma : float
        Noise scale parameter.
    threshold : float
        Failure threshold on the same scale as mu.
    noise_type : {"additive", "multiplicative"}
        - "additive": y ~ Normal(mu, sigma^2)
        - "multiplicative": log y ~ Normal(log mu, sigma^2)
    scale_type : {"damage", "performance"}
        - "damage": increasing damage, fail if D >= Df
        - "performance": decreasing performance, fail if R <= Rf

    Returns
    -------
    G : array-like or scalar
        Failure CDF at this time.
    """
    mu = np.asarray(mu, float)
    if sigma <= 0:
        return np.full_like(mu, np.nan, dtype=float)

    if noise_type == "additive":
        z = (threshold - mu) / sigma
        if scale_type == "damage":
            # P(D >= Df) = 1 - Φ((Df - mu)/σ)
            return 1.0 - norm.cdf(z)
        else:
            # performance: fail if R <= Rf => P(R <= Rf) = Φ((Rf - mu)/σ)
            return norm.cdf(z)

    elif noise_type == "multiplicative":
        mu_pos = np.clip(mu, 1e-12, None)
        thr_pos = max(threshold, 1e-12)
        z = (np.log(thr_pos) - np.log(mu_pos)) / sigma
        if scale_type == "damage":
            # P(D >= Df) = 1 - Φ(z)
            return 1.0 - norm.cdf(z)
        else:
            # P(R <= Rf) = Φ(z)
            return norm.cdf(z)

    else:
        raise ValueError("noise_type must be 'additive' or 'multiplicative'.")


def sample_ttf_from_posterior(mu_fun_factory,
                              theta_samples,
                              sigma_samples,
                              threshold,
                              noise_type="additive",
                              scale_type="damage",
                              n_samps=8000,
                              seed=24,
                              t_hi_max=1e12):
    """
    Sample Time-To-Failure (TTF) from a posterior over (theta, sigma)
    using numerical inversion of the failure CDF.

    Parameters
    ----------
    mu_fun_factory : callable
        Factory that takes a single theta vector and returns a function
            mu_fun(t): array-like
        which computes the deterministic model mean at time t (and appropriate
        stress), already specialised to the eval stress condition.
        Example (inside a class):
            def mu_fun_factory(theta):
                def mu_fun(t):
                    return self._mu_D(theta, t, T_eval_K, self.T_use_K)
                return mu_fun
    theta_samples : array-like, shape (N, p)
        Posterior draws for model parameters theta.
    sigma_samples : array-like, shape (N,)
        Posterior draws for observation noise sigma.
    threshold : float
        Failure threshold (Df or Rf).
    noise_type : {"additive", "multiplicative"}
        Noise model on the observed scale.
    scale_type : {"damage", "performance"}
        Scale type (see failure_cdf).
    n_samps : int
        Number of TTF samples to generate (with replacement across posterior draws).
    seed : int
        RNG seed.
    t_hi_max : float
        Upper bound cap for search.

    Returns
    -------
    ttf_samples : np.ndarray
        Array of TTF samples.
    """

    theta_samples = np.asarray(theta_samples)
    sigma_samples = np.asarray(sigma_samples, float)

    n_draws = theta_samples.shape[0]
    if n_draws == 0:
        raise RuntimeError("No posterior draws provided.")

    rng = np.random.default_rng(seed)
    idx = rng.integers(low=0, high=n_draws, size=n_samps)

    ttf_list = []

    def grow_bracket(G_draw, u, t_hi0, t_hi_max=t_hi_max):
        """Grow t_hi until G_draw(t_hi) >= u."""
        t_hi = max(1.0, t_hi0)
        val = G_draw(t_hi)
        it = 0
        while val < u and t_hi < t_hi_max and it < 60:
            t_hi *= 2.0
            val = G_draw(t_hi)
            it += 1
        return t_hi if val >= u else np.nan

    for i in idx:
        theta_i = theta_samples[i]
        sigma_i = sigma_samples[i]

        mu_fun = mu_fun_factory(theta_i)

        def G_draw(t):
            mu_t = mu_fun(t)
            return failure_cdf(
                mu_t,
                sigma_i,
                threshold=threshold,
                noise_type=noise_type,
                scale_type=scale_type,
            )

        u = rng.random()

        # already failed at t=0?
        if G_draw(0.0) >= u:
            ttf_list.append(0.0)
            continue

        # crude starting bracket
        t_hi0 = 1.0
        t_hi = grow_bracket(G_draw, u, t_hi0)
        if not np.isfinite(t_hi):
            # try again from 1 (already does) or bail
            continue

        try:
            t_root = brentq(lambda tt: G_draw(tt) - u, 0.0, t_hi, maxiter=200)
            if np.isfinite(t_root) and t_root >= 0:
                ttf_list.append(t_root)
        except ValueError:
            # no sign change / numerical issue; skip this draw
            continue

    return np.asarray(ttf_list, float)


def summarize_ttf(samples, hdi_prob=0.95,
                  label="TTF",
                  unit_label="time units",
                  as_markdown=True):
    """
    Summarise posterior TTF samples with mean, median, HDI and ETI.
    """
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("No valid TTF samples to summarise.")

    mean = float(s.mean())
    median = float(np.median(s))
    lo_hdi, hi_hdi = hdi(s, cred_mass=hdi_prob)

    eti_lo, eti_hi = np.percentile(
        s,
        [(1 - hdi_prob) / 2 * 100,
         (1 - (1 - hdi_prob) / 2) * 100]
    )

    hdi_pct = int(hdi_prob * 100)

    if as_markdown:
        lines = []
        lines.append(f"### Posterior summary for {label}\n")
        lines.append(f"- Mean life: {mean:.4e} {unit_label}")
        lines.append(f"- Median life: {median:.4e} {unit_label}")
        lines.append(f"- {hdi_pct}% HDI: [{lo_hdi:.4e}, {hi_hdi:.4e}] {unit_label}")
        lines.append(f"- {hdi_pct}% ETI: [{eti_lo:.4e}, {eti_hi:.4e}] {unit_label}")
        display(Markdown("\n".join(lines)))

    return {
        "mean": float(f"{mean:.4e}"),
        "median": float(f"{median:.4e}"),
        "hdi": (float(f"{lo_hdi:.4e}"), float(f"{hi_hdi:.4e}")),
        "eti": (float(f"{eti_lo:.4e}"), float(f"{eti_hi:.4e}")),
    }


def plot_ttf_hist_with_fits(ttf_samples,
                            unit_label="time units",
                            title="TTF posterior and best distribution fits",
                            bins=60):
    """
    Plot TTF histogram and fit Weibull / Gamma / Lognormal using reliability.Fitters.Fit_Everything.

    If Fit_Everything fails, falls back to a histogram-only plot.
    """
    ttf_samples = np.asarray(ttf_samples, float)
    ttf_clean = ttf_samples[np.isfinite(ttf_samples)]
    if ttf_clean.size == 0:
        raise ValueError("No finite TTF samples for plotting.")

    try:
        from reliability.Fitters import Fit_Everything

        Best_fit = Fit_Everything(
            failures=ttf_clean,
            show_histogram_plot=False,
            show_probability_plot=False,
            show_PP_plot=False,
            show_best_distribution_probability_plot=False,
        )
    except Exception as e:
        print("Fit_Everything failed; showing histogram only. Error:", e)
        plt.figure(figsize=(10, 6))
        plt.hist(ttf_clean, bins=bins, density=True, alpha=0.7)
        plt.xlabel(f"TTF ({unit_label})")
        plt.ylabel("Posterior density")
        plt.title("TTF posterior (histogram only)")
        plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        plt.show()
        return

    # x-grid for PDFs
    x = np.linspace(0.01, np.max(ttf_clean), 500)

    # PDFs using SciPy, parameterised by Best_fit
    W_pdf = weibull_min.pdf(
        x,
        c=Best_fit.Weibull_2P_beta,
        scale=Best_fit.Weibull_2P_alpha,
    )
    G_pdf = gamma.pdf(
        x,
        a=Best_fit.Gamma_2P_beta,
        scale=Best_fit.Gamma_2P_alpha,
        loc=0,
    )
    L_pdf = lognorm.pdf(
        x,
        s=Best_fit.Lognormal_2P_sigma,
        scale=np.exp(Best_fit.Lognormal_2P_mu),
        loc=0,
    )

    plt.figure(figsize=(10, 8))
    plt.hist(ttf_clean, bins=bins, density=True, alpha=0.6,
             label="Posterior TTF samples")

    plt.plot(
        x, L_pdf, c="black",
        label=fr'Lognormal ($\sigma$={Best_fit.Lognormal_2P_sigma:.2f}, '
              fr'$\exp(\mu)$={np.exp(Best_fit.Lognormal_2P_mu):.0f})'
    )
    plt.plot(
        x, G_pdf, c="orange",
        label=fr'Gamma ($\beta$={Best_fit.Gamma_2P_beta:.2f}, '
              fr'$\alpha$={Best_fit.Gamma_2P_alpha:.0f})'
    )
    plt.plot(
        x, W_pdf, c="blue",
        label=fr'Weibull ($\beta$={Best_fit.Weibull_2P_beta:.2f}, '
              fr'$\alpha$={Best_fit.Weibull_2P_alpha:.0f})'
    )

    plt.xlabel(f"TTF ({unit_label})")
    plt.ylabel("Posterior density")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

K_BOLTZ_eV = 8.617333262e-5  # eV/K 

class ADTDataGenerator:
    """
    Synthetic ADT data generator with unit-level parameter variation (Option B).

    For each model, we assume a population parameter vector theta_pop and then:
        - For each *unit* j, draw unit-specific theta_j around theta_pop
        - For each observation (t_ij, S_ij) simulate:
              D_ij ~ Noise( mu(t_ij, S_ij; theta_j) )

    Supported models
    ----------------
    "sqrt_arrhenius":
        D(t, T) = g0 + g1 * sqrt(t) * exp( Ea / k * (1/T_use - 1/T) )
        theta = (g0, g1, Ea[eV])

    "power_power":
        D(t, S) = b * S^n * t^m
        theta = (b, n, m)

    "power_exponential":
        D(t, S) = b * exp(a / S) * t^n
        theta = (b, a, n)
        
    "exponential_arrhenius":
        D(t, T) = b * exp(a * t) * exp( Ea / k * (1/T_use - 1/T) )
        theta = (b, a, Ea[eV])


    Noise model
    -----------
    noise = "additive":
        D_ij = mu_ij + eps_ij,  eps_ij ~ N(0, sigma_meas^2)

    noise = "multiplicative":
        log D_ij = log mu_ij + eps_ij,  eps_ij ~ N(0, sigma_meas^2)

    Unit-level parameter variation (Option B)
    -----------------------------------------
    For parameters constrained to be > 0:
        log(theta_j[i]) ~ Normal( log(theta_pop[i]), unit_log_sd^2 )

    For signed parameters (e.g. 'a' in power_exponential):
        theta_j[i] ~ Normal( theta_pop[i], unit_signed_sd^2 * scale^2 )
        where scale = max(|theta_pop[i]|, 1.0)

    Parameters
    ----------
    model : {"sqrt_arrhenius", "power_power", "power_exponential"}
        Degradation model form.
    theta : sequence of floats
        Population parameter vector (see above).
    sigma_meas : float
        Measurement noise scale (on D or log D depending on `noise`).
    noise : {"additive", "multiplicative"}, default "additive"
        Observation noise model.
    stress_use : float, optional
        Use stress (°C for sqrt_arrhenius, same units as S otherwise).
        Required for sqrt_arrhenius; optional otherwise.
    Df : float, optional
        Failure threshold on *damage* scale (fail when D >= Df).
        Used if `truncate_at_failure=True` in `generate`.
    unit_cv_pos : float, default 0.15
        Rough coefficient-of-variation for positive parameters; internally
        converted to a log-space standard deviation.
    unit_cv_signed : float, default 0.15
        Relative standard deviation for signed parameters (e.g. 'a').
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        model,
        theta,
        sigma_meas,
        noise="additive",
        stress_use=None,
        Df=None,
        unit_cv_pos=0.15,
        unit_cv_signed=0.15,
        seed=None,
    ):
        
        self.model = model.lower()
        if self.model not in (
            "sqrt_arrhenius",
            "power_power",
            "power_exponential",
            "exponential_arrhenius",
        ):
            raise ValueError(f"Unsupported model: {model}")


        self.theta_pop = np.asarray(theta, float)
        self.p = len(self.theta_pop)

        self.sigma_meas = float(sigma_meas)
        self.noise = noise.lower()
        if self.noise not in ("additive", "multiplicative"):
            raise ValueError("noise must be 'additive' or 'multiplicative'")

        self.stress_use = stress_use
        if self.model in ("sqrt_arrhenius", "exponential_arrhenius"):
            if stress_use is None:
                raise ValueError(f"{self.model} requires stress_use (°C).")
            self.T_use_K = float(stress_use) + 273.15

        self.Df = Df

        # Unit-level variability controls
        # For positive parameters we use lognormal with log-sd from unit_cv_pos.
        self.unit_log_sd = np.log(1.0 + float(unit_cv_pos))
        # For signed parameters we use Normal with sd = unit_cv_signed * |theta_pop|
        self.unit_signed_rel_sd = float(unit_cv_signed)

        self.rng = np.random.default_rng(seed)

        # Which parameters are constrained positive vs signed (by index)
        if self.model == "sqrt_arrhenius":
            # (g0, g1, Ea) all positive in your fits
            self.pos_idx = {0, 1, 2}
        elif self.model == "power_power":
            # (b, n, m) all > 0
            self.pos_idx = {0, 1, 2}
        elif self.model == "power_exponential":
            # (b, a, n) with a allowed to be signed
            self.pos_idx = {0, 2}
        elif self.model == "exponential_arrhenius":
            # (b, a, Ea) all > 0 in this formulation
            self.pos_idx = {0, 1, 2}
        # signed indexes are the complement
        self.signed_idx = {i for i in range(self.p) if i not in self.pos_idx}

    # ------------------------------------------------------------------
    # Public helper: stress grid
    # ------------------------------------------------------------------
    @staticmethod
    def default_stress_grid(stress_use, deltas):
        """
        E.g. stress_use=50, deltas=[-40, -10, +50] -> [10, 40, 100].

        Just a convenience helper.
        """
        stress_use = float(stress_use)
        return np.asarray([stress_use + d for d in deltas], float)

    # ------------------------------------------------------------------
    # Core: unit-level parameter sampling
    # ------------------------------------------------------------------
    def _sample_theta_unit(self):
        theta_u = np.empty_like(self.theta_pop)
        for i, theta_i in enumerate(self.theta_pop):
            if i in self.pos_idx:
                # positive → lognormal
                base = max(theta_i, 1e-12)
                mu_log = np.log(base)
                theta_u[i] = np.exp(self.rng.normal(mu_log, self.unit_log_sd))
            else:
                # signed → Normal around mean, *relative* scale
                # OLD: scale = max(abs(theta_i), 1.0)
                scale = max(abs(theta_i), 1e-8)   # tiny floor to avoid exactly zero
                sd = self.unit_signed_rel_sd * scale
                theta_u[i] = self.rng.normal(theta_i, sd)
        return theta_u


    # ------------------------------------------------------------------
    # Core: mean degradation model
    # ------------------------------------------------------------------
    def _mu(self, theta, t, stress):
        """
        Mean degradation mu(t, stress; theta) for the chosen model.

        t      : scalar or array
        stress : scalar or array broadcastable to t
        """
        t = np.asarray(t, float)
        S = np.asarray(stress, float)

        if self.model == "sqrt_arrhenius":
            g0, g1, Ea = theta
            T_C = S
            T_K = T_C + 273.15
            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K))
            return g0 + g1 * np.sqrt(t) * accel

        elif self.model == "power_power":
            b, n, m = theta
            return b * (S ** n) * (t ** m)

        elif self.model == "power_exponential":
            b, a, n = theta
            t_pos = np.clip(t, 1e-12, None)
            S_pos = np.clip(S, 1e-12, None)
            a_over_S = a / S_pos
            expo = np.clip(a_over_S, -50.0, 50.0)   # numerical safety
            return b * np.exp(expo) * (t_pos ** n)
            
        elif self.model == "exponential_arrhenius":
            # theta = (b, a, Ea), stress = T in °C
            b, a, Ea = theta
            t = np.asarray(t, float)
            S = np.asarray(stress, float)
            t_pos = np.clip(t, 0.0, None)
            # Convert °C -> K
            T_C = S
            T_K = T_C + 273.15
            # Arrhenius acceleration relative to use temperature
            accel = np.exp(Ea / K_BOLTZ_eV * (1.0 / self.T_use_K - 1.0 / T_K))
            # Exponential in time * Arrhenius accel
            return b * np.exp(a * t_pos) * accel

        else:
            raise RuntimeError("Unknown model inside _mu.")

    # ------------------------------------------------------------------
    # Core: one observation with noise
    # ------------------------------------------------------------------
    def _draw_observation(self, mu):
        if self.noise == "additive":
            return mu + self.sigma_meas * self.rng.normal()
        else:  # multiplicative
            mu_pos = max(mu, 1e-12)
            log_y = np.log(mu_pos) + self.sigma_meas * self.rng.normal()
            return np.exp(log_y)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def generate(
        self,
        stress_levels,
        times,
        n_units_per_stress,
        truncate_at_failure=False,
    ):
        """
        Generate synthetic ADT degradation data.

        Parameters
        ----------
        stress_levels : array-like
            Unique stress levels (temperature, weight, etc.).
        times : array-like
            Observation times (same for all units at a given stress).
        n_units_per_stress : int
            Number of units to simulate at each stress.
        truncate_at_failure : bool, default False
            If True and Df is set, stop generating observations for a unit
            once D >= Df.

        Returns
        -------
        data : dict of np.ndarray
            Keys: "Time", "Stress", "Degradation", "Unit"
        """
        S_levels = np.asarray(stress_levels, float).ravel()
        times = np.asarray(times, float).ravel()

        t_list = []
        s_list = []
        d_list = []
        u_list = []

        unit_id = 0

        for S in S_levels:
            for _ in range(int(n_units_per_stress)):
                theta_u = self._sample_theta_unit()

                for t in times:
                    mu = self._mu(theta_u, t, S)
                    y = self._draw_observation(mu)

                    t_list.append(t)
                    s_list.append(S)
                    d_list.append(y)
                    u_list.append(unit_id)

                    if truncate_at_failure and (self.Df is not None):
                        if y >= self.Df:
                            break

                unit_id += 1

        data = {
            "Time": np.asarray(t_list, float),
            "Stress": np.asarray(s_list, float),
            "Degradation": np.asarray(d_list, float),
            "Unit": np.asarray(u_list, int),
        }
        return data

############################################################################################################################
#                                                      START ALT ADDITIONS
############################################################################################################################

# Domain checking per L-S Fn and Distribution
POSITIVE = "positive"
REAL = "real"

_MODEL_PARAM_DOMAIN = {
    "Power":            {"a": POSITIVE, "n": REAL},
    "Exponential":      {"b": POSITIVE, "a": REAL},
    "Eyring":           {"a": REAL, "c": REAL},
    "Dual_Exponential": {"c": POSITIVE, "a": REAL, "b": REAL},
    "Dual_Power":       {"c": POSITIVE, "m": REAL, "n": REAL},
    "Power_Exponential":{"c": POSITIVE, "a": REAL, "n": REAL},
    "Sqrt_Arrhenius_ADT":        {"g0": POSITIVE,  "g1": POSITIVE,  "Ea": POSITIVE},
    "Power_Power_ADT":           {"b": POSITIVE, "n_S": REAL, "m_t": REAL},
    "Power_Exponential_ADT":     {"b": POSITIVE, "a": REAL, "n": REAL},
    "Power_Arrhenius_ADT":       {"a": POSITIVE, "n": REAL, "Ea": POSITIVE},
    "Exponential_Arrhenius_ADT": {"b": POSITIVE, "a": POSITIVE, "Ea": POSITIVE},
    "Linear_Arrhenius_ADT":      {"a": REAL, "b": POSITIVE, "Ea": POSITIVE},
    "Mitsuom_Arrhenius_ADT":     {"a": POSITIVE, "b": POSITIVE, "Ea": POSITIVE},
    "Mitsuom_Arrhenius_Power1_ADT":     {"a": POSITIVE, "b": POSITIVE, "Ea": POSITIVE, "n": POSITIVE, "c": REAL},
}

_DIST_PARAM_DOMAIN = {
    "Weibull":   {"beta": POSITIVE},
    "Lognormal": {"sigma": POSITIVE},
    "Normal":    {"sigma": POSITIVE},
    "NormalError": {"sigma": POSITIVE},
}

def get_param_domain(model: str, dist: str, name: str, override: str | None = None) -> str:
    if override in (POSITIVE, REAL):
        return override
    d_dom = _DIST_PARAM_DOMAIN.get(str(dist), {}).get(name)
    if d_dom:
        return d_dom
    m_dom = _MODEL_PARAM_DOMAIN.get(str(model), {}).get(name)
    return m_dom if m_dom else REAL

# Generic Prior generation   
def log_prior_single(name: str, x: float, prior, domain: str | None = None) -> float:
    dom = domain if domain in (POSITIVE, REAL) else REAL
    if prior is None:
        if dom == POSITIVE:
            return ss.norm.logpdf(np.log(x), loc=0.0, scale=5.0) - np.log(x) if x > 0 else -np.inf
        return ss.norm.logpdf(x, loc=0.0, scale=5.0)

    if isinstance(prior, (list, tuple)) and len(prior) >= 2:
        kind = str(prior[0]).lower(); seq = prior; mp = None
    elif isinstance(prior, dict) and "type" in prior:
        kind = str(prior["type"]).lower(); seq = None; mp = prior
    else:
        raise ValueError(f"Unrecognized prior format for {name}: {prior}")

    if dom == POSITIVE and x <= 0:
        return -np.inf

    if kind == "uniform":
        if seq is not None:
            if len(seq) < 3: raise ValueError('Uniform needs ("Uniform", low, high).')
            low, high = float(seq[1]), float(seq[2])
        else:
            low, high = float(mp["low"]), float(mp["high"])
        return ss.uniform.logpdf(x, low, high - low) if low <= x <= high else -np.inf

    if kind == "normal":
        if seq is not None:
            if len(seq) < 3: raise ValueError('Normal needs ("Normal", mu, sigma).')
            mu, sigma = float(seq[1]), float(seq[2])
        else:
            mu, sigma = float(mp["mu"]), float(mp["sigma"])
        return ss.norm.logpdf(x, mu, sigma)

    if kind == "lognormal":
        if x <= 0: return -np.inf
        if seq is not None:
            if len(seq) < 3: raise ValueError('Lognormal needs ("Lognormal", mu_log, sigma_log).')
            mu_log, sigma_log = float(seq[1]), float(seq[2])
        else:
            mu_log, sigma_log = float(mp["mu"]), float(mp["sigma"])
        return ss.norm.logpdf(np.log(x), mu_log, sigma_log) - np.log(x)

    if kind in ("loguniform", "log-uniform", "log_uniform"):
        if seq is not None:
            if len(seq) < 3: raise ValueError('LogUniform needs ("LogUniform", low, high).')
            low, high = float(seq[1]), float(seq[2])
        else:
            low, high = float(mp["low"]), float(mp["high"])
        if not (low > 0 and high > 0 and high > low):
            raise ValueError("LogUniform bounds must satisfy 0 < low < high.")
        if not (low <= x <= high): return -np.inf
        return -np.log(x) - np.log(np.log(high) - np.log(low))

    if kind in ("triangular", "triangle", "triang"):
        if seq is not None:
            if   len(seq) == 3: low, high = float(seq[1]), float(seq[2]); mode = 0.5*(low+high)
            elif len(seq) == 4: low, mode, high = float(seq[1]), float(seq[2]), float(seq[3])
            else: raise ValueError('Triangular needs ("Triangular", low, high) or ("Triangular", low, mode, high).')
        else:
            low = float(mp["low"]); high = float(mp["high"])
            mode = float(mp["mode"]) if "mode" in mp else 0.5*(low+high)
        if not (low < high and low <= mode <= high): raise ValueError("Triangular: low < high and low <= mode <= high.")
        if x < low or x > high: return -np.inf
        if x == mode:  # safe peak log-pdf via one-sided limit
            eps = 1e-300
            left  = np.log(2.0) + np.log(max(x-low, eps)) - np.log(high-low) - np.log(max(mode-low, eps))
            right = np.log(2.0) + np.log(max(high-x, eps)) - np.log(high-low) - np.log(max(high-mode, eps))
            return max(left, right)
        if x < mode:
            return np.log(2.0) + np.log(x-low) - np.log(high-low) - np.log(mode-low)
        return np.log(2.0) + np.log(high-x) - np.log(high-low) - np.log(high-mode)

    raise ValueError(f"Unsupported prior kind '{kind}' for {name}")

def log_prior_vector(model: str, dist: str, theta_dict: dict, priors: dict | None) -> float:
    total = 0.0
    for name, val in theta_dict.items():
        pr = None if priors is None else priors.get(name)
        dom = get_param_domain(model, dist, name)
        lp = log_prior_single(name, float(val), pr, domain=dom)
        if not np.isfinite(lp):
            return -np.inf
        total += lp
    return total
    
############################################################################################################################
#                                                      END ADDITIONS
############################################################################################################################

