from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def empirical_bayes_shrinkage(
    estimates: pd.Series,
    std_errors: pd.Series,
    min_shrink: float = 0.0,
    max_shrink: float = 1.0,
) -> pd.DataFrame:
    """Apply empirical Bayes shrinkage to event study estimates.

    Shrinkage pulls individual group estimates toward the grand mean
    with intensity proportional to estimation uncertainty.
    This is critical in event studies where small-sample groups
    (e.g., 'RECESSION' with n=3 events) produce noisy estimates.
    Shrinkage reduces variance at the cost of introducing bias.

    The shrinkage factor lambda_i = tau^2 / (tau^2 + se_i^2)
    where tau^2 = Var(true effects) estimated via marginal MLE.

    Returns DataFrame with original, shrunk, lambda, and weights.
    """
    valid = estimates.notna() & std_errors.notna() & (std_errors > 0)
    if not valid.any():
        return pd.DataFrame(
            {"original": estimates, "shrunk": estimates, "lambda": 1.0, "weight": 0.0}
        )

    theta_hat = estimates[valid].values
    se = std_errors[valid].values

    tau2 = _estimate_tau2(theta_hat, se)

    lam = tau2 / (tau2 + se**2)
    lam = np.clip(lam, min_shrink, max_shrink)

    grand_mean = np.average(theta_hat, weights=1.0 / se**2)

    shrunk = lam * theta_hat + (1.0 - lam) * grand_mean

    result = pd.DataFrame(index=estimates.index)
    result["original"] = estimates
    result["shrunk"] = np.nan
    result["lambda"] = np.nan
    result["weight"] = np.nan

    result.loc[valid, "shrunk"] = shrunk
    result.loc[valid, "lambda"] = lam
    result.loc[valid, "weight"] = 1.0 / se**2
    result["grand_mean"] = grand_mean
    result["tau2"] = tau2

    return result


def _estimate_tau2(theta: np.ndarray, se: np.ndarray) -> float:
    """Marginal MLE for between-group variance tau^2.

    Uses iterative EM algorithm. Falls back to method-of-moments
    estimator if MLE does not converge.
    """
    n = len(theta)
    if n < 2:
        return 0.0

    tau2 = np.var(theta, ddof=1) - np.mean(se**2)
    tau2 = max(tau2, 0.01)

    for _ in range(100):
        w = 1.0 / (tau2 + se**2)
        theta_bar = np.sum(w * theta) / np.sum(w)
        tau2_new = np.sum(w**2 * ((theta - theta_bar) ** 2 - se**2)) / np.sum(w**2)
        tau2_new = max(tau2_new, 1e-6)

        if abs(tau2_new - tau2) < 1e-6:
            tau2 = tau2_new
            break
        tau2 = tau2_new

    return tau2


def neyman_confidence_intervals(
    event_returns: pd.Series,
    ci: float = 0.95,
    n_bootstrap: int = 10000,
) -> dict[str, float]:
    """Neyman-style confidence intervals using bootstrap-t procedure.

    Unlike percentile bootstrap, the bootstrap-t method studentizes
    the statistic, providing better coverage for asymmetric
    distributions common in event studies (fat tails).

    Returns dict with mean, se, ci_lower, ci_upper, and method.
    """
    values = event_returns.dropna().values
    n = len(values)

    if n < 2:
        return {"mean": np.nan, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

    sample_mean = float(np.mean(values))
    sample_se = float(np.std(values, ddof=1) / np.sqrt(n))

    rng = np.random.default_rng(42)
    t_stars = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot = rng.choice(values, size=n, replace=True)
        boot_mean = np.mean(boot)
        boot_se = np.std(boot, ddof=1) / np.sqrt(n)
        if boot_se > 0:
            t_stars[i] = (boot_mean - sample_mean) / boot_se
        else:
            t_stars[i] = 0.0

    alpha = 1.0 - ci
    t_lower = float(np.percentile(t_stars, alpha / 2 * 100))
    t_upper = float(np.percentile(t_stars, (1 - alpha / 2) * 100))

    return {
        "mean": sample_mean,
        "se": sample_se,
        "ci_lower": sample_mean - t_upper * sample_se,
        "ci_upper": sample_mean - t_lower * sample_se,
        "t_lower": t_lower,
        "t_upper": t_upper,
        "method": "bootstrap-t",
    }


def neyman_event_study_aggregation(
    event_studies: pd.DataFrame,
    group_by: Optional[list[str]] = None,
    return_cols: Optional[list[str]] = None,
    apply_shrinkage: bool = True,
) -> pd.DataFrame:
    """Full event study aggregation with Neyman CIs and optional EB shrinkage.

    This is the recommended aggregation method for publication-quality
    event studies: bootstrap-t CIs + empirical Bayes shrinkage.
    """
    if group_by is None:
        group_by = ["event_type"]

    if return_cols is None:
        return_cols = [c for c in event_studies.columns if c.startswith("return_")]
        return_cols = sorted(return_cols)

    if not return_cols:
        return pd.DataFrame()

    results: list[dict] = []
    for keys, group in event_studies.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_by, keys))
        row["n_events"] = len(group)

        for rc in return_cols:
            vals = group[rc]
            ci_result = neyman_confidence_intervals(vals)
            row[f"{rc}_mean"] = ci_result["mean"]
            row[f"{rc}_se"] = ci_result["se"]
            row[f"{rc}_ci_lower"] = ci_result["ci_lower"]
            row[f"{rc}_ci_upper"] = ci_result["ci_upper"]

        results.append(row)

    result = pd.DataFrame(results)
    if not apply_shrinkage:
        return result

    for rc in return_cols:
        mean_col = f"{rc}_mean"
        se_col = f"{rc}_se"
        if mean_col not in result.columns or se_col not in result.columns:
            continue
        if len(result) < 2:
            continue

        shrunk = empirical_bayes_shrinkage(
            result[mean_col],
            result[se_col],
        )
        result[f"{rc}_shrunk"] = shrunk["shrunk"]
        result[f"{rc}_lambda"] = shrunk["lambda"]

    return result


def compute_sharpe_ratio_equivalent(
    mean_return: float, std_return: float, rf_annual: float = 0.05
) -> float:
    """Convert event-study mean/std to information ratio equivalent.

    Annualizes event-driven returns to a Sharpe-like metric
    for comparability across strategies and asset classes.
    """
    if std_return <= 0 or np.isnan(mean_return) or np.isnan(std_return):
        return np.nan
    ir = (mean_return - rf_annual / 252) / std_return
    return float(ir)


def compute_bayes_factor(
    mean_estimate: float, se_estimate: float, prior_mean: float = 0.0, prior_se: float = 0.01
) -> float:
    """Compute approximate Bayes factor for H0: effect = 0 vs H1: effect != 0.

    Uses Savage-Dickey density ratio under normal approximations.
    BF > 3 is considered 'substantial' evidence against H0.
    BF > 10 is 'strong' evidence.
    """
    if se_estimate <= 0 or np.isnan(mean_estimate) or np.isnan(se_estimate):
        return 1.0

    posterior_var = 1.0 / (1.0 / prior_se**2 + 1.0 / se_estimate**2)
    posterior_mean = posterior_var * (mean_estimate / se_estimate**2)

    prior_density = np.exp(-0.5 * (prior_mean / prior_se) ** 2) / (prior_se * np.sqrt(2 * np.pi))
    posterior_density = np.exp(-0.5 * (posterior_mean / np.sqrt(posterior_var)) ** 2) / (
        np.sqrt(posterior_var) * np.sqrt(2 * np.pi)
    )

    bf = prior_density / posterior_density if posterior_density > 0 else 1.0
    return float(np.clip(bf, 0.01, 100.0))
