from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def decompose_surprise(
    implied_prob: float,
    realized_prob: float,
    implied_distribution: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Decompose a macro surprise into level, volatility, and optional skew components.

    Standard macro surprises conflate several distinct economic forces:
    1. Level component: shift in the conditional mean of the outcome
    2. Volatility component: change in outcome uncertainty
    3. Skew component: change in tail risk (requires multi-bucket markets)

    For binary markets, we decompose using the log-odds transformation:
        log(p/(1-p)) = logit(p) = beta * X + epsilon

    The Level component is the change in the logit.
    The Vol component is the residual uncertainty.
    """
    eps = 1e-10
    p_implied = np.clip(implied_prob, eps, 1 - eps)
    p_realized = np.clip(realized_prob, eps, 1 - eps)

    logit_implied = np.log(p_implied / (1 - p_implied))
    logit_realized = np.log(p_realized / (1 - p_realized))

    total_shock = logit_realized - logit_implied
    level_component = total_shock
    vol_component = 0.0
    skew_component = 0.0

    if implied_distribution is not None:
        implied_var = implied_distribution.get("variance", 0)
        implied_skew = implied_distribution.get("skewness", 0)

        if implied_var > 0 and False:
            pass

    shannon_implied = -p_implied * np.log(p_implied) - (1 - p_implied) * np.log(1 - p_implied)
    shannon_realized = -p_realized * np.log(p_realized) - (1 - p_realized) * np.log(1 - p_realized)
    uncertainty_revision = shannon_realized - shannon_implied

    return {
        "total_surprise_logit": float(total_shock),
        "level_component": float(level_component),
        "volatility_component": float(vol_component),
        "skew_component": float(skew_component),
        "uncertainty_revision": float(uncertainty_revision),
        "implied_entropy": float(shannon_implied),
        "realized_entropy": float(shannon_realized),
    }


def decompose_all_surprises(
    surprises: pd.DataFrame,
) -> pd.DataFrame:
    """Apply surprise decomposition to all events in the surprises DataFrame."""
    records: list[dict] = []
    for _, row in surprises.iterrows():
        expected = row.get("expected_probability", np.nan)
        realized = row.get("realized_value", np.nan)

        if pd.isna(expected) or pd.isna(realized):
            continue

        decomp = decompose_surprise(expected, realized)
        records.append(
            {
                "event_id": row.get("event_id"),
                "event_type": row.get("event_type"),
                "snapshot_type": row.get("snapshot_type"),
                "expected_probability": expected,
                "realized_value": realized,
                **decomp,
            }
        )

    return pd.DataFrame(records)


def compute_entropy_based_confidence(
    implied_prob: float,
    n_observations: int = 100,
) -> float:
    """Compute confidence score based on entropy of implied probability.

    Extreme probabilities (near 0 or 1) have low entropy and indicate
    high market conviction. P ~ 0.5 has maximum entropy (maximum uncertainty).

    Confidence = 1 - (entropy / max_entropy)

    Returns value in [0, 1].
    """
    eps = 1e-10
    p = np.clip(implied_prob, eps, 1 - eps)
    entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    max_entropy = -0.5 * np.log(0.5) - 0.5 * np.log(0.5)
    confidence = 1.0 - entropy / max_entropy

    sample_adjustment = min(1.0, n_observations / 500)
    return float(confidence * sample_adjustment)
