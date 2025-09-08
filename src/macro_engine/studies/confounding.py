from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def detect_confounding_events(
    event_calendar: pd.DataFrame,
    max_gap_hours: float = 24.0,
) -> pd.DataFrame:
    """Detect confounding events occurring within a window of each other.

    Confounding events are a first-order concern in macro event studies:
    when CPI and FOMC fall on the same day, observed asset returns
    reflect both surprises simultaneously. Without controlling for
    confounds, attribution is biased.

    This function flags events within `max_gap_hours` of another event
    and groups them into confounding clusters.
    """
    cal = event_calendar.copy()
    cal["release_datetime"] = pd.to_datetime(cal["release_datetime"])
    cal = cal.sort_values("release_datetime").reset_index(drop=True)

    max_gap = pd.Timedelta(hours=max_gap_hours)
    cal["confounded"] = False
    cal["confounding_group"] = -1
    cal["confounding_events"] = ""
    cal["distance_to_nearest_hours"] = np.inf

    for i in range(len(cal)):
        dt_i = cal.loc[i, "release_datetime"]
        min_dist = np.inf

        for j in range(len(cal)):
            if i == j:
                continue
            dt_j = cal.loc[j, "release_datetime"]
            dist = abs((dt_i - dt_j).total_seconds()) / 3600.0

            if dist < max_gap_hours:
                cal.at[i, "confounded"] = True
                if dist < min_dist:
                    min_dist = dist
                existing = cal.at[i, "confounding_events"]
                other = cal.loc[j, "event_type"]
                cal.at[i, "confounding_events"] = f"{existing}, {other}" if existing else other

        if min_dist < np.inf:
            cal.at[i, "distance_to_nearest_hours"] = min_dist

    group_id = 0
    for i in range(len(cal)):
        if cal.loc[i, "confounded"] and cal.loc[i, "confounding_group"] == -1:
            cluster_start = cal.loc[i, "release_datetime"] - max_gap
            cluster_end = cal.loc[i, "release_datetime"] + max_gap
            mask = (
                (cal["release_datetime"] >= cluster_start)
                & (cal["release_datetime"] <= cluster_end)
                & (cal["confounding_group"] == -1)
            )
            cal.loc[mask, "confounding_group"] = group_id
            group_id += 1

    return cal


def flag_confounded_surprises(
    surprises: pd.DataFrame,
    confounded_events: pd.DataFrame,
) -> pd.DataFrame:
    """Merge confounding flags into surprises DataFrame."""
    s = surprises.copy()
    s = s.merge(
        confounded_events[["event_id", "confounded", "confounding_group", "confounding_events"]],
        on="event_id",
        how="left",
    )
    s["confounded"] = s["confounded"].fillna(False)
    s["confounding_group"] = s["confounding_group"].fillna(-1).astype(int)
    s["confounding_events"] = s["confounding_events"].fillna("")
    return s


def compute_confounding_robustness(
    event_studies: pd.DataFrame,
    return_col: str = "return_1D",
) -> dict[str, float]:
    """Test whether confounded events drive the main results.

    Compares mean returns for confounded vs. unconfounded events.
    If confounded events produce systematically different returns,
    the event study results may be biased.
    """
    if "confounded" not in event_studies.columns or event_studies.empty:
        return {}

    confounded = event_studies[event_studies["confounded"] == True][return_col].dropna()
    unconfounded = event_studies[event_studies["confounded"] == False][return_col].dropna()

    if len(confounded) < 2 or len(unconfounded) < 2:
        return {}

    from scipy import stats as sp_stats

    tstat, pval = sp_stats.ttest_ind(confounded, unconfounded, equal_var=False)

    return {
        "n_confounded": len(confounded),
        "n_unconfounded": len(unconfounded),
        "mean_confounded": float(confounded.mean()),
        "mean_unconfounded": float(unconfounded.mean()),
        "diff_confounded_unconfounded": float(confounded.mean() - unconfounded.mean()),
        "welch_tstat": float(tstat),
        "welch_pvalue": float(pval),
        "significant_diff_5pct": bool(pval < 0.05),
    }


def residualize_event_returns(
    event_studies: pd.DataFrame,
    confounded_events: pd.DataFrame,
    return_col: str = "return_1D",
) -> pd.DataFrame:
    """Residualize event returns by subtracting confounded-event effects.

    Uses a simple linear model: return_i = alpha + beta * n_confounds_i + epsilon_i
    The residual epsilon_i is the 'clean' return after controlling for confounds.
    """
    es = event_studies.copy()
    merge_cols = []
    for c in ["confounded", "confounding_group"]:
        if c in confounded_events.columns and c not in es.columns:
            merge_cols.append(c)
    if not merge_cols:
        merge_cols = ["confounded", "confounding_group"]

    cols_to_merge = [
        c for c in ["event_id", "confounded", "confounding_group"] if c in confounded_events.columns
    ]
    if "event_id" in es.columns and "event_id" in confounded_events.columns:
        suffix_cols = [c for c in cols_to_merge if c != "event_id" and c in es.columns]
        if suffix_cols:
            es = es.merge(
                confounded_events[cols_to_merge], on="event_id", how="left", suffixes=("_orig", "")
            )
            for c in suffix_cols:
                if f"{c}_orig" in es.columns and c in es.columns:
                    es[c] = es[f"{c}_orig"].fillna(es[c])
                    es = es.drop(columns=[f"{c}_orig"])
        else:
            es = es.merge(confounded_events[cols_to_merge], on="event_id", how="left")

    if "confounded" not in es.columns:
        es["confounded"] = False

    n_confounds = es.groupby("event_id")["confounded"].transform("sum")
    es["n_confounds"] = n_confounds.astype(int)

    if es["n_confounds"].nunique() < 2:
        es["residualized_return"] = es[return_col]
        return es

    y = es[return_col].fillna(0).values
    X = np.column_stack([np.ones(len(es)), es["n_confounds"].values])

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        es["residualized_return"] = residuals
        es["confound_beta"] = beta[1]
        es["confound_intercept"] = beta[0]
    except np.linalg.LinAlgError:
        es["residualized_return"] = es[return_col]

    return es
