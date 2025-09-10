from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings

logger = logging.getLogger(__name__)

_INFLATION_EVENTS = {"CPI", "PCE"}
_EMPLOYMENT_EVENTS = {"NFP", "UNEMPLOYMENT"}
_GROWTH_EVENTS = {"GDP", "RECESSION"}
_POLICY_EVENTS = {"FOMC"}


def compute_raw_surprise(realized: float, expected: float) -> float:
    return realized - expected


def compute_standardized_surprise(realized: float, expected: float, std: float) -> float:
    if std is None or std == 0 or np.isnan(std) or np.isinf(std):
        return np.nan
    return (realized - expected) / std


def compute_percent_surprise(realized: float, expected: float) -> float:
    if expected == 0 or np.isnan(expected):
        return np.nan
    return (realized / expected) - 1.0


def label_surprise(
    event_type: str, raw_surprise: float, standardized_surprise: float
) -> dict[str, str]:
    direction = "neutral"
    qualitative = "neutral"
    risk_label = "neutral"

    threshold = 0.5
    if not np.isnan(standardized_surprise):
        neutral_flag = abs(standardized_surprise) < threshold
    else:
        neutral_flag = abs(raw_surprise) < 1e-6

    if neutral_flag:
        return {"direction": direction, "qualitative": qualitative, "risk_label": risk_label}

    if raw_surprise > 0:
        direction = "above_expectations"
    else:
        direction = "below_expectations"

    if event_type in _INFLATION_EVENTS:
        if direction == "above_expectations":
            qualitative = "inflation_hot"
            risk_label = "risk_off"
        else:
            qualitative = "inflation_cool"
            risk_label = "risk_on"

    elif event_type in _EMPLOYMENT_EVENTS:
        if event_type == "UNEMPLOYMENT":
            if direction == "above_expectations":
                qualitative = "labor_weak"
                risk_label = "risk_off"
            else:
                qualitative = "labor_strong"
                risk_label = "risk_on"
        else:
            if direction == "above_expectations":
                qualitative = "labor_strong"
                risk_label = "risk_on"
            else:
                qualitative = "labor_weak"
                risk_label = "risk_off"

    elif event_type in _GROWTH_EVENTS:
        if direction == "above_expectations":
            qualitative = "growth_strong"
            risk_label = "risk_on"
        else:
            qualitative = "growth_weak"
            risk_label = "risk_off"

    elif event_type in _POLICY_EVENTS:
        if direction == "above_expectations":
            qualitative = "policy_hawkish"
            risk_label = "risk_off"
        else:
            qualitative = "policy_dovish"
            risk_label = "risk_on"

    return {"direction": direction, "qualitative": qualitative, "risk_label": risk_label}


def _series_needs_yoy(series_id: str) -> bool:
    """Check if a series is a price index whose value should be YoY % change."""
    return series_id in {"CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "GDPC1", "GDP"}


def _compute_display_value(
    macro_df: pd.DataFrame, series_id: str, event_date
) -> tuple[float, float]:
    """Compute the realized value for an event.

    For price index series (CPI, PCE, GDP): returns YoY % change.
    For other series (NFP, unemployment, FEDFUNDS): returns raw value.
    Returns (value, std_estimate).
    """
    series_data = macro_df[macro_df["series_id"] == series_id].sort_values("date")
    if series_data.empty:
        return np.nan, np.nan

    latest = series_data.iloc[-1]
    raw_value = float(latest["value"])

    if _series_needs_yoy(series_id):
        if len(series_data) >= 13:
            recent = series_data.iloc[-13:]
            yoy = (recent["value"].iloc[-1] / recent["value"].iloc[0] - 1.0) * 100
            return yoy, abs(yoy) * 0.1 + 0.1
        return raw_value, abs(raw_value) * 0.05 + 0.1

    if series_id == "PAYEMS":
        return raw_value, abs(raw_value) * 0.05 + 1000

    return raw_value, abs(raw_value) * 0.05 + 0.01


def compute_all_surprises(
    implied_expectations: pd.DataFrame,
    macro_data: pd.DataFrame,
    event_calendar: pd.DataFrame,
    market_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Compute surprises by comparing implied expectations to realized outcomes.

    For price index series (CPI, PCE, GDP), realized values are computed as
    YoY percent changes to match the scale of prediction market questions.
    """
    records: list[dict] = []

    for _, exp_row in implied_expectations.iterrows():
        event_id = exp_row["event_id"]
        event_type = exp_row["event_type"]

        event_row = event_calendar[event_calendar["event_id"] == event_id]
        if event_row.empty:
            continue

        ticker = event_row.iloc[0]["ticker"]
        event_time = exp_row.get("event_time")

        realized_val, surprise_std = _find_realized_value(
            macro_data, ticker, event_time, event_type
        )

        expected_prob = exp_row.get("implied_probability", np.nan)

        if pd.isna(expected_prob) or pd.isna(realized_val):
            continue

        expected_val = expected_prob

        raw = compute_raw_surprise(realized_val, expected_val)
        std_surprise = compute_standardized_surprise(realized_val, expected_val, surprise_std)
        pct = compute_percent_surprise(realized_val, expected_val)

        labels = label_surprise(
            event_type,
            raw if not np.isnan(raw) else 0.0,
            std_surprise if not np.isnan(std_surprise) else 0.0,
        )

        records.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "market_ticker": exp_row.get("market_ticker", ""),
                "snapshot_type": exp_row.get("snapshot_type", ""),
                "snapshot_time": exp_row.get("snapshot_time"),
                "event_time": event_time,
                "expected_probability": expected_val,
                "realized_value": realized_val,
                "raw_surprise": raw,
                "standardized_surprise": std_surprise,
                "percent_surprise": pct,
                "direction": labels["direction"],
                "qualitative": labels["qualitative"],
                "risk_label": labels["risk_label"],
                "confidence_score": exp_row.get("confidence_score", 0.0),
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result
    sort_cols = [c for c in ["event_id", "snapshot_type"] if c in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)
    return result


def _find_realized_value(
    macro_df: pd.DataFrame, ticker: str, event_time: Optional[datetime], event_type: str
) -> tuple[float, float]:
    """Find realized macro value nearest to event time.

    Returns (value, std_estimate).
    For price indices, returns YoY % change.
    """
    if macro_df.empty or event_time is None:
        return np.nan, np.nan

    if isinstance(event_time, str):
        event_time = datetime.fromisoformat(event_time)

    macro = macro_df.copy()
    macro["date"] = pd.to_datetime(macro["date"])

    series_candidates = _get_series_ids(event_type)
    filtered = (
        macro[macro["series_id"].isin(series_candidates)] if "series_id" in macro.columns else macro
    )

    if filtered.empty:
        filtered = macro

    event_date = event_time.date() if hasattr(event_time, "date") else event_time
    mask = filtered["date"].dt.date <= event_date
    before = filtered[mask].sort_values("date", ascending=False)

    if before.empty:
        return np.nan, np.nan

    # Use the best candidate series
    best_series = None
    for candidate in series_candidates:
        sub = before[before["series_id"] == candidate]
        if not sub.empty:
            best_series = candidate
            break

    if best_series is None:
        best_series = before.iloc[0]["series_id"]

    return _compute_display_value(macro, best_series, event_date)


def _get_series_ids(event_type: str) -> list[str]:
    mapping = {
        "CPI": ["CPIAUCSL", "CPILFESL"],
        "NFP": ["PAYEMS"],
        "UNEMPLOYMENT": ["UNRATE"],
        "GDP": ["GDPC1", "GDP"],
        "PCE": ["PCEPI", "PCEPILFE"],
        "FOMC": ["FEDFUNDS"],
        "RECESSION": ["RECPROUSM156N"],
    }
    return mapping.get(event_type, [])


def _bootstrap_ci(
    series: pd.Series, n_bootstrap: int = 10000, ci: float = 0.95
) -> tuple[float, float, float]:
    values = series.dropna().values
    if len(values) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    boot_means = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return float(boot_means.mean()), lower, upper


def aggregate_surprises_with_ci(
    surprises: pd.DataFrame,
    group_by: Optional[list[str]] = None,
) -> pd.DataFrame:
    if group_by is None:
        group_by = ["event_type"]

    results: list[dict] = []
    for group_keys, group_df in surprises.groupby(group_by):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row: dict = dict(zip(group_by, group_keys))
        raw = group_df["raw_surprise"].dropna()
        std = group_df["standardized_surprise"].dropna()

        row["n_events"] = len(raw)
        row["raw_mean"] = raw.mean() if len(raw) > 0 else np.nan
        row["raw_std"] = raw.std(ddof=1) if len(raw) > 1 else np.nan
        row["std_mean"] = std.mean() if len(std) > 0 else np.nan
        row["std_std"] = std.std(ddof=1) if len(std) > 1 else np.nan

        if len(raw) >= 2:
            _, ci_lo, ci_hi = _bootstrap_ci(raw)
            row["raw_ci_lower"] = ci_lo
            row["raw_ci_upper"] = ci_hi
        else:
            row["raw_ci_lower"] = np.nan
            row["raw_ci_upper"] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def save_surprises(df: pd.DataFrame, config: Optional[EngineConfig] = None) -> Path:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.surprises_file
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d surprise records to %s", len(df), path)
    return path


def load_surprises(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.surprises_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
