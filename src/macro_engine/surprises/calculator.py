from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings

# ---------------------------------------------------------------------------
# Surprise computation
# ---------------------------------------------------------------------------


def compute_raw_surprise(realized: float, expected: float) -> float:
    """Compute the raw surprise as realized minus expected."""
    return realized - expected


def compute_standardized_surprise(realized: float, expected: float, std: float) -> float:
    """Compute the standardized (z-score) surprise.

    surprise = (realized - expected) / std
    If std is 0 or NaN, uses historical volatility of expectations.
    """
    if std is None or std == 0 or np.isnan(std) or np.isinf(std):
        return np.nan
    return (realized - expected) / std


def compute_percent_surprise(realized: float, expected: float) -> float:
    """Compute percentage surprise: (realized / expected) - 1."""
    if expected == 0 or np.isnan(expected):
        return np.nan
    return (realized / expected) - 1.0


# ---------------------------------------------------------------------------
# Surprise labeling
# ---------------------------------------------------------------------------

_INFLATION_EVENTS = {"CPI", "PCE"}
_EMPLOYMENT_EVENTS = {"NFP", "UNEMPLOYMENT"}
_GROWTH_EVENTS = {"GDP", "RECESSION"}
_POLICY_EVENTS = {"FOMC"}


def label_surprise(
    event_type: str, raw_surprise: float, standardized_surprise: float
) -> dict[str, str]:
    """Label a macro surprise with directional and qualitative tags.

    Returns dict with:
        - direction: "above_expectations" / "below_expectations" / "neutral"
        - qualitative: narrative label (e.g. "inflation_hot", "growth_weak")
        - risk_label: "risk_on" / "risk_off" / "neutral"
    """
    direction = "neutral"
    qualitative = "neutral"
    risk_label = "neutral"

    if (
        abs(standardized_surprise) < 0.5
        if not np.isnan(standardized_surprise)
        else abs(raw_surprise) < 1e-6
    ):
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
            # Higher unemployment = bad
            if direction == "above_expectations":
                qualitative = "labor_weak"
                risk_label = "risk_off"
            else:
                qualitative = "labor_strong"
                risk_label = "risk_on"
        else:
            # More jobs = good
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


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def compute_all_surprises(
    implied_expectations: pd.DataFrame,
    macro_data: pd.DataFrame,
    event_calendar: pd.DataFrame,
    market_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Combine implied expectations with realized macro data to compute surprises.

    Steps:
    1. For each event from implied_expectations, find the corresponding realized value.
    2. Compute raw and standardized surprises.
    3. Label each surprise.
    4. Merge in mapping metadata.
    """
    records: list[dict] = []

    for _, exp_row in implied_expectations.iterrows():
        event_id = exp_row["event_id"]
        event_type = exp_row["event_type"]

        # Find matched macro data point
        event_row = event_calendar[event_calendar["event_id"] == event_id]
        if event_row.empty:
            continue

        ticker = event_row.iloc[0]["ticker"]
        event_time = exp_row.get("event_time")

        # Find the realized value from macro data around the event date
        realized_val = _find_realized_value(macro_data, ticker, event_time, event_type)

        expected_prob = exp_row.get("implied_probability", np.nan)
        if pd.isna(expected_prob):
            continue

        # Expected value from probability: for binary markets,
        # the implied probability IS the market's expectation.
        expected_val = expected_prob

        raw = (
            compute_raw_surprise(realized_val, expected_val)
            if not np.isnan(realized_val)
            else np.nan
        )
        std_surprise = (
            compute_standardized_surprise(realized_val, expected_val, 0.1)
            if not np.isnan(realized_val)
            else np.nan
        )
        pct = (
            compute_percent_surprise(realized_val, expected_val)
            if not np.isnan(realized_val)
            else np.nan
        )

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
    return result.sort_values(["event_id", "snapshot_type"]).reset_index(drop=True)


def _find_realized_value(
    macro_df: pd.DataFrame, ticker: str, event_time: Optional[datetime], event_type: str
) -> float:
    """Find the realized macro value closest to the event time."""
    if macro_df.empty or event_time is None:
        return np.nan

    if isinstance(event_time, str):
        event_time = datetime.fromisoformat(event_time)

    macro = macro_df.copy()
    macro["date"] = pd.to_datetime(macro["date"])

    # Filter by series that matches the event
    series_candidates = _get_series_ids(event_type)
    filtered = (
        macro[macro["series_id"].isin(series_candidates)] if "series_id" in macro.columns else macro
    )

    if filtered.empty:
        filtered = macro

    # Find the nearest observation before or on the event date
    event_date = event_time.date() if hasattr(event_time, "date") else event_time
    mask = filtered["date"].dt.date <= event_date
    before = filtered[mask].sort_values("date", ascending=False)

    if before.empty:
        return np.nan

    return float(before.iloc[0]["value"])


def _get_series_ids(event_type: str) -> list[str]:
    mapping = {
        "CPI": ["CPIAUCSL", "CPILFESL"],
        "NFP": ["PAYEMS"],
        "UNEMPLOYMENT": ["UNRATE"],
        "GDP": ["GDP", "GDPC1"],
        "PCE": ["PCEPI", "PCEPILFE"],
        "FOMC": ["FEDFUNDS"],
        "RECESSION": ["RECPROUSM156N"],
    }
    return mapping.get(event_type, [])


def save_surprises(df: pd.DataFrame, config: Optional[EngineConfig] = None) -> Path:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.surprises_file
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_surprises(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.surprises_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
