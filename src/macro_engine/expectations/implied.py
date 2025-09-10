from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from macro_engine.config.settings import EngineConfig, get_settings

# ---------------------------------------------------------------------------
# Probability conversion
# ---------------------------------------------------------------------------


def binary_to_probability(yes_bid: float, yes_ask: float, no_bid: float, no_ask: float) -> float:
    """Convert binary market prices to an implied probability using mid-price.

    Uses the mid-price of the Yes contract as the implied probability.
    If the spread is too wide, weights by spread.

    Returns a probability in [0, 1].
    """
    if yes_bid is None or yes_ask is None:
        return np.nan

    mid = (yes_bid + yes_ask) / 2.0
    # Use No side if available as a consistency check
    if no_bid is not None and no_ask is not None:
        no_mid = (no_bid + no_ask) / 2.0
        implied_no = 1.0 - no_mid
        # Blend if they disagree (arbitrage window)
        if abs(mid + no_mid - 1.0) > 0.02:
            mid = (mid + implied_no) / 2.0

    return float(np.clip(mid, 0.001, 0.999))


def calculate_implied_mean_from_prices(
    prices: list[float], buckets: list[tuple[float, float]]
) -> float:
    """Calculate implied mean from a set of bucket probabilities."""
    if not prices or not buckets or len(prices) != len(buckets):
        return np.nan

    total = sum(prices)
    if total <= 0:
        return np.nan

    probs = np.array(prices) / total
    means = np.array([(lo + hi) / 2.0 for lo, hi in buckets])
    return float(np.sum(probs * means))


def calculate_implied_variance(
    prices: list[float], buckets: list[tuple[float, float]], mean: float
) -> float:
    """Calculate implied variance from bucket probabilities."""
    if not prices or not buckets or len(prices) != len(buckets):
        return np.nan

    total = sum(prices)
    if total <= 0:
        return np.nan

    probs = np.array(prices) / total
    means = np.array([(lo + hi) / 2.0 for lo, hi in buckets])
    variance = float(np.sum(probs * (means - mean) ** 2))
    return variance


def fit_normal_from_prices(
    prices: list[float], buckets: list[tuple[float, float]]
) -> tuple[float, float]:
    """Fit a normal distribution to bucket probabilities."""
    mean = calculate_implied_mean_from_prices(prices, buckets)
    var = calculate_implied_variance(prices, buckets, mean)
    std = np.sqrt(var) if var > 0 else 0.0
    return mean, std


def bucketed_to_distribution(
    prices: list[float],
    buckets: list[tuple[float, float]],
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert bucketed market prices into an implied probability distribution.

    Returns (x_values, pdf_values).
    """
    if not prices or not buckets:
        return np.array([]), np.array([])

    mean, std = fit_normal_from_prices(prices, buckets)

    if std <= 0:
        return np.array([]), np.array([])

    lo = min(b[0] for b in buckets)
    hi = max(b[1] for b in buckets)
    margin = std * 3
    x = np.linspace(lo - margin, hi + margin, n_points)
    pdf = sp_stats.norm.pdf(x, mean, std)
    return x, pdf


# ---------------------------------------------------------------------------
# Pre-event snapshots
# ---------------------------------------------------------------------------


def create_expectation_snapshot(
    market_prices: pd.DataFrame,
    event_time: datetime,
    snapshot_offset_days: int = 1,
    snapshot_offset_hours: Optional[float] = None,
) -> pd.DataFrame:
    """Create a snapshot of market-implied expectations before an event.

    Takes the last available price before `event_time - snapshot_offset`.
    For daily offset: uses the close from `snapshot_offset_days` trading days before.
    For hourly offset: uses the price `snapshot_offset_hours` hours before.
    """
    if market_prices.empty:
        return pd.DataFrame()

    prices = market_prices.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])

    if snapshot_offset_hours is not None:
        cutoff = event_time - timedelta(hours=snapshot_offset_hours)
    else:
        cutoff = event_time - timedelta(days=snapshot_offset_days)

    before = prices[prices["timestamp"] < cutoff].copy()
    if before.empty:
        warnings.warn(f"No prices found before cutoff {cutoff} for event at {event_time}")
        return pd.DataFrame()

    # Take the last observation for each ticker
    idx = before.groupby("ticker")["timestamp"].idxmax()
    snapshot = before.loc[idx].reset_index(drop=True)
    snapshot["snapshot_time"] = cutoff
    snapshot["event_time"] = event_time

    # Compute implied probability for each market
    snapshot["implied_probability"] = snapshot.apply(
        lambda r: binary_to_probability(
            r.get("yes_bid", np.nan),
            r.get("yes_ask", np.nan),
            r.get("no_bid", np.nan),
            r.get("no_ask", np.nan),
        ),
        axis=1,
    )

    return snapshot


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def compute_implied_expectations(
    event_calendar: pd.DataFrame,
    kalshi_prices: pd.DataFrame,
    market_mapping: pd.DataFrame,
    config: Optional[EngineConfig] = None,
) -> pd.DataFrame:
    """Compute implied expectations for all mapped events.

    Returns a DataFrame with pre-event snapshots:
        event_id, event_type, market_ticker, snapshot_time, implied_probability,
        implied_mean, implied_std, implied_distribution
    """
    records: list[dict] = []

    for _, mapping_row in market_mapping.iterrows():
        event_id = mapping_row["event_id"]
        event_row = event_calendar[event_calendar["event_id"] == event_id]
        if event_row.empty:
            continue

        event_time = event_row.iloc[0]["release_datetime"]
        if isinstance(event_time, str):
            event_time = datetime.fromisoformat(event_time)

        ticker = mapping_row["market_ticker"]

        # Daily snapshot (T-1 day)
        daily_snap = create_expectation_snapshot(
            kalshi_prices[kalshi_prices["ticker"] == ticker],
            event_time,
            snapshot_offset_days=1,
        )
        if not daily_snap.empty:
            for _, snap in daily_snap.iterrows():
                records.append(
                    {
                        "event_id": event_id,
                        "event_type": event_row.iloc[0]["event_type"],
                        "market_ticker": ticker,
                        "market_title": mapping_row.get("market_title", ""),
                        "snapshot_type": "T-1_day",
                        "snapshot_time": snap["snapshot_time"],
                        "event_time": event_time,
                        "implied_probability": snap["implied_probability"],
                        "implied_mean": np.nan,
                        "implied_std": np.nan,
                        "confidence_score": mapping_row.get("confidence_score", 0.0),
                    }
                )

        # Hourly snapshot (T-1 hour)
        hourly_snap = create_expectation_snapshot(
            kalshi_prices[kalshi_prices["ticker"] == ticker],
            event_time,
            snapshot_offset_hours=1.0,
        )
        if not hourly_snap.empty:
            for _, snap in hourly_snap.iterrows():
                records.append(
                    {
                        "event_id": event_id,
                        "event_type": event_row.iloc[0]["event_type"],
                        "market_ticker": ticker,
                        "market_title": mapping_row.get("market_title", ""),
                        "snapshot_type": "T-1_hour",
                        "snapshot_time": snap["snapshot_time"],
                        "event_time": event_time,
                        "implied_probability": snap["implied_probability"],
                        "implied_mean": np.nan,
                        "implied_std": np.nan,
                        "confidence_score": mapping_row.get("confidence_score", 0.0),
                    }
                )

    result = pd.DataFrame(records)
    if result.empty:
        return result
    return result.sort_values(["event_id", "snapshot_type"]).reset_index(drop=True)


def save_implied_expectations(
    df: pd.DataFrame,
    config: Optional[EngineConfig] = None,
    distribution_df: Optional[pd.DataFrame] = None,
) -> tuple[Path, Optional[Path]]:
    """Save implied expectations and optional distributions."""
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.implied_expectations_file
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    dist_path = None
    if distribution_df is not None and not distribution_df.empty:
        dist_path = cfg.data_dir / cfg.implied_distributions_file
        distribution_df.to_parquet(dist_path, index=False)

    return path, dist_path


def load_implied_expectations(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.implied_expectations_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
