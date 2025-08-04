from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings

# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------


_TD_LOOKUP = {
    "1D": 1,
    "2D": 2,
    "3D": 3,
    "5D": 5,
    "10D": 10,
    "21D": 21,
}


def compute_event_returns(
    price_data: pd.DataFrame,
    event_time: datetime,
    tickers: list[str],
    pre_window: str = "1D",
    post_windows: Optional[list[str]] = None,
) -> dict[str, dict[str, float]]:
    """Compute returns around an event for each ticker.

    Returns dict: {ticker: {"return_1D": 0.01, ...}}
    """
    if post_windows is None:
        post_windows = ["1D", "3D", "5D"]

    if price_data.empty:
        return {}

    prices = price_data.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    if isinstance(event_time, str):
        event_time = datetime.fromisoformat(event_time)

    event_date = event_time.date() if hasattr(event_time, "date") else event_time

    results: dict[str, dict[str, float]] = {}

    for ticker in tickers:
        ticker_data = prices[prices["ticker"] == ticker].sort_values("date")
        if ticker_data.empty:
            continue

        close_col = "close" if "close" in ticker_data.columns else "Close"
        dates = ticker_data["date"].values
        closes = ticker_data[close_col].values

        # Find closest price before or on event date
        pre_mask = ticker_data["date"] <= pd.Timestamp(event_date)
        pre_data = ticker_data[pre_mask]

        if pre_data.empty:
            continue

        pre_close = pre_data.iloc[-1][close_col]
        pre_date = pre_data.iloc[-1]["date"]

        returns: dict[str, float] = {}

        for pw in post_windows:
            days = _TD_LOOKUP.get(pw, 1)
            target_date = pre_date + timedelta(days=days)

            # Find closest price after target
            post_mask = ticker_data["date"] >= target_date
            post_data = ticker_data[post_mask]

            if post_data.empty:
                continue

            post_close = post_data.iloc[0][close_col]

            if pre_close != 0 and not np.isnan(pre_close) and not np.isnan(post_close):
                ret = (post_close / pre_close) - 1.0
                returns[f"return_{pw}"] = float(ret)

        if returns:
            results[ticker] = returns

    return results


def open_event_window(
    price_data: pd.DataFrame,
    event_time: datetime,
    ticker: str,
    window_days: int = 21,
) -> Optional[pd.DataFrame]:
    """Extract a window of price data around an event for one ticker."""
    prices = price_data.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    ticker_data = prices[prices["ticker"] == ticker].sort_values("date")

    if isinstance(event_time, str):
        event_time = datetime.fromisoformat(event_time)

    start = event_time - timedelta(days=window_days)
    end = event_time + timedelta(days=window_days)

    window_data = ticker_data[
        (ticker_data["date"] >= pd.Timestamp(start)) & (ticker_data["date"] <= pd.Timestamp(end))
    ]
    if window_data.empty:
        return None

    close_col = "close" if "close" in window_data.columns else "Close"
    window_data["event_time"] = event_time
    window_data["cum_return"] = window_data[close_col] / window_data[close_col].iloc[0] - 1.0
    return window_data


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_event_study(
    event_returns: pd.DataFrame,
    group_by: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Aggregate event study returns by group.

    Computes mean return, standard error, t-statistic, and hit rate (fraction positive).
    """
    if event_returns.empty:
        return pd.DataFrame()

    if group_by is None:
        group_by = ["event_type"]

    return_cols = [c for c in event_returns.columns if c.startswith("return_")]

    groups = event_returns.groupby(group_by)
    results: list[dict] = []

    for group_keys, group_df in groups:
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row: dict = dict(zip(group_by, group_keys))
        n_obs = len(group_df)
        row["n_events"] = n_obs

        for rc in return_cols:
            vals = group_df[rc].dropna()
            if len(vals) < 2:
                continue
            mean_ret = vals.mean()
            std_ret = vals.std(ddof=1)
            se = std_ret / np.sqrt(len(vals))
            t_stat = mean_ret / se if se != 0 else 0.0
            hit_rate = (vals > 0).mean()

            row[f"{rc}_mean"] = mean_ret
            row[f"{rc}_se"] = se
            row[f"{rc}_tstat"] = t_stat
            row[f"{rc}_hit_rate"] = hit_rate

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_event_studies(
    surprises: pd.DataFrame,
    price_data: pd.DataFrame,
    tickers: Optional[list[str]] = None,
    post_windows: Optional[list[str]] = None,
    config: Optional[EngineConfig] = None,
) -> pd.DataFrame:
    """Run event studies for all events with surprises.

    For each event, compute asset returns across multiple windows.
    """
    cfg = config or get_settings()
    if tickers is None:
        tickers = list(cfg.etf_tickers)
    if post_windows is None:
        post_windows = cfg.post_event_windows

    if surprises.empty or price_data.empty:
        return pd.DataFrame()

    records: list[dict] = []

    for _, surprise_row in surprises.iterrows():
        event_time = surprise_row.get("event_time")
        if event_time is None:
            continue

        if isinstance(event_time, str):
            try:
                event_time = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
            except ValueError:
                continue

        returns = compute_event_returns(
            price_data=price_data,
            event_time=event_time,
            tickers=tickers,
            post_windows=post_windows,
        )

        for ticker, ret_dict in returns.items():
            record = {
                "event_id": surprise_row["event_id"],
                "event_type": surprise_row["event_type"],
                "snapshot_type": surprise_row.get("snapshot_type", ""),
                "event_time": event_time,
                "ticker": ticker,
                "direction": surprise_row.get("direction", ""),
                "qualitative": surprise_row.get("qualitative", ""),
                "risk_label": surprise_row.get("risk_label", ""),
                "raw_surprise": surprise_row.get("raw_surprise", np.nan),
                "standardized_surprise": surprise_row.get("standardized_surprise", np.nan),
            }
            record.update(ret_dict)
            records.append(record)

    result = pd.DataFrame(records)
    return result.sort_values(["event_id", "ticker"]).reset_index(drop=True)


def save_event_studies(df: pd.DataFrame, config: Optional[EngineConfig] = None) -> Path:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.event_studies_file
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_event_studies(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.event_studies_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
