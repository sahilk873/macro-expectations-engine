from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings

logger = logging.getLogger(__name__)


def _precompute_ticker_data(price_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pre-process price data for fast lookup by ticker."""
    prices = price_data.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    close_col = "close" if "close" in prices.columns else "Close"
    result: dict[str, pd.DataFrame] = {}
    for ticker in prices["ticker"].unique():
        tdata = prices[prices["ticker"] == ticker].sort_values("date")[["date", close_col]].copy()
        tdata.rename(columns={close_col: "close"}, inplace=True)
        result[ticker] = tdata
    return result


def _quick_return(ticker_data: pd.DataFrame, event_date, window: str) -> float:
    """Fast return computation for a single ticker and window."""
    window_days = {"1D": 1, "2D": 2, "3D": 3, "5D": 5, "10D": 10, "21D": 21}
    days = window_days.get(window, 1)

    pre_mask = ticker_data["date"] <= pd.Timestamp(event_date)
    pre_rows = ticker_data[pre_mask]
    if len(pre_rows) == 0:
        return np.nan

    pre_close = float(pre_rows["close"].iloc[-1])
    pre_date = pre_rows["date"].iloc[-1]

    target = pre_date + pd.Timedelta(days=days)
    post_rows = ticker_data[ticker_data["date"] >= target]
    if len(post_rows) == 0:
        return np.nan

    post_close = float(post_rows["close"].iloc[0])
    if pre_close <= 0 or np.isnan(pre_close) or np.isnan(post_close):
        return np.nan

    return post_close / pre_close - 1.0


def placebo_test_random_dates(
    surprises: pd.DataFrame,
    price_data: pd.DataFrame,
    tickers: list[str],
    n_iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Placebo test: randomize event dates while preserving surprise values."""
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    event_times = surprises["event_time"].dropna().unique()
    if len(event_times) < 2:
        return pd.DataFrame()

    all_times_list = sorted(pd.to_datetime(event_times))
    ticker_data = _precompute_ticker_data(price_data)
    available_tickers = [t for t in tickers if t in ticker_data]

    windows = ["1D", "3D", "5D"]

    for i in range(n_iterations):
        shuffled = rng.permutation(all_times_list)
        for orig_time, new_time in zip(all_times_list, shuffled):
            new_date = new_time.date() if hasattr(new_time, "date") else new_time
            for ticker in available_tickers:
                td = ticker_data[ticker]
                for pw in windows:
                    ret = _quick_return(td, new_date, pw)
                    if not np.isnan(ret):
                        records.append(
                            {
                                "iteration": i,
                                "test_type": "random_dates",
                                "original_event_time": orig_time,
                                "placebo_event_time": new_time,
                                "ticker": ticker,
                                "window": f"return_{pw}",
                                "placebo_return": ret,
                            }
                        )

        if (i + 1) % 25 == 0:
            logger.info("  Placebo date test: iteration %d/%d", i + 1, n_iterations)

    return pd.DataFrame(records)


def placebo_test_random_signs(
    surprises: pd.DataFrame,
    price_data: pd.DataFrame,
    tickers: list[str],
    n_iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Placebo test: randomize surprise signs while preserving magnitudes."""
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    ticker_data = _precompute_ticker_data(price_data)
    available_tickers = [t for t in tickers if t in ticker_data]
    windows = ["1D", "3D", "5D"]

    event_ids = surprises["event_id"].unique()

    for i in range(n_iterations):
        for eid in event_ids:
            event_surprises = surprises[surprises["event_id"] == eid]
            if event_surprises.empty:
                continue

            event_time = event_surprises.iloc[0]["event_time"]
            if event_time is None:
                continue

            event_date = pd.Timestamp(event_time).date()
            flip = rng.random() > 0.5
            sign = 1.0 if flip else -1.0

            for ticker in available_tickers:
                td = ticker_data[ticker]
                for pw in windows:
                    ret = _quick_return(td, event_date, pw)
                    if not np.isnan(ret):
                        records.append(
                            {
                                "iteration": i,
                                "test_type": "random_signs",
                                "event_id": eid,
                                "flipped": flip,
                                "ticker": ticker,
                                "window": f"return_{pw}",
                                "placebo_return": ret * sign,
                            }
                        )

        if (i + 1) % 25 == 0:
            logger.info("  Placebo sign test: iteration %d/%d", i + 1, n_iterations)

    return pd.DataFrame(records)


def run_robustness_checks(
    surprises: pd.DataFrame,
    price_data: pd.DataFrame,
    event_studies: pd.DataFrame,
    config: Optional[EngineConfig] = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or get_settings()
    tickers = list(
        set(event_studies["ticker"].unique()) if not event_studies.empty else cfg.etf_tickers
    )

    logger.info(
        "Running placebo date randomization test (%d iterations)...", cfg.placebo_n_iterations
    )
    date_results = placebo_test_random_dates(
        surprises,
        price_data,
        tickers,
        n_iterations=cfg.placebo_n_iterations,
    )

    logger.info(
        "Running placebo sign randomization test (%d iterations)...", cfg.placebo_n_iterations
    )
    sign_results = placebo_test_random_signs(
        surprises,
        price_data,
        tickers,
        n_iterations=cfg.placebo_n_iterations,
    )

    return {
        "placebo_dates": date_results,
        "placebo_signs": sign_results,
        "placebo_summary": _summarize_placebo(event_studies, date_results, sign_results),
    }


def _summarize_placebo(
    actual: pd.DataFrame,
    placebo_dates: pd.DataFrame,
    placebo_signs: pd.DataFrame,
) -> pd.DataFrame:
    if actual.empty:
        return pd.DataFrame()

    return_cols = [c for c in actual.columns if c.startswith("return_")]
    summary: list[dict] = []

    for rc in return_cols:
        actual_mean = actual[rc].mean()
        actual_std = actual[rc].std()

        combined_placebo: list[float] = []
        for df in [placebo_dates, placebo_signs]:
            if not df.empty and "window" in df.columns and "placebo_return" in df.columns:
                subset = df[df["window"] == rc]["placebo_return"].dropna()
                combined_placebo.extend(subset.values)

        all_placebo = np.array(combined_placebo)
        if len(all_placebo) > 0:
            p_value = float((abs(all_placebo) >= abs(actual_mean)).mean())
        else:
            p_value = 1.0

        summary.append(
            {
                "window": rc,
                "actual_mean_return": actual_mean,
                "actual_std": actual_std,
                "placebo_n_obs": len(all_placebo),
                "p_value": round(p_value, 4),
                "significant_5pct": p_value < 0.05,
            }
        )

    return pd.DataFrame(summary)


def save_robustness_results(
    results: dict[str, pd.DataFrame],
    config: Optional[EngineConfig] = None,
) -> dict[str, Path]:
    cfg = config or get_settings()
    paths = {}

    for key, df in results.items():
        if key == "placebo_summary":
            path = cfg.data_dir / cfg.placebo_results_file
        else:
            path = cfg.data_dir / f"{key}.parquet"
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        paths[key] = path

    return paths


def load_robustness_results(config: Optional[EngineConfig] = None) -> dict[str, pd.DataFrame]:
    cfg = config or get_settings()
    results = {}
    for key in ["placebo_dates", "placebo_signs"]:
        path = cfg.data_dir / f"{key}.parquet"
        if path.exists():
            results[key] = pd.read_parquet(path)
    path = cfg.data_dir / cfg.placebo_results_file
    if path.exists():
        results["placebo_summary"] = pd.read_parquet(path)
    return results
