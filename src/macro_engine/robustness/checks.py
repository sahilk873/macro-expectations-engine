from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings
from macro_engine.studies.event_study import compute_event_returns


def placebo_test_random_dates(
    surprises: pd.DataFrame,
    price_data: pd.DataFrame,
    tickers: list[str],
    n_iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Placebo test: randomize event dates while preserving surprise values.

    For each iteration, shuffle event times and recompute event-study returns.
    Returns the distribution of placebo average returns.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    event_times = surprises["event_time"].dropna().unique()
    if len(event_times) < 2:
        return pd.DataFrame()

    all_times_list = sorted(pd.to_datetime(event_times))

    for i in range(n_iterations):
        # Shuffle event times
        shuffled = rng.permutation(all_times_list)

        placebo_returns: list[float] = []

        for orig_time, new_time in zip(all_times_list, shuffled):
            matching = surprises[pd.to_datetime(surprises["event_time"]) == orig_time]
            if matching.empty:
                continue

            returns = compute_event_returns(
                price_data=price_data,
                event_time=new_time,
                tickers=tickers,
                post_windows=["1D", "3D", "5D"],
            )

            for ticker, ret_dict in returns.items():
                for pw, ret in ret_dict.items():
                    placebo_returns.append(ret)
                    records.append(
                        {
                            "iteration": i,
                            "test_type": "random_dates",
                            "original_event_time": orig_time,
                            "placebo_event_time": new_time,
                            "ticker": ticker,
                            "window": pw,
                            "placebo_return": ret,
                        }
                    )

        if (i + 1) % 100 == 0:
            pass  # progress tracking

    result = pd.DataFrame(records)
    return result


def placebo_test_random_signs(
    surprises: pd.DataFrame,
    price_data: pd.DataFrame,
    tickers: list[str],
    n_iterations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Placebo test: randomize surprise signs while preserving magnitudes.

    For each iteration, flip a random subset of surprise signs and
    re-run the event study.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    event_ids = surprises["event_id"].unique()

    for i in range(n_iterations):
        for eid in event_ids:
            event_surprises = surprises[surprises["event_id"] == eid]
            if event_surprises.empty:
                continue

            event_time = event_surprises.iloc[0]["event_time"]
            if event_time is None:
                continue

            # Randomly flip direction
            flip = rng.random() > 0.5
            sign = 1.0 if flip else -1.0

            returns = compute_event_returns(
                price_data=price_data,
                event_time=event_time,
                tickers=tickers,
                post_windows=["1D", "3D", "5D"],
            )

            for ticker, ret_dict in returns.items():
                for pw, ret in ret_dict.items():
                    flipped_ret = ret * sign
                    records.append(
                        {
                            "iteration": i,
                            "test_type": "random_signs",
                            "event_id": eid,
                            "flipped": flip,
                            "ticker": ticker,
                            "window": pw,
                            "placebo_return": flipped_ret,
                        }
                    )

    return pd.DataFrame(records)


def run_robustness_checks(
    surprises: pd.DataFrame,
    price_data: pd.DataFrame,
    event_studies: pd.DataFrame,
    config: Optional[EngineConfig] = None,
) -> dict[str, pd.DataFrame]:
    """Run comprehensive robustness checks.

    Returns dict of DataFrames:
        - placebo_dates: results from date-randomization test
        - placebo_signs: results from sign-randomization test
        - placebo_summary: comparison of actual vs. placebo distributions
    """
    cfg = config or get_settings()
    tickers = list(
        set(event_studies["ticker"].unique()) if not event_studies.empty else cfg.etf_tickers
    )

    date_results = placebo_test_random_dates(
        surprises,
        price_data,
        tickers,
        n_iterations=cfg.placebo_n_iterations,
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
    """Compare actual event study results to placebo distributions."""
    if actual.empty:
        return pd.DataFrame()

    return_cols = [c for c in actual.columns if c.startswith("return_")]
    summary: list[dict] = []

    for rc in return_cols:
        actual_mean = actual[rc].mean()
        actual_std = actual[rc].std()
        window_label = rc.replace("return_", "")

        # Compute mean placebo return per iteration for each test type
        def _iteration_means(df: pd.DataFrame, window: str) -> pd.Series:
            if df.empty or "window" not in df.columns or "placebo_return" not in df.columns:
                return pd.Series(dtype=float)
            subset = df[df["window"] == window]
            if subset.empty:
                return pd.Series(dtype=float)
            return subset.groupby("iteration")["placebo_return"].mean()

        date_means = _iteration_means(placebo_dates, rc)
        sign_means = _iteration_means(placebo_signs, rc)

        # Calculate p-values
        all_placebo = pd.concat([date_means, sign_means]).dropna()
        if len(all_placebo) > 0:
            p_value = float((abs(all_placebo.values) >= abs(actual_mean)).mean())
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
    """Save robustness check results."""
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
