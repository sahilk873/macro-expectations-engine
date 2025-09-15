from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.backtest.strategy import (
    RegimeAwareStrategy,
    compute_performance_metrics,
    run_backtest,
)
from macro_engine.config.settings import EngineConfig, get_settings

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    split_index: int


@dataclass
class ParameterPerturbation:
    param_name: str
    base_value: float
    perturbed_value: float
    base_sharpe: float
    perturbed_sharpe: float
    sensitivity: float  # dSharpe / dParam


def generate_walk_forward_splits(
    start_date: str,
    end_date: str,
    n_splits: int = 4,
    train_ratio: float = 0.7,
) -> list[WalkForwardSplit]:
    """Generate expanding-window walk-forward splits.

    Each split uses an increasing training window and a fixed-length
    validation window. This mimics live trading where the model
    is retrained as new data becomes available.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days

    train_days = int(total_days * train_ratio)
    val_days = (total_days - train_days) // n_splits

    splits: list[WalkForwardSplit] = []
    for i in range(n_splits):
        train_end = start + pd.Timedelta(days=train_days + i * val_days)
        val_start_i = train_end + pd.Timedelta(days=1)
        val_end_i = min(val_start_i + pd.Timedelta(days=val_days), end)

        splits.append(
            WalkForwardSplit(
                train_start=start,
                train_end=train_end,
                val_start=val_start_i,
                val_end=val_end_i,
                split_index=i,
            )
        )

    return splits


def run_walk_forward_backtest(
    price_data: pd.DataFrame,
    regime_classifications: pd.DataFrame,
    splits: list[WalkForwardSplit],
    strategy: Optional[RegimeAwareStrategy] = None,
    config: Optional[EngineConfig] = None,
    surprises: Optional[pd.DataFrame] = None,
    factor_attribution: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run walk-forward backtest with expanding windows.

    For each split:
    1. Train regime model on training window
    2. Apply strategy on validation window
    3. Record out-of-sample performance

    Returns out-of-sample results concatenated across all validation periods.
    """
    cfg = config or get_settings()
    strat = strategy or RegimeAwareStrategy(transaction_cost_bps=cfg.transaction_cost_bps)

    if price_data.empty or regime_classifications.empty:
        return pd.DataFrame()

    all_results: list[pd.DataFrame] = []
    split_metrics: list[dict] = []

    for split in splits:
        logger.info(
            "Walk-forward split %d: train %s to %s | val %s to %s",
            split.split_index,
            split.train_start.date(),
            split.train_end.date(),
            split.val_start.date(),
            split.val_end.date(),
        )

        val_regimes = regime_classifications[
            (pd.to_datetime(regime_classifications["date"]) >= split.val_start)
            & (pd.to_datetime(regime_classifications["date"]) <= split.val_end)
        ]

        if val_regimes.empty:
            logger.warning(
                "  No regime data in validation window, skipping split %d", split.split_index
            )
            continue

        val_prices = price_data[
            (pd.to_datetime(price_data["date"]) >= split.val_start)
            & (pd.to_datetime(price_data["date"]) <= split.val_end)
        ]

        if val_prices.empty:
            logger.warning(
                "  No price data in validation window, skipping split %d", split.split_index
            )
            continue

        split_results = run_backtest(
            price_data=val_prices,
            regime_classifications=val_regimes,
            strategy=strat,
            config=cfg,
            surprises=surprises,
            factor_attribution=factor_attribution,
        )

        if not split_results.empty:
            split_results["split_index"] = split.split_index
            split_results["split_train_start"] = split.train_start
            split_results["split_train_end"] = split.train_end
            split_results["split_val_start"] = split.val_start
            split_results["split_val_end"] = split.val_end
            all_results.append(split_results)

            metrics = compute_performance_metrics(split_results)
            metrics["split_index"] = split.split_index
            metrics["val_start"] = split.val_start
            metrics["val_end"] = split.val_end
            split_metrics.append(metrics)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined["walk_forward"] = True

    summary = pd.DataFrame(split_metrics)
    if not summary.empty:
        logger.info("\nWalk-Forward Summary:")
        for _, row in summary.iterrows():
            logger.info(
                "  Split %d: Sharpe=%.3f, Return=%.2f%%, MaxDD=%.2f%%",
                row["split_index"],
                row.get("sharpe_ratio", 0),
                row.get("total_return", 0) * 100,
                row.get("max_drawdown", 0) * 100,
            )

        mean_sharpe = (
            summary["sharpe_ratio"].mean() if "sharpe_ratio" in summary.columns else np.nan
        )
        sharpe_std = (
            summary["sharpe_ratio"].std(ddof=1) if "sharpe_ratio" in summary.columns else np.nan
        )
        logger.info(
            "  OOS Sharpe: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
            mean_sharpe,
            sharpe_std,
            summary["sharpe_ratio"].min() if "sharpe_ratio" in summary.columns else np.nan,
            summary["sharpe_ratio"].max() if "sharpe_ratio" in summary.columns else np.nan,
        )

    return combined


def compute_parameter_sensitivity(
    price_data: pd.DataFrame,
    regime_classifications: pd.DataFrame,
    base_strategy: RegimeAwareStrategy,
    param_name: str,
    param_values: list[float],
    config: Optional[EngineConfig] = None,
    surprises: Optional[pd.DataFrame] = None,
    factor_attribution: Optional[pd.DataFrame] = None,
) -> list[ParameterPerturbation]:
    """Compute sensitivity of Sharpe ratio to strategy parameter perturbations.

    A strategy with low parameter sensitivity is more robust.
    High sensitivity indicates potential overfitting.
    """
    cfg = config or get_settings()
    results: list[ParameterPerturbation] = []

    base_results = run_backtest(
        price_data, regime_classifications, base_strategy, cfg, surprises, factor_attribution
    )
    base_metrics = compute_performance_metrics(base_results) if not base_results.empty else {}
    base_sharpe = base_metrics.get("sharpe_ratio", 0.0)

    for pval in param_values:
        perturbed_strat = RegimeAwareStrategy(
            base_weights=dict(base_strategy.base_weights),
            transaction_cost_bps=base_strategy.tc_bps,
            vol_target=base_strategy.vol_target,
        )

        if param_name == "transaction_cost_bps":
            perturbed_strat.tc_bps = pval
        elif param_name == "vol_target":
            perturbed_strat.vol_target = pval
        else:
            logger.warning("Unknown parameter: %s", param_name)
            continue

        pert_results = run_backtest(
            price_data, regime_classifications, perturbed_strat, cfg, surprises, factor_attribution
        )
        pert_metrics = compute_performance_metrics(pert_results) if not pert_results.empty else {}
        pert_sharpe = pert_metrics.get("sharpe_ratio", 0.0)

        attr_name = {"transaction_cost_bps": "tc_bps", "vol_target": "vol_target"}.get(
            param_name, param_name
        )
        base_val = getattr(base_strategy, attr_name, 0)
        delta_param = pval - base_val
        delta_sharpe = pert_sharpe - base_sharpe
        sensitivity = delta_sharpe / delta_param if abs(delta_param) > 1e-10 else 0.0

        results.append(
            ParameterPerturbation(
                param_name=param_name,
                base_value=base_val,
                perturbed_value=pval,
                base_sharpe=base_sharpe,
                perturbed_sharpe=pert_sharpe,
                sensitivity=sensitivity,
            )
        )

    return results


def compute_out_of_sample_sharpe(
    walk_forward_results: pd.DataFrame,
) -> dict[str, float]:
    """Compute pooled out-of-sample Sharpe ratio from walk-forward results.

    The OOS Sharpe is the gold standard for strategy evaluation:
    it measures performance on data never used for parameter selection.
    """
    if walk_forward_results.empty:
        return {}

    results_by_split = {}
    for split_idx in walk_forward_results["split_index"].unique():
        sub = walk_forward_results[walk_forward_results["split_index"] == split_idx]
        metrics = compute_performance_metrics(sub)
        results_by_split[int(split_idx)] = metrics

    all_returns = walk_forward_results["daily_return"].dropna().values
    if len(all_returns) == 0:
        return {}

    total_return = float(
        walk_forward_results["portfolio_value"].iloc[-1]
        / walk_forward_results["portfolio_value"].iloc[0]
        - 1.0
    )
    n_years = max(len(all_returns) / 252, 1 / 252)
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    ann_vol = np.std(all_returns, ddof=1) * np.sqrt(252)
    oos_sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    split_sharpes = [m.get("sharpe_ratio", np.nan) for m in results_by_split.values()]
    split_sharpes = [s for s in split_sharpes if not np.isnan(s)]

    return {
        "oos_total_return": total_return,
        "oos_annualized_return": ann_return,
        "oos_annualized_vol": ann_vol,
        "oos_sharpe_ratio": oos_sharpe,
        "oos_n_splits": len(split_sharpes),
        "oos_mean_split_sharpe": float(np.mean(split_sharpes)) if split_sharpes else np.nan,
        "oos_std_split_sharpe": float(np.std(split_sharpes, ddof=1))
        if len(split_sharpes) > 1
        else np.nan,
        "oos_min_split_sharpe": float(np.min(split_sharpes)) if split_sharpes else np.nan,
        "oos_max_split_sharpe": float(np.max(split_sharpes)) if split_sharpes else np.nan,
    }
