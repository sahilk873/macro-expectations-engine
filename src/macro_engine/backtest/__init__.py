"""Regime-aware ETF allocation backtest with walk-forward validation."""

from macro_engine.backtest.strategy import (
    RegimeAwareStrategy,
    compute_performance_metrics,
    load_backtest_results,
    run_backtest,
    save_backtest_results,
)
from macro_engine.backtest.walk_forward import (
    compute_out_of_sample_sharpe,
    compute_parameter_sensitivity,
    generate_walk_forward_splits,
    run_walk_forward_backtest,
)

__all__ = [
    "RegimeAwareStrategy",
    "run_backtest",
    "save_backtest_results",
    "load_backtest_results",
    "compute_performance_metrics",
    "generate_walk_forward_splits",
    "run_walk_forward_backtest",
    "compute_out_of_sample_sharpe",
    "compute_parameter_sensitivity",
]
