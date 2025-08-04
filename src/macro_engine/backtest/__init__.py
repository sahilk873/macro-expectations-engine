"""Regime-aware ETF allocation backtest."""

from macro_engine.backtest.strategy import (
    RegimeAwareStrategy,
    compute_performance_metrics,
    load_backtest_results,
    run_backtest,
    save_backtest_results,
)

__all__ = [
    "RegimeAwareStrategy",
    "run_backtest",
    "save_backtest_results",
    "load_backtest_results",
    "compute_performance_metrics",
]
