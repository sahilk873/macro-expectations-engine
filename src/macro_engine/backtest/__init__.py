"""Regime-aware ETF allocation backtest with walk-forward validation and surprise-based tactical overlays."""

from macro_engine.backtest.signals import (
    SurpriseSignal,
    build_surprise_signals,
    build_surprise_tilts,
    get_active_signals,
)
from macro_engine.backtest.strategy import (
    RegimeAwareStrategy,
    SurpriseTacticalStrategy,
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
    "SurpriseTacticalStrategy",
    "SurpriseSignal",
    "build_surprise_signals",
    "build_surprise_tilts",
    "get_active_signals",
    "run_backtest",
    "save_backtest_results",
    "load_backtest_results",
    "compute_performance_metrics",
    "generate_walk_forward_splits",
    "run_walk_forward_backtest",
    "compute_out_of_sample_sharpe",
    "compute_parameter_sensitivity",
]
