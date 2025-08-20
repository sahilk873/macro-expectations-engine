#!/usr/bin/env python3
"""Phase 11: Run regime-aware backtest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config.settings import get_settings
from macro_engine.prices.providers import load_price_data
from macro_engine.regime.model import load_regime_classifications
from macro_engine.backtest.strategy import (
    RegimeAwareStrategy,
    run_backtest,
    save_backtest_results,
    compute_performance_metrics,
)


def main():
    cfg = get_settings()

    print("Loading data...")
    price_data = load_price_data()
    regimes = load_regime_classifications()

    if price_data.empty or regimes.empty:
        print("Missing required data. Ensure price data and regime classifications exist.")
        return

    print("Running regime-aware backtest...")
    strategy = RegimeAwareStrategy(transaction_cost_bps=cfg.transaction_cost_bps)
    results = run_backtest(price_data, regimes, strategy, config=cfg)

    if not results.empty:
        path = save_backtest_results(results)
        print(f"Saved {len(results)} backtest records to {path}")

        metrics = compute_performance_metrics(results)
        print("\nBacktest Performance:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(
                    f"  {k:25s}: {v:>8.4f}"
                    if "ratio" in k.lower() or k == "n_periods"
                    else f"  {k:25s}: {v:>8.2%}"
                )
            else:
                print(f"  {k:25s}: {v}")
    else:
        print("No backtest results")

    print("Done.")


if __name__ == "__main__":
    main()
