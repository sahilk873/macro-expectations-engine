#!/usr/bin/env python3
"""Phase 11: Run regime-aware backtest."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config import setup_logging
from macro_engine.config.settings import get_settings
from macro_engine.backtest.strategy import (
    RegimeAwareStrategy,
    compute_performance_metrics,
    run_backtest,
    save_backtest_results,
)
from macro_engine.prices.providers import load_price_data
from macro_engine.regime.model import load_regime_classifications

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_settings()

    logger.info("Loading data...")
    price_data = load_price_data()
    regimes = load_regime_classifications()

    if price_data.empty or regimes.empty:
        logger.error("Missing required data.")
        return

    logger.info("Running regime-aware backtest...")
    strategy = RegimeAwareStrategy(transaction_cost_bps=cfg.transaction_cost_bps)
    results = run_backtest(price_data, regimes, strategy, config=cfg)

    if not results.empty:
        path = save_backtest_results(results)
        logger.info("Saved %d backtest records to %s", len(results), path)

        metrics = compute_performance_metrics(results)
        logger.info("Backtest Performance:")
        for k, v in metrics.items():
            if isinstance(v, float):
                if "ratio" in k.lower() or k == "n_periods":
                    logger.info("  %25s: %8.4f", k, v)
                else:
                    logger.info("  %25s: %8.2f%%", k, v * 100)
            else:
                logger.info("  %25s: %s", k, v)
    else:
        logger.warning("No backtest results")


if __name__ == "__main__":
    main()
