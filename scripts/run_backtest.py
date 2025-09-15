#!/usr/bin/env python3
"""Phase 11: Run regime-aware backtest with optional surprise tactical overlay."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config import setup_logging
from macro_engine.config.settings import get_settings
from macro_engine.backtest.strategy import (
    SurpriseTacticalStrategy,
    compute_performance_metrics,
    run_backtest,
    save_backtest_results,
)
from macro_engine.factors.model import load_factor_attribution
from macro_engine.prices.providers import load_price_data
from macro_engine.regime.model import load_regime_classifications
from macro_engine.surprises.calculator import load_surprises

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_settings()

    logger.info("Loading data...")
    price_data = load_price_data()
    regimes = load_regime_classifications()
    surprises = load_surprises()
    factor_attribution = load_factor_attribution()

    if price_data.empty or regimes.empty:
        logger.error("Missing required data.")
        return

    has_surprises = not surprises.empty
    has_factors = not factor_attribution.empty

    logger.info(
        "Running surprise-enhanced tactical backtest%s...",
        "" if has_surprises else " (no surprise data — using regime-only strategy)",
    )
    if has_surprises:
        logger.info(
            "Loaded %d surprise events across %d types",
            len(surprises),
            surprises["event_type"].nunique() if "event_type" in surprises.columns else 0,
        )
    if has_factors:
        logger.info(
            "Loaded factor attribution for %d (event, asset) pairs", len(factor_attribution)
        )

    strategy = SurpriseTacticalStrategy(transaction_cost_bps=cfg.transaction_cost_bps)
    results = run_backtest(
        price_data,
        regimes,
        strategy,
        config=cfg,
        surprises=surprises if has_surprises else None,
        factor_attribution=factor_attribution if has_factors else None,
    )

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
