#!/usr/bin/env python3
"""Phase 12: Run robustness checks and placebo tests."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config import setup_logging
from macro_engine.config.settings import get_settings
from macro_engine.prices.providers import load_price_data
from macro_engine.robustness.checks import run_robustness_checks, save_robustness_results
from macro_engine.studies.event_study import load_event_studies
from macro_engine.surprises.calculator import load_surprises

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_settings()

    logger.info("Loading data...")
    surprises = load_surprises()
    price_data = load_price_data()
    event_studies = load_event_studies()

    if surprises.empty or price_data.empty:
        logger.error("Missing required data.")
        return

    logger.info("Running robustness checks...")
    results = run_robustness_checks(surprises, price_data, event_studies, config=cfg)

    paths = save_robustness_results(results)
    for key, p in paths.items():
        df = results.get(key)
        if df is not None:
            logger.info("  %20s: %d records -> %s", key, len(df), p)

    summary = results.get("placebo_summary")
    if summary is not None and not summary.empty:
        logger.info("Placebo Test Summary:")
        for _, row in summary.iterrows():
            sig = " ***" if row.get("significant_5pct", False) else ""
            logger.info(
                "  %15s  actual=%+.4f  p=%.4f%s",
                row["window"],
                row["actual_mean_return"],
                row["p_value"],
                sig,
            )


if __name__ == "__main__":
    main()
