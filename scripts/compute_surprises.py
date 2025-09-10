#!/usr/bin/env python3
"""Phase 8: Compute macro surprises from implied vs. realized."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config import setup_logging
from macro_engine.events.calendar import load_event_calendar
from macro_engine.expectations.implied import load_implied_expectations
from macro_engine.macro.sources import load_macro_data
from macro_engine.mapping.mapper import load_market_mapping
from macro_engine.surprises.calculator import (
    aggregate_surprises_with_ci,
    compute_all_surprises,
    save_surprises,
)

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    logger.info("Loading data...")
    event_cal = load_event_calendar()
    mapping = load_market_mapping()
    expectations = load_implied_expectations()
    macro_data = load_macro_data()

    if expectations.empty:
        logger.error("No implied expectations found.")
        return

    logger.info("Computing surprises for %d expectation snapshots...", len(expectations))
    surprises = compute_all_surprises(expectations, macro_data, event_cal, mapping)

    if not surprises.empty:
        path = save_surprises(surprises)
        logger.info("Saved %d surprise records to %s", len(surprises), path)
        logger.info("Direction breakdown: %s", surprises["direction"].value_counts().to_dict())

        agg = aggregate_surprises_with_ci(surprises)
        logger.info("Aggregated surprises with bootstrap CI:\n%s", agg.to_string(index=False))
    else:
        logger.warning("No surprises computed (check macro data availability)")


if __name__ == "__main__":
    main()
