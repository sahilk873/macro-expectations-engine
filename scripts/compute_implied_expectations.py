#!/usr/bin/env python3
"""Phase 7: Compute implied expectations from prediction markets."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from macro_engine.config import setup_logging
from macro_engine.config.settings import get_settings
from macro_engine.events.calendar import load_event_calendar
from macro_engine.expectations.implied import (
    compute_implied_expectations,
    save_implied_expectations,
)
from macro_engine.mapping.mapper import load_market_mapping

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_settings()

    logger.info("Loading data...")
    event_cal = load_event_calendar()
    mapping = load_market_mapping()

    prices_path = cfg.kalshi_dir / cfg.kalshi_prices_file
    if prices_path.exists():
        kalshi_prices = pd.read_parquet(prices_path)
    else:
        logger.error("No Kalshi prices found. Run fetch_kalshi.py first.")
        return

    if event_cal.empty or mapping.empty or kalshi_prices.empty:
        logger.error("Missing required data.")
        return

    logger.info("Computing implied expectations for %d mappings...", len(mapping))
    expectations = compute_implied_expectations(event_cal, kalshi_prices, mapping)

    exp_path, dist_path = save_implied_expectations(expectations)
    logger.info("Saved %d expectation snapshots to %s", len(expectations), exp_path)
    if not expectations.empty:
        logger.info("Snapshot types: %s", expectations["snapshot_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
