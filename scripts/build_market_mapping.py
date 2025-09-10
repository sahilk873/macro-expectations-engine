#!/usr/bin/env python3
"""Phase 4: Build market-to-event mappings."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from macro_engine.config import setup_logging
from macro_engine.config.settings import get_settings
from macro_engine.events.calendar import load_event_calendar
from macro_engine.mapping.mapper import (
    build_market_mapping,
    export_low_confidence,
    load_manual_overrides,
    save_market_mapping,
)

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_settings()

    logger.info("Loading event calendar...")
    event_cal = load_event_calendar()
    if event_cal.empty:
        logger.error("No event calendar found. Run build_event_calendar.py first.")
        return

    logger.info("Loading Kalshi markets...")
    markets_path = cfg.kalshi_dir / cfg.kalshi_markets_file
    if not markets_path.exists():
        logger.error("No Kalshi markets found. Run fetch_kalshi.py first.")
        return

    kalshi_markets = pd.read_parquet(markets_path)
    overrides = load_manual_overrides()

    logger.info(
        "Building mappings for %d markets x %d events...", len(kalshi_markets), len(event_cal)
    )
    mappings = build_market_mapping(event_cal, kalshi_markets, overrides)

    path = save_market_mapping(mappings)
    logger.info("Saved %d mappings to %s", len(mappings), path)

    low_path = export_low_confidence(mappings, threshold=0.5)
    low_count = sum(1 for m in mappings if m.confidence_score < 0.5)
    logger.info("Exported %d low-confidence mappings to %s", low_count, low_path)


if __name__ == "__main__":
    main()
