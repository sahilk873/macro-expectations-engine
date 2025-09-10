#!/usr/bin/env python3
"""Phase 9: Run event-study analysis."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config import setup_logging
from macro_engine.config.settings import get_settings
from macro_engine.prices.providers import load_price_data
from macro_engine.studies.event_study import run_event_studies, save_event_studies
from macro_engine.surprises.calculator import load_surprises

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_settings()

    logger.info("Loading data...")
    surprises = load_surprises()
    price_data = load_price_data()

    if surprises.empty or price_data.empty:
        logger.error("Missing required data.")
        return

    logger.info("Running event studies for %d surprise records...", len(surprises))
    studies = run_event_studies(surprises, price_data, config=cfg)

    if not studies.empty:
        path = save_event_studies(studies)
        logger.info("Saved %d event study records to %s", len(studies), path)
        logger.info("Tickers: %d unique", studies["ticker"].nunique())
        logger.info("Event types: %s", studies["event_type"].value_counts().to_dict())
    else:
        logger.warning("No event studies computed")


if __name__ == "__main__":
    main()
