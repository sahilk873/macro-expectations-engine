#!/usr/bin/env python3
"""Phase 5: Fetch official macro data from BLS, BEA, FRED."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config import setup_logging
from macro_engine.macro.sources import build_macro_dataset, save_macro_data

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    logger.info("Fetching official macro data...")
    df = build_macro_dataset()
    if not df.empty:
        path = save_macro_data(df)
        logger.info("Saved %d records to %s", len(df), path)
        if "series_id" in df.columns:
            logger.info("Series: %d unique", df["series_id"].nunique())
    else:
        logger.warning("No macro data fetched (check API keys)")


if __name__ == "__main__":
    main()
