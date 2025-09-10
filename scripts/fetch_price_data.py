#!/usr/bin/env python3
"""Phase 6: Fetch ETF price data."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config import setup_logging
from macro_engine.prices.providers import get_price_data, save_price_data

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    logger.info("Fetching ETF price data...")
    df = get_price_data()
    if not df.empty:
        path = save_price_data(df)
        logger.info("Saved %d records to %s", len(df), path)
        if "ticker" in df.columns:
            logger.info("Tickers: %d unique", df["ticker"].nunique())
        if "date" in df.columns:
            logger.info("Date range: %s to %s", df["date"].min(), df["date"].max())
    else:
        logger.warning("No price data fetched")


if __name__ == "__main__":
    main()
