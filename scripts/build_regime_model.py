#!/usr/bin/env python3
"""Phase 10: Build macro regime model."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config import setup_logging
from macro_engine.config.settings import get_settings
from macro_engine.macro.sources import load_macro_data
from macro_engine.prices.providers import load_price_data
from macro_engine.regime.model import compute_macro_regime, save_regime_classifications

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_settings()

    logger.info("Loading data...")
    macro_data = load_macro_data()
    price_data = load_price_data()

    if macro_data.empty and price_data.empty:
        logger.error("No macro or price data available.")
        return

    logger.info("Computing macro regime classifications...")
    regimes = compute_macro_regime(macro_data, price_data, config=cfg)

    if not regimes.empty:
        path = save_regime_classifications(regimes)
        logger.info("Saved %d regime classifications to %s", len(regimes), path)
        logger.info("Growth regimes: %s", regimes["growth_regime"].value_counts().to_dict())
        logger.info("Inflation regimes: %s", regimes["inflation_regime"].value_counts().to_dict())
        logger.info("Risk regimes: %s", regimes["risk_regime"].value_counts().to_dict())
    else:
        logger.warning("No regime classifications computed")


if __name__ == "__main__":
    main()
