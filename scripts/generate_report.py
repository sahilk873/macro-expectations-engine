#!/usr/bin/env python3
"""Phase 13: Generate reports, tables, and figures."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config import setup_logging
from macro_engine.backtest.strategy import load_backtest_results
from macro_engine.report.generator import (
    generate_all_figures,
    generate_all_tables,
)
from macro_engine.robustness.checks import load_robustness_results
from macro_engine.studies.event_study import load_event_studies
from macro_engine.surprises.calculator import load_surprises

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    logger.info("Loading results...")
    surprises = load_surprises()
    event_studies = load_event_studies()
    backtest_results = load_backtest_results()
    robustness_results = load_robustness_results()

    logger.info("Generating tables...")
    table_paths = generate_all_tables(
        surprises, event_studies, backtest_results, robustness_results
    )
    for name, path in table_paths.items():
        logger.info("  %20s: %s", name, path)

    logger.info("Generating figures...")
    figure_paths = generate_all_figures(surprises, event_studies, backtest_results)
    for name, path in figure_paths.items():
        logger.info("  %25s: %s", name, path)


if __name__ == "__main__":
    main()
