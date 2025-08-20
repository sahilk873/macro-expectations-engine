#!/usr/bin/env python3
"""Phase 13: Generate reports, tables, and figures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config.settings import get_settings
from macro_engine.surprises.calculator import load_surprises
from macro_engine.studies.event_study import load_event_studies
from macro_engine.backtest.strategy import load_backtest_results
from macro_engine.robustness.checks import load_robustness_results
from macro_engine.report.generator import (
    generate_all_tables,
    generate_all_figures,
    generate_research_report,
)


def main():
    cfg = get_settings()

    print("Loading results...")
    surprises = load_surprises()
    event_studies = load_event_studies()
    backtest_results = load_backtest_results()
    robustness_results = load_robustness_results()

    print("Generating tables...")
    table_paths = generate_all_tables(
        surprises, event_studies, backtest_results, robustness_results
    )
    for name, path in table_paths.items():
        print(f"  {name:20s}: {path}")

    print("Generating figures...")
    figure_paths = generate_all_figures(surprises, event_studies, backtest_results)
    for name, path in figure_paths.items():
        print(f"  {name:25s}: {path}")

    print("Done. Reports saved to {cfg.tables_dir} and {cfg.figures_dir}")


if __name__ == "__main__":
    main()
