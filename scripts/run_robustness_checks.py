#!/usr/bin/env python3
"""Phase 12: Run robustness checks and placebo tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config.settings import get_settings
from macro_engine.surprises.calculator import load_surprises
from macro_engine.prices.providers import load_price_data
from macro_engine.studies.event_study import load_event_studies
from macro_engine.robustness.checks import run_robustness_checks, save_robustness_results


def main():
    cfg = get_settings()

    print("Loading data...")
    surprises = load_surprises()
    price_data = load_price_data()
    event_studies = load_event_studies()

    if surprises.empty or price_data.empty:
        print("Missing required data.")
        return

    print("Running robustness checks...")
    results = run_robustness_checks(surprises, price_data, event_studies, config=cfg)

    paths = save_robustness_results(results)
    for key, p in paths.items():
        df = results.get(key, None)
        if df is not None:
            print(f"  {key:20s}: {len(df)} records -> {p}")

    summary = results.get("placebo_summary", None)
    if summary is not None and not summary.empty:
        print("\nPlacebo Test Summary:")
        for _, row in summary.iterrows():
            sig = " ***" if row.get("significant_5pct", False) else ""
            print(
                f"  {row['window']:15s}  actual={row['actual_mean_return']:+.4f}  p={row['p_value']:.4f}{sig}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
