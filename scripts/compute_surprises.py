#!/usr/bin/env python3
"""Phase 8: Compute macro surprises from implied vs. realized."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.events.calendar import load_event_calendar
from macro_engine.mapping.mapper import load_market_mapping
from macro_engine.expectations.implied import load_implied_expectations
from macro_engine.macro.sources import load_macro_data
from macro_engine.surprises.calculator import compute_all_surprises, save_surprises


def main():
    print("Loading data...")
    event_cal = load_event_calendar()
    mapping = load_market_mapping()
    expectations = load_implied_expectations()
    macro_data = load_macro_data()

    if expectations.empty:
        print("No implied expectations found. Run compute_implied_expectations.py first.")
        return

    print(f"Computing surprises for {len(expectations)} expectation snapshots...")
    surprises = compute_all_surprises(expectations, macro_data, event_cal, mapping)

    if not surprises.empty:
        path = save_surprises(surprises)
        print(f"Saved {len(surprises)} surprise records to {path}")
        print(f"Direction breakdown: {surprises['direction'].value_counts().to_dict()}")
        print(f"Qualitative breakdown: {surprises['qualitative'].value_counts().to_dict()}")
    else:
        print("No surprises computed (check macro data availability)")

    print("Done.")


if __name__ == "__main__":
    main()
