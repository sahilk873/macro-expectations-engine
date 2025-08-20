#!/usr/bin/env python3
"""Phase 7: Compute implied expectations from prediction markets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.events.calendar import load_event_calendar
from macro_engine.kalshi.client import KalshiClient
from macro_engine.mapping.mapper import load_market_mapping
from macro_engine.expectations.implied import (
    compute_implied_expectations,
    save_implied_expectations,
)
from macro_engine.config.settings import get_settings


def main():
    cfg = get_settings()

    print("Loading data...")
    event_cal = load_event_calendar()
    mapping = load_market_mapping()

    # Load Kalshi prices
    prices_path = cfg.kalshi_dir / cfg.kalshi_prices_file
    import pandas as pd

    if prices_path.exists():
        kalshi_prices = pd.read_parquet(prices_path)
    else:
        print("No Kalshi prices found. Run fetch_kalshi.py first.")
        return

    if event_cal.empty or mapping.empty or kalshi_prices.empty:
        print("Missing required data. Ensure event calendar, mapping, and prices exist.")
        return

    print(f"Computing implied expectations for {len(mapping)} mappings...")
    expectations = compute_implied_expectations(event_cal, kalshi_prices, mapping)

    exp_path, dist_path = save_implied_expectations(expectations)
    print(f"Saved {len(expectations)} expectation snapshots to {exp_path}")

    print(f"Snapshot types: {expectations['snapshot_type'].value_counts().to_dict()}")
    print("Done.")


if __name__ == "__main__":
    main()
