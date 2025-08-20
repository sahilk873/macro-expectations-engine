#!/usr/bin/env python3
"""Phase 4: Build market-to-event mappings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.events.calendar import load_event_calendar
from macro_engine.kalshi.client import KalshiClient
from macro_engine.mapping.mapper import (
    build_market_mapping,
    save_market_mapping,
    export_low_confidence,
    load_manual_overrides,
)
from macro_engine.config.settings import get_settings


def main():
    cfg = get_settings()

    print("Loading event calendar...")
    event_cal = load_event_calendar()
    if event_cal.empty:
        print("No event calendar found. Run build_event_calendar.py first.")
        return

    print("Loading Kalshi markets...")
    markets_path = cfg.kalshi_dir / cfg.kalshi_markets_file
    if not markets_path.exists():
        print("No Kalshi markets found. Run fetch_kalshi.py first.")
        return
    import pandas as pd

    kalshi_markets = pd.read_parquet(markets_path)

    print("Loading manual overrides...")
    overrides = load_manual_overrides()

    print(f"Building mappings for {len(kalshi_markets)} markets x {len(event_cal)} events...")
    mappings = build_market_mapping(event_cal, kalshi_markets, overrides)

    path = save_market_mapping(mappings)
    print(f"Saved {len(mappings)} mappings to {path}")

    # Export low-confidence for review
    low_path = export_low_confidence(mappings, threshold=0.5)
    low_count = sum(1 for m in mappings if m.confidence_score < 0.5)
    print(f"Exported {low_count} low-confidence mappings to {low_path}")

    print("Done.")


if __name__ == "__main__":
    main()
