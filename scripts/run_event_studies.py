#!/usr/bin/env python3
"""Phase 9: Run event-study analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config.settings import get_settings
from macro_engine.surprises.calculator import load_surprises
from macro_engine.prices.providers import load_price_data
from macro_engine.studies.event_study import run_event_studies, save_event_studies


def main():
    cfg = get_settings()

    print("Loading data...")
    surprises = load_surprises()
    price_data = load_price_data()

    if surprises.empty or price_data.empty:
        print("Missing required data. Ensure surprises and price data exist.")
        return

    print(f"Running event studies for {len(surprises)} surprise records...")
    studies = run_event_studies(surprises, price_data, config=cfg)

    if not studies.empty:
        path = save_event_studies(studies)
        print(f"Saved {len(studies)} event study records to {path}")
        print(f"Tickers: {studies['ticker'].nunique()}")
        print(f"Event types: {studies['event_type'].value_counts().to_dict()}")
    else:
        print("No event studies computed")

    print("Done.")


if __name__ == "__main__":
    main()
