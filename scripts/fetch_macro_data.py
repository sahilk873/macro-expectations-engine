#!/usr/bin/env python3
"""Phase 5: Fetch official macro data from BLS, BEA, FRED."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.macro.sources import build_macro_dataset, save_macro_data


def main():
    print("Fetching official macro data...")
    df = build_macro_dataset()
    if not df.empty:
        path = save_macro_data(df)
        print(f"Saved {len(df)} records to {path}")
        print(f"Series: {df['series_id'].nunique() if 'series_id' in df.columns else 'N/A'}")
    else:
        print("No macro data fetched (check API keys)")

    print("Done.")


if __name__ == "__main__":
    main()
