#!/usr/bin/env python3
"""Phase 6: Fetch ETF price data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.prices.providers import get_price_data, save_price_data


def main():
    print("Fetching ETF price data...")
    df = get_price_data()
    if not df.empty:
        path = save_price_data(df)
        print(f"Saved {len(df)} records to {path}")
        print(f"Tickers: {df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'}")
        print(
            f"Date range: {df['date'].min() if 'date' in df.columns else 'N/A'} to {df['date'].max() if 'date' in df.columns else 'N/A'}"
        )
    else:
        print("No price data fetched")

    print("Done.")


if __name__ == "__main__":
    main()
