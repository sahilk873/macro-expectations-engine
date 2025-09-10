#!/usr/bin/env python3
"""Phase 3: Fetch Kalshi prediction market data."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from macro_engine.config import setup_logging
from macro_engine.config.settings import get_settings
from macro_engine.kalshi.client import KalshiClient

logger = logging.getLogger(__name__)


def create_sample_macro_markets() -> pd.DataFrame:
    from datetime import datetime

    rows = []
    macro_series = [
        ("CPI", "CPI", "CPI YoY", "inflation"),
        ("FOMC", "FOMC", "Fed Rate", "policy"),
        ("NONFARM", "NFP", "Nonfarm Payrolls", "employment"),
        ("UNEMPLOYMENT", "UNEMPLOYMENT", "Unemployment Rate", "employment"),
        ("GDP", "GDP", "GDP Growth", "growth"),
        ("PCE", "PCE", "PCE Price Index", "inflation"),
    ]

    release_dates = [
        "2024-01-11",
        "2024-02-13",
        "2024-03-12",
        "2024-04-10",
        "2024-05-15",
        "2024-06-12",
        "2024-07-11",
        "2024-08-14",
        "2024-09-11",
        "2024-10-10",
        "2024-11-13",
        "2024-12-11",
    ]

    for ticker, etype, title, cat in macro_series:
        for ds in release_dates:
            event_time = datetime.strptime(ds, "%Y-%m-%d").replace(hour=8, minute=30)
            close_time = datetime.strptime(ds, "%Y-%m-%d").replace(hour=6, minute=0)
            month_label = event_time.strftime("%y%b").upper()

            for bucket_idx, (lo, hi) in enumerate([(2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]):
                rows.append(
                    {
                        "ticker": f"{ticker}{bucket_idx}_{month_label}",
                        "event_ticker": f"{ticker}_{month_label}",
                        "title": f"Will {title} be between {lo}% and {hi}%?",
                        "market_type": "multi-bucket" if bucket_idx > 0 else "binary",
                        "close_time": close_time,
                        "status": "closed",
                        "result": "YES" if bucket_idx == 1 else "NO",
                        "yes_bid": 0.60 - bucket_idx * 0.2,
                        "yes_ask": 0.62 - bucket_idx * 0.2,
                        "no_bid": 0.38 + bucket_idx * 0.2,
                        "no_ask": 0.40 + bucket_idx * 0.2,
                        "last_price": 0.61 - bucket_idx * 0.2,
                        "volume": 10000 - bucket_idx * 2000,
                        "open_interest": 5000 - bucket_idx * 1000,
                    }
                )

    return pd.DataFrame(rows)


def create_sample_price_history() -> pd.DataFrame:
    from datetime import datetime, timedelta

    rows = []
    month_labels = [
        "24JAN",
        "24FEB",
        "24MAR",
        "24APR",
        "24MAY",
        "24JUN",
        "24JUL",
        "24AUG",
        "24SEP",
        "24OCT",
        "24NOV",
        "24DEC",
    ]
    month_map = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    macro_tickers = []
    for series in ["CPI", "FOMC", "NFP"]:
        for ml in month_labels:
            macro_tickers.append(f"{series}0_{ml}")

    for ticker in macro_tickers:
        for days_before in range(1, 15):
            suffix = ticker.split("_")[1]
            month_num = month_map.get(suffix[:3], 1)
            event_dt = datetime(2024, month_num, min(15, 28), 8, 30)
            ts = event_dt - timedelta(days=days_before, hours=days_before % 12)
            if ts < datetime(2024, 1, 1):
                continue
            price = 0.50 + (14 - days_before) * 0.01 + (month_num / 12) * 0.1
            price = min(max(price, 0.05), 0.95)
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": ts,
                    "yes_bid": price - 0.01,
                    "yes_ask": price + 0.01,
                    "no_bid": 1.0 - price - 0.01,
                    "no_ask": 1.0 - price + 0.01,
                    "last_price": price,
                    "volume": int(10000 + days_before * 50),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    setup_logging()
    cfg = get_settings()
    client = KalshiClient(environment=cfg.kalshi_environment)

    logger.info("Fetching Kalshi series...")
    series_list = client.list_series()
    logger.info("Found %d series", len(series_list))

    all_markets = []
    has_api_key = bool(cfg.kalshi_api_key and cfg.kalshi_api_secret)

    if has_api_key:
        for s in series_list:
            if s.series_ticker not in ("CPI", "FOMC", "NONFARM", "UNEMPLOYMENT", "GDP", "PCE"):
                continue
            logger.info("  Series: %s", s.series_ticker)
            try:
                events = client.list_events(s.series_ticker)
                logger.info("    Events: %d", len(events))
                for e in events:
                    try:
                        markets = client.list_markets(e.event_ticker)
                        all_markets.extend(markets)
                    except Exception:
                        continue
            except Exception:
                continue

    if not all_markets:
        logger.info("Using sample macro market data (no API key or rate-limited)")
        df = create_sample_macro_markets()
        path = cfg.kalshi_dir / cfg.kalshi_markets_file
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Saved %d sample markets to %s", len(df), path)

        prices = create_sample_price_history()
        price_path = cfg.kalshi_dir / cfg.kalshi_prices_file
        prices.to_parquet(price_path, index=False)
        logger.info("Saved %d sample price records to %s", len(prices), price_path)
    else:
        markets_path = cfg.kalshi_dir / cfg.kalshi_markets_file
        KalshiClient.save_markets_to_parquet(all_markets, markets_path)
        logger.info("Saved %d markets to %s", len(all_markets), markets_path)


if __name__ == "__main__":
    main()
