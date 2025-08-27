"""Tests for event-time alignment and pre-event snapshots."""

from datetime import datetime, timedelta

import pandas as pd

from macro_engine.expectations.implied import create_expectation_snapshot


class TestCreateExpectationSnapshot:
    def test_daily_snapshot(self):
        event_time = datetime(2024, 6, 12, 8, 30)
        prices = pd.DataFrame(
            {
                "ticker": ["CPIYOY_24JUN"] * 5,
                "timestamp": [
                    datetime(2024, 6, 10, 16, 0),
                    datetime(2024, 6, 11, 10, 0),
                    datetime(2024, 6, 11, 16, 0),
                    datetime(2024, 6, 12, 8, 0),
                    datetime(2024, 6, 12, 9, 0),
                ],
                "yes_bid": [0.50, 0.55, 0.58, 0.60, 0.62],
                "yes_ask": [0.52, 0.57, 0.60, 0.62, 0.64],
                "no_bid": [0.48, 0.43, 0.40, 0.38, 0.36],
                "no_ask": [0.50, 0.45, 0.42, 0.40, 0.38],
                "last_price": [0.51, 0.56, 0.59, 0.61, 0.63],
            }
        )

        snapshot = create_expectation_snapshot(prices, event_time, snapshot_offset_days=1)

        assert not snapshot.empty
        assert snapshot.iloc[0]["snapshot_time"] <= event_time - timedelta(days=1)
        assert "implied_probability" in snapshot.columns

    def test_hourly_snapshot(self):
        event_time = datetime(2024, 6, 12, 14, 0)
        prices = pd.DataFrame(
            {
                "ticker": ["FEDTARGET_24JUN"] * 4,
                "timestamp": [
                    datetime(2024, 6, 12, 8, 0),
                    datetime(2024, 6, 12, 10, 0),
                    datetime(2024, 6, 12, 12, 0),
                    datetime(2024, 6, 12, 13, 30),
                ],
                "yes_bid": [0.70, 0.72, 0.75, 0.78],
                "yes_ask": [0.72, 0.74, 0.77, 0.80],
                "no_bid": [0.28, 0.26, 0.23, 0.20],
                "no_ask": [0.30, 0.28, 0.25, 0.22],
                "last_price": [0.71, 0.73, 0.76, 0.79],
            }
        )

        snapshot = create_expectation_snapshot(prices, event_time, snapshot_offset_hours=1.0)

        assert not snapshot.empty
        # Should take the 12:00 price (before 13:00 cutoff = event_time - 1hr)
        assert snapshot.iloc[0]["timestamp"] < event_time - timedelta(hours=1)

    def test_snapshot_no_prices_before_cutoff(self):
        event_time = datetime(2024, 6, 12, 8, 30)
        prices = pd.DataFrame(
            {
                "ticker": ["TEST"] * 3,
                "timestamp": [
                    datetime(2024, 6, 12, 9, 0),
                    datetime(2024, 6, 12, 10, 0),
                    datetime(2024, 6, 12, 11, 0),
                ],
                "yes_bid": [0.50, 0.55, 0.60],
                "yes_ask": [0.52, 0.57, 0.62],
                "no_bid": [0.48, 0.43, 0.40],
                "no_ask": [0.50, 0.45, 0.42],
                "last_price": [0.51, 0.56, 0.61],
            }
        )

        snapshot = create_expectation_snapshot(prices, event_time, snapshot_offset_days=1)
        assert snapshot.empty

    def test_last_price_taken(self):
        event_time = datetime(2024, 6, 12, 8, 30)
        prices = pd.DataFrame(
            {
                "ticker": ["TEST"] * 3,
                "timestamp": [
                    datetime(2024, 6, 10, 10, 0),
                    datetime(2024, 6, 10, 14, 0),
                    datetime(2024, 6, 10, 16, 0),
                ],
                "yes_bid": [0.50, 0.55, 0.60],
                "yes_ask": [0.52, 0.57, 0.62],
                "no_bid": [0.48, 0.43, 0.40],
                "no_ask": [0.50, 0.45, 0.42],
                "last_price": [0.51, 0.56, 0.61],
            }
        )

        snapshot = create_expectation_snapshot(prices, event_time, snapshot_offset_days=1)
        assert not snapshot.empty
        # Should pick the 16:00 price as it's the last
        assert snapshot.iloc[0]["timestamp"] == datetime(2024, 6, 10, 16, 0)

    def test_empty_prices(self):
        snap = create_expectation_snapshot(pd.DataFrame(), datetime.now(), snapshot_offset_days=1)
        assert snap.empty

    def test_multiple_tickers(self):
        event_time = datetime(2024, 6, 12, 8, 30)
        prices = pd.DataFrame(
            {
                "ticker": ["TICKER_A", "TICKER_A", "TICKER_B", "TICKER_B"],
                "timestamp": [
                    datetime(2024, 6, 10, 16, 0),
                    datetime(2024, 6, 11, 14, 0),
                    datetime(2024, 6, 10, 15, 0),
                    datetime(2024, 6, 11, 16, 0),
                ],
                "yes_bid": [0.50, 0.55, 0.60, 0.65],
                "yes_ask": [0.52, 0.57, 0.62, 0.67],
                "no_bid": [0.48, 0.43, 0.38, 0.33],
                "no_ask": [0.50, 0.45, 0.40, 0.35],
                "last_price": [0.51, 0.56, 0.61, 0.66],
            }
        )

        snapshot = create_expectation_snapshot(prices, event_time, snapshot_offset_days=1)
        assert len(snapshot) == 2  # One per ticker
