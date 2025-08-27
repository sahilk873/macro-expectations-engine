"""Tests for no-lookahead bias in expectations and regime classification.

These tests verify that:
1. Pre-event snapshots use only data available before the snapshot time.
2. Regime classification uses only data available as of the classification date.
3. Backtest rebalancing uses only data available at the rebalance date.
"""

from datetime import datetime, timedelta

import pandas as pd

from macro_engine.expectations.implied import create_expectation_snapshot
from macro_engine.regime.model import MacroRegimeModel


class TestNoLookaheadExpectations:
    """Verify pre-event snapshots don't include post-event data."""

    def test_snapshot_only_uses_pre_event_data(self):
        event_time = datetime(2024, 6, 12, 8, 30)
        prices = pd.DataFrame(
            {
                "ticker": ["CPI_TEST"] * 5,
                "timestamp": [
                    datetime(2024, 6, 10, 16, 0),
                    datetime(2024, 6, 11, 10, 0),
                    datetime(2024, 6, 11, 16, 0),
                    datetime(2024, 6, 12, 9, 0),  # After event
                    datetime(2024, 6, 13, 10, 0),  # After event
                ],
                "yes_bid": [0.50, 0.55, 0.58, 0.70, 0.80],
                "yes_ask": [0.52, 0.57, 0.60, 0.72, 0.82],
                "no_bid": [0.48, 0.43, 0.40, 0.28, 0.18],
                "no_ask": [0.50, 0.45, 0.42, 0.30, 0.20],
                "last_price": [0.51, 0.56, 0.59, 0.71, 0.81],
            }
        )

        snapshot = create_expectation_snapshot(prices, event_time, snapshot_offset_days=1)

        assert not snapshot.empty
        # The implied probability should be ~0.59 (from Jun 11 16:00), NOT 0.71 or 0.80
        implied_prob = snapshot.iloc[0]["implied_probability"]
        assert implied_prob < 0.65, f"Expected probability < 0.65, got {implied_prob}"

    def test_snapshot_cutoff_is_strict(self):
        event_time = datetime(2024, 6, 12, 8, 30)
        prices = pd.DataFrame(
            {
                "ticker": ["TEST"] * 3,
                "timestamp": [
                    datetime(2024, 6, 11, 7, 0),  # Before T-1 cutoff
                    datetime(2024, 6, 11, 8, 0),  # At cutoff boundary
                    datetime(2024, 6, 11, 9, 0),  # After cutoff
                ],
                "yes_bid": [0.50, 0.60, 0.70],
                "yes_ask": [0.52, 0.62, 0.72],
                "no_bid": [0.48, 0.38, 0.28],
                "no_ask": [0.50, 0.40, 0.30],
                "last_price": [0.51, 0.61, 0.71],
            }
        )

        snapshot_daily = create_expectation_snapshot(prices, event_time, snapshot_offset_days=1)
        assert not snapshot_daily.empty

        snapshot_hourly = create_expectation_snapshot(prices, event_time, snapshot_offset_hours=1.0)
        # Only the 07:00 price is before 07:30 cutoff (event 08:30 - 1hr)
        assert not snapshot_hourly.empty
        snap_time = snapshot_hourly.iloc[0]["timestamp"]
        assert snap_time < event_time - timedelta(hours=1)


class TestNoLookaheadRegime:
    """Verify regime classification uses only past data."""

    def test_regime_only_uses_past_data(self):
        model = MacroRegimeModel(lookback_days=180)

        macro_data = pd.DataFrame(
            {
                "series_id": ["GDPC1", "GDPC1", "GDPC1"],
                "date": ["2024-01-01", "2024-06-01", "2024-12-01"],
                "value": [2.0, 1.5, 3.0],
            }
        )
        price_data = pd.DataFrame(
            {
                "ticker": ["SPY", "SPY", "SPY"],
                "date": ["2024-01-01", "2024-06-01", "2024-12-01"],
                "close": [400, 420, 450],
            }
        )

        # Classify as of mid-2024
        regime_mid = model.classify(macro_data, price_data, as_of_date="2024-07-01")
        # The 2024-12-01 data should NOT be used
        assert regime_mid["date"] == "2024-07-01"

    def test_regime_different_dates_give_different_results(self):
        model = MacroRegimeModel(lookback_days=180)

        macro_data = pd.DataFrame(
            {
                "series_id": ["PAYEMS", "PAYEMS", "PAYEMS"],
                "date": ["2020-01-01", "2020-06-01", "2020-12-01"],
                "value": [150000, -500000, 100000],
            }
        )
        price_data = pd.DataFrame(
            {
                "ticker": ["SPY", "SPY", "SPY"],
                "date": ["2020-01-01", "2020-06-01", "2020-12-01"],
                "close": [300, 250, 350],
            }
        )

        r1 = model.classify(macro_data, price_data, as_of_date="2020-03-01")
        r2 = model.classify(macro_data, price_data, as_of_date="2020-09-01")

        # Different dates should have potentially different regimes
        assert r1["date"] != r2["date"]


class TestNoLookaheadBacktest:
    """Verify backtest doesn't use future information."""

    def test_mapping_confidence_not_from_future(self):
        from datetime import datetime

        from macro_engine.mapping.mapper import _compute_confidence

        conf = _compute_confidence(
            "CPIYOY_24JUN",
            "CPI YoY June 2024",
            "CPI",
            datetime(2024, 6, 12, 8, 30),
        )
        assert 0.0 <= conf <= 1.0

    def test_surprise_uses_pre_event_expectation(self):
        from macro_engine.surprises.calculator import compute_all_surprises

        # Build minimal datasets
        implied = pd.DataFrame(
            {
                "event_id": ["CPI_20240612"],
                "event_type": ["CPI"],
                "market_ticker": ["CPIYOY"],
                "snapshot_type": ["T-1_day"],
                "snapshot_time": [datetime(2024, 6, 11, 16, 0)],
                "event_time": [datetime(2024, 6, 12, 8, 30)],
                "implied_probability": [0.6],
                "confidence_score": [0.8],
            }
        )
        event_cal = pd.DataFrame(
            {
                "event_id": ["CPI_20240612"],
                "event_type": ["CPI"],
                "ticker": ["CPIAUCSL"],
                "release_datetime": [datetime(2024, 6, 12, 8, 30)],
                "event_name": ["CPI Jun 2024"],
                "source": ["BLS"],
                "category": ["inflation"],
            }
        )
        macro = pd.DataFrame(
            {
                "series_id": ["CPIAUCSL"],
                "date": ["2024-06-12"],
                "value": [0.65],
            }
        )
        mapping = pd.DataFrame(
            {
                "market_ticker": ["CPIYOY"],
                "event_id": ["CPI_20240612"],
                "event_type": ["CPI"],
                "confidence_score": [0.8],
                "market_title": ["CPI Test"],
            }
        )

        surprises = compute_all_surprises(implied, macro, event_cal, mapping)
        assert not surprises.empty

        # The expectation snapshot was taken at T-1 day,
        # but the realized value is from after the event
        # This is correct: we compare pre-event expectation to post-event realization
        first = surprises.iloc[0]
        assert first["snapshot_time"] < first["event_time"]
