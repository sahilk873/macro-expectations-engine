"""Tests for event return window computation."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from macro_engine.studies.event_study import (
    aggregate_event_study,
    compute_event_returns,
    open_event_window,
)


@pytest.fixture
def sample_price_data():
    """Create sample daily price data for multiple tickers."""
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="B")
    rows = []
    np.random.seed(42)
    for ticker in ["SPY", "TLT"]:
        price = 400.0 if ticker == "SPY" else 100.0
        for dt in dates:
            change = np.random.randn() * 0.005
            price *= 1 + change
            rows.append({"date": dt, "ticker": ticker, "close": price, "volume": 1e7})
    return pd.DataFrame(rows)


class TestComputeEventReturns:
    def test_basic_event_returns(self, sample_price_data):
        event_time = datetime(2024, 1, 15, 8, 30)
        results = compute_event_returns(
            sample_price_data,
            event_time,
            ["SPY", "TLT"],
            post_windows=["1D", "3D"],
        )

        assert "SPY" in results
        assert "TLT" in results
        assert "return_1D" in results["SPY"]
        assert "return_3D" in results["SPY"]

    def test_return_sign(self, sample_price_data):
        event_time = datetime(2024, 1, 10, 8, 30)
        results = compute_event_returns(
            sample_price_data,
            event_time,
            ["SPY"],
            post_windows=["1D"],
        )
        ret = results["SPY"]["return_1D"]
        assert isinstance(ret, float)
        assert not np.isnan(ret)

    def test_event_before_data(self, sample_price_data):
        event_time = datetime(2019, 1, 1, 8, 30)
        results = compute_event_returns(
            sample_price_data,
            event_time,
            ["SPY"],
            post_windows=["1D"],
        )
        # No pre-event data available, so SPY should not be in results
        assert results == {}

    def test_empty_price_data(self):
        results = compute_event_returns(
            pd.DataFrame(),
            datetime.now(),
            ["SPY"],
        )
        assert results == {}

    def test_unknown_ticker(self, sample_price_data):
        results = compute_event_returns(
            sample_price_data,
            datetime(2024, 1, 15, 8, 30),
            ["UNKNOWN"],
        )
        assert results == {}


class TestOpenEventWindow:
    def test_window_creation(self, sample_price_data):
        event_time = datetime(2024, 1, 15, 8, 30)
        window = open_event_window(sample_price_data, event_time, "SPY", window_days=10)

        assert window is not None
        assert "cum_return" in window.columns
        assert window["cum_return"].iloc[0] == 0.0

    def test_window_missing_ticker(self, sample_price_data):
        window = open_event_window(sample_price_data, datetime.now(), "UNKNOWN")
        assert window is None

    def test_window_cum_return_progression(self, sample_price_data):
        event_time = datetime(2024, 1, 15, 8, 30)
        window = open_event_window(sample_price_data, event_time, "SPY", window_days=5)

        assert window is not None
        cum_rets = window["cum_return"].values
        assert cum_rets[0] == 0.0


class TestAggregateEventStudy:
    def test_basic_aggregation(self):
        data = pd.DataFrame(
            {
                "event_type": ["CPI", "CPI", "FOMC", "FOMC"],
                "event_id": ["E1", "E2", "E3", "E4"],
                "ticker": ["SPY", "SPY", "SPY", "SPY"],
                "return_1D": [0.01, -0.005, 0.02, -0.01],
                "return_3D": [0.015, 0.0, 0.025, -0.015],
            }
        )

        agg = aggregate_event_study(data, group_by=["event_type"])
        assert not agg.empty
        assert "return_1D_mean" in agg.columns
        assert len(agg) == 2  # CPI, FOMC

    def test_aggregation_with_direction(self):
        data = pd.DataFrame(
            {
                "event_type": ["CPI", "CPI", "CPI", "CPI"],
                "direction": [
                    "above_expectations",
                    "above_expectations",
                    "below_expectations",
                    "below_expectations",
                ],
                "event_id": ["E1", "E2", "E3", "E4"],
                "ticker": ["SPY", "TLT", "SPY", "TLT"],
                "return_1D": [0.01, -0.005, -0.01, 0.005],
            }
        )

        agg = aggregate_event_study(data, group_by=["event_type", "direction"])
        assert not agg.empty
        assert len(agg) == 2

    def test_empty_data(self):
        agg = aggregate_event_study(pd.DataFrame())
        assert agg.empty
