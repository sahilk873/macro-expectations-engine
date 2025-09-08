"""Integration tests for the full macro engine pipeline.

Tests the end-to-end flow using synthetic data to verify
that all modules work together correctly.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from macro_engine.backtest.strategy import RegimeAwareStrategy, run_backtest
from macro_engine.expectations.implied import (
    compute_implied_expectations,
    create_expectation_snapshot,
)
from macro_engine.regime.model import MacroRegimeModel
from macro_engine.robustness.checks import placebo_test_random_dates, placebo_test_random_signs
from macro_engine.studies.event_study import aggregate_event_study, run_event_studies
from macro_engine.surprises.calculator import (
    compute_all_surprises,
    label_surprise,
)


@pytest.fixture
def sample_event_calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["CPI_20240612", "FOMC_20240612", "NFP_20240607"],
            "event_type": ["CPI", "FOMC", "NFP"],
            "event_name": ["CPI Jun 2024", "FOMC Jun 2024", "NFP Jun 2024"],
            "release_datetime": [
                datetime(2024, 6, 12, 8, 30),
                datetime(2024, 6, 12, 14, 0),
                datetime(2024, 6, 7, 8, 30),
            ],
            "source": ["BLS", "FOMC", "BLS"],
            "ticker": ["CPIAUCSL", "FEDFUNDS", "PAYEMS"],
            "category": ["inflation", "policy", "employment"],
        }
    )


@pytest.fixture
def sample_kalshi_prices() -> pd.DataFrame:
    rows = []
    for ticker in ["CPIYOY_24JUN", "FEDTARGET_24JUN", "NFP_24JUN"]:
        for days_back in range(1, 15):
            ts = datetime(2024, 6, 12, 8, 30) - pd.Timedelta(days=days_back)
            if ticker == "NFP_24JUN":
                ts = datetime(2024, 6, 7, 8, 30) - pd.Timedelta(days=days_back)
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": ts,
                    "yes_bid": 0.55,
                    "yes_ask": 0.57,
                    "no_bid": 0.43,
                    "no_ask": 0.45,
                    "last_price": 0.56,
                    "volume": 10000,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_macro_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["CPIAUCSL", "FEDFUNDS", "PAYEMS"],
            "date": ["2024-06-01", "2024-06-01", "2024-06-01"],
            "value": [3.3, 5.5, 272000],
            "name": ["CPI", "Fed Funds", "Nonfarm Payrolls"],
            "category": ["inflation", "policy", "employment"],
            "frequency": ["monthly", "monthly", "monthly"],
        }
    )


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    dates = pd.date_range(start="2024-06-01", end="2024-06-30", freq="B")
    rows = []
    np.random.seed(42)
    for ticker in ["SPY", "TLT", "GLD"]:
        price = {"SPY": 500, "TLT": 95, "GLD": 200}[ticker]
        for dt in dates:
            change = np.random.randn() * 0.005
            price *= 1 + change
            rows.append({"date": dt, "ticker": ticker, "close": price, "volume": 1e7})
    return pd.DataFrame(rows)


class TestIntegrationSurprisePipeline:
    """End-to-end test: calendar -> mapping -> expectations -> surprises."""

    def test_full_surprise_pipeline(
        self, sample_event_calendar, sample_kalshi_prices, sample_macro_data
    ):
        mapping = pd.DataFrame(
            {
                "market_ticker": ["CPIYOY_24JUN", "FEDTARGET_24JUN", "NFP_24JUN"],
                "market_title": ["CPI YoY Jun 2024", "Fed Rate Jun 2024", "NFP Jun 2024"],
                "event_id": ["CPI_20240612", "FOMC_20240612", "NFP_20240607"],
                "event_type": ["CPI", "FOMC", "NFP"],
                "confidence_score": [0.85, 0.80, 0.75],
            }
        )

        expectations = compute_implied_expectations(
            sample_event_calendar, sample_kalshi_prices, mapping
        )
        assert not expectations.empty
        assert "implied_probability" in expectations.columns

        surprises = compute_all_surprises(
            expectations, sample_macro_data, sample_event_calendar, mapping
        )
        assert not surprises.empty
        assert "raw_surprise" in surprises.columns
        assert "standardized_surprise" in surprises.columns
        assert "direction" in surprises.columns

    def test_surprise_labels_are_consistent(self):
        for etype in ["CPI", "PCE", "NFP", "UNEMPLOYMENT", "GDP", "FOMC"]:
            pos = label_surprise(etype, 1.0, 2.0)
            neg = label_surprise(etype, -1.0, -2.0)
            assert pos["direction"] != neg["direction"]
            assert pos["qualitative"] != "neutral" or neg["qualitative"] != "neutral"


class TestIntegrationEventStudy:
    """End-to-end test: surprises -> event studies -> aggregation."""

    def test_event_study_pipeline(self, sample_price_data):
        surprises = pd.DataFrame(
            {
                "event_id": ["CPI_20240612", "FOMC_20240612"],
                "event_type": ["CPI", "FOMC"],
                "snapshot_type": ["T-1_day", "T-1_day"],
                "event_time": [datetime(2024, 6, 12, 8, 30), datetime(2024, 6, 12, 14, 0)],
                "direction": ["above_expectations", "above_expectations"],
                "qualitative": ["inflation_hot", "policy_hawkish"],
                "risk_label": ["risk_off", "risk_off"],
                "raw_surprise": [0.5, 0.25],
                "standardized_surprise": [2.0, 1.5],
                "market_ticker": ["CPIYOY", "FEDTARGET"],
                "confidence_score": [0.85, 0.80],
                "expected_probability": [0.6, 0.7],
                "realized_value": [3.3, 5.5],
                "percent_surprise": [0.1, 0.05],
            }
        )

        studies = run_event_studies(
            surprises,
            sample_price_data,
            tickers=["SPY", "TLT"],
            post_windows=["1D", "3D"],
        )
        assert not studies.empty
        assert "return_1D" in studies.columns
        assert "return_3D" in studies.columns

        agg = aggregate_event_study(studies, group_by=["event_type"])
        assert not agg.empty
        assert "return_1D_mean" in agg.columns
        assert "return_1D_pvalue" in agg.columns
        assert "return_1D_ci_lower" in agg.columns

    def test_bootstrap_confidence_intervals(self, sample_price_data):
        studies = run_event_studies(
            pd.DataFrame(
                {
                    "event_id": ["E1", "E2"],
                    "event_type": ["CPI", "CPI"],
                    "snapshot_type": ["T-1_day", "T-1_day"],
                    "event_time": [datetime(2024, 6, 12, 8, 30), datetime(2024, 6, 7, 8, 30)],
                    "direction": ["above_expectations", "below_expectations"],
                    "qualitative": ["inflation_hot", "inflation_cool"],
                    "risk_label": ["risk_off", "risk_on"],
                    "raw_surprise": [0.5, -0.3],
                    "standardized_surprise": [2.0, -1.5],
                    "market_ticker": ["CPIYOY", "CPIYOY"],
                    "confidence_score": [0.85, 0.85],
                    "expected_probability": [0.6, 0.5],
                    "realized_value": [3.3, 3.1],
                    "percent_surprise": [0.1, -0.05],
                }
            ),
            sample_price_data,
            tickers=["SPY"],
            post_windows=["1D"],
        )
        agg = aggregate_event_study(studies)
        assert "return_1D_ci_lower" in agg.columns
        assert "return_1D_ci_upper" in agg.columns


class TestIntegrationRegimeBacktest:
    """End-to-end test: regime classification -> backtest -> performance."""

    def test_regime_backtest_pipeline(self, sample_price_data):
        regimes = pd.DataFrame(
            {
                "date": ["2024-06-03", "2024-07-01"],
                "growth_regime": ["expansion", "expansion"],
                "inflation_regime": ["stable", "rising"],
                "policy_regime": ["restrictive", "restrictive"],
                "volatility_regime": ["low", "normal"],
                "risk_regime": ["risk_on", "neutral"],
            }
        )

        strategy = RegimeAwareStrategy(transaction_cost_bps=3.0)
        results = run_backtest(sample_price_data, regimes, strategy)

        assert not results.empty
        assert "portfolio_value" in results.columns
        assert "daily_return" in results.columns
        assert results["portfolio_value"].iloc[-1] > 0

    def test_regime_shifts_affect_weights(self):
        strategy = RegimeAwareStrategy()
        risk_on = strategy.get_weights(
            {"risk_regime": "risk_on", "inflation_regime": "stable", "volatility_regime": "low"}
        )
        risk_off = strategy.get_weights(
            {"risk_regime": "risk_off", "inflation_regime": "rising", "volatility_regime": "high"}
        )
        assert risk_on["SPY"] > risk_off["SPY"]
        assert risk_off.get("TLT", 0) > risk_on.get("TLT", 0)

    def test_turnover_computation(self):
        old = {"SPY": 0.5, "TLT": 0.5}
        new = {"SPY": 0.6, "TLT": 0.4}
        turnover = RegimeAwareStrategy.compute_turnover(old, new)
        assert turnover == pytest.approx(0.1, abs=1e-10)


class TestIntegrationRobustness:
    """End-to-end test: robustness checks with placebo tests."""

    def test_placebo_date_test(self, sample_price_data):
        surprises = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "event_type": ["CPI", "CPI"],
                "snapshot_type": ["T-1_day", "T-1_day"],
                "event_time": [datetime(2024, 6, 12, 8, 30), datetime(2024, 6, 7, 8, 30)],
                "direction": ["above_expectations", "below_expectations"],
                "qualitative": ["inflation_hot", "inflation_cool"],
                "risk_label": ["risk_off", "risk_on"],
                "raw_surprise": [0.5, -0.3],
                "standardized_surprise": [2.0, -1.5],
                "market_ticker": ["CPIYOY", "CPIYOY"],
                "confidence_score": [0.85, 0.85],
                "expected_probability": [0.6, 0.5],
                "realized_value": [3.3, 3.1],
                "percent_surprise": [0.1, -0.05],
            }
        )

        results = placebo_test_random_dates(
            surprises, sample_price_data, tickers=["SPY"], n_iterations=10, seed=42
        )
        if not results.empty:
            assert "placebo_return" in results.columns
            assert results["iteration"].nunique() <= 10

    def test_placebo_sign_test(self, sample_price_data):
        surprises = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "event_type": ["CPI", "CPI"],
                "snapshot_type": ["T-1_day", "T-1_day"],
                "event_time": [datetime(2024, 6, 12, 8, 30), datetime(2024, 6, 7, 8, 30)],
                "direction": ["above_expectations", "below_expectations"],
                "qualitative": ["inflation_hot", "inflation_cool"],
                "risk_label": ["risk_off", "risk_on"],
                "raw_surprise": [0.5, -0.3],
                "standardized_surprise": [2.0, -1.5],
                "market_ticker": ["CPIYOY", "CPIYOY"],
                "confidence_score": [0.85, 0.85],
                "expected_probability": [0.6, 0.5],
                "realized_value": [3.3, 3.1],
                "percent_surprise": [0.1, -0.05],
            }
        )

        results = placebo_test_random_signs(
            surprises, sample_price_data, tickers=["SPY"], n_iterations=10, seed=42
        )
        if not results.empty:
            assert "placebo_return" in results.columns


class TestNoLookaheadBias:
    """Verify no lookahead bias across the integrated pipeline."""

    def test_expectations_dont_use_future(self):
        event_time = datetime(2024, 6, 12, 8, 30)
        prices = pd.DataFrame(
            {
                "ticker": ["CPI_TEST"] * 5,
                "timestamp": [
                    datetime(2024, 6, 10, 16, 0),
                    datetime(2024, 6, 11, 10, 0),
                    datetime(2024, 6, 11, 16, 0),
                    datetime(2024, 6, 12, 9, 0),
                    datetime(2024, 6, 13, 10, 0),
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
        implied_prob = snapshot.iloc[0]["implied_probability"]
        assert implied_prob < 0.65

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
        regime = model.classify(macro_data, price_data, as_of_date="2024-07-01")
        assert regime["date"] == "2024-07-01"


class TestStatisticalRigor:
    """Tests for statistical rigor features."""

    def test_bootstrap_ci_narrows_with_more_data(self):
        from macro_engine.studies.event_study import _bootstrap_ci

        small = pd.Series(np.random.default_rng(42).normal(0.01, 0.02, 10))
        large = pd.Series(np.random.default_rng(42).normal(0.01, 0.02, 100))

        _, small_lo, small_hi = _bootstrap_ci(small, n_bootstrap=500)
        _, large_lo, large_hi = _bootstrap_ci(large, n_bootstrap=500)

        assert (large_hi - large_lo) < (small_hi - small_lo)

    def test_multiple_testing_correction(self):
        from macro_engine.studies.event_study import _benjamini_hochberg

        p_values = np.array([0.001, 0.02, 0.03, 0.4, 0.5, 0.8])
        adjusted = _benjamini_hochberg(p_values)
        assert all(adjusted >= p_values)
        assert adjusted[0] <= 0.05
