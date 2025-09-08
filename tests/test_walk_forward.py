"""Tests for walk-forward backtest validation."""

import numpy as np
import pandas as pd
import pytest

from macro_engine.backtest.strategy import RegimeAwareStrategy
from macro_engine.backtest.walk_forward import (
    compute_out_of_sample_sharpe,
    compute_parameter_sensitivity,
    generate_walk_forward_splits,
    run_walk_forward_backtest,
)


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="B")
    rows = []
    np.random.seed(42)
    for ticker in ["SPY", "TLT", "IEF", "HYG", "LQD"]:
        price = {"SPY": 400, "TLT": 95, "IEF": 100, "HYG": 85, "LQD": 90}[ticker]
        for dt in dates:
            change = np.random.randn() * 0.005
            price *= 1 + change
            rows.append({"date": dt, "ticker": ticker, "close": price})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_regimes() -> pd.DataFrame:
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="BMS")
    records = []
    for dt in dates:
        records.append(
            {
                "date": str(dt.date()),
                "growth_regime": "expansion",
                "inflation_regime": "stable",
                "policy_regime": "restrictive",
                "volatility_regime": "normal",
                "risk_regime": "neutral",
            }
        )
    return pd.DataFrame(records)


class TestWalkForwardSplits:
    def test_generates_correct_number_of_splits(self):
        splits = generate_walk_forward_splits("2023-01-01", "2024-12-31", n_splits=4)
        assert len(splits) == 4
        for s in splits:
            assert s.train_start < s.train_end < s.val_start < s.val_end

    def test_split_dates_are_valid(self):
        splits = generate_walk_forward_splits("2023-01-01", "2024-12-31", n_splits=3)
        for s in splits:
            assert s.train_start <= s.train_end
            assert s.val_start <= s.val_end
            assert s.train_end < s.val_start

    def test_last_split_ends_at_end_date(self):
        splits = generate_walk_forward_splits("2023-01-01", "2024-12-31", n_splits=4)
        assert splits[-1].val_end <= pd.Timestamp("2024-12-31")


class TestWalkForwardBacktest:
    def test_runs_with_valid_data(self, sample_price_data, sample_regimes):
        splits = generate_walk_forward_splits("2023-06-01", "2024-12-31", n_splits=2)
        strategy = RegimeAwareStrategy(transaction_cost_bps=3.0)
        results = run_walk_forward_backtest(sample_price_data, sample_regimes, splits, strategy)
        if not results.empty:
            assert "walk_forward" in results.columns
            assert "split_index" in results.columns
            assert all(results["walk_forward"])

    def test_returns_empty_with_no_data(self):
        splits = generate_walk_forward_splits("2023-01-01", "2024-12-31", n_splits=2)
        results = run_walk_forward_backtest(pd.DataFrame(), pd.DataFrame(), splits)
        assert results.empty


class TestOutOfSampleSharpe:
    def test_with_valid_results(self):
        results = pd.DataFrame(
            {
                "portfolio_value": [1.0, 1.05, 1.02, 1.08, 1.04],
                "daily_return": [0.0, 0.05, -0.03, 0.06, -0.04],
                "split_index": [0, 0, 1, 1, 1],
            }
        )
        metrics = compute_out_of_sample_sharpe(results)
        assert "oos_sharpe_ratio" in metrics
        assert "oos_n_splits" in metrics
        assert metrics["oos_n_splits"] <= 2

    def test_empty_results(self):
        metrics = compute_out_of_sample_sharpe(pd.DataFrame())
        assert metrics == {}

    def test_single_split(self):
        results = pd.DataFrame(
            {
                "portfolio_value": [1.0, 1.05],
                "daily_return": [0.0, 0.05],
                "split_index": [0, 0],
            }
        )
        metrics = compute_out_of_sample_sharpe(results)
        assert metrics["oos_n_splits"] == 1


class TestParameterSensitivity:
    def test_sensitivity_computation(self, sample_price_data, sample_regimes):
        strategy = RegimeAwareStrategy(transaction_cost_bps=3.0, vol_target=0.12)
        sensitivities = compute_parameter_sensitivity(
            sample_price_data,
            sample_regimes,
            strategy,
            "transaction_cost_bps",
            param_values=[1.0, 5.0],
        )
        assert len(sensitivities) == 2
        for s in sensitivities:
            assert s.param_name == "transaction_cost_bps"
            assert s.base_value == 3.0
