"""Tests for the multi-factor surprise attribution model."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from macro_engine.factors.model import (
    SurpriseFactorModel,
    compute_cumulative_abnormal_returns,
    compute_factor_attribution,
)


@pytest.fixture
def sample_event_studies() -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for i in range(20):
        month = 1 + i // 2
        day = 1 + (i * 7) % 25
        if month > 12:
            break
        dt = datetime(2024, month, day, 8, 30)
        rows.append(
            {
                "event_id": f"CPI_{i}",
                "event_type": "CPI",
                "ticker": "SPY",
                "event_time": dt,
                "return_1D": np.random.randn() * 0.005,
                "return_3D": np.random.randn() * 0.008,
                "standardized_surprise": np.random.randn() * 1.5,
                "raw_surprise": np.random.randn() * 0.3,
            }
        )
    for i in range(15):
        month = 1 + i // 2
        day = 5 + (i * 5) % 20
        if month > 12:
            break
        dt = datetime(2024, month, day, 14, 0)
        rows.append(
            {
                "event_id": f"FOMC_{i}",
                "event_type": "FOMC",
                "ticker": "TLT",
                "event_time": dt,
                "return_1D": np.random.randn() * 0.004,
                "return_3D": np.random.randn() * 0.006,
                "standardized_surprise": np.random.randn() * 1.2,
                "raw_surprise": np.random.randn() * 0.2,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="B")
    rows = []
    np.random.seed(42)
    for ticker in ["SPY", "TLT", "IEF", "HYG", "LQD", "UUP", "GLD", "USO"]:
        price = {
            "SPY": 500,
            "TLT": 95,
            "IEF": 100,
            "HYG": 85,
            "LQD": 90,
            "UUP": 25,
            "GLD": 180,
            "USO": 60,
        }[ticker]
        for dt in dates:
            change = np.random.randn() * 0.005
            price *= 1 + change
            rows.append({"date": dt, "ticker": ticker, "close": price})
    return pd.DataFrame(rows)


class TestSurpriseFactorModel:
    def test_estimate_returns_dataframe(self, sample_event_studies, sample_price_data):
        model = SurpriseFactorModel(min_events=2)
        result = model.estimate(sample_event_studies, sample_price_data)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        expected_cols = ["surprise_type", "ticker", "total_effect", "alpha", "alpha_tstat"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_estimate_includes_bh_correction(self, sample_event_studies, sample_price_data):
        model = SurpriseFactorModel(min_events=2)
        result = model.estimate(sample_event_studies, sample_price_data)
        assert "bh_adjusted_p" in result.columns
        assert all(result["bh_adjusted_p"] >= 0)
        assert all(result["bh_adjusted_p"] <= 1)

    def test_min_events_filter(self, sample_event_studies, sample_price_data):
        model = SurpriseFactorModel(min_events=100)
        result = model.estimate(sample_event_studies, sample_price_data)
        assert result.empty

    def test_r_squared_in_bounds(self, sample_event_studies, sample_price_data):
        model = SurpriseFactorModel(min_events=2)
        result = model.estimate(sample_event_studies, sample_price_data)
        assert all((result["r_squared"] >= 0) | result["r_squared"].isna())
        assert all((result["r_squared"] <= 1) | result["r_squared"].isna())

    def test_factor_cols_identifies_factors(self):
        data = pd.DataFrame({"factor_market": [1], "factor_rates": [2], "return_1D": [0]})
        cols = SurpriseFactorModel._factor_cols(data)
        assert "factor_market" in cols
        assert "factor_rates" in cols
        assert "return_1D" not in cols


class TestCumulativeAbnormalReturns:
    def test_car_computation(self, sample_event_studies):
        car = compute_cumulative_abnormal_returns(sample_event_studies, group_by=["event_type"])
        assert not car.empty
        assert "car_mean" in car.columns
        assert "car_tstat" in car.columns
        assert "car_ci_lower" in car.columns
        assert "car_ci_upper" in car.columns

    def test_car_initialized(self, sample_event_studies):
        car = compute_cumulative_abnormal_returns(sample_event_studies, group_by=["event_type"])
        for _, row in car.iterrows():
            assert row["car_ci_lower"] <= row["car_ci_upper"]

    def test_empty_input(self):
        result = compute_cumulative_abnormal_returns(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_single_return_col(self):
        df = pd.DataFrame({"event_type": ["CPI"], "return_1D": [0.01]})
        car = compute_cumulative_abnormal_returns(df)
        assert car.empty


class TestComputeFactorAttribution:
    def test_high_level_function(self, sample_event_studies, sample_price_data):
        result = compute_factor_attribution(sample_event_studies, sample_price_data, min_events=2)
        assert isinstance(result, pd.DataFrame)

    def test_benjamini_hochberg_correction(self):
        from macro_engine.factors.model import SurpriseFactorModel

        p_values = np.array([0.001, 0.01, 0.04, 0.2, 0.5, 0.9])
        adjusted = SurpriseFactorModel._benjamini_hochberg(p_values)
        assert all(adjusted >= p_values)
        assert adjusted[0] <= 0.05
