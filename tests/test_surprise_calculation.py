"""Tests for surprise calculation and labeling."""

import numpy as np
import pytest

from macro_engine.surprises.calculator import (
    compute_percent_surprise,
    compute_raw_surprise,
    compute_standardized_surprise,
    label_surprise,
)


class TestSurpriseCalculation:
    def test_raw_surprise_positive(self):
        assert compute_raw_surprise(5.0, 3.0) == 2.0

    def test_raw_surprise_negative(self):
        assert compute_raw_surprise(2.0, 5.0) == -3.0

    def test_raw_surprise_zero(self):
        assert compute_raw_surprise(3.0, 3.0) == 0.0

    def test_standardized_surprise(self):
        s = compute_standardized_surprise(5.0, 3.0, 1.0)
        assert s == 2.0

    def test_standardized_surprise_zero_std(self):
        s = compute_standardized_surprise(5.0, 3.0, 0.0)
        assert np.isnan(s)

    def test_standardized_surprise_nan_std(self):
        s = compute_standardized_surprise(5.0, 3.0, np.nan)
        assert np.isnan(s)

    def test_percent_surprise(self):
        s = compute_percent_surprise(6.0, 5.0)
        assert s == pytest.approx(0.2)

    def test_percent_surprise_zero_expected(self):
        s = compute_percent_surprise(5.0, 0.0)
        assert np.isnan(s)


class TestSurpriseLabeling:
    def test_cpi_hot(self):
        labels = label_surprise("CPI", 0.5, 2.0)
        assert labels["direction"] == "above_expectations"
        assert labels["qualitative"] == "inflation_hot"
        assert labels["risk_label"] == "risk_off"

    def test_cpi_cool(self):
        labels = label_surprise("CPI", -0.5, -2.0)
        assert labels["direction"] == "below_expectations"
        assert labels["qualitative"] == "inflation_cool"
        assert labels["risk_label"] == "risk_on"

    def test_nfp_strong(self):
        labels = label_surprise("NFP", 100000, 2.0)
        assert labels["qualitative"] == "labor_strong"
        assert labels["risk_label"] == "risk_on"

    def test_unemployment_high(self):
        labels = label_surprise("UNEMPLOYMENT", 0.5, 2.0)
        assert labels["qualitative"] == "labor_weak"
        assert labels["risk_label"] == "risk_off"

    def test_gdp_strong(self):
        labels = label_surprise("GDP", 1.0, 2.0)
        assert labels["qualitative"] == "growth_strong"

    def test_gdp_weak(self):
        labels = label_surprise("GDP", -1.0, -2.0)
        assert labels["qualitative"] == "growth_weak"

    def test_fomc_hawkish(self):
        labels = label_surprise("FOMC", 0.25, 2.0)
        assert labels["qualitative"] == "policy_hawkish"
        assert labels["risk_label"] == "risk_off"

    def test_fomc_dovish(self):
        labels = label_surprise("FOMC", -0.25, -2.0)
        assert labels["qualitative"] == "policy_dovish"
        assert labels["risk_label"] == "risk_on"

    def test_neutral_small_surprise(self):
        labels = label_surprise("CPI", 0.001, 0.1)
        assert labels["direction"] == "neutral"
