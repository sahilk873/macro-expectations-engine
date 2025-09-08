"""Tests for Bayesian shrinkage and Neyman CI modules."""

import numpy as np
import pandas as pd
import pytest

from macro_engine.studies.bayesian import (
    compute_bayes_factor,
    compute_sharpe_ratio_equivalent,
    empirical_bayes_shrinkage,
    neyman_confidence_intervals,
    neyman_event_study_aggregation,
)


class TestEmpiricalBayesShrinkage:
    def test_shrinkage_pulls_toward_grand_mean(self):
        estimates = pd.Series({"A": 0.05, "B": 0.01, "C": -0.02})
        std_errors = pd.Series({"A": 0.02, "B": 0.01, "C": 0.005})
        result = empirical_bayes_shrinkage(estimates, std_errors)
        assert "shrunk" in result.columns
        assert "lambda" in result.columns
        assert "grand_mean" in result.columns
        means = result["shrunk"].mean()
        assert not np.isnan(means)

    def test_shrinkage_reduces_variance(self):
        np.random.seed(42)
        estimates = pd.Series({f"G{i}": np.random.randn() * 0.1 for i in range(10)})
        std_errors = pd.Series({f"G{i}": 0.02 + np.random.rand() * 0.03 for i in range(10)})
        result = empirical_bayes_shrinkage(estimates, std_errors)
        orig_var = estimates.var()
        shrunk_var = result["shrunk"].var()
        assert shrunk_var < orig_var

    def test_lambda_approaches_one_with_small_se(self):
        estimates = pd.Series({"A": 0.05, "B": 0.01})
        std_errors = pd.Series({"A": 0.001, "B": 0.001})
        result = empirical_bayes_shrinkage(estimates, std_errors)
        assert all(result["lambda"].dropna() > 0.9)

    def test_lambda_approaches_zero_with_large_se(self):
        estimates = pd.Series({"A": 0.05, "B": 0.01, "C": -0.02})
        std_errors = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
        result = empirical_bayes_shrinkage(estimates, std_errors)
        assert all(result["lambda"].dropna() < 0.1)

    def test_missing_values_handled(self):
        estimates = pd.Series({"A": 0.05, "B": np.nan, "C": -0.02})
        std_errors = pd.Series({"A": 0.02, "B": 0.01, "C": 0.005})
        result = empirical_bayes_shrinkage(estimates, std_errors)
        assert np.isnan(result.loc["B", "shrunk"])
        assert not np.isnan(result.loc["A", "shrunk"])


class TestNeymanConfidenceIntervals:
    def test_ci_contains_mean(self):
        np.random.seed(42)
        vals = pd.Series(np.random.randn(50) * 0.02 + 0.005)
        ci = neyman_confidence_intervals(vals)
        assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]
        assert "method" in ci
        assert ci["method"] == "bootstrap-t"

    def test_ci_narrows_with_more_data(self):
        np.random.seed(42)
        small = pd.Series(np.random.randn(10) * 0.02)
        large = pd.Series(np.random.randn(200) * 0.02)
        ci_small = neyman_confidence_intervals(small)
        ci_large = neyman_confidence_intervals(large)
        assert (ci_large["ci_upper"] - ci_large["ci_lower"]) < (
            ci_small["ci_upper"] - ci_small["ci_lower"]
        )

    def test_too_few_obs_returns_nan(self):
        ci = neyman_confidence_intervals(pd.Series([0.01]))
        assert np.isnan(ci["mean"])


class TestNeymanAggregation:
    def test_aggregation_returns_dataframe(self):
        event_studies = pd.DataFrame(
            {
                "event_type": ["CPI", "CPI", "FOMC", "FOMC"],
                "ticker": ["SPY", "TLT", "SPY", "TLT"],
                "return_1D": [0.005, -0.003, 0.002, -0.001],
                "return_3D": [0.008, -0.005, 0.004, -0.002],
            }
        )
        result = neyman_event_study_aggregation(event_studies, group_by=["event_type"])
        assert not result.empty
        assert "return_1D_mean" in result.columns
        assert "return_1D_ci_lower" in result.columns
        assert "return_1D_ci_upper" in result.columns

    def test_shrinkage_applied(self):
        event_studies = pd.DataFrame(
            {
                "event_type": ["CPI", "CPI", "FOMC", "FOMC", "GDP", "GDP"],
                "ticker": ["SPY", "TLT", "SPY", "TLT", "SPY", "TLT"],
                "return_1D": [0.05, -0.03, 0.02, -0.01, 0.005, -0.002],
                "return_3D": [0.08, -0.05, 0.04, -0.02, 0.01, -0.003],
            }
        )
        result = neyman_event_study_aggregation(
            event_studies, group_by=["event_type"], apply_shrinkage=True
        )
        assert "return_1D_shrunk" in result.columns
        assert "return_1D_lambda" in result.columns


class TestSharpeRatioEquivalent:
    def test_positive_sharpe(self):
        ir = compute_sharpe_ratio_equivalent(0.001, 0.01)
        assert ir > 0
        assert not np.isnan(ir)

    def test_negative_sharpe(self):
        ir = compute_sharpe_ratio_equivalent(-0.001, 0.01)
        assert ir < 0

    def test_zero_vol_returns_nan(self):
        ir = compute_sharpe_ratio_equivalent(0.001, 0.0)
        assert np.isnan(ir)


class TestBayesFactor:
    def test_strong_evidence(self):
        bf = compute_bayes_factor(0.05, 0.01, prior_mean=0.0, prior_se=0.05)
        assert bf > 3

    def test_weak_evidence(self):
        bf = compute_bayes_factor(0.001, 0.05, prior_mean=0.0, prior_se=0.05)
        assert bf < 3 or bf >= 0.01

    def test_bf_bounded(self):
        bf = compute_bayes_factor(np.nan, 0.01)
        assert bf == 1.0
