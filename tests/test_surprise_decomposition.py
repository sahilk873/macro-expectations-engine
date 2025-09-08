"""Tests for surprise decomposition module."""

import numpy as np
import pandas as pd
import pytest

from macro_engine.surprises.decomposition import (
    compute_entropy_based_confidence,
    decompose_all_surprises,
    decompose_surprise,
)


class TestSurpriseDecomposition:
    def test_decompose_returns_expected_keys(self):
        result = decompose_surprise(0.6, 0.8)
        expected_keys = [
            "total_surprise_logit",
            "level_component",
            "volatility_component",
            "skew_component",
            "uncertainty_revision",
            "implied_entropy",
            "realized_entropy",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_logit_sign_positive_surprise(self):
        result = decompose_surprise(0.5, 0.8)
        assert result["total_surprise_logit"] > 0

    def test_logit_sign_negative_surprise(self):
        result = decompose_surprise(0.8, 0.5)
        assert result["total_surprise_logit"] < 0

    def test_zero_surprise(self):
        result = decompose_surprise(0.5, 0.5)
        assert abs(result["total_surprise_logit"]) < 1e-6

    def test_entropy_higher_at_uncertainty(self):
        p_mid = decompose_surprise(0.5, 0.6)
        p_extreme = decompose_surprise(0.9, 0.95)
        assert p_mid["implied_entropy"] > p_extreme["implied_entropy"]

    def test_extreme_probabilities(self):
        result = decompose_surprise(0.01, 0.99)
        assert not np.isnan(result["total_surprise_logit"])
        assert abs(result["total_surprise_logit"]) > 0


class TestDecomposeAll:
    def test_batch_decomposition(self):
        surprises = pd.DataFrame(
            {
                "event_id": ["E1", "E2"],
                "event_type": ["CPI", "FOMC"],
                "snapshot_type": ["T-1_day", "T-1_day"],
                "expected_probability": [0.6, 0.7],
                "realized_value": [0.8, 0.5],
            }
        )
        result = decompose_all_surprises(surprises)
        assert len(result) == 2
        assert "total_surprise_logit" in result.columns

    def test_missing_data_skipped(self):
        surprises = pd.DataFrame(
            {
                "event_id": ["E1"],
                "event_type": ["CPI"],
                "snapshot_type": ["T-1_day"],
                "expected_probability": [np.nan],
                "realized_value": [0.8],
            }
        )
        result = decompose_all_surprises(surprises)
        assert result.empty


class TestEntropyConfidence:
    def test_extreme_prob_high_confidence(self):
        c = compute_entropy_based_confidence(0.99, n_observations=500)
        assert c > 0.5

    def test_mid_prob_low_confidence(self):
        c = compute_entropy_based_confidence(0.5, n_observations=500)
        assert c < 0.1

    def test_confidence_bounded(self):
        for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            c = compute_entropy_based_confidence(p, n_observations=100)
            assert 0 <= c <= 1

    def test_sample_size_adjustment(self):
        c_few = compute_entropy_based_confidence(0.3, n_observations=10)
        c_many = compute_entropy_based_confidence(0.3, n_observations=1000)
        assert c_many >= c_few
