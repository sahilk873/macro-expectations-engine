"""Tests for probability conversion from binary markets."""

import numpy as np
import pytest

from macro_engine.expectations.implied import (
    binary_to_probability,
    bucketed_to_distribution,
    calculate_implied_mean_from_prices,
    calculate_implied_variance,
    fit_normal_from_prices,
)


class TestBinaryToProbability:
    def test_mid_price_yes(self):
        p = binary_to_probability(0.60, 0.62, 0.38, 0.40)
        assert 0.60 <= p <= 0.62

    def test_no_arbitrage(self):
        p = binary_to_probability(0.50, 0.50, 0.50, 0.50)
        assert p == pytest.approx(0.50, abs=0.01)

    def test_spread_arbitrage_blend(self):
        # Large spread where yes_mid + no_mid != 1, should blend
        p = binary_to_probability(0.55, 0.65, 0.30, 0.40)
        assert 0.40 <= p <= 0.65

    def test_clipping(self):
        p = binary_to_probability(-0.1, 0.0, 1.0, 1.1)
        assert p == pytest.approx(0.001, abs=0.001)

    def test_nan_input(self):
        p = binary_to_probability(np.nan, 0.60, 0.40, 0.40)
        assert np.isnan(p)

    def test_extreme_probability(self):
        p = binary_to_probability(0.99, 0.999, 0.001, 0.01)
        assert 0.98 <= p <= 1.0


class TestBucketedToDistribution:
    def test_implied_mean_simple(self):
        prices = [10, 20, 30, 20, 10]
        buckets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        mean = calculate_implied_mean_from_prices(prices, buckets)
        assert mean == pytest.approx(2.5, abs=0.1)

    def test_implied_mean_skewed(self):
        prices = [50, 30, 15, 5]
        buckets = [(0, 1), (1, 2), (2, 3), (3, 4)]
        mean = calculate_implied_mean_from_prices(prices, buckets)
        assert mean < 1.6

    def test_implied_variance(self):
        prices = [10, 20, 40, 20, 10]
        buckets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        mean = calculate_implied_mean_from_prices(prices, buckets)
        var = calculate_implied_variance(prices, buckets, mean)
        assert var > 0

    def test_implied_variance_zero(self):
        prices = [100]
        buckets = [(2, 3)]
        mean = calculate_implied_mean_from_prices(prices, buckets)
        var = calculate_implied_variance(prices, buckets, mean)
        assert np.isnan(var) or var == 0.0

    def test_empty_inputs(self):
        assert np.isnan(calculate_implied_mean_from_prices([], []))
        assert np.isnan(calculate_implied_variance([], [], 0.0))

    def test_fit_normal(self):
        prices = [10, 20, 40, 20, 10]
        buckets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        mean, std = fit_normal_from_prices(prices, buckets)
        assert 2.0 <= mean <= 3.0
        assert std > 0

    def test_distribution_output(self):
        prices = [10, 20, 40, 20, 10]
        buckets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        x, pdf = bucketed_to_distribution(prices, buckets)
        assert len(x) == 1000
        assert len(pdf) == 1000
        assert np.all(pdf >= 0)

    def test_distribution_empty(self):
        x, pdf = bucketed_to_distribution([], [])
        assert len(x) == 0
        assert len(pdf) == 0
