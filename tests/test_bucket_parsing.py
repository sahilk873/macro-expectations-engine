"""Tests for bucket range parsing from Kalshi market data."""

import pytest

from macro_engine.kalshi.client import KalshiClient


class TestBucketParsing:
    def test_count_range_buckets(self):
        market = {"count_range": [0.0, 10.0]}
        buckets = KalshiClient._parse_buckets(market)
        assert len(buckets) == 10
        assert buckets[0] == pytest.approx((0.0, 1.0))
        assert buckets[-1] == pytest.approx((9.0, 10.0))

    def test_native_value_options(self):
        market = {
            "native_value_options": [
                {"lower": 0.0, "upper": 1.0},
                {"lower": 1.0, "upper": 2.0},
                {"lower": 2.0, "upper": 3.0},
            ]
        }
        buckets = KalshiClient._parse_buckets(market)
        assert len(buckets) == 3
        assert buckets[0] == (0.0, 1.0)
        assert buckets[2] == (2.0, 3.0)

    def test_no_buckets(self):
        market = {}
        buckets = KalshiClient._parse_buckets(market)
        assert len(buckets) == 0

    def test_partial_bucket_info(self):
        market = {
            "native_value_options": [
                {"lower": 0.0},  # missing upper
                {"upper": 5.0},  # missing lower
            ]
        }
        buckets = KalshiClient._parse_buckets(market)
        assert len(buckets) == 0

    @staticmethod
    def test_low_hi_keys():
        market = {
            "native_value_options": [
                {"low": 0, "high": 10},
                {"low": 10, "high": 20},
            ]
        }
        buckets = KalshiClient._parse_buckets(market)
        assert len(buckets) == 2
        assert buckets[0] == (0.0, 10.0)
