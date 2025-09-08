"""Tests for prediction market microstructure analysis."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from macro_engine.microstructure.analysis import (
    MicrostructureMetrics,
    compute_all_microstructure_metrics,
    compute_market_depth_ratio,
    compute_price_discovery_ratio,
    compute_spread_metrics,
    compute_vwap_probability,
    detect_arbitrage,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    rows = []
    for ticker in ["CPIYOY_24JUN", "FEDTARGET_24JUN"]:
        for days_back in range(1, 20):
            ts = datetime(2024, 6, 12, 8, 30) - pd.Timedelta(days=days_back)
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": ts,
                    "yes_bid": 0.55,
                    "yes_ask": 0.57,
                    "no_bid": 0.43,
                    "no_ask": 0.45,
                    "last_price": 0.56,
                    "volume": 10000 + days_back * 100,
                    "open_interest": 50000,
                }
            )
    return pd.DataFrame(rows)


class TestVWAPProbability:
    def test_vwap_in_bounds(self, sample_prices):
        vwap, vol = compute_vwap_probability(sample_prices)
        assert 0 <= vwap <= 1
        assert vol > 0

    def test_vwap_by_ticker(self, sample_prices):
        vwap_1, _ = compute_vwap_probability(sample_prices, ticker="CPIYOY_24JUN")
        vwap_2, _ = compute_vwap_probability(sample_prices, ticker="FEDTARGET_24JUN")
        assert not np.isnan(vwap_1)
        assert not np.isnan(vwap_2)

    def test_empty_data(self):
        vwap, vol = compute_vwap_probability(pd.DataFrame())
        assert np.isnan(vwap)
        assert vol == 0.0


class TestSpreadMetrics:
    def test_spread_positive(self, sample_prices):
        metrics = compute_spread_metrics(sample_prices)
        assert metrics["mean_spread_bps"] > 0
        assert metrics["median_spread_bps"] > 0

    def test_spread_volatility_non_negative(self, sample_prices):
        metrics = compute_spread_metrics(sample_prices)
        assert metrics["spread_volatility"] >= 0

    def test_empty_data(self):
        metrics = compute_spread_metrics(pd.DataFrame())
        assert np.isnan(metrics["mean_spread_bps"])

    def test_wide_spread_market(self):
        data = pd.DataFrame(
            {
                "ticker": ["TEST"],
                "yes_bid": [0.10],
                "yes_ask": [0.90],
            }
        )
        metrics = compute_spread_metrics(data)
        assert metrics["mean_spread_bps"] > 100


class TestArbitrageDetection:
    def test_no_arbitrage_in_normal_data(self, sample_prices):
        arb = detect_arbitrage(sample_prices)
        assert len(arb) == 0

    def test_detects_overpriced_market(self):
        data = pd.DataFrame(
            {
                "ticker": ["TEST"],
                "timestamp": [datetime(2024, 1, 1)],
                "yes_bid": [0.60],
                "yes_ask": [0.62],
                "no_bid": [0.50],
                "no_ask": [0.52],
            }
        )
        arb = detect_arbitrage(data)
        assert len(arb) > 0
        assert arb[0]["type"] == "overpriced"

    def test_detects_underpriced_market(self):
        data = pd.DataFrame(
            {
                "ticker": ["TEST"],
                "timestamp": [datetime(2024, 1, 1)],
                "yes_bid": [0.30],
                "yes_ask": [0.32],
                "no_bid": [0.60],
                "no_ask": [0.62],
            }
        )
        arb = detect_arbitrage(data)
        assert len(arb) > 0

    def test_empty_data(self):
        arb = detect_arbitrage(pd.DataFrame())
        assert len(arb) == 0


class TestMarketDepth:
    def test_depth_ratio_positive(self, sample_prices):
        depth = compute_market_depth_ratio(sample_prices)
        assert depth > 0
        assert not np.isnan(depth)

    def test_empty_data(self):
        depth = compute_market_depth_ratio(pd.DataFrame())
        assert np.isnan(depth)


class TestPriceDiscovery:
    def test_discovery_ratio_bounded(self, sample_prices):
        pdr = compute_price_discovery_ratio(sample_prices)
        if not np.isnan(pdr):
            assert 0 <= pdr <= 1

    def test_empty_data(self):
        pdr = compute_price_discovery_ratio(pd.DataFrame())
        assert np.isnan(pdr)

    def test_single_obs(self):
        data = pd.DataFrame(
            {
                "ticker": ["TEST"],
                "timestamp": [datetime(2024, 1, 1)],
                "yes_bid": [0.5],
                "yes_ask": [0.6],
                "last_price": [0.55],
            }
        )
        pdr = compute_price_discovery_ratio(data)
        assert np.isnan(pdr)


class TestAllMetrics:
    def test_all_metrics(self, sample_prices):
        tickers = ["CPIYOY_24JUN", "FEDTARGET_24JUN"]
        results = compute_all_microstructure_metrics(sample_prices, tickers=tickers)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, MicrostructureMetrics)
            assert r.ticker in tickers
            assert r.n_observations > 0
            assert r.mean_spread_bps > 0
            assert r.vwap_volume > 0
            assert r.arbitrage_opportunities == 0

    def test_empty_prices(self):
        results = compute_all_microstructure_metrics(pd.DataFrame())
        assert len(results) == 0
