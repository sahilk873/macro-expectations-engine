"""Tests for market-to-event mapping with confidence scoring."""

from datetime import datetime

import pandas as pd

from macro_engine.mapping.mapper import (
    MarketEventMapping,
    _compute_confidence,
    _extract_date_from_text,
    _match_keywords,
    build_market_mapping,
    mappings_to_dataframe,
)


class TestMappingHelpers:
    def test_keyword_match_cpi(self):
        assert _match_keywords("CPI YoY Report", "CPI")
        assert _match_keywords("Consumer Price Index", "CPI")
        assert not _match_keywords("GDP Report", "CPI")

    def test_keyword_match_fomc(self):
        assert _match_keywords("FOMC Rate Decision", "FOMC")
        assert _match_keywords("Fed Funds Target", "FOMC")

    def test_keyword_match_nfp(self):
        assert _match_keywords("Nonfarm Payrolls Report", "NFP")
        assert _match_keywords("Jobs Report", "NFP")

    def test_date_extraction_confidence(self):
        dt = datetime(2024, 6, 12, 8, 30)
        conf = _extract_date_from_text("CPI Report June 2024", dt)
        assert conf > 0.5

    def test_date_extraction_no_match(self):
        dt = datetime(2024, 6, 12, 8, 30)
        conf = _extract_date_from_text("Some Random Market", dt)
        assert conf < 0.3

    def test_confidence_computation(self):
        conf = _compute_confidence(
            "CPIYOY24JUN", "CPI YoY June 2024", "CPI", datetime(2024, 6, 12, 8, 30)
        )
        assert 0.0 <= conf <= 1.0
        assert conf > 0.3


class TestBuildMarketMapping:
    def test_basic_mapping_creation(self):
        event_cal = pd.DataFrame(
            {
                "event_id": ["CPI_20240612"],
                "event_type": ["CPI"],
                "release_datetime": [datetime(2024, 6, 12, 8, 30)],
                "ticker": ["CPIAUCSL"],
                "event_name": ["CPI Jun 2024"],
                "source": ["BLS"],
                "category": ["inflation"],
            }
        )

        kalshi_markets = pd.DataFrame(
            {
                "ticker": ["CPIYOY_24JUN"],
                "title": ["CPI YoY June 2024"],
                "close_time": [datetime(2024, 6, 11, 12, 0)],
                "event_ticker": ["CPI_24JUN"],
                "market_type": ["binary"],
                "status": ["closed"],
            }
        )

        mappings = build_market_mapping(event_cal, kalshi_markets)
        assert len(mappings) > 0

    def test_manual_override(self):
        event_cal = pd.DataFrame(
            {
                "event_id": ["CPI_20240612", "PCE_20240612"],
                "event_type": ["CPI", "PCE"],
                "release_datetime": [datetime(2024, 6, 12, 8, 30), datetime(2024, 6, 12, 8, 30)],
                "ticker": ["CPIAUCSL", "PCEPI"],
                "event_name": ["CPI Jun 2024", "PCE Jun 2024"],
                "source": ["BLS", "BEA"],
                "category": ["inflation", "inflation"],
            }
        )

        kalshi_markets = pd.DataFrame(
            {
                "ticker": ["TEST_MARKET"],
                "title": ["Some Market"],
                "close_time": [datetime(2024, 6, 11, 12, 0)],
                "event_ticker": ["TEST"],
                "market_type": ["binary"],
                "status": ["open"],
            }
        )

        overrides = pd.DataFrame(
            {
                "market_ticker": ["TEST_MARKET"],
                "event_id": ["CPI_20240612"],
            }
        )

        mappings = build_market_mapping(event_cal, kalshi_markets, manual_overrides=overrides)
        manual_mappings = [m for m in mappings if m.manual_override]
        assert len(manual_mappings) == 1
        assert manual_mappings[0].confidence_score == 1.0
        assert manual_mappings[0].mapping_method == "manual"

    def test_empty_markets_returns_empty(self):
        event_cal = pd.DataFrame(
            {
                "event_id": ["CPI_20240612"],
                "event_type": ["CPI"],
                "release_datetime": [datetime(2024, 6, 12, 8, 30)],
                "ticker": ["CPIAUCSL"],
                "event_name": ["CPI Jun 2024"],
                "source": ["BLS"],
                "category": ["inflation"],
            }
        )
        kalshi_markets = pd.DataFrame(columns=["ticker", "title"])
        mappings = build_market_mapping(event_cal, kalshi_markets)
        assert len(mappings) == 0

    def test_mapping_dataframe_conversion(self):
        mappings = [
            MarketEventMapping(
                market_ticker="CPIYOY",
                market_title="CPI YoY",
                event_id="CPI_2024",
                event_type="CPI",
                confidence_score=0.85,
                mapping_method="fuzzy",
            )
        ]
        df = mappings_to_dataframe(mappings)
        assert len(df) == 1
        assert df.iloc[0]["confidence_score"] == 0.85
