"""Tests for confounding event detection and control."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from macro_engine.studies.confounding import (
    compute_confounding_robustness,
    detect_confounding_events,
    flag_confounded_surprises,
    residualize_event_returns,
)


@pytest.fixture
def event_calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [
                "CPI_20240612",
                "FOMC_20240612",
                "NFP_20240607",
                "GDP_20240627",
            ],
            "event_type": ["CPI", "FOMC", "NFP", "GDP"],
            "release_datetime": [
                datetime(2024, 6, 12, 8, 30),
                datetime(2024, 6, 12, 14, 0),
                datetime(2024, 6, 7, 8, 30),
                datetime(2024, 6, 27, 8, 30),
            ],
        }
    )


@pytest.fixture
def sample_surprises() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["CPI_20240612", "FOMC_20240612", "NFP_20240607"],
            "event_type": ["CPI", "FOMC", "NFP"],
            "raw_surprise": [0.5, 0.25, -0.1],
        }
    )


@pytest.fixture
def sample_event_studies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["CPI_20240612", "FOMC_20240612", "NFP_20240607"],
            "event_type": ["CPI", "FOMC", "NFP"],
            "return_1D": [0.005, -0.003, 0.002],
            "confounded": [True, True, False],
        }
    )


class TestDetectConfounding:
    def test_detects_same_day_events(self, event_calendar):
        result = detect_confounding_events(event_calendar, max_gap_hours=24)
        cpi_row = result[result["event_id"] == "CPI_20240612"]
        fomc_row = result[result["event_id"] == "FOMC_20240612"]
        nfp_row = result[result["event_id"] == "NFP_20240607"]

        assert cpi_row["confounded"].iloc[0]
        assert fomc_row["confounded"].iloc[0]
        assert not nfp_row["confounded"].iloc[0]
        assert cpi_row["confounding_group"].iloc[0] >= 0
        assert fomc_row["confounding_group"].iloc[0] >= 0

    def test_same_confounding_group(self, event_calendar):
        result = detect_confounding_events(event_calendar, max_gap_hours=24)
        cpi_group = result[result["event_id"] == "CPI_20240612"]["confounding_group"].iloc[0]
        fomc_group = result[result["event_id"] == "FOMC_20240612"]["confounding_group"].iloc[0]
        assert cpi_group == fomc_group


class TestFlagConfounded:
    def test_merges_confounding_flags(self, sample_surprises, event_calendar):
        confounded = detect_confounding_events(event_calendar)
        flagged = flag_confounded_surprises(sample_surprises, confounded)
        assert "confounded" in flagged.columns
        assert "confounding_group" in flagged.columns
        cpi_row = flagged[flagged["event_id"] == "CPI_20240612"]
        assert cpi_row["confounded"].iloc[0]


class TestConfoundingRobustness:
    def test_comparison_with_limited_data(self, sample_event_studies):
        result = compute_confounding_robustness(sample_event_studies, "return_1D")
        assert isinstance(result, dict)

    def test_empty_data(self):
        result = compute_confounding_robustness(pd.DataFrame())
        assert result == {}

    def test_single_group_returns_empty(self):
        df = pd.DataFrame({"confounded": [True, True], "return_1D": [0.01, -0.01]})
        result = compute_confounding_robustness(df, "return_1D")
        assert isinstance(result, dict)


class TestResidualize:
    def test_residualization_returns_dataframe(self, sample_event_studies, event_calendar):
        confounded = detect_confounding_events(event_calendar)
        result = residualize_event_returns(sample_event_studies, confounded)
        assert "residualized_return" in result.columns

    def test_residualized_values(self, sample_event_studies, event_calendar):
        confounded = detect_confounding_events(event_calendar)
        result = residualize_event_returns(sample_event_studies, confounded)
        assert len(result) == len(sample_event_studies)
        assert "n_confounds" in result.columns
