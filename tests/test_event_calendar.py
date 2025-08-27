"""Tests for the event calendar."""

from datetime import datetime

import pandas as pd

from macro_engine.events.calendar import EventRecord, build_event_calendar


class TestBuildEventCalendar:
    def test_calendar_has_events(self):
        df = build_event_calendar()
        assert not df.empty
        assert len(df) > 100  # Should have many events across years

    def test_calendar_columns(self):
        df = build_event_calendar()
        expected = {
            "event_id",
            "event_type",
            "event_name",
            "release_datetime",
            "source",
            "ticker",
            "category",
        }
        assert expected.issubset(df.columns)

    def test_event_types_present(self):
        df = build_event_calendar()
        expected_types = {"CPI", "FOMC", "NFP", "UNEMPLOYMENT", "GDP", "PCE"}
        assert expected_types.issubset(set(df["event_type"]))

    def test_release_datetime_is_datetime(self):
        df = build_event_calendar()
        assert pd.api.types.is_datetime64_any_dtype(df["release_datetime"])

    def test_categories_assigned(self):
        df = build_event_calendar()
        assert df["category"].isin(["inflation", "employment", "growth", "policy", "other"]).all()

    def test_no_duplicate_event_ids(self):
        df = build_event_calendar()
        assert df["event_id"].is_unique

    def test_sorted_by_date(self):
        df = build_event_calendar()
        dates = df["release_datetime"]
        assert dates.is_monotonic_increasing


class TestEventRecord:
    def test_category_inference(self):
        rec = EventRecord(
            event_id="CPI_TEST",
            event_type="CPI",
            event_name="CPI Test",
            release_datetime=datetime(2024, 1, 1),
            source="BLS",
            ticker="CPIAUCSL",
        )
        assert rec.category == "inflation"

    def test_unknown_event_type_category(self):
        rec = EventRecord(
            event_id="TEST",
            event_type="UNKNOWN",
            event_name="Test",
            release_datetime=datetime(2024, 1, 1),
            source="TEST",
            ticker="TEST",
        )
        assert rec.category == "other"
