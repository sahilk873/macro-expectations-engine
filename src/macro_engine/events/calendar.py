from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class EventRecord:
    """A single macro event on the calendar."""

    event_id: str
    event_type: str
    event_name: str
    release_datetime: datetime
    source: str
    ticker: str  # e.g. CPIYOY, FEDTARGET
    description: str = ""
    previous_value: Optional[float] = None
    forecast_median: Optional[float] = None
    data_url: str = ""
    category: str = ""  # inflation, employment, growth, policy, other

    def __post_init__(self) -> None:
        if not self.category:
            self.category = _infer_category(self.event_type)


def _infer_category(event_type: str) -> str:
    mapping = {
        "CPI": "inflation",
        "PCE": "inflation",
        "NFP": "employment",
        "UNEMPLOYMENT": "employment",
        "FOMC": "policy",
        "GDP": "growth",
        "RECESSION": "growth",
    }
    return mapping.get(event_type, "other")


# ---------------------------------------------------------------------------
# Hard-coded schedule data for known recurring macro events
# ---------------------------------------------------------------------------

CPI_MONTHS = [
    "2020-01-14",
    "2020-02-13",
    "2020-03-11",
    "2020-04-10",
    "2020-05-13",
    "2020-06-10",
    "2020-07-14",
    "2020-08-12",
    "2020-09-11",
    "2020-10-14",
    "2020-11-13",
    "2020-12-10",
    "2021-01-13",
    "2021-02-10",
    "2021-03-10",
    "2021-04-13",
    "2021-05-12",
    "2021-06-10",
    "2021-07-13",
    "2021-08-11",
    "2021-09-14",
    "2021-10-13",
    "2021-11-10",
    "2021-12-10",
    "2022-01-12",
    "2022-02-10",
    "2022-03-10",
    "2022-04-12",
    "2022-05-11",
    "2022-06-10",
    "2022-07-13",
    "2022-08-10",
    "2022-09-13",
    "2022-10-13",
    "2022-11-10",
    "2022-12-13",
    "2023-01-12",
    "2023-02-14",
    "2023-03-14",
    "2023-04-12",
    "2023-05-10",
    "2023-06-13",
    "2023-07-12",
    "2023-08-10",
    "2023-09-13",
    "2023-10-12",
    "2023-11-14",
    "2023-12-12",
    "2024-01-11",
    "2024-02-13",
    "2024-03-12",
    "2024-04-10",
    "2024-05-15",
    "2024-06-12",
    "2024-07-11",
    "2024-08-14",
    "2024-09-11",
    "2024-10-10",
    "2024-11-13",
    "2024-12-11",
    "2025-01-15",
    "2025-02-12",
    "2025-03-12",
    "2025-04-10",
    "2025-05-13",
    "2025-06-11",
]

FOMC_DATES = [
    "2020-01-29",
    "2020-03-03",
    "2020-03-15",
    "2020-04-29",
    "2020-06-10",
    "2020-07-29",
    "2020-09-16",
    "2020-11-05",
    "2020-12-16",
    "2021-01-27",
    "2021-03-17",
    "2021-04-28",
    "2021-06-16",
    "2021-07-28",
    "2021-09-22",
    "2021-11-03",
    "2021-12-15",
    "2022-01-26",
    "2022-03-16",
    "2022-05-04",
    "2022-06-15",
    "2022-07-27",
    "2022-09-21",
    "2022-11-02",
    "2022-12-14",
    "2023-02-01",
    "2023-03-22",
    "2023-05-03",
    "2023-06-14",
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13",
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
]

NFP_DATES = [
    "2020-01-10",
    "2020-02-07",
    "2020-03-06",
    "2020-04-03",
    "2020-05-08",
    "2020-06-05",
    "2020-07-02",
    "2020-08-07",
    "2020-09-04",
    "2020-10-02",
    "2020-11-06",
    "2020-12-04",
    "2021-01-08",
    "2021-02-05",
    "2021-03-05",
    "2021-04-02",
    "2021-05-07",
    "2021-06-04",
    "2021-07-02",
    "2021-08-06",
    "2021-09-03",
    "2021-10-08",
    "2021-11-05",
    "2021-12-03",
    "2022-01-07",
    "2022-02-04",
    "2022-03-04",
    "2022-04-01",
    "2022-05-06",
    "2022-06-03",
    "2022-07-08",
    "2022-08-05",
    "2022-09-02",
    "2022-10-07",
    "2022-11-04",
    "2022-12-02",
    "2023-01-06",
    "2023-02-03",
    "2023-03-10",
    "2023-04-07",
    "2023-05-05",
    "2023-06-02",
    "2023-07-07",
    "2023-08-04",
    "2023-09-01",
    "2023-10-06",
    "2023-11-03",
    "2023-12-08",
    "2024-01-05",
    "2024-02-02",
    "2024-03-08",
    "2024-04-05",
    "2024-05-03",
    "2024-06-07",
    "2024-07-05",
    "2024-08-02",
    "2024-09-06",
    "2024-10-04",
    "2024-11-01",
    "2024-12-06",
    "2025-01-10",
    "2025-02-07",
    "2025-03-07",
    "2025-04-04",
    "2025-05-02",
    "2025-06-06",
]

GDP_DATES = [
    "2020-01-30",
    "2020-02-27",
    "2020-03-26",
    "2020-04-29",
    "2020-05-28",
    "2020-06-25",
    "2020-07-30",
    "2020-08-27",
    "2020-09-30",
    "2020-10-29",
    "2020-11-25",
    "2020-12-22",
    "2021-01-28",
    "2021-02-25",
    "2021-03-25",
    "2021-04-29",
    "2021-05-27",
    "2021-06-24",
    "2021-07-29",
    "2021-08-26",
    "2021-09-30",
    "2021-10-28",
    "2021-11-24",
    "2021-12-22",
    "2022-01-27",
    "2022-02-24",
    "2022-03-30",
    "2022-04-28",
    "2022-05-26",
    "2022-06-29",
    "2022-07-28",
    "2022-08-25",
    "2022-09-29",
    "2022-10-27",
    "2022-11-30",
    "2022-12-22",
    "2023-01-26",
    "2023-02-23",
    "2023-03-30",
    "2023-04-27",
    "2023-05-25",
    "2023-06-29",
    "2023-07-27",
    "2023-08-30",
    "2023-09-28",
    "2023-10-26",
    "2023-11-29",
    "2023-12-21",
    "2024-01-25",
    "2024-02-28",
    "2024-03-28",
    "2024-04-25",
    "2024-05-30",
    "2024-06-27",
    "2024-07-25",
    "2024-08-29",
    "2024-09-26",
    "2024-10-30",
    "2024-11-27",
    "2024-12-19",
    "2025-01-30",
    "2025-02-27",
    "2025-03-27",
    "2025-04-30",
    "2025-05-29",
    "2025-06-26",
]

UNEMPLOYMENT_DATES = NFP_DATES  # Released same day as NFP

PCE_DATES = [
    "2020-01-31",
    "2020-02-28",
    "2020-03-27",
    "2020-04-30",
    "2020-05-29",
    "2020-06-26",
    "2020-07-31",
    "2020-08-28",
    "2020-09-30",
    "2020-10-30",
    "2020-11-25",
    "2020-12-23",
    "2021-01-29",
    "2021-02-26",
    "2021-03-26",
    "2021-04-30",
    "2021-05-28",
    "2021-06-25",
    "2021-07-30",
    "2021-08-27",
    "2021-09-30",
    "2021-10-29",
    "2021-11-24",
    "2021-12-23",
    "2022-01-28",
    "2022-02-25",
    "2022-03-31",
    "2022-04-29",
    "2022-05-27",
    "2022-06-30",
    "2022-07-29",
    "2022-08-26",
    "2022-09-30",
    "2022-10-28",
    "2022-11-30",
    "2022-12-23",
    "2023-01-27",
    "2023-02-24",
    "2023-03-31",
    "2023-04-28",
    "2023-05-26",
    "2023-06-30",
    "2023-07-28",
    "2023-08-31",
    "2023-09-29",
    "2023-10-27",
    "2023-11-30",
    "2023-12-22",
    "2024-01-26",
    "2024-02-29",
    "2024-03-29",
    "2024-04-26",
    "2024-05-31",
    "2024-06-28",
    "2024-07-26",
    "2024-08-30",
    "2024-09-27",
    "2024-10-31",
    "2024-11-27",
    "2024-12-20",
    "2025-01-31",
    "2025-02-28",
    "2025-03-28",
    "2025-04-30",
    "2025-05-30",
    "2025-06-27",
]


# ---------------------------------------------------------------------------
# Build the calendar
# ---------------------------------------------------------------------------


def _make_events(
    dates: list[str],
    event_type: str,
    ticker: str,
    source: str,
    base_name: str,
    release_hour: int = 8,
    release_minute: int = 30,
) -> list[EventRecord]:
    """Create EventRecords from a list of date strings."""
    records: list[EventRecord] = []
    for i, ds in enumerate(dates):
        dt = datetime.strptime(ds, "%Y-%m-%d").replace(hour=release_hour, minute=release_minute)
        records.append(
            EventRecord(
                event_id=f"{event_type}_{dt.strftime('%Y%m%d')}",
                event_type=event_type,
                event_name=f"{base_name} {dt.strftime('%b %Y')}",
                release_datetime=dt,
                source=source,
                ticker=ticker,
            )
        )
    return records


def build_event_calendar(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    """Build a comprehensive macro event calendar.

    Returns a DataFrame with columns:
        event_id, event_type, event_name, release_datetime,
        source, ticker, category, previous_value, forecast_median
    """
    records: list[EventRecord] = []

    # CPI
    records.extend(_make_events(CPI_MONTHS, "CPI", "CPIYOY", "BLS", "CPI YoY"))
    # FOMC
    records.extend(
        _make_events(
            FOMC_DATES,
            "FOMC",
            "FEDTARGET",
            "FOMC",
            "FOMC Decision",
            release_hour=14,
            release_minute=0,
        )
    )
    # NFP
    records.extend(_make_events(NFP_DATES, "NFP", "NFP", "BLS", "Nonfarm Payrolls"))
    # Unemployment
    records.extend(
        _make_events(UNEMPLOYMENT_DATES, "UNEMPLOYMENT", "UNRATE", "BLS", "Unemployment Rate")
    )
    # GDP
    for i, ds in enumerate(GDP_DATES):
        estimate = ["Advance", "Second", "Third", "Advance", "Second", "Third"][i % 6]
        dt = datetime.strptime(ds, "%Y-%m-%d").replace(hour=8, minute=30)
        records.append(
            EventRecord(
                event_id=f"GDP_{dt.strftime('%Y%m%d')}",
                event_type="GDP",
                event_name=f"GDP {estimate} {dt.strftime('%b %Y')}",
                release_datetime=dt,
                source="BEA",
                ticker="GDP",
            )
        )
    # PCE
    records.extend(_make_events(PCE_DATES, "PCE", "PCEPI", "BEA", "PCE Price Index"))
    # Recession indicator (tracked but not scheduled)
    records.append(
        EventRecord(
            event_id="RECESSION_BASE",
            event_type="RECESSION",
            event_name="Recession Probability Assessment",
            release_datetime=datetime(2020, 1, 1, 8, 30),
            source="NBER",
            ticker="RECESSION",
            description="Recurring NBER recession assessments",
        )
    )

    df = pd.DataFrame([r.__dict__ for r in records])
    df = df.sort_values("release_datetime").reset_index(drop=True)
    return df


def save_event_calendar(df: pd.DataFrame, config: Optional[EngineConfig] = None) -> Path:
    """Save event calendar to parquet."""
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.event_calendar_file
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_event_calendar(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    """Load event calendar from parquet."""
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.event_calendar_file
    if not path.exists():
        warnings.warn(f"Event calendar not found at {path}. Run build_event_calendar first.")
        return pd.DataFrame()
    return pd.read_parquet(path)
