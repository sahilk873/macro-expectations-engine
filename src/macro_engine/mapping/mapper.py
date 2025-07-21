from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings


@dataclass
class MarketEventMapping:
    """Mapping between a Kalshi market and a macro event."""

    market_ticker: str
    market_title: str
    event_id: str
    event_type: str
    confidence_score: float  # 0.0 to 1.0
    mapping_method: str  # "exact", "fuzzy", "regex", "manual"
    notes: str = ""
    manual_override: bool = False


# ---------------------------------------------------------------------------
# Keyword-driven matching rules
# ---------------------------------------------------------------------------

_EVENT_KEYWORDS: dict[str, list[str]] = {
    "CPI": ["cpi", "consumer price", "inflation", "cpiyoy", "cpi_index"],
    "FOMC": [
        "fomc",
        "fed rate",
        "fed funds",
        "interest rate",
        "fed target",
        "federal reserve",
        "fedtarget",
    ],
    "NFP": ["nonfarm", "non-farm", "payroll", "nfp", "jobs", "employment", "nonfarm_payrolls"],
    "UNEMPLOYMENT": ["unemployment", "unrate", "jobless", "unemployment rate"],
    "GDP": ["gdp", "gross domestic product"],
    "PCE": ["pce", "core pce", "personal consumption", "pcepi"],
    "RECESSION": ["recession", "recession probability"],
}

_MONTH_NAMES = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]


def _match_keywords(text: str, event_type: str) -> bool:
    """Check if text contains keywords for a given event type."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in _EVENT_KEYWORDS.get(event_type, []))


def _extract_date_from_text(text: str, event_datetime: datetime) -> float:
    """Estimate confidence that text refers to a specific event date."""

    month_str = event_datetime.strftime("%b").lower()
    year_str = str(event_datetime.year)
    text_lower = text.lower()

    # Check year
    has_year = year_str in text_lower or str(event_datetime.year % 100) in text_lower
    # Check month
    has_month = month_str in text_lower or _month_number(month_str) in text_lower

    if has_year and has_month:
        return 0.8
    if has_year:
        return 0.5
    if has_month:
        return 0.3
    return 0.1


def _month_number(month_abbr: str) -> str:
    mapping = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    return mapping.get(month_abbr, "")


def _parse_ticker_date(ticker: str) -> Optional[str]:
    """Try to extract a date like '23jan24' or '202401' from a ticker."""
    patterns = [
        r"(\d{2})(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d{2,4})",
        r"(\d{4})(\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, ticker.lower())
        if m:
            return m.group(0)
    return None


def _compute_confidence(
    market_ticker: str, market_title: str, event_type: str, event_datetime: datetime
) -> float:
    """Compute a confidence score for a market-event pair."""
    scores: list[float] = []

    # 1. Keyword match in title
    if _match_keywords(market_title, event_type):
        scores.append(0.6)
    elif _match_keywords(market_ticker, event_type):
        scores.append(0.4)
    else:
        scores.append(0.0)

    # 2. Date alignment
    date_conf = _extract_date_from_text(market_title, event_datetime)
    scores.append(date_conf)

    # 3. Ticker date parsing
    ticker_date = _parse_ticker_date(market_ticker)
    if ticker_date:
        scores.append(0.3)
    else:
        scores.append(0.1)

    # 4. Event type in ticker
    et_lower = event_type.lower()
    if et_lower in market_ticker.lower():
        scores.append(0.4)
    elif et_lower[:3] in market_ticker.lower():
        scores.append(0.2)

    if not scores:
        return 0.0

    # Weighted average
    weights = [0.35, 0.30, 0.15, 0.20]
    weighted = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    return float(np.clip(weighted, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main mapping builder
# ---------------------------------------------------------------------------


def build_market_mapping(
    event_calendar: pd.DataFrame,
    kalshi_markets: pd.DataFrame,
    manual_overrides: Optional[pd.DataFrame] = None,
    min_confidence: float = 0.3,
) -> list[MarketEventMapping]:
    """Build mappings between Kalshi markets and macro events.

    Returns a list of MarketEventMapping with computed confidence scores.
    Mappings below `min_confidence` are still included but flagged for review.
    """
    mappings: list[MarketEventMapping] = []
    overrides: dict[str, str] = {}

    if manual_overrides is not None and not manual_overrides.empty:
        for _, row in manual_overrides.iterrows():
            overrides[row["market_ticker"]] = row["event_id"]

    for _, market_row in kalshi_markets.iterrows():
        mticker = market_row.get("ticker", "")
        mtitle = market_row.get("title", "")

        # Manual override takes precedence
        if mticker in overrides:
            eid = overrides[mticker]
            event_row = event_calendar[event_calendar["event_id"] == eid]
            if not event_row.empty:
                mappings.append(
                    MarketEventMapping(
                        market_ticker=mticker,
                        market_title=mtitle,
                        event_id=eid,
                        event_type=event_row.iloc[0]["event_type"],
                        confidence_score=1.0,
                        mapping_method="manual",
                        manual_override=True,
                    )
                )
                continue

        best_event_id: Optional[str] = None
        best_event_type: str = ""
        best_confidence: float = 0.0

        for _, event_row in event_calendar.iterrows():
            etype = event_row["event_type"]
            edt = event_row["release_datetime"]
            if isinstance(edt, str):
                edt = datetime.fromisoformat(edt)

            conf = _compute_confidence(mticker, mtitle, etype, edt)

            # Check if market close time aligns with event release
            close_time = market_row.get("close_time")
            if close_time is not None:
                if isinstance(close_time, str):
                    try:
                        close_time = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                    except ValueError:
                        close_time = None
                if close_time is not None and isinstance(edt, datetime):
                    diff = abs((close_time - edt).total_seconds())
                    if diff < 86400:  # within 1 day
                        conf = min(conf + 0.2, 1.0)

            if conf > best_confidence:
                best_confidence = conf
                best_event_id = event_row["event_id"]
                best_event_type = etype

        if best_event_id is not None and best_confidence > 0:
            mappings.append(
                MarketEventMapping(
                    market_ticker=mticker,
                    market_title=mtitle,
                    event_id=best_event_id,
                    event_type=best_event_type,
                    confidence_score=round(best_confidence, 3),
                    mapping_method="fuzzy" if best_confidence >= 0.5 else "regex",
                )
            )

    return mappings


def mappings_to_dataframe(mappings: list[MarketEventMapping]) -> pd.DataFrame:
    rows = []
    for m in mappings:
        rows.append(
            {
                "market_ticker": m.market_ticker,
                "market_title": m.market_title,
                "event_id": m.event_id,
                "event_type": m.event_type,
                "confidence_score": m.confidence_score,
                "mapping_method": m.mapping_method,
                "notes": m.notes,
                "manual_override": m.manual_override,
            }
        )
    return pd.DataFrame(rows)


def save_market_mapping(
    mappings: list[MarketEventMapping], config: Optional[EngineConfig] = None
) -> Path:
    cfg = config or get_settings()
    df = mappings_to_dataframe(mappings)
    path = cfg.data_dir / cfg.market_mapping_file
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_market_mapping(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.market_mapping_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def export_low_confidence(
    mappings: list[MarketEventMapping],
    threshold: float = 0.5,
    config: Optional[EngineConfig] = None,
) -> Path:
    """Export mappings below threshold for manual review."""
    cfg = config or get_settings()
    low_conf = [m for m in mappings if m.confidence_score < threshold]
    path = cfg.data_dir / cfg.low_confidence_mapping_file
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "market_ticker",
                "market_title",
                "event_id",
                "event_type",
                "confidence_score",
                "mapping_method",
            ]
        )
        for m in low_conf:
            writer.writerow(
                [
                    m.market_ticker,
                    m.market_title,
                    m.event_id,
                    m.event_type,
                    m.confidence_score,
                    m.mapping_method,
                ]
            )
    return path


def save_manual_overrides(overrides: list[dict], config: Optional[EngineConfig] = None) -> Path:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.manual_overrides_file
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["market_ticker", "event_id", "notes"])
        writer.writeheader()
        writer.writerows(overrides)
    return path


def load_manual_overrides(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.manual_overrides_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
