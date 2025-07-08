from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class KalshiSeries:
    """A Kalshi series (category of markets, e.g. 'CPI')."""

    series_ticker: str
    series_title: str
    description: str = ""
    category: str = ""


@dataclass
class KalshiEvent:
    """A Kalshi event (e.g. 'CPI for Jan 2024')."""

    event_ticker: str
    series_ticker: str
    title: str
    event_strike_date: Optional[datetime] = None
    settlement_date: Optional[datetime] = None
    status: str = ""
    category: str = ""


@dataclass
class KalshiMarket:
    """A Kalshi market (e.g. 'Will CPI YoY be above 3.0%?')."""

    ticker: str
    event_ticker: str
    title: str
    market_type: str = ""  # binary, multi-bucket, etc
    close_time: Optional[datetime] = None
    status: str = ""
    result: Optional[str] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    last_price: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    settlement_timestamp: Optional[datetime] = None
    bucket_ranges: list[tuple[float, float]] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class KalshiPricePoint:
    """A single price observation for a Kalshi market."""

    ticker: str
    timestamp: datetime
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    last_price: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
