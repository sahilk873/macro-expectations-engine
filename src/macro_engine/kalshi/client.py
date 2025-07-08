from __future__ import annotations

import hashlib
import hmac
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from macro_engine.config.settings import get_settings
from macro_engine.kalshi.models import KalshiEvent, KalshiMarket, KalshiSeries

BASE_URLS = {
    "production": "https://api.elections.kalshi.com/trade-api/v2",
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
}


class KalshiAuth(requests.auth.AuthBase):
    """HMAC-based authentication for Kalshi API."""

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret

    def __call__(self, r: requests.Request) -> requests.Request:
        ts = int(time.time() * 1000)
        method = r.method.upper() if r.method else "GET"
        path = r.path_url
        body = r.body or b""
        if isinstance(body, str):
            body = body.encode()
        msg = f"{ts}{method}{path}{body.decode()}".encode()
        sig = hmac.new(self.api_secret.encode(), msg, hashlib.sha256).hexdigest()
        r.headers["Authorization"] = f"{self.api_key}:{sig}:{ts}"
        r.headers["Content-Type"] = (
            "application/json" if method in ("POST", "PUT") else "application/octet-stream"
        )
        return r


class KalshiClient:
    """Client for fetching Kalshi prediction market data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        environment: str = "production",
    ):
        cfg = get_settings()
        self.api_key = api_key or cfg.kalshi_api_key or ""
        self.api_secret = api_secret or cfg.kalshi_api_secret or ""
        self.environment = environment
        self.base_url = BASE_URLS.get(environment, BASE_URLS["production"])
        self.session = requests.Session()

        if self.api_key and self.api_secret:
            self.session.auth = KalshiAuth(self.api_key, self.api_secret)

    # ------------------------------------------------------------------
    # Public / unauthenticated endpoints
    # ------------------------------------------------------------------

    def list_series(self, limit: int = 100) -> list[KalshiSeries]:
        """List available series (categories like CPI, GDP, etc)."""
        url = f"{self.base_url}/series"
        params: dict[str, Any] = {"limit": min(limit, 1000)}
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            warnings.warn(f"Failed to list series: {e}")
            return self._fallback_series()

        series_list = []
        for s in data.get("series", []):
            series_list.append(
                KalshiSeries(
                    series_ticker=s.get("series_ticker", ""),
                    series_title=s.get("title", ""),
                    description=s.get("description", ""),
                    category=s.get("category", ""),
                )
            )
        if not series_list:
            return self._fallback_series()
        return series_list

    def list_events(self, series_ticker: str, limit: int = 100) -> list[KalshiEvent]:
        """List events for a given series."""
        url = f"{self.base_url}/events"
        params: dict[str, Any] = {"series_ticker": series_ticker, "limit": min(limit, 1000)}
        events: list[KalshiEvent] = []
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for e in data.get("events", []):
                events.append(
                    KalshiEvent(
                        event_ticker=e.get("event_ticker", ""),
                        series_ticker=series_ticker,
                        title=e.get("title", ""),
                        event_strike_date=self._parse_dt(e.get("event_strike_date")),
                        settlement_date=self._parse_dt(e.get("settlement_date")),
                        status=e.get("status", ""),
                        category=e.get("category", ""),
                    )
                )
        except Exception as ex:
            warnings.warn(f"Failed to list events for {series_ticker}: {ex}")
        return events

    def list_markets(self, event_ticker: str, limit: int = 100) -> list[KalshiMarket]:
        """List markets for a given event."""
        url = f"{self.base_url}/markets"
        params: dict[str, Any] = {"event_ticker": event_ticker, "limit": min(limit, 1000)}
        markets: list[KalshiMarket] = []
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for m in data.get("markets", []):
                buckets = self._parse_buckets(m)
                markets.append(
                    KalshiMarket(
                        ticker=m.get("ticker", ""),
                        event_ticker=event_ticker,
                        title=m.get("title", ""),
                        market_type=m.get("market_type", ""),
                        close_time=self._parse_dt(m.get("close_time")),
                        status=m.get("status", ""),
                        result=m.get("result"),
                        yes_bid=m.get("yes_bid"),
                        yes_ask=m.get("yes_ask"),
                        no_bid=m.get("no_bid"),
                        no_ask=m.get("no_ask"),
                        last_price=m.get("last_price"),
                        volume=m.get("volume"),
                        open_interest=m.get("open_interest"),
                        settlement_timestamp=self._parse_dt(m.get("settlement_timestamp")),
                        bucket_ranges=buckets,
                        raw_data=m,
                    )
                )
        except Exception as ex:
            warnings.warn(f"Failed to list markets for {event_ticker}: {ex}")
        return markets

    def get_market(self, ticker: str) -> Optional[KalshiMarket]:
        """Get a single market by ticker."""
        url = f"{self.base_url}/markets/{ticker}"
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            m = resp.json().get("market", {})
            return KalshiMarket(
                ticker=m.get("ticker", ""),
                event_ticker=m.get("event_ticker", ""),
                title=m.get("title", ""),
                market_type=m.get("market_type", ""),
                close_time=self._parse_dt(m.get("close_time")),
                status=m.get("status", ""),
                result=m.get("result"),
                yes_bid=m.get("yes_bid"),
                yes_ask=m.get("yes_ask"),
                no_bid=m.get("no_bid"),
                no_ask=m.get("no_ask"),
                last_price=m.get("last_price"),
                volume=m.get("volume"),
                open_interest=m.get("open_interest"),
                bucket_ranges=self._parse_buckets(m),
                raw_data=m,
            )
        except Exception as ex:
            warnings.warn(f"Failed to get market {ticker}: {ex}")
            return None

    def get_price_history(self, ticker: str, limit: int = 200) -> pd.DataFrame:
        """Get historical price ticks for a market."""
        url = f"{self.base_url}/markets/{ticker}/prices"
        params = {"limit": min(limit, 10000)}
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("prices", [])
            rows = []
            for p in data:
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp": self._parse_dt(p.get("t")),
                        "yes_bid": p.get("yes_bid"),
                        "yes_ask": p.get("yes_ask"),
                        "no_bid": p.get("no_bid"),
                        "no_ask": p.get("no_ask"),
                        "last_price": p.get("last_price"),
                        "volume": p.get("volume"),
                    }
                )
            return pd.DataFrame(rows)
        except Exception as ex:
            warnings.warn(f"Failed to get price history for {ticker}: {ex}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dt(val: Any) -> Optional[datetime]:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        try:
            return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_buckets(market: dict[str, Any]) -> list[tuple[float, float]]:
        """Parse bucket ranges from a multi-bucket market."""
        buckets: list[tuple[float, float]] = []
        raw = market.get("cap_native_value") or market.get("native_value_options")
        if raw and isinstance(raw, list):
            for opt in raw:
                if isinstance(opt, dict):
                    lo = opt.get("lower", opt.get("low"))
                    hi = opt.get("upper", opt.get("high"))
                    if lo is not None and hi is not None:
                        buckets.append((float(lo), float(hi)))
        # Fallback: try count ranges
        count_range = market.get("count_range")
        if not buckets and count_range and isinstance(count_range, list) and len(count_range) == 2:
            num_buckets = 10
            lo, hi = float(count_range[0]), float(count_range[1])
            step = (hi - lo) / num_buckets
            for i in range(num_buckets):
                buckets.append((lo + i * step, lo + (i + 1) * step))
        return buckets

    def _fallback_series(self) -> list[KalshiSeries]:
        """Return hard-coded fallback series for known macro categories."""
        return [
            KalshiSeries(
                series_ticker="CPI", series_title="Consumer Price Index", category="inflation"
            ),
            KalshiSeries(
                series_ticker="FOMC", series_title="Federal Funds Rate", category="policy"
            ),
            KalshiSeries(
                series_ticker="NONFARM", series_title="Nonfarm Payrolls", category="employment"
            ),
            KalshiSeries(
                series_ticker="UNEMPLOYMENT",
                series_title="Unemployment Rate",
                category="employment",
            ),
            KalshiSeries(
                series_ticker="GDP", series_title="Gross Domestic Product", category="growth"
            ),
            KalshiSeries(
                series_ticker="PCE",
                series_title="Personal Consumption Expenditures",
                category="inflation",
            ),
            KalshiSeries(
                series_ticker="RECESSION", series_title="Recession Probability", category="growth"
            ),
        ]

    @staticmethod
    def save_markets_to_parquet(markets: list[KalshiMarket], path: Path) -> None:
        """Save markets list to parquet."""
        rows = []
        for m in markets:
            d = {
                "ticker": m.ticker,
                "event_ticker": m.event_ticker,
                "title": m.title,
                "market_type": m.market_type,
                "close_time": m.close_time,
                "status": m.status,
                "result": m.result,
                "yes_bid": m.yes_bid,
                "yes_ask": m.yes_ask,
                "no_bid": m.no_bid,
                "no_ask": m.no_ask,
                "last_price": m.last_price,
                "volume": m.volume,
                "open_interest": m.open_interest,
                "bucket_ranges": str(m.bucket_ranges) if m.bucket_ranges else "",
            }
            rows.append(d)
        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    @staticmethod
    def save_price_history_to_parquet(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
