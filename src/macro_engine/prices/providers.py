from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings


class PriceProvider(ABC):
    """Abstract base for price data providers."""

    @abstractmethod
    def fetch_prices(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame: ...


class YFinanceProvider(PriceProvider):
    """Price provider using yfinance."""

    def fetch_prices(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            warnings.warn("yfinance not installed. Install with: pip install yfinance")
            return pd.DataFrame()

        all_data: list[pd.DataFrame] = []
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(start=start_date, end=end_date, auto_adjust=True)
                if hist.empty:
                    continue
                df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df["ticker"] = ticker
                df = df.reset_index()
                df.rename(columns={"Date": "date"}, inplace=True)
                all_data.append(df)
            except Exception as e:
                warnings.warn(f"Failed to fetch {ticker}: {e}")

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


class DummyPriceProvider(PriceProvider):
    """Fallback price provider that generates sample data for testing."""

    def fetch_prices(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        rows: list[dict] = []
        np.random.seed(42)
        base_prices = {
            "SPY": 400,
            "QQQ": 300,
            "IWM": 200,
            "TLT": 100,
            "IEF": 95,
            "SHY": 80,
            "HYG": 85,
            "LQD": 90,
            "UUP": 25,
            "GLD": 180,
            "USO": 60,
        }
        for ticker in tickers:
            price = base_prices.get(ticker, 100.0)
            for dt in dates:
                change = np.random.randn() * 0.01
                price = price * (1 + change)
                rows.append(
                    {
                        "date": dt,
                        "ticker": ticker,
                        "open": price * 0.998,
                        "high": price * 1.005,
                        "low": price * 0.995,
                        "close": price,
                        "volume": int(np.random.uniform(1e6, 1e8)),
                    }
                )
        return pd.DataFrame(rows)


def get_price_data(
    tickers: Optional[list[str]] = None,
    start_date: str = "2019-01-01",
    end_date: str = "2025-12-31",
    provider: Optional[PriceProvider] = None,
) -> pd.DataFrame:
    """Fetch price data from the specified provider.

    Falls back to DummyPriceProvider if yfinance is unavailable.
    """
    cfg = get_settings()
    if tickers is None:
        tickers = list(cfg.etf_tickers)

    if provider is not None:
        return provider.fetch_prices(tickers, start_date, end_date)

    # Try yfinance first
    yfp = YFinanceProvider()
    df = yfp.fetch_prices(tickers, start_date, end_date)
    if not df.empty:
        return df

    warnings.warn("yfinance returned no data, using dummy data provider for development")
    dummy = DummyPriceProvider()
    return dummy.fetch_prices(tickers, start_date, end_date)


def save_price_data(df: pd.DataFrame, config: Optional[EngineConfig] = None) -> Path:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.price_data_file
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_price_data(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.price_data_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
