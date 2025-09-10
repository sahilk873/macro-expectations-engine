from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EngineConfig:
    """Central configuration for the macro engine."""

    # Paths
    repo_root: Path = Path(__file__).resolve().parent.parent.parent.parent
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    reports_dir: Path = field(default_factory=lambda: Path.cwd() / "reports")
    config_dir: Path = field(default_factory=lambda: Path.cwd() / "config")

    # Kalshi
    kalshi_api_key: Optional[str] = field(default_factory=lambda: os.getenv("KALSHI_API_KEY"))
    kalshi_api_secret: Optional[str] = field(default_factory=lambda: os.getenv("KALSHI_API_SECRET"))
    kalshi_environment: str = field(
        default_factory=lambda: os.getenv("KALSHI_ENVIRONMENT", "production")
    )

    # FRED
    fred_api_key: Optional[str] = field(default_factory=lambda: os.getenv("FRED_API_KEY"))

    # BLS
    bls_api_key: Optional[str] = field(default_factory=lambda: os.getenv("BLS_API_KEY"))

    # Event study defaults
    pre_event_windows_days: list[int] = field(default_factory=lambda: [1])
    pre_event_windows_hours: list[float] = field(default_factory=lambda: [1.0])
    post_event_windows: list[str] = field(default_factory=lambda: ["1D", "3D", "5D", "10D", "21D"])

    # Regime
    regime_lookback_days: int = 180

    # Backtest
    backtest_start: str = "2022-01-01"
    backtest_end: str = "2025-12-31"
    transaction_cost_bps: float = 3.0
    rebalance_freq: str = "monthly"

    # Robustness
    placebo_n_iterations: int = 100
    bootstrap_n_iterations: int = 10000

    # Output files
    event_calendar_file: str = "macro_event_calendar.parquet"
    kalshi_markets_file: str = "kalshi_markets.parquet"
    kalshi_prices_file: str = "kalshi_prices.parquet"
    market_mapping_file: str = "market_event_mapping.parquet"
    low_confidence_mapping_file: str = "low_confidence_mapping.csv"
    manual_overrides_file: str = "manual_overrides.csv"
    macro_data_file: str = "official_macro_data.parquet"
    price_data_file: str = "etf_prices.parquet"
    implied_expectations_file: str = "implied_expectations.parquet"
    implied_distributions_file: str = "implied_distributions.parquet"
    surprises_file: str = "macro_surprises.parquet"
    event_studies_file: str = "event_studies.parquet"
    regime_classifications_file: str = "regime_classifications.parquet"
    backtest_results_file: str = "backtest_results.parquet"
    placebo_results_file: str = "placebo_results.parquet"
    robustness_results_file: str = "robustness_results.parquet"

    # Event types we track
    event_types: tuple[str, ...] = (
        "CPI",
        "FOMC",
        "NFP",
        "UNEMPLOYMENT",
        "GDP",
        "PCE",
        "RECESSION",
    )

    # ETF tickers
    etf_tickers: tuple[str, ...] = (
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
        "IEF",
        "SHY",
        "HYG",
        "LQD",
        "UUP",
        "GLD",
        "USO",
        "XLE",
        "XLF",
        "XLK",
        "XLV",
        "XLI",
        "XLP",
        "XLY",
        "XLB",
        "XLU",
        "XLRE",
        "MTUM",
        "QUAL",
        "SIZE",
        "USMV",
    )

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.reports_dir = Path(self.reports_dir)
        self.config_dir = Path(self.config_dir)

    @property
    def kalshi_dir(self) -> Path:
        return self.data_dir / "kalshi"

    @property
    def macro_dir(self) -> Path:
        return self.data_dir / "macro"

    @property
    def prices_dir(self) -> Path:
        return self.data_dir / "prices"

    @property
    def manual_dir(self) -> Path:
        return self.data_dir / "manual"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def tables_dir(self) -> Path:
        return self.reports_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def notebooks_dir(self) -> Path:
        return self.reports_dir / "notebooks"


_settings: Optional[EngineConfig] = None


def get_settings() -> EngineConfig:
    global _settings
    if _settings is None:
        _settings = EngineConfig()
    return _settings
