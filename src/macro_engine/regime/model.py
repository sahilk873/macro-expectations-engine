from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings


class MacroRegimeModel:
    """Classifies macro regimes based on available macro and financial data.

    Regime dimensions:
        - Growth regime: expansion / neutral / contraction
        - Inflation regime: rising / stable / falling
        - Policy regime: accommodative / neutral / restrictive
        - Volatility regime: low / normal / high
        - Risk regime: risk-on / neutral / risk-off

    Uses only information available at the classification date (no lookahead).
    """

    def __init__(self, lookback_days: int = 180):
        self.lookback_days = lookback_days

    def classify(
        self,
        macro_data: pd.DataFrame,
        price_data: pd.DataFrame,
        as_of_date: Optional[str] = None,
    ) -> dict[str, str]:
        """Classify the current regime as of a given date.

        Returns dict with regime labels for each dimension.
        """
        as_of = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.now()
        lookback_start = as_of - pd.Timedelta(days=self.lookback_days)

        # Filter macro data to lookback window
        macro = macro_data.copy()
        if "date" in macro.columns:
            macro["date"] = pd.to_datetime(macro["date"])
            macro = macro[macro["date"].between(lookback_start, as_of)]

        # Growth regime
        growth_regime = self._classify_growth(macro)

        # Inflation regime
        inflation_regime = self._classify_inflation(macro)

        # Policy regime
        policy_regime = self._classify_policy(macro)

        # Volatility regime
        vol_regime = self._classify_volatility(price_data, as_of)

        # Risk regime
        risk_regime = self._classify_risk(macro, price_data, as_of)

        return {
            "date": str(as_of.date()),
            "growth_regime": growth_regime,
            "inflation_regime": inflation_regime,
            "policy_regime": policy_regime,
            "volatility_regime": vol_regime,
            "risk_regime": risk_regime,
        }

    def _classify_growth(self, macro: pd.DataFrame) -> str:
        """Classify growth regime using GDP and employment data."""
        gdp = (
            macro[macro["series_id"].isin(["GDP", "GDPC1"])]
            if "series_id" in macro.columns
            else pd.DataFrame()
        )
        emp = (
            macro[macro["series_id"] == "PAYEMS"]
            if "series_id" in macro.columns
            else pd.DataFrame()
        )

        if not emp.empty and len(emp) >= 2:
            emp_sorted = emp.sort_values("date")
            recent_change = emp_sorted["value"].diff().tail(3).mean()
            if not np.isnan(recent_change):
                if recent_change > 150000:
                    return "expansion"
                elif recent_change < -50000:
                    return "contraction"
                else:
                    return "neutral"

        if not gdp.empty:
            latest = gdp.sort_values("date").iloc[-1]["value"]
            if latest > 2.5:
                return "expansion"
            elif latest < 0:
                return "contraction"
        return "neutral"

    def _classify_inflation(self, macro: pd.DataFrame) -> str:
        """Classify inflation regime using CPI and PCE data."""
        cpi = (
            macro[macro["series_id"] == "CPIAUCSL"]
            if "series_id" in macro.columns
            else pd.DataFrame()
        )
        pce = (
            macro[macro["series_id"] == "PCEPI"] if "series_id" in macro.columns else pd.DataFrame()
        )

        for df, label in [(cpi, "cpi"), (pce, "pce")]:
            if not df.empty and len(df) >= 2:
                sorted_df = df.sort_values("date")
                annualized = (
                    sorted_df["value"].pct_change(12).iloc[-1] * 100
                    if label == "cpi"
                    else sorted_df["value"].pct_change(12).iloc[-1] * 100
                )
                if not np.isnan(annualized):
                    if annualized > 3.0:
                        return "rising"
                    elif annualized < 1.0:
                        return "falling"
                    else:
                        return "stable"
        return "stable"

    def _classify_policy(self, macro: pd.DataFrame) -> str:
        """Classify policy regime using Fed Funds rate."""
        ff = (
            macro[macro["series_id"] == "FEDFUNDS"]
            if "series_id" in macro.columns
            else pd.DataFrame()
        )

        if not ff.empty:
            latest = ff.sort_values("date").iloc[-1]["value"]
            if latest > 4.0:
                return "restrictive"
            elif latest > 1.0:
                return "neutral"
            else:
                return "accommodative"
        return "neutral"

    def _classify_volatility(self, price_data: pd.DataFrame, as_of: pd.Timestamp) -> str:
        """Classify volatility regime using VIX proxy."""
        if price_data.empty:
            return "normal"

        spy = price_data[price_data["ticker"] == "SPY"].copy()
        if spy.empty:
            return "normal"

        spy["date"] = pd.to_datetime(spy["date"])
        spy = spy[spy["date"] <= as_of]

        if len(spy) < 20:
            return "normal"

        close_col = "close" if "close" in spy.columns else "Close"
        spy["return"] = spy[close_col].pct_change()
        realized_vol = spy["return"].tail(20).std() * np.sqrt(252)

        if realized_vol > 0.30:
            return "high"
        elif realized_vol < 0.15:
            return "low"
        return "normal"

    def _classify_risk(
        self, macro: pd.DataFrame, price_data: pd.DataFrame, as_of: pd.Timestamp
    ) -> str:
        """Classify risk regime combining growth, vol, and policy."""
        growth = self._classify_growth(macro)
        vol = self._classify_volatility(price_data, as_of)

        if growth == "contraction" or vol == "high":
            return "risk_off"
        elif growth == "expansion" and vol == "low":
            return "risk_on"
        # Check drawdown
        if not price_data.empty:
            spy = price_data[price_data["ticker"] == "SPY"]
            if not spy.empty:
                spy["date"] = pd.to_datetime(spy["date"])
                spy = spy[spy["date"] <= as_of]
                close_col = "close" if "close" in spy.columns else "Close"
                if len(spy) > 0:
                    peak = spy[close_col].max()
                    current = spy[close_col].iloc[-1]
                    if peak > 0 and current / peak < 0.9:
                        return "risk_off"
        return "neutral"

    def classify_series(
        self,
        macro_data: pd.DataFrame,
        price_data: pd.DataFrame,
        dates: list[str],
    ) -> pd.DataFrame:
        """Classify regime for a series of dates."""
        records = []
        for d in dates:
            records.append(self.classify(macro_data, price_data, as_of_date=d))
        return pd.DataFrame(records)


def compute_macro_regime(
    macro_data: pd.DataFrame,
    price_data: pd.DataFrame,
    config: Optional[EngineConfig] = None,
) -> pd.DataFrame:
    """Compute macro regime classifications for the full backtest period."""
    cfg = config or get_settings()
    model = MacroRegimeModel(lookback_days=cfg.regime_lookback_days)

    # Generate monthly dates for regime classification
    start = pd.Timestamp(cfg.backtest_start)
    end = pd.Timestamp(cfg.backtest_end)
    dates = pd.date_range(start=start, end=end, freq="BMS").strftime("%Y-%m-%d").tolist()

    return model.classify_series(macro_data, price_data, dates)


def save_regime_classifications(df: pd.DataFrame, config: Optional[EngineConfig] = None) -> Path:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.regime_classifications_file
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_regime_classifications(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.regime_classifications_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
