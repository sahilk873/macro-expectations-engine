from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.backtest.signals import (
    SurpriseSignal,
    build_surprise_signals,
    build_surprise_tilts,
    get_active_signals,
)
from macro_engine.config.settings import EngineConfig, get_settings

logger = logging.getLogger(__name__)


class RegimeAwareStrategy:
    """A regime-aware ETF allocation strategy with volatility targeting.

    Adjusts portfolio weights based on the prevailing macro regime,
    using only information available at the rebalance date.
    """

    def __init__(
        self,
        base_weights: Optional[dict[str, float]] = None,
        transaction_cost_bps: float = 3.0,
        vol_target: float = 0.12,
    ):
        self.base_weights = base_weights or self._default_base_weights()
        self.tc_bps = transaction_cost_bps
        self.vol_target = vol_target

    @staticmethod
    def _default_base_weights() -> dict[str, float]:
        return {
            "SPY": 0.30,
            "QQQ": 0.10,
            "IWM": 0.05,
            "TLT": 0.15,
            "IEF": 0.10,
            "HYG": 0.05,
            "LQD": 0.05,
            "GLD": 0.10,
            "UUP": 0.05,
            "USO": 0.05,
        }

    def get_weights(self, regime: dict[str, str]) -> dict[str, float]:
        """Get target portfolio weights given a regime classification, with vol targeting."""
        weights = dict(self.base_weights)

        risk_regime = regime.get("risk_regime", "neutral")
        infl_regime = regime.get("inflation_regime", "stable")
        vol_regime = regime.get("volatility_regime", "normal")

        if risk_regime == "risk_off":
            for ticker in ["SPY", "QQQ", "IWM", "HYG", "USO"]:
                weights[ticker] = weights.get(ticker, 0) * 0.5
            for ticker in ["TLT", "IEF", "SHY", "GLD", "LQD"]:
                weights[ticker] = weights.get(ticker, 0) * 1.5
        elif risk_regime == "risk_on":
            for ticker in ["SPY", "QQQ", "IWM"]:
                weights[ticker] = weights.get(ticker, 0) * 1.3
            for ticker in ["TLT", "IEF", "SHY"]:
                weights[ticker] = weights.get(ticker, 0) * 0.7

        if infl_regime == "rising":
            for ticker in ["TLT", "IEF", "LQD"]:
                weights[ticker] = weights.get(ticker, 0) * 0.7
            for ticker in ["GLD", "USO"]:
                weights[ticker] = weights.get(ticker, 0) * 1.4
        elif infl_regime == "falling":
            for ticker in ["TLT", "IEF"]:
                weights[ticker] = weights.get(ticker, 0) * 1.3

        if vol_regime == "high":
            for ticker in ["QQQ", "IWM", "HYG"]:
                weights[ticker] = weights.get(ticker, 0) * 0.6
            weights["USMV"] = weights.get("USMV", 0) + 0.05

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    @staticmethod
    def compute_turnover(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        turnover = 0.0
        for t in all_tickers:
            turnover += abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0))
        return turnover / 2.0


class SurpriseTacticalStrategy(RegimeAwareStrategy):
    """Two-layer strategy combining regime-aware allocation with surprise-based tactical tilts.

    Layer 1 — Strategic (inherited)
        Base asset allocation adjusted for the prevailing macro regime
        (growth, inflation, policy, volatility, and risk dimensions).

    Layer 2 — Tactical
        On each rebalance date, the strategy checks for recent prediction-market
        macro surprises.  For each active surprise it computes ticker-level tilts
        using two sources of information:

        * **Risk-regime tilts** — risk-off / risk-on label from the surprise
          (e.g., an inflation surprise labelled ``risk_off`` reduces equity
          exposure and increases safe-haven Treasuries/gold).

        * **Factor-model tilts** — when a :class:`~macro_engine.factors.model.SurpriseFactorModel`
          attribution table is available, per-ticker tilts are proportional to
          the product of the surprise's standardized magnitude and the
          historically estimated factor alpha for that (surprise_type, ticker)
          pair.  This directly connects the trading logic to the project's
          core research output: the marginal impact of a macro surprise on
          each asset, controlling for standard risk factors.

    All tilts are time-decayed (linear decay over ``signal_decay_days``) so
    that older surprises have less influence.  Confounded surprises (events
    that co-occur within 24 h of another release) are downweighted by 50 %.
    """

    def __init__(
        self,
        base_weights: Optional[dict[str, float]] = None,
        transaction_cost_bps: float = 3.0,
        vol_target: float = 0.12,
        surprise_tilt_strength: float = 1.0,
        signal_decay_days: int = 5,
        min_surprise_confidence: float = 0.3,
    ):
        super().__init__(base_weights, transaction_cost_bps, vol_target)
        self.surprise_tilt_strength = surprise_tilt_strength
        self.signal_decay_days = signal_decay_days
        self.min_surprise_confidence = min_surprise_confidence

    def get_weights(
        self,
        regime: dict[str, str],
        active_signals: Optional[list[SurpriseSignal]] = None,
        factor_attribution: Optional[pd.DataFrame] = None,
    ) -> dict[str, float]:
        """Compute target weights combining regime positioning and surprise tilts."""
        weights = dict(super().get_weights(regime))

        if not active_signals:
            return weights

        tilts = build_surprise_tilts(
            active_signals,
            factor_attribution=factor_attribution,
            tilt_strength=self.surprise_tilt_strength,
            min_confidence=self.min_surprise_confidence,
        )

        for ticker, tilt in tilts.items():
            weights[ticker] = weights.get(ticker, 0.0) * (1.0 + tilt)

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights if weights else dict(self.base_weights)


def run_backtest(
    price_data: pd.DataFrame,
    regime_classifications: pd.DataFrame,
    strategy: Optional[RegimeAwareStrategy] = None,
    config: Optional[EngineConfig] = None,
    surprises: Optional[pd.DataFrame] = None,
    factor_attribution: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run a regime-aware ETF allocation backtest with daily P&L tracking.

    When *surprises* is provided and *strategy* is a
    :class:`SurpriseTacticalStrategy`, the backtest will also incorporate
    tactical tilts from prediction-market macro surprises on each rebalance
    date, using the factor-model attribution to determine per-asset
    directional exposures.
    """
    cfg = config or get_settings()
    strat = strategy or RegimeAwareStrategy(transaction_cost_bps=cfg.transaction_cost_bps)

    if price_data.empty or regime_classifications.empty:
        return pd.DataFrame()

    # Pre-build surprise signals if the strategy supports them
    all_signals: list[SurpriseSignal] = []
    use_surprise_overlay = (
        isinstance(strat, SurpriseTacticalStrategy)
        and surprises is not None
        and not surprises.empty
    )
    if use_surprise_overlay:
        all_signals = build_surprise_signals(surprises)

    prices = price_data.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    close_col = "close" if "close" in prices.columns else "Close"
    price_pivot = prices.pivot_table(
        index="date", columns="ticker", values=close_col, aggfunc="first"
    )
    price_pivot = price_pivot.sort_index().ffill()

    regime_lookup: dict[str, dict] = {}
    for _, r in regime_classifications.iterrows():
        regime_lookup[str(r["date"])[:10]] = r.to_dict()

    all_dates = price_pivot.index
    regime_dates = pd.to_datetime(list(regime_lookup.keys()))
    rebal_dates = sorted(set(all_dates) & set(regime_dates))
    rebal_dates = sorted(rebal_dates)

    if not rebal_dates:
        return pd.DataFrame()

    records: list[dict] = []
    portfolio_value = 1.0
    current_weights: dict[str, float] = {}

    for i, rebal_date in enumerate(rebal_dates):
        date_str = str(rebal_date.date())
        regime = regime_lookup.get(date_str, {})
        if not regime:
            continue

        if use_surprise_overlay:
            active_signals = get_active_signals(
                all_signals,
                rebal_date.to_pydatetime(),
                lookback_days=cfg.signal_lookback_days,
                decay_days=strat.signal_decay_days,
            )
            new_weights = strat.get_weights(regime, active_signals, factor_attribution)
        else:
            new_weights = strat.get_weights(regime)

        if current_weights:
            turnover = strat.compute_turnover(current_weights, new_weights)
            tc = turnover * (strat.tc_bps / 10000.0)
        else:
            turnover = 1.0
            tc = 0.0

        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else all_dates[-1]
        window = price_pivot.loc[rebal_date:next_date]

        if len(window) < 1:
            continue

        daily_returns = window.pct_change().fillna(0.0)

        for j in range(len(window)):
            current_date = window.index[j]
            if j == 0:
                daily_ret = 0.0
            else:
                daily_ret = sum(
                    new_weights.get(t, 0.0) * daily_returns.loc[current_date, t]
                    for t in new_weights
                    if t in daily_returns.columns
                    and not np.isnan(daily_returns.loc[current_date, t])
                )

            portfolio_value *= 1.0 + daily_ret

            if j == 0 and tc > 0:
                portfolio_value *= 1.0 - tc

            record = {
                "date": current_date,
                "portfolio_value": portfolio_value,
                "daily_return": daily_ret,
                "turnover": turnover if j == 0 else 0.0,
                "transaction_cost": tc if j == 0 else 0.0,
            }
            for t in price_pivot.columns:
                record[f"weight_{t}"] = new_weights.get(t, 0.0)
            for k, v in regime.items():
                if k != "date":
                    record[f"regime_{k}"] = v

            if use_surprise_overlay and active_signals:
                active_etypes = list({s.event_type for s in active_signals})
                record["active_surprise_types"] = ",".join(sorted(active_etypes))
                record["n_active_signals"] = len(active_signals)
                record["avg_signal_decay"] = float(
                    np.mean([s.effective_weight for s in active_signals])
                )

            records.append(record)

        current_weights = new_weights

    return pd.DataFrame(records)


def compute_performance_metrics(backtest_results: pd.DataFrame) -> dict[str, float]:
    """Compute key performance metrics from backtest results."""
    if backtest_results.empty:
        return {}

    returns = backtest_results["daily_return"].dropna().values
    if len(returns) == 0:
        return {}

    total_return = float(
        backtest_results["portfolio_value"].iloc[-1] / backtest_results["portfolio_value"].iloc[0]
        - 1.0
    )
    n_years = max(len(returns) / 252, 1 / 252)
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    ann_vol = np.std(returns, ddof=1) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    portfolio_values = backtest_results["portfolio_value"].values
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_dd = np.min(drawdown)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.0

    avg_turnover = (
        backtest_results.loc[backtest_results["turnover"] > 0, "turnover"].mean()
        if "turnover" in backtest_results.columns
        else 0.0
    )

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "avg_turnover": avg_turnover,
        "n_periods": len(returns),
    }


def save_backtest_results(df: pd.DataFrame, config: Optional[EngineConfig] = None) -> Path:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.backtest_results_file
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d backtest records to %s", len(df), path)
    return path


def load_backtest_results(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.backtest_results_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
