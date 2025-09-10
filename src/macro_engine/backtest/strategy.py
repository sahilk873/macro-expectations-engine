from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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


def run_backtest(
    price_data: pd.DataFrame,
    regime_classifications: pd.DataFrame,
    strategy: Optional[RegimeAwareStrategy] = None,
    config: Optional[EngineConfig] = None,
) -> pd.DataFrame:
    """Run a regime-aware ETF allocation backtest with daily P&L tracking."""
    cfg = config or get_settings()
    strat = strategy or RegimeAwareStrategy(transaction_cost_bps=cfg.transaction_cost_bps)

    if price_data.empty or regime_classifications.empty:
        return pd.DataFrame()

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
