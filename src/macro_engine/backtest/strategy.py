from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings


class RegimeAwareStrategy:
    """A regime-aware ETF allocation strategy.

    Adjusts portfolio weights based on the prevailing macro regime,
    using only information available at the rebalance date.

    Base (neutral) weights and regime adjustment matrices define the strategy.
    """

    def __init__(
        self,
        base_weights: Optional[dict[str, float]] = None,
        transaction_cost_bps: float = 3.0,
    ):
        self.base_weights = base_weights or self._default_base_weights()
        self.tc_bps = transaction_cost_bps

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
        """Get target portfolio weights given a regime classification."""
        weights = dict(self.base_weights)

        risk_regime = regime.get("risk_regime", "neutral")
        growth_regime = regime.get("growth_regime", "neutral")
        infl_regime = regime.get("inflation_regime", "stable")
        vol_regime = regime.get("volatility_regime", "normal")

        if risk_regime == "risk_off":
            # Reduce equities, increase bonds and gold
            for ticker in ["SPY", "QQQ", "IWM", "HYG", "USO"]:
                weights[ticker] = weights.get(ticker, 0) * 0.5
            for ticker in ["TLT", "IEF", "SHY", "GLD", "LQD"]:
                weights[ticker] = weights.get(ticker, 0) * 1.5
        elif risk_regime == "risk_on":
            # Increase equities, reduce bonds
            for ticker in ["SPY", "QQQ", "IWM"]:
                weights[ticker] = weights.get(ticker, 0) * 1.3
            for ticker in ["TLT", "IEF", "SHY"]:
                weights[ticker] = weights.get(ticker, 0) * 0.7

        if infl_regime == "rising":
            # Reduce bonds, increase commodities
            for ticker in ["TLT", "IEF", "LQD"]:
                weights[ticker] = weights.get(ticker, 0) * 0.7
            for ticker in ["GLD", "USO"]:
                weights[ticker] = weights.get(ticker, 0) * 1.4
        elif infl_regime == "falling":
            # Increase bonds
            for ticker in ["TLT", "IEF"]:
                weights[ticker] = weights.get(ticker, 0) * 1.3

        if vol_regime == "high":
            for ticker in ["QQQ", "IWM", "HYG"]:
                weights[ticker] = weights.get(ticker, 0) * 0.6
            weights["USMV"] = weights.get("USMV", 0) + 0.05

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    @staticmethod
    def compute_turnover(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
        """Compute portfolio turnover from weight changes."""
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
    """Run a regime-aware ETF allocation backtest.

    Steps:
    1. For each rebalance date, get the regime classification.
    2. Compute target weights from the strategy.
    3. Apply transaction costs on turnover.
    4. Track portfolio value over time.
    5. Compare against benchmarks (SPY, 60/40).
    """
    cfg = config or get_settings()
    strat = strategy or RegimeAwareStrategy(transaction_cost_bps=cfg.transaction_cost_bps)

    if price_data.empty or regime_classifications.empty:
        return pd.DataFrame()

    prices = price_data.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    # Prepare price pivot
    close_col = "close" if "close" in prices.columns else "Close"
    price_pivot = prices.pivot_table(
        index="date", columns="ticker", values=close_col, aggfunc="first"
    )
    price_pivot = price_pivot.sort_index().ffill()

    # Generate rebalance dates (monthly)
    all_dates = price_pivot.index
    rebal_dates = all_dates[
        all_dates.isin(regime_classifications["date"].values) | (all_dates.day == 1)
    ]
    rebal_dates = sorted(set(rebal_dates) & set(price_pivot.index))

    if not rebal_dates:
        # Fallback: use first of each month
        rebal_dates = all_dates[
            all_dates.isin(pd.to_datetime(regime_classifications["date"].values))
        ]
        rebal_dates = sorted(set(rebal_dates))

    if not rebal_dates:
        warnings.warn("No matching rebalance dates found")
        return pd.DataFrame()

    # Run backtest
    portfolio_value = 1.0
    current_weights: dict[str, float] = {}
    records: list[dict] = []

    regime_lookup = {}
    for _, r in regime_classifications.iterrows():
        regime_lookup[str(r["date"])[:10]] = r.to_dict()

    for i, date in enumerate(rebal_dates):
        date_str = str(date.date()) if hasattr(date, "date") else str(date)[:10]
        regime = regime_lookup.get(date_str, {})

        if not regime:
            continue

        new_weights = strat.get_weights(regime)

        # Compute transaction costs
        if current_weights:
            turnover = strat.compute_turnover(current_weights, new_weights)
            tc = turnover * (strat.tc_bps / 10000.0)
        else:
            turnover = 1.0
            tc = 0.0

        # Record portfolio state
        row = {
            "date": date,
            "portfolio_value": portfolio_value,
            "turnover": turnover,
            "transaction_cost": tc,
            **{f"weight_{t}": new_weights.get(t, 0.0) for t in price_pivot.columns},
        }
        row.update({f"regime_{k}": v for k, v in regime.items() if k != "date"})
        records.append(row)

        current_weights = new_weights

        # --- Daily returns until next rebalance ---
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else all_dates[-1]
        daily_prices = price_pivot.loc[date:next_date]

        if len(daily_prices) <= 1:
            continue

        for j in range(1, len(daily_prices)):
            prev_prices = daily_prices.iloc[j - 1]
            curr_prices = daily_prices.iloc[j]

            ret = 0.0
            for ticker, weight in current_weights.items():
                if ticker in prev_prices and ticker in curr_prices:
                    p_prev = prev_prices[ticker]
                    p_curr = curr_prices[ticker]
                    if p_prev != 0 and not np.isnan(p_prev) and not np.isnan(p_curr):
                        ret += weight * (p_curr / p_prev - 1.0)

            portfolio_value *= 1.0 + ret

            # Apply transaction costs on rebalance day
            day_date = daily_prices.index[j]
            if day_date == date and tc > 0 and j == 1:
                portfolio_value *= 1.0 - tc

        # Update portfolio value for final day
        if len(daily_prices) > 0:
            portfolio_value = portfolio_value

    return pd.DataFrame(records)


def compute_performance_metrics(backtest_results: pd.DataFrame) -> dict[str, float]:
    """Compute key performance metrics from backtest results."""
    if backtest_results.empty:
        return {}

    portfolio_values = backtest_results["portfolio_value"].values
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    if len(returns) == 0:
        return {}

    total_return = portfolio_values[-1] / portfolio_values[0] - 1.0

    n_years = len(returns) / 252 if len(returns) > 0 else 1.0
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol = np.std(returns, ddof=1) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_dd = np.min(drawdown)

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.0

    # Average turnover
    avg_turnover = (
        backtest_results["turnover"].mean() if "turnover" in backtest_results.columns else 0.0
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
    return path


def load_backtest_results(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.backtest_results_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
