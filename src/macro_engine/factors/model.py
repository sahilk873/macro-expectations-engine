from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from macro_engine.config.settings import EngineConfig, get_settings

logger = logging.getLogger(__name__)

RISK_FACTOR_MAP: dict[str, str] = {
    "SPY": "market",
    "TLT": "rates",
    "IEF": "rates",
    "HYG": "credit",
    "LQD": "credit",
    "UUP": "dollar",
    "GLD": "commodities",
    "USO": "commodities",
}


@dataclass
class FactorAttribution:
    surprise_type: str
    ticker: str
    total_effect: float
    alpha: float
    alpha_tstat: float
    alpha_pvalue: float
    alpha_std: float
    market_beta: float
    rates_beta: float
    credit_beta: float
    dollar_beta: float
    commodity_beta: float
    r_squared: float
    n_events: int
    bh_adjusted_p: float


class SurpriseFactorModel:
    """Panel regression: asset returns = alpha + beta_mkt * R_mkt + beta_surprise * S + epsilon.

    For each (event_type, ticker) pair, we estimate the marginal impact
    of a surprise shock controlling for standard risk factors.
    This isolates the 'abnormal return' attributable to macro news,
    addressing a core critique: are apparent event-study returns
    simply compensation for routine factor exposure?
    """

    def __init__(
        self,
        min_events: int = 5,
        include_factors: Optional[list[str]] = None,
    ):
        self.min_events = min_events
        self.include_factors = include_factors or ["market", "rates", "credit", "dollar"]

    def estimate(
        self,
        event_studies: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run panel regression for each (event_type, ticker) pair.

        Returns factor-attributed surprise effects with t-stats and BH-adjusted p-values.
        """
        if event_studies.empty or price_data.empty:
            return pd.DataFrame()

        price_data = price_data.copy()
        price_data["date"] = pd.to_datetime(price_data["date"])
        close_col = "close" if "close" in price_data.columns else "Close"

        factor_returns = self._compute_factor_returns(price_data, close_col)
        if factor_returns.empty:
            return pd.DataFrame()

        event_returns = self._prepare_event_returns(event_studies, price_data, close_col)
        if event_returns.empty:
            return pd.DataFrame()

        merged = event_returns.merge(
            factor_returns, left_on="event_date", right_index=True, how="left"
        )
        merged = merged.dropna(subset=self._factor_cols(merged))

        results: list[FactorAttribution] = []
        groups = merged.groupby(["event_type", "ticker"])

        for (etype, ticker), group_df in groups:
            if len(group_df) < self.min_events:
                continue
            row = self._run_regression(etype, ticker, group_df)
            if row is not None:
                results.append(row)

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame([r.__dict__ for r in results])
        p_col = "alpha_pvalue"
        if p_col in result_df.columns:
            p_vals = result_df[p_col].fillna(1.0).values
            adjusted = self._benjamini_hochberg(p_vals)
            result_df["bh_adjusted_p"] = adjusted

        return result_df.sort_values("alpha_pvalue").reset_index(drop=True)

    def _run_regression(
        self, etype: str, ticker: str, data: pd.DataFrame
    ) -> Optional[FactorAttribution]:
        y = data["event_return"].values
        factor_cols = self._factor_cols(data)
        X = data[factor_cols].values
        surprise = data["standardized_surprise"].values

        X_design = np.column_stack([np.ones(len(X)), X, surprise])
        n, k = X_design.shape

        if n < k + 1:
            return None

        try:
            beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
            residuals = y - X_design @ beta
            mse = np.sum(residuals**2) / (n - k)
            var_beta = mse * np.linalg.inv(X_design.T @ X_design)
            se = np.sqrt(np.diag(var_beta))
        except np.linalg.LinAlgError:
            return None

        alpha_idx = 0
        surprise_idx = len(factor_cols) + 1 - 1
        alpha = beta[alpha_idx]
        alpha_se = se[alpha_idx]
        alpha_tstat = alpha / alpha_se if alpha_se > 0 else 0.0
        alpha_pval = 2.0 * (1.0 - sp_stats.t.cdf(abs(alpha_tstat), df=n - k))

        market_idx = factor_cols.index("factor_market") if "factor_market" in factor_cols else -1
        rates_idx = factor_cols.index("factor_rates") if "factor_rates" in factor_cols else -1
        credit_idx = factor_cols.index("factor_credit") if "factor_credit" in factor_cols else -1
        dollar_idx = factor_cols.index("factor_dollar") if "factor_dollar" in factor_cols else -1
        comm_idx = (
            factor_cols.index("factor_commodities") if "factor_commodities" in factor_cols else -1
        )

        def _beta(i):
            return beta[i] if i >= 0 else np.nan

        ss_total = np.sum((y - y.mean()) ** 2)
        ss_resid = np.sum(residuals**2)
        r_sq = 1.0 - ss_resid / ss_total if ss_total > 0 else 0.0

        return FactorAttribution(
            surprise_type=etype,
            ticker=ticker,
            total_effect=beta[surprise_idx],
            alpha=alpha,
            alpha_tstat=alpha_tstat,
            alpha_pvalue=alpha_pval,
            alpha_std=alpha_se,
            market_beta=_beta(market_idx),
            rates_beta=_beta(rates_idx),
            credit_beta=_beta(credit_idx),
            dollar_beta=_beta(dollar_idx),
            commodity_beta=_beta(comm_idx),
            r_squared=r_sq,
            n_events=n,
            bh_adjusted_p=1.0,
        )

    def _compute_factor_returns(self, price_data: pd.DataFrame, close_col: str) -> pd.DataFrame:
        fac_map: dict[str, list[str]] = {
            "market": ["SPY"],
            "rates": ["TLT", "IEF"],
            "credit": ["HYG", "LQD"],
            "dollar": ["UUP"],
            "commodities": ["GLD", "USO"],
        }

        pivots = {}
        for fac_name, tickers in fac_map.items():
            if fac_name not in self.include_factors:
                continue
            vals = []
            for t in tickers:
                sub = price_data[price_data["ticker"] == t][["date", close_col]].copy()
                sub = sub.rename(columns={close_col: t})
                if not sub.empty:
                    vals.append(sub.set_index("date"))
            if vals:
                combined = pd.concat(vals, axis=1).mean(axis=1)
                pivots[f"factor_{fac_name}"] = combined.pct_change()

        if not pivots:
            return pd.DataFrame()
        result = pd.DataFrame(pivots)
        if result.empty:
            return result
        result.index = pd.to_datetime(result.index)
        return result

    def _prepare_event_returns(
        self, event_studies: pd.DataFrame, price_data: pd.DataFrame, close_col: str
    ) -> pd.DataFrame:
        records: list[dict] = []
        return_col = "return_1D"

        for _, row in event_studies.iterrows():
            event_time = row.get("event_time")
            ticker = row.get("ticker")
            s_surprise = row.get("standardized_surprise", np.nan)
            event_ret = row.get(return_col, np.nan)

            if pd.isna(event_ret) or pd.isna(s_surprise):
                continue

            if isinstance(event_time, str):
                event_time = pd.Timestamp(event_time)
            event_date = pd.Timestamp(event_time).normalize()

            records.append(
                {
                    "event_type": row.get("event_type"),
                    "ticker": ticker,
                    "event_date": event_date,
                    "event_return": event_ret,
                    "standardized_surprise": s_surprise,
                }
            )

        return pd.DataFrame(records)

    @staticmethod
    def _factor_cols(data: pd.DataFrame) -> list[str]:
        return [c for c in data.columns if c.startswith("factor_")]

    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        ranks = np.arange(1, n + 1)
        bh_thresholds = ranks / n * 0.05
        rejected = sorted_p <= bh_thresholds
        if not rejected.any():
            return np.ones(n)
        max_rejected = np.where(rejected)[0].max()
        adjusted = np.ones(n)
        adjusted[: max_rejected + 1] = sorted_p[: max_rejected + 1] * n / ranks[: max_rejected + 1]
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        unadjusted = np.zeros(n)
        unadjusted[sorted_idx] = adjusted
        return np.clip(unadjusted, 0, 1)


def compute_factor_attribution(
    event_studies: pd.DataFrame,
    price_data: pd.DataFrame,
    min_events: int = 5,
) -> pd.DataFrame:
    model = SurpriseFactorModel(min_events=min_events)
    return model.estimate(event_studies, price_data)


def compute_cumulative_abnormal_returns(
    event_studies: pd.DataFrame,
    group_by: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute cumulative abnormal returns (CAR) across event windows.

    For each event, we compute the CAR as the sum of returns across all available
    post-event windows, testing H0: CAR = 0 via bootstrap.
    """
    if group_by is None:
        group_by = ["event_type"]

    return_cols = sorted(c for c in event_studies.columns if c.startswith("return_"))
    if len(return_cols) < 2:
        return pd.DataFrame()

    results: list[dict] = []
    for keys, group in event_studies.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_by, keys))

        cars = group[return_cols].sum(axis=1, min_count=1).dropna()
        if len(cars) < 2:
            continue

        row["n_events"] = len(cars)
        row["car_mean"] = cars.mean()
        row["car_std"] = cars.std(ddof=1)
        row["car_se"] = row["car_std"] / np.sqrt(len(cars))
        row["car_tstat"] = row["car_mean"] / row["car_se"] if row["car_se"] > 0 else 0.0
        row["car_pvalue"] = 2.0 * (1.0 - sp_stats.t.cdf(abs(row["car_tstat"]), df=len(cars) - 1))
        row["car_hit_rate"] = (cars > 0).mean()

        rng = np.random.default_rng(42)
        boot_means = np.array(
            [rng.choice(cars.values, size=len(cars), replace=True).mean() for _ in range(10000)]
        )
        row["car_ci_lower"] = float(np.percentile(boot_means, 2.5))
        row["car_ci_upper"] = float(np.percentile(boot_means, 97.5))

        results.append(row)

    return pd.DataFrame(results)


def save_factor_attribution(df: pd.DataFrame, config: Optional[EngineConfig] = None) -> Path:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.factor_attribution_file
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d factor attribution records to %s", len(df), path)
    return path


def load_factor_attribution(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.factor_attribution_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def compute_car_test(car_results: pd.DataFrame) -> dict[str, float]:
    """Joint test: are CARs jointly significantly different from zero?

    Uses Hotelling's T^2 or a simple F-test on the vector of mean CARs.
    """
    car_cols = [c for c in car_results.columns if c.endswith("_mean") and c.startswith("car")]
    if not car_cols:
        return {}

    means = car_results[car_cols].values.flatten()
    n_groups = len(means)
    if n_groups < 2:
        return {}

    pooled_mean = means.mean()
    pooled_std = means.std(ddof=1) if n_groups > 1 else 1.0
    tstat = pooled_mean / (pooled_std / np.sqrt(n_groups)) if pooled_std > 0 else 0.0
    pval = 2.0 * (1.0 - sp_stats.t.cdf(abs(tstat), df=n_groups - 1))

    return {
        "n_groups": n_groups,
        "pooled_car_mean": float(pooled_mean),
        "pooled_car_std": float(pooled_std),
        "cross_sectional_tstat": float(tstat),
        "cross_sectional_pvalue": float(pval),
    }
