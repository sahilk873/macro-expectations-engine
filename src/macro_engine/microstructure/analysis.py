from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MicrostructureMetrics:
    ticker: str
    n_observations: int
    mean_mid_price: float
    mean_spread_bps: float
    median_spread_bps: float
    mean_relative_spread: float
    spread_volatility: float
    vwap_probability: float
    vwap_volume: float
    arbitrage_opportunities: int
    arbitrage_rate: float
    depth_ratio: float
    price_discovery_ratio: float
    mean_volume: float
    total_volume: float
    mean_open_interest: float


def compute_vwap_probability(
    prices: pd.DataFrame,
    ticker: Optional[str] = None,
) -> tuple[float, float]:
    """Volume-weighted average probability from trade prices.

    VWAP = sum(volume_i * price_i) / sum(volume_i)

    Uses mid-price as proxy when only quote data is available,
    weighted by volume to reduce impact of low-liquidity prints.

    Returns (vwap_probability, total_volume).
    """
    df = prices.copy()
    if ticker is not None:
        df = df[df["ticker"] == ticker]

    if df.empty:
        return np.nan, 0.0

    df["mid"] = df[["yes_bid", "yes_ask"]].mean(axis=1, skipna=True)
    volume = df["volume"].fillna(0).values
    mid = df["mid"].values

    valid = ~np.isnan(mid) & (volume > 0)
    if not valid.any():
        return float(np.nanmean(mid)) if np.nanmean(mid) is not None else np.nan, float(
            volume.sum()
        )

    vwap = float(np.average(mid[valid], weights=volume[valid]))
    total_vol = float(volume.sum())
    return vwap, total_vol


def compute_spread_metrics(prices: pd.DataFrame, ticker: Optional[str] = None) -> dict[str, float]:
    """Compute bid-ask spread statistics.

    Spread analysis is critical for assessing market quality:
    - Wide spreads = low liquidity = less reliable implied probabilities
    - Spread volatility = uncertainty/risk in the market
    - Relative spread = spread / mid-price (for cross-market comparison)

    Returns dict with mean_spread_bps, median_spread_bps, spread_volatility.
    """
    df = prices.copy()
    if ticker is not None:
        df = df[df["ticker"] == ticker]

    if df.empty:
        return {"mean_spread_bps": np.nan, "median_spread_bps": np.nan, "spread_volatility": np.nan}

    yes_ask = df["yes_ask"].values
    yes_bid = df["yes_bid"].values
    mid = (yes_bid + yes_ask) / 2.0
    spread = yes_ask - yes_bid

    valid = ~np.isnan(spread) & ~np.isnan(mid) & (mid > 0)
    if not valid.any():
        return {"mean_spread_bps": np.nan, "median_spread_bps": np.nan, "spread_volatility": np.nan}

    rel_spread = spread[valid] / mid[valid]
    spread_bps = rel_spread * 10000

    return {
        "mean_spread_bps": float(np.mean(spread_bps)),
        "median_spread_bps": float(np.median(spread_bps)),
        "mean_relative_spread": float(np.mean(rel_spread)),
        "spread_volatility": float(np.std(spread_bps, ddof=1)),
    }


def detect_arbitrage(prices: pd.DataFrame, ticker: Optional[str] = None) -> list[dict]:
    """Detect arbitrage opportunities in binary markets.

    In a no-arbitrage binary market:
        P(yes) + P(no) = 1
        OR: yes_bid + no_bid <= 1 <= yes_ask + no_ask

    An arbitrage exists when:
        yes_bid > 1 - no_ask (long yes, short no is profitable)
        no_bid > 1 - yes_ask (long no, short yes is profitable)

    These represent market inefficiencies and prediction markets
    that fail no-arbitrage should be flagged.
    """
    df = prices.copy()
    if ticker is not None:
        df = df[df["ticker"] == ticker]

    if df.empty:
        return []

    arb_opportunities: list[dict] = []
    for _, row in df.iterrows():
        yb = row.get("yes_bid")
        ya = row.get("yes_ask")
        nb = row.get("no_bid")
        na = row.get("no_ask")

        if any(v is None for v in [yb, ya, nb, na]):
            continue

        mid_yes = (yb + ya) / 2.0
        mid_no = (nb + na) / 2.0
        implied_sum = mid_yes + mid_no

        if abs(implied_sum - 1.0) > 0.02:
            arb_opportunities.append(
                {
                    "ticker": row.get("ticker"),
                    "timestamp": row.get("timestamp"),
                    "yes_mid": mid_yes,
                    "no_mid": mid_no,
                    "implied_sum": implied_sum,
                    "arb_size": abs(implied_sum - 1.0),
                    "type": "overpriced" if implied_sum > 1.0 else "underpriced",
                }
            )

    return arb_opportunities


def compute_market_depth_ratio(prices: pd.DataFrame, ticker: Optional[str] = None) -> float:
    """Compute market depth ratio: open_interest / volume.

    Higher depth/volume ratios indicate deeper markets with
    more committed capital. Low ratios suggest transient liquidity
    and less reliable price signals.
    """
    df = prices.copy()
    if ticker is not None:
        df = df[df["ticker"] == ticker]

    if df.empty:
        return np.nan

    oi = df["open_interest"].fillna(0).values
    vol = df["volume"].fillna(0).values

    valid = vol > 0
    if not valid.any():
        return np.nan

    ratios = oi[valid] / vol[valid]
    return float(np.median(ratios))


def compute_price_discovery_ratio(prices: pd.DataFrame, ticker: Optional[str] = None) -> float:
    """Compute price discovery ratio: change in mid / change in last.

    Measures how much information is incorporated into quoted prices
    vs. trade prices. Values near 1 suggest quotes reflect all available info.
    Values < 1 suggest trades contain additional information (asymmetric info).
    """
    df = prices.copy()
    if ticker is not None:
        df = df[df["ticker"] == ticker]

    if df.empty or len(df) < 2:
        return np.nan

    df = df.sort_values("timestamp")
    mid = (df["yes_bid"].values + df["yes_ask"].values) / 2.0
    last = df["last_price"].values

    delta_mid = np.diff(mid)
    delta_last = np.diff(last)

    var_last = np.var(delta_last)
    if var_last == 0:
        return np.nan

    cov = np.cov(delta_mid, delta_last)
    ratio = cov[0, 1] / var_last if var_last > 0 else np.nan
    return float(np.clip(ratio, 0, 1))


def compute_all_microstructure_metrics(
    prices: pd.DataFrame,
    tickers: Optional[list[str]] = None,
) -> list[MicrostructureMetrics]:
    """Compute complete microstructure analysis for all tickers."""
    if prices.empty:
        return []
    if tickers is None:
        tickers = prices["ticker"].unique().tolist()

    results: list[MicrostructureMetrics] = []
    for ticker in tickers:
        sub = prices[prices["ticker"] == ticker]
        if sub.empty:
            continue

        vwap_prob, vwap_vol = compute_vwap_probability(prices, ticker)
        spread = compute_spread_metrics(prices, ticker)
        arb_opps = detect_arbitrage(prices, ticker)
        depth = compute_market_depth_ratio(prices, ticker)
        pdr = compute_price_discovery_ratio(prices, ticker)

        total_obs = len(sub)
        arb_count = len(arb_opps)
        arb_rate = arb_count / total_obs if total_obs > 0 else 0.0

        mean_vol = float(sub["volume"].mean()) if "volume" in sub.columns else 0.0
        total_vol = float(sub["volume"].sum()) if "volume" in sub.columns else 0.0
        mean_oi = float(sub["open_interest"].mean()) if "open_interest" in sub.columns else 0.0

        results.append(
            MicrostructureMetrics(
                ticker=ticker,
                n_observations=total_obs,
                mean_mid_price=vwap_prob,
                mean_spread_bps=spread.get("mean_spread_bps", np.nan),
                median_spread_bps=spread.get("median_spread_bps", np.nan),
                mean_relative_spread=spread.get("mean_relative_spread", np.nan),
                spread_volatility=spread.get("spread_volatility", np.nan),
                vwap_probability=vwap_prob,
                vwap_volume=vwap_vol,
                arbitrage_opportunities=arb_count,
                arbitrage_rate=arb_rate,
                depth_ratio=depth,
                price_discovery_ratio=pdr,
                mean_volume=mean_vol,
                total_volume=total_vol,
                mean_open_interest=mean_oi,
            )
        )

    return results
