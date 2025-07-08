from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from macro_engine.config.settings import EngineConfig, get_settings

# ---------------------------------------------------------------------------
# FRED Data
# ---------------------------------------------------------------------------

FRED_BASE = "https://api.stlouisfed.org/fred"

_FRED_SERIES_MAP: dict[str, dict[str, Any]] = {
    "CPIAUCSL": {"name": "CPI All Urban Consumers", "freq": "monthly", "category": "inflation"},
    "CPILFESL": {
        "name": "CPI Core (less food & energy)",
        "freq": "monthly",
        "category": "inflation",
    },
    "UNRATE": {"name": "Unemployment Rate", "freq": "monthly", "category": "employment"},
    "PAYEMS": {"name": "Nonfarm Payrolls", "freq": "monthly", "category": "employment"},
    "GDP": {"name": "Gross Domestic Product", "freq": "quarterly", "category": "growth"},
    "GDPC1": {"name": "Real GDP", "freq": "quarterly", "category": "growth"},
    "PCEPI": {"name": "PCE Price Index", "freq": "monthly", "category": "inflation"},
    "PCEPILFE": {"name": "Core PCE Price Index", "freq": "monthly", "category": "inflation"},
    "FEDFUNDS": {"name": "Federal Funds Rate", "freq": "monthly", "category": "policy"},
    "DGS10": {"name": "10-Year Treasury Yield", "freq": "daily", "category": "rates"},
    "DGS2": {"name": "2-Year Treasury Yield", "freq": "daily", "category": "rates"},
    "T10YIE": {"name": "10-Year Breakeven Inflation", "freq": "daily", "category": "rates"},
    "T5YIE": {"name": "5-Year Breakeven Inflation", "freq": "daily", "category": "rates"},
    "SP500": {"name": "S&P 500 Index", "freq": "daily", "category": "financial"},
    "VIXCLS": {"name": "CBOE Volatility Index", "freq": "daily", "category": "financial"},
    "RECPROUSM156N": {
        "name": "Recession Probability (Smooth)",
        "freq": "monthly",
        "category": "growth",
    },
}


def fetch_fred_series(
    series_id: str,
    api_key: Optional[str] = None,
    observation_start: str = "2019-01-01",
    observation_end: str = "",
) -> pd.DataFrame:
    """Fetch a single FRED series."""
    key = api_key or get_settings().fred_api_key
    if not key:
        warnings.warn("No FRED API key set. Returning empty DataFrame.")
        return pd.DataFrame()

    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": observation_start,
    }
    if observation_end:
        params["observation_end"] = observation_end

    url = f"{FRED_BASE}/series/observations"
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for obs in data.get("observations", []):
            val = obs.get("value")
            if val and val != ".":
                rows.append(
                    {
                        "date": obs["date"],
                        "series_id": series_id,
                        "value": float(val),
                        "name": _FRED_SERIES_MAP.get(series_id, {}).get("name", series_id),
                        "category": _FRED_SERIES_MAP.get(series_id, {}).get("category", ""),
                        "frequency": _FRED_SERIES_MAP.get(series_id, {}).get("freq", ""),
                    }
                )
        return pd.DataFrame(rows)
    except Exception as e:
        warnings.warn(f"Failed to fetch FRED series {series_id}: {e}")
        return pd.DataFrame()


def download_fred_data(
    series_ids: Optional[list[str]] = None,
    config: Optional[EngineConfig] = None,
) -> pd.DataFrame:
    """Download multiple FRED series."""
    if series_ids is None:
        series_ids = list(_FRED_SERIES_MAP.keys())
    all_dfs: list[pd.DataFrame] = []
    for sid in series_ids:
        df = fetch_fred_series(sid)
        if not df.empty:
            all_dfs.append(df)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# BLS Data
# ---------------------------------------------------------------------------

BLS_BASE = "https://api.bls.gov/publicAPI/v2"


def _bls_request(
    series_ids: list[str],
    start_year: str = "2019",
    end_year: str = "2025",
    api_key: Optional[str] = None,
) -> list[dict]:
    """Make a BLS API request for multiple series."""
    key = api_key or get_settings().bls_api_key
    payload: dict[str, Any] = {
        "seriesid": series_ids,
        "startyear": start_year,
        "endyear": end_year,
        "registrationKey": key or "",
    }
    try:
        resp = requests.post(f"{BLS_BASE}/timeseries/data/", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("Results", {}).get("series", [])
    except Exception as e:
        warnings.warn(f"BLS API request failed: {e}")
        return []


def _parse_bls_series(series_data: dict) -> pd.DataFrame:
    rows = []
    sid = series_data.get("seriesID", "")
    for item in series_data.get("data", []):
        try:
            val = float(item["value"])
            year = item["year"]
            period = item["period"]
            if period == "M13":
                continue
            month = period.replace("M", "")
            date_str = f"{year}-{month}-01"
            rows.append(
                {
                    "series_id": sid,
                    "date": date_str,
                    "value": val,
                    "footnote": item.get("footnotes", [{}])[0].get("text", ""),
                }
            )
        except (ValueError, KeyError):
            continue
    return pd.DataFrame(rows)


def fetch_bls_cpi(api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch CPI data from BLS."""
    series = _bls_request(["CUUR0000SA0", "CUUR0000SA0L1E"], api_key=api_key)
    dfs = [_parse_bls_series(s) for s in series]
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        result["name"] = "CPI"
        result["category"] = "inflation"
        return result
    return pd.DataFrame()


def fetch_bls_jobs(api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch nonfarm payrolls from BLS."""
    series = _bls_request(["CES0000000001"], api_key=api_key)
    dfs = [_parse_bls_series(s) for s in series]
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        result["name"] = "Nonfarm Payrolls"
        result["category"] = "employment"
        return result
    return pd.DataFrame()


def fetch_bls_unemployment(api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch unemployment rate from BLS."""
    series = _bls_request(["LNS14000000"], api_key=api_key)
    dfs = [_parse_bls_series(s) for s in series]
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        result["name"] = "Unemployment Rate"
        result["category"] = "employment"
        return result
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# BEA Data (simulated via FRED for now)
# ---------------------------------------------------------------------------


def fetch_bea_gdp(api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch GDP data (uses FRED as proxy for BEA)."""
    return fetch_fred_series("GDPC1", api_key=api_key)


def fetch_bea_pce(api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch PCE data (uses FRED as proxy for BEA)."""
    return fetch_fred_series("PCEPI", api_key=api_key)


# ---------------------------------------------------------------------------
# Combined macro dataset
# ---------------------------------------------------------------------------


def build_macro_dataset(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    """Build a combined dataset of official macro releases."""
    cfg = config or get_settings()
    all_dfs: list[pd.DataFrame] = []

    fred_df = download_fred_data(config=cfg)
    if not fred_df.empty:
        all_dfs.append(fred_df)

    if cfg.bls_api_key:
        cpi = fetch_bls_cpi(cfg.bls_api_key)
        jobs = fetch_bls_jobs(cfg.bls_api_key)
        unemp = fetch_bls_unemployment(cfg.bls_api_key)
        for df in [cpi, jobs, unemp]:
            if not df.empty:
                all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        return combined.sort_values("date").reset_index(drop=True)
    return pd.DataFrame()


def save_macro_data(df: pd.DataFrame, config: Optional[EngineConfig] = None) -> Path:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.macro_data_file
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_macro_data(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / cfg.macro_data_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
