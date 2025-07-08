"""Official macro data fetchers (BLS, BEA, FRED, Fed, Treasury)."""

from macro_engine.macro.sources import (
    build_macro_dataset,
    download_fred_data,
    fetch_bea_gdp,
    fetch_bea_pce,
    fetch_bls_cpi,
    fetch_bls_jobs,
    fetch_bls_unemployment,
    fetch_fred_series,
    load_macro_data,
    save_macro_data,
)

__all__ = [
    "fetch_bls_cpi",
    "fetch_bls_jobs",
    "fetch_bls_unemployment",
    "fetch_bea_gdp",
    "fetch_bea_pce",
    "fetch_fred_series",
    "download_fred_data",
    "build_macro_dataset",
    "save_macro_data",
    "load_macro_data",
]
