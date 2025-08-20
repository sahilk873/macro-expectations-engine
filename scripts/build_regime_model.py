#!/usr/bin/env python3
"""Phase 10: Build macro regime model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macro_engine.config.settings import get_settings
from macro_engine.macro.sources import load_macro_data
from macro_engine.prices.providers import load_price_data
from macro_engine.regime.model import compute_macro_regime, save_regime_classifications


def main():
    cfg = get_settings()

    print("Loading data...")
    macro_data = load_macro_data()
    price_data = load_price_data()

    if macro_data.empty and price_data.empty:
        print("No macro or price data available.")
        return

    print("Computing macro regime classifications...")
    regimes = compute_macro_regime(macro_data, price_data, config=cfg)

    if not regimes.empty:
        path = save_regime_classifications(regimes)
        print(f"Saved {len(regimes)} regime classifications to {path}")
        print(f"Growth regimes: {regimes['growth_regime'].value_counts().to_dict()}")
        print(f"Inflation regimes: {regimes['inflation_regime'].value_counts().to_dict()}")
        print(f"Risk regimes: {regimes['risk_regime'].value_counts().to_dict()}")
    else:
        print("No regime classifications computed")

    print("Done.")


if __name__ == "__main__":
    main()
