"""Multi-factor cross-asset surprise attribution model."""

from macro_engine.factors.model import (
    SurpriseFactorModel,
    compute_car_test,
    compute_cumulative_abnormal_returns,
    compute_factor_attribution,
    load_factor_attribution,
    save_factor_attribution,
)

__all__ = [
    "SurpriseFactorModel",
    "compute_factor_attribution",
    "compute_cumulative_abnormal_returns",
    "compute_car_test",
    "save_factor_attribution",
    "load_factor_attribution",
]
