"""Multi-factor cross-asset surprise attribution model."""

from macro_engine.factors.model import (
    SurpriseFactorModel,
    compute_factor_attribution,
    compute_cumulative_abnormal_returns,
    compute_car_test,
)

__all__ = [
    "SurpriseFactorModel",
    "compute_factor_attribution",
    "compute_cumulative_abnormal_returns",
    "compute_car_test",
]
