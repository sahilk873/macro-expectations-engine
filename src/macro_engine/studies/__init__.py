"""Event-study analysis with Bayesian shrinkage and confounding control."""

from macro_engine.studies.bayesian import (
    compute_bayes_factor,
    compute_sharpe_ratio_equivalent,
    empirical_bayes_shrinkage,
    neyman_confidence_intervals,
    neyman_event_study_aggregation,
)
from macro_engine.studies.confounding import (
    compute_confounding_robustness,
    detect_confounding_events,
    flag_confounded_surprises,
    residualize_event_returns,
)
from macro_engine.studies.event_study import (
    aggregate_event_study,
    compute_event_returns,
    load_event_studies,
    open_event_window,
    run_event_studies,
    save_event_studies,
)

__all__ = [
    "compute_event_returns",
    "aggregate_event_study",
    "run_event_studies",
    "save_event_studies",
    "load_event_studies",
    "open_event_window",
    "empirical_bayes_shrinkage",
    "neyman_confidence_intervals",
    "neyman_event_study_aggregation",
    "compute_bayes_factor",
    "compute_sharpe_ratio_equivalent",
    "detect_confounding_events",
    "flag_confounded_surprises",
    "compute_confounding_robustness",
    "residualize_event_returns",
]
