"""Event-study analysis."""

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
]
