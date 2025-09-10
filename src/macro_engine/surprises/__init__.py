"""Macro surprise computation, labeling, and decomposition."""

from macro_engine.surprises.calculator import (
    compute_all_surprises,
    compute_raw_surprise,
    compute_standardized_surprise,
    label_surprise,
    load_surprises,
    save_surprises,
)
from macro_engine.surprises.decomposition import (
    compute_entropy_based_confidence,
    decompose_all_surprises,
    decompose_surprise,
)

__all__ = [
    "compute_raw_surprise",
    "compute_standardized_surprise",
    "compute_all_surprises",
    "label_surprise",
    "save_surprises",
    "load_surprises",
    "decompose_surprise",
    "decompose_all_surprises",
    "compute_entropy_based_confidence",
]
