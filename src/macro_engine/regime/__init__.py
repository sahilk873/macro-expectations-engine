"""Macro regime classification models.

Provides both rule-based (MacroRegimeModel) and data-driven
(HMMRegimeModel) approaches to regime classification.
"""

from macro_engine.regime.hmm_model import (
    HMMRegimeModel,
    compute_hmm_regime,
    load_hmm_classifications,
    save_hmm_classifications,
)
from macro_engine.regime.model import (
    MacroRegimeModel,
    compute_macro_regime,
    load_regime_classifications,
    save_regime_classifications,
)

__all__ = [
    "MacroRegimeModel",
    "compute_macro_regime",
    "save_regime_classifications",
    "load_regime_classifications",
    "HMMRegimeModel",
    "compute_hmm_regime",
    "save_hmm_classifications",
    "load_hmm_classifications",
]
