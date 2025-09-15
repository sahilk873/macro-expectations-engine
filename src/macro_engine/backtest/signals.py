from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SurpriseSignal:
    event_id: str
    event_type: str
    event_time: datetime
    standardized_surprise: float
    raw_surprise: float
    direction: str
    risk_label: str
    qualitative: str
    confidence: float
    level_component: float
    uncertainty_revision: float
    is_confounded: bool
    effective_weight: float = 1.0


def build_surprise_signals(
    surprises: pd.DataFrame,
    min_abs_standardized: float = 0.5,
) -> list[SurpriseSignal]:
    """Convert raw macro surprise data into structured signal objects.

    Filters to surprises exceeding the minimum standardized threshold
    and attaches all available metadata for downstream use.
    """
    if surprises.empty:
        return []

    signals: list[SurpriseSignal] = []
    for _, row in surprises.iterrows():
        std_surp = row.get("standardized_surprise", 0)
        if pd.isna(std_surp) or abs(std_surp) < min_abs_standardized:
            continue

        event_time = row.get("event_time")
        if isinstance(event_time, str):
            event_time = datetime.fromisoformat(str(event_time).replace("Z", "+00:00"))
        if event_time is None:
            continue

        confidence = row.get("entropy_confidence", 1.0)
        if pd.isna(confidence):
            confidence = 1.0

        signals.append(
            SurpriseSignal(
                event_id=str(row.get("event_id", "")),
                event_type=str(row.get("event_type", "")),
                event_time=event_time,
                standardized_surprise=float(std_surp),
                raw_surprise=float(row.get("raw_surprise", 0)),
                direction=str(row.get("direction", "")),
                risk_label=str(row.get("risk_label", "neutral")),
                qualitative=str(row.get("qualitative", "")),
                confidence=float(confidence),
                level_component=float(row.get("level_component", 0)),
                uncertainty_revision=float(row.get("uncertainty_revision", 0)),
                is_confounded=bool(row.get("confounded", False)),
            )
        )

    return signals


def get_active_signals(
    signals: list[SurpriseSignal],
    current_date: datetime,
    lookback_days: int = 5,
    decay_days: int = 5,
) -> list[SurpriseSignal]:
    """Return signals active as of *current_date* with time-decayed weights.

    A signal is active if its event occurred within ``lookback_days``
    before *current_date*.  Its ``effective_weight`` is linearly decayed
    from 1.0 (event day) to 0.0 (``decay_days`` after the event).
    """
    active: list[SurpriseSignal] = []
    for signal in signals:
        if signal.event_time is None:
            continue
        days_since = (current_date - signal.event_time).days
        if days_since < 0 or days_since > lookback_days:
            continue
        import copy

        s = copy.copy(signal)
        s.effective_weight = max(0.0, 1.0 - days_since / max(decay_days, 1))
        active.append(s)
    return active


def build_surprise_tilts(
    active_signals: list[SurpriseSignal],
    factor_attribution: Optional[pd.DataFrame] = None,
    tilt_strength: float = 1.0,
    min_factor_alpha: float = 0.001,
    min_confidence: float = 0.3,
) -> dict[str, float]:
    """Aggregate all active surprise signals into a single set of per-ticker tilts.

    Two sources of tilts:

    1. **Risk-regime tilts** – based on each surprise's ``risk_label``.
       Risk-off surprises scale down risky assets and scale up safe havens.
       Risk-on surprises do the opposite.

    2. **Factor-model tilts** – when ``factor_attribution`` is provided,
       the tilt for a ticker is proportional to the product of the surprise's
       standardized magnitude and the estimated factor alpha for that
       (surprise_type, ticker) pair.  This connects the strategy directly
       to the project's core research output.

    All tilts are multiplied by ``effective_weight`` (time decay) and
    ``tilt_strength`` (global scaling parameter).
    """
    combined: dict[str, float] = {}

    for signal in active_signals:
        if signal.confidence < min_confidence:
            continue
        decay = signal.effective_weight
        magnitude = abs(signal.standardized_surprise)
        direction = np.sign(signal.standardized_surprise)
        strength = tilt_strength * decay

        # --- 1. Risk-regime tilts ---
        if signal.risk_label == "risk_off":
            for t in ["SPY", "QQQ", "IWM", "HYG", "USO"]:
                combined[t] = combined.get(t, 0.0) - 0.05 * magnitude * strength
            for t in ["TLT", "IEF", "GLD"]:
                combined[t] = combined.get(t, 0.0) + 0.05 * magnitude * strength
        elif signal.risk_label == "risk_on":
            for t in ["SPY", "QQQ", "IWM"]:
                combined[t] = combined.get(t, 0.0) + 0.03 * magnitude * strength
            for t in ["TLT", "IEF"]:
                combined[t] = combined.get(t, 0.0) - 0.03 * magnitude * strength

        # --- 2. Factor-model tilts ---
        if factor_attribution is not None and not factor_attribution.empty:
            fa = factor_attribution[factor_attribution["surprise_type"] == signal.event_type]
            for _, row in fa.iterrows():
                ticker = row["ticker"]
                alpha = row.get("total_effect", 0)
                if abs(alpha) < min_factor_alpha:
                    continue
                combined[ticker] = (
                    combined.get(ticker, 0.0) + alpha * direction * magnitude * strength
                )

    return combined
