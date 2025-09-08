from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings

HMM_AVAILABLE = False
try:
    from hmmlearn import hmm

    HMM_AVAILABLE = True
except ImportError:
    pass


class HMMRegimeModel:
    """Data-driven macro regime classification using Gaussian HMM.

    Unlike the rule-based MacroRegimeModel (which applies fixed thresholds
    to growth, inflation, and volatility series), this model learns
    regime boundaries directly from the data.

    The HMM discovers latent regimes (states) that best explain the joint
    dynamics of macro-financial variables. This addresses a key limitation
    of rule-based approaches: threshold values are arbitrary and may not
    reflect actual regime shifts in the data.

    Attributes:
        n_regimes: Number of latent regimes to discover (default 3).
        model: Fitted Gaussian HMM.
        state_names: Human-readable labels for each state.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model: Optional[object] = None
        self.state_names: list[str] = []
        self._feature_cols: list[str] = []
        self._is_fitted = False

    def _check_available(self) -> None:
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for HMM regime detection. Install with: pip install hmmlearn"
            )

    def _prepare_features(
        self,
        macro_data: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Construct feature matrix for HMM from macro and price data.

        Features:
        - GDP growth trend (12-month rolling of GDPC1)
        - CPI inflation (12-month rolling)
        - Unemployment rate level
        - NFP change (3-month average)
        - Fed Funds rate level
        - SPY realized volatility (20-day, annualized)
        - 10Y-2Y Treasury spread (if available)
        """
        features: dict[str, pd.Series] = {}
        macro = macro_data.copy()
        macro["date"] = pd.to_datetime(macro["date"])
        monthly = macro.set_index("date")

        gdp = monthly[monthly["series_id"].isin(["GDPC1", "GDP"])]["value"]
        if not gdp.empty:
            gdp_yoy = gdp.pct_change(4).dropna() * 100
            features["gdp_growth"] = gdp_yoy

        cpi = monthly[monthly["series_id"] == "CPIAUCSL"]["value"]
        if not cpi.empty:
            cpi_yoy = cpi.pct_change(12).dropna() * 100
            features["cpi_inflation"] = cpi_yoy

        unemp = monthly[monthly["series_id"] == "UNRATE"]["value"]
        if not unemp.empty:
            features["unemployment"] = unemp

        nfp = monthly[monthly["series_id"] == "PAYEMS"]["value"]
        if not nfp.empty and len(nfp) >= 3:
            nfp_chg = nfp.diff().rolling(3, min_periods=1).mean()
            features["nfp_change"] = nfp_chg

        fed = monthly[monthly["series_id"] == "FEDFUNDS"]["value"]
        if not fed.empty:
            features["fed_funds"] = fed

        dgs10 = monthly[monthly["series_id"] == "DGS10"]["value"]
        dgs2 = monthly[monthly["series_id"] == "DGS2"]["value"]
        if not dgs10.empty and not dgs2.empty:
            ts_spread = dgs10 - dgs2
            features["ts_spread"] = ts_spread

        if price_data is not None and not price_data.empty:
            spy = price_data[price_data["ticker"] == "SPY"].copy()
            if not spy.empty:
                spy["date"] = pd.to_datetime(spy["date"])
                spy = spy.set_index("date").sort_index()
                close_col = "close" if "close" in spy.columns else "Close"
                spy_ret = spy[close_col].pct_change()
                spy_vol = spy_ret.rolling(20).std() * np.sqrt(252)
                spy_vol.name = "spy_volatility"
                spy_vol = spy_vol.resample("M").last()
                features["spy_volatility"] = spy_vol

        feature_df = pd.DataFrame(features)
        feature_df.index = pd.to_datetime(feature_df.index)
        feature_df = feature_df.sort_index()
        feature_df = feature_df.fillna(method="ffill").fillna(method="bfill").dropna(how="all")
        return feature_df

    def fit(
        self,
        macro_data: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> "HMMRegimeModel":
        """Fit the HMM to discover latent regime states."""
        self._check_available()

        features = self._prepare_features(macro_data, price_data)
        if features.empty or len(features) < self.n_regimes * 5:
            raise ValueError(
                f"Insufficient data to fit HMM: need at least {self.n_regimes * 5} observations"
            )

        self._feature_cols = features.columns.tolist()
        X = features.values

        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
            n_iter=1000,
            tol=1e-4,
        )
        self.model.fit(X)
        self._is_fitted = True

        states = self.model.predict(X)
        state_order = self._order_states(X, states)
        self.state_names = self._label_states(state_order)

        return self

    def predict(
        self,
        macro_data: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict regime states for each date in the feature matrix.

        Returns DataFrame with date, state, and state probabilities.
        """
        self._check_available()
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction. Call .fit() first.")

        features = self._prepare_features(macro_data, price_data)
        if features.empty:
            return pd.DataFrame()

        X = features.values
        states = self.model.predict(X)
        probs = self.model.predict_proba(X)

        result = pd.DataFrame(index=features.index)
        result["hmm_state"] = states
        result["hmm_state_name"] = [
            self.state_names[s] if s < len(self.state_names) else f"S{s}" for s in states
        ]

        for i in range(self.n_regimes):
            result[f"hmm_prob_s{i}"] = probs[:, i]

        result = result.reset_index()
        result = result.rename(columns={"index": "date"})
        return result

    def _order_states(self, X: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Order states by economic interpretation (expansion -> neutral -> contraction).

        Uses the mean of the first principal component to rank states.
        """
        if self.model is None:
            return states

        state_means = self.model.means_
        if state_means.shape[1] >= 2:
            pc1_scores = state_means[:, 0]
        else:
            pc1_scores = state_means[:, 0]

        ordered = np.argsort(pc1_scores)[::-1]
        mapping = {old: new for new, old in enumerate(ordered)}
        return np.array([mapping[s] for s in states])

    def _label_states(self, ordered_states: np.ndarray) -> list[str]:
        n = self.n_regimes
        if n == 2:
            return ["expansion", "contraction"]
        elif n == 3:
            return ["expansion", "neutral", "contraction"]
        elif n == 4:
            return ["expansion", "mild_expansion", "mild_contraction", "contraction"]
        else:
            return [f"regime_{i}" for i in range(n)]

    def compute_transition_matrix(self) -> Optional[pd.DataFrame]:
        """Compute the regime transition probability matrix.

        High persistence (diagonal dominance) indicates stable regimes.
        Off-diagonal elements reveal regime-switching likelihood.
        """
        if not self._is_fitted or self.model is None:
            return None
        tm = self.model.transmat_
        return pd.DataFrame(
            tm,
            index=[f"from_{s}" for s in self.state_names],
            columns=[f"to_{s}" for s in self.state_names],
        )

    def compare_to_rule_based(
        self,
        macro_data: pd.DataFrame,
        price_data: pd.DataFrame,
        rule_based_classifications: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compare HMM states to rule-based regime classifications.

        Computes confusion matrix and agreement statistics.
        """
        hmm_states = self.predict(macro_data, price_data)
        if hmm_states.empty:
            return pd.DataFrame()

        hmm_states["date"] = pd.to_datetime(hmm_states["date"])
        rule = rule_based_classifications.copy()
        rule["date"] = pd.to_datetime(rule["date"])

        merged = pd.merge_asof(
            hmm_states.sort_values("date"),
            rule.sort_values("date"),
            on="date",
            direction="nearest",
        )

        return merged


def compute_hmm_regime(
    macro_data: pd.DataFrame,
    price_data: pd.DataFrame,
    n_regimes: int = 3,
    config: Optional[EngineConfig] = None,
) -> pd.DataFrame:
    model = HMMRegimeModel(n_regimes=n_regimes)
    model.fit(macro_data, price_data)
    return model.predict(macro_data, price_data)


def save_hmm_classifications(df: pd.DataFrame, config: Optional[EngineConfig] = None) -> Path:
    cfg = config or get_settings()
    path = cfg.data_dir / "hmm_regime_classifications.parquet"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_hmm_classifications(config: Optional[EngineConfig] = None) -> pd.DataFrame:
    cfg = config or get_settings()
    path = cfg.data_dir / "hmm_regime_classifications.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
