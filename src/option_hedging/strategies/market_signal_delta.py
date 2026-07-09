"""Market-signal adjusted Black-Scholes delta hedging utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Real

import numpy as np
import pandas as pd


RiskState = str

NORMAL: RiskState = "normal"
ELEVATED: RiskState = "elevated"
STRESS: RiskState = "stress"
RISK_STATES: tuple[RiskState, ...] = (NORMAL, ELEVATED, STRESS)


def _validate_features(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if features.empty:
        raise ValueError("features cannot be empty.")
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise KeyError(f"features is missing required columns: {', '.join(missing)}")

    clean = features.loc[:, columns].astype(float)
    if clean.isna().any().any():
        raise ValueError("features contains missing values.")
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("features must contain only finite values.")
    return clean


def _validate_base_volatility(
    base_volatility: float | pd.Series,
    index: pd.Index,
) -> pd.Series:
    if isinstance(base_volatility, Real):
        value = float(base_volatility)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("base_volatility must be finite and strictly positive.")
        return pd.Series(value, index=index, name="base_volatility")

    if not isinstance(base_volatility, pd.Series):
        raise TypeError("base_volatility must be a positive number or pandas Series.")

    aligned = base_volatility.astype(float).reindex(index)
    if aligned.isna().any():
        raise ValueError("base_volatility is missing one or more feature dates.")
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("base_volatility must contain only finite values.")
    if (aligned <= 0).any():
        raise ValueError("base_volatility must be strictly positive.")
    aligned.name = "base_volatility"
    return aligned


@dataclass
class MarketSignalRiskModel:
    """Convert point-in-time market signals into hedge risk states.

    The model is intentionally simple and transparent: features are standardized
    on the training sample, combined through user-supplied weights, and mapped
    into normal/elevated/stress states using training-sample score quantiles.
    """

    weights: Mapping[str, float]
    elevated_quantile: float = 0.70
    stress_quantile: float = 0.90
    volatility_multipliers: Mapping[RiskState, float] = field(
        default_factory=lambda: {
            NORMAL: 1.00,
            ELEVATED: 1.20,
            STRESS: 1.50,
        }
    )

    feature_names_: tuple[str, ...] | None = None
    feature_mean_: pd.Series | None = None
    feature_scale_: pd.Series | None = None
    elevated_threshold_: float | None = None
    stress_threshold_: float | None = None

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weights cannot be empty.")
        for name, value in self.weights.items():
            if not isinstance(name, str) or not name:
                raise ValueError("weight names must be nonempty strings.")
            if not math.isfinite(float(value)):
                raise ValueError("all weights must be finite.")
        if not 0 < self.elevated_quantile < self.stress_quantile < 1:
            raise ValueError(
                "quantiles must satisfy 0 < elevated_quantile < "
                "stress_quantile < 1."
            )
        missing_states = set(RISK_STATES).difference(self.volatility_multipliers)
        if missing_states:
            raise KeyError(
                "volatility_multipliers is missing state(s): "
                + ", ".join(sorted(missing_states))
            )
        for state in RISK_STATES:
            value = float(self.volatility_multipliers[state])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("volatility multipliers must be positive.")

    @property
    def feature_names(self) -> list[str]:
        """Return feature names in deterministic weight order."""

        return list(self.weights.keys())

    def fit(self, features: pd.DataFrame) -> "MarketSignalRiskModel":
        """Fit standardization and risk-state thresholds on training features."""

        x = _validate_features(features, self.feature_names)
        self.feature_names_ = tuple(self.feature_names)
        self.feature_mean_ = x.mean(axis=0)
        self.feature_scale_ = x.std(axis=0, ddof=0).replace(0.0, 1.0)

        training_scores = self.score(features)
        self.elevated_threshold_ = float(
            training_scores.quantile(self.elevated_quantile)
        )
        self.stress_threshold_ = float(training_scores.quantile(self.stress_quantile))
        return self

    def _check_fitted(self) -> None:
        if (
            self.feature_names_ is None
            or self.feature_mean_ is None
            or self.feature_scale_ is None
            or self.elevated_threshold_ is None
            or self.stress_threshold_ is None
        ):
            raise RuntimeError("The model must be fitted before use.")

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Return standardized weighted risk scores."""

        if self.feature_mean_ is None or self.feature_scale_ is None:
            x = _validate_features(features, self.feature_names)
            mean = x.mean(axis=0)
            scale = x.std(axis=0, ddof=0).replace(0.0, 1.0)
        else:
            x = _validate_features(features, list(self.feature_names_ or ()))
            mean = self.feature_mean_
            scale = self.feature_scale_

        standardized = (x - mean) / scale
        weights = pd.Series(self.weights, dtype=float).reindex(standardized.columns)
        scores = standardized.mul(weights, axis=1).sum(axis=1)
        scores.name = "risk_score"
        return scores

    def classify(self, features: pd.DataFrame) -> pd.Series:
        """Classify features into normal, elevated, and stress states."""

        self._check_fitted()
        scores = self.score(features)
        states = pd.Series(NORMAL, index=scores.index, name="risk_state", dtype=object)
        states.loc[scores >= float(self.elevated_threshold_)] = ELEVATED
        states.loc[scores >= float(self.stress_threshold_)] = STRESS
        return states

    def multipliers(self, states: pd.Series) -> pd.Series:
        """Map risk states to volatility multipliers."""

        if not isinstance(states, pd.Series):
            raise TypeError("states must be a pandas Series.")
        invalid = sorted(set(states.dropna()).difference(RISK_STATES))
        if invalid:
            raise ValueError(f"unknown risk state(s): {', '.join(invalid)}")

        mapped = states.map(
            {
                state: float(self.volatility_multipliers[state])
                for state in RISK_STATES
            }
        )
        if mapped.isna().any():
            raise ValueError("states contains missing values.")
        mapped.name = "volatility_multiplier"
        return mapped.astype(float)

    def adjusted_volatility(
        self,
        features: pd.DataFrame,
        base_volatility: float | pd.Series,
    ) -> pd.Series:
        """Return the MSA-Delta volatility path for hedge dates."""

        states = self.classify(features)
        base = _validate_base_volatility(base_volatility, states.index)
        adjusted = base * self.multipliers(states)
        adjusted.name = "msa_delta_volatility"
        return adjusted

    def diagnostic_frame(
        self,
        features: pd.DataFrame,
        base_volatility: float | pd.Series,
    ) -> pd.DataFrame:
        """Return score, state, multiplier, and adjusted volatility by date."""

        self._check_fitted()
        scores = self.score(features)
        states = self.classify(features)
        multipliers = self.multipliers(states)
        base = _validate_base_volatility(base_volatility, states.index)
        adjusted = base * multipliers
        adjusted.name = "msa_delta_volatility"
        return pd.concat([scores, states, multipliers, base, adjusted], axis=1)
