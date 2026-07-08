"""Signal-based volatility forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


def _validate_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if features.empty:
        raise ValueError("features cannot be empty.")
    clean = features.astype(float)
    if clean.isna().any().any():
        raise ValueError("features contains missing values.")
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("features must contain only finite values.")
    return clean


def _validate_target(target: pd.Series, index: pd.Index) -> pd.Series:
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series.")
    aligned = target.reindex(index).astype(float)
    if aligned.isna().any():
        raise ValueError("target is missing one or more feature dates.")
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("target must contain only finite values.")
    if (aligned <= 0).any():
        raise ValueError("volatility targets must be strictly positive.")
    return aligned


@dataclass
class RidgeVolatilityForecaster:
    """Ridge regression for positive volatility forecasts.

    By default the model predicts log-volatility.  This guarantees positive
    forecasts after exponentiation and usually stabilizes the target variance.
    Features are standardized using training-sample statistics, and the
    intercept is not penalized.
    """

    alpha: float = 1.0
    log_target: bool = True
    minimum_volatility: float = 1e-4

    feature_names_: tuple[str, ...] | None = None
    feature_mean_: pd.Series | None = None
    feature_scale_: pd.Series | None = None
    intercept_: float | None = None
    coefficients_: pd.Series | None = None

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative.")
        if self.minimum_volatility <= 0:
            raise ValueError("minimum_volatility must be positive.")

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> "RidgeVolatilityForecaster":
        """Fit the forecaster on one chronological training sample."""

        x = _validate_feature_frame(features)
        y = _validate_target(target, x.index)

        mean = x.mean(axis=0)
        scale = x.std(axis=0, ddof=0).replace(0.0, 1.0)
        z = (x - mean) / scale

        response = np.log(y.to_numpy()) if self.log_target else y.to_numpy()
        design = np.column_stack([np.ones(len(z)), z.to_numpy()])

        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        gram = design.T @ design + penalty
        rhs = design.T @ response

        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram) @ rhs

        self.feature_names_ = tuple(x.columns)
        self.feature_mean_ = mean
        self.feature_scale_ = scale
        self.intercept_ = float(beta[0])
        self.coefficients_ = pd.Series(
            beta[1:],
            index=x.columns,
            name="coefficient",
        )
        return self

    def _check_fitted(self) -> None:
        if (
            self.feature_names_ is None
            or self.feature_mean_ is None
            or self.feature_scale_ is None
            or self.intercept_ is None
            or self.coefficients_ is None
        ):
            raise RuntimeError("The model must be fitted before prediction.")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return positive annualized volatility forecasts."""

        self._check_fitted()
        x = _validate_feature_frame(features)
        expected = list(self.feature_names_ or ())
        missing = [column for column in expected if column not in x.columns]
        unexpected = [column for column in x.columns if column not in expected]
        if missing or unexpected:
            raise ValueError(
                "Prediction columns must match training columns exactly. "
                f"Missing={missing}; unexpected={unexpected}."
            )

        x = x.loc[:, expected]
        z = (x - self.feature_mean_) / self.feature_scale_
        linear_prediction = self.intercept_ + z @ self.coefficients_
        prediction = (
            np.exp(linear_prediction)
            if self.log_target
            else linear_prediction
        )
        result = pd.Series(
            np.maximum(prediction, self.minimum_volatility),
            index=x.index,
            name="predicted_volatility",
            dtype=float,
        )
        return result

    def coefficient_table(self) -> pd.DataFrame:
        """Return standardized coefficients ordered by absolute magnitude."""

        self._check_fitted()
        table = self.coefficients_.to_frame()
        table["absolute_coefficient"] = table["coefficient"].abs()
        return table.sort_values("absolute_coefficient", ascending=False)


def expanding_window_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    initial_train_size: int,
    refit_every: int = 20,
    model_factory: Callable[[], RidgeVolatilityForecaster] | None = None,
) -> pd.Series:
    """Generate chronological expanding-window forecasts without look-ahead.

    The first prediction is produced at ``initial_train_size``.  The model is
    refitted every ``refit_every`` observations and predicts only dates that
    occur after its training sample.
    """

    x = _validate_feature_frame(features)
    y = _validate_target(target, x.index)

    if initial_train_size < 2:
        raise ValueError("initial_train_size must be at least 2.")
    if initial_train_size >= len(x):
        raise ValueError("initial_train_size must be smaller than the dataset.")
    if refit_every < 1:
        raise ValueError("refit_every must be positive.")

    factory = model_factory or (lambda: RidgeVolatilityForecaster())
    predictions = pd.Series(
        np.nan,
        index=x.index,
        name="predicted_volatility",
        dtype=float,
    )

    model: RidgeVolatilityForecaster | None = None
    for position in range(initial_train_size, len(x)):
        should_refit = model is None or (
            (position - initial_train_size) % refit_every == 0
        )
        if should_refit:
            model = factory()
            model.fit(x.iloc[:position], y.iloc[:position])

        predictions.iloc[position] = model.predict(
            x.iloc[[position]]
        ).iloc[0]

    return predictions
