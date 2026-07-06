"""Historical volatility estimators.

The estimators in this module use observed return data only. Returns should be
expressed in decimal form; for example, 0.01 represents a one-percent return.
Volatility estimates are annualized by multiplying the sample standard
deviation by the square root of the annualization factor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


def _validate_returns(returns: pd.Series, *, minimum_length: int) -> pd.Series:
    """Return a clean float series after validating the input."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    clean = returns.astype(float)

    if clean.isna().any():
        raise ValueError(
            "returns contains missing values. Remove or handle them before fitting."
        )

    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("returns must contain only finite values.")

    if len(clean) < minimum_length:
        raise ValueError(
            f"At least {minimum_length} return observations are required."
        )

    return clean


def annualized_historical_volatility(
    returns: pd.Series,
    *,
    annualization_factor: int = 252,
    ddof: int = 1,
) -> float:
    """Estimate annualized volatility from a return sample.

    Parameters
    ----------
    returns:
        A pandas Series of periodic simple or log returns.
    annualization_factor:
        Number of return periods per year. Use 252 for daily trading data.
    ddof:
        Delta degrees of freedom used by the standard-deviation estimator.

    Returns
    -------
    float
        Annualized sample volatility.
    """
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive.")
    if ddof < 0:
        raise ValueError("ddof cannot be negative.")

    minimum_length = max(2, ddof + 1)
    clean = _validate_returns(returns, minimum_length=minimum_length)

    periodic_volatility = float(clean.std(ddof=ddof))
    return periodic_volatility * math.sqrt(annualization_factor)


@dataclass
class FixedHistoricalVolatility:
    """Estimate one volatility value and hold it fixed.

    The estimator uses the most recent ``lookback`` returns supplied to ``fit``.
    This will serve as the constant-volatility Black-Scholes benchmark.
    """

    lookback: int = 20
    annualization_factor: int = 252
    ddof: int = 1

    volatility_: float | None = None

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2.")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive.")
        if self.ddof < 0:
            raise ValueError("ddof cannot be negative.")
        if self.lookback <= self.ddof:
            raise ValueError("lookback must be greater than ddof.")

    def fit(self, returns: pd.Series) -> "FixedHistoricalVolatility":
        """Fit the model using the most recent lookback window."""
        clean = _validate_returns(returns, minimum_length=self.lookback)
        estimation_window = clean.iloc[-self.lookback :]

        self.volatility_ = annualized_historical_volatility(
            estimation_window,
            annualization_factor=self.annualization_factor,
            ddof=self.ddof,
        )
        return self

    def predict(self) -> float:
        """Return the fitted constant volatility estimate."""
        if self.volatility_ is None:
            raise RuntimeError("The model must be fitted before predict is called.")
        return self.volatility_


@dataclass
class RollingGaussianVolatility:
    """Compute rolling annualized volatility from recent returns.

    At time t, the estimate uses returns ending at time t. It never uses returns
    dated after t, which is essential for avoiding look-ahead bias.
    """

    window: int = 20
    annualization_factor: int = 252
    ddof: int = 1

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2.")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive.")
        if self.ddof < 0:
            raise ValueError("ddof cannot be negative.")
        if self.window <= self.ddof:
            raise ValueError("window must be greater than ddof.")

    def transform(self, returns: pd.Series) -> pd.Series:
        """Return a time-indexed rolling annualized volatility series."""
        clean = _validate_returns(returns, minimum_length=self.window)

        volatility = (
            clean.rolling(window=self.window, min_periods=self.window)
            .std(ddof=self.ddof)
            .mul(math.sqrt(self.annualization_factor))
        )
        volatility.name = f"rolling_volatility_{self.window}"
        return volatility

    def latest(self, returns: pd.Series) -> float:
        """Return the most recent rolling volatility estimate."""
        volatility = self.transform(returns)
        latest_value = volatility.iloc[-1]

        if pd.isna(latest_value):
            raise RuntimeError("Unable to compute the latest volatility estimate.")

        return float(latest_value)
