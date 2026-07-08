"""Utilities for loading and validating historical market prices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validate_prices(prices: pd.Series, *, minimum_length: int = 2) -> pd.Series:
    """Validate and return a floating-point price series."""
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")
    if minimum_length < 1:
        raise ValueError("minimum_length must be positive.")
    if len(prices) < minimum_length:
        raise ValueError(f"At least {minimum_length} price observations are required.")
    if prices.index.has_duplicates:
        raise ValueError("prices must have a unique index.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be ordered from earliest to latest.")

    clean = prices.astype(float)
    if clean.isna().any():
        raise ValueError("prices contains missing values.")
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("prices must contain only finite values.")
    if (clean <= 0).any():
        raise ValueError("all prices must be strictly positive.")

    clean.name = prices.name or "price"
    return clean


def simple_returns(prices: pd.Series, *, dropna: bool = True) -> pd.Series:
    """Compute simple returns, S_t / S_{t-1} - 1."""
    clean = validate_prices(prices)
    returns = clean.pct_change(fill_method=None)
    returns.name = "simple_return"
    return returns.dropna() if dropna else returns


def log_returns(prices: pd.Series, *, dropna: bool = True) -> pd.Series:
    """Compute log returns, log(S_t / S_{t-1})."""
    clean = validate_prices(prices)
    returns = np.log(clean / clean.shift(1))
    returns.name = "log_return"
    return returns.dropna() if dropna else returns


def load_price_csv(
    path: str | Path,
    *,
    date_column: str = "Date",
    price_column: str = "Adj Close",
) -> pd.Series:
    """Load one adjusted-price series from a CSV file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Price file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = [c for c in (date_column, price_column) if c not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")

    prices = pd.Series(
        frame[price_column].to_numpy(),
        index=pd.to_datetime(frame[date_column], errors="raise"),
        name=price_column,
        dtype=float,
    ).sort_index()
    return validate_prices(prices)
