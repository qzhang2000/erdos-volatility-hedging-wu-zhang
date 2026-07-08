"""Chronological panel transforms used by signal definitions."""

from __future__ import annotations

from typing import Any, Iterable


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The signal library requires pandas. Install project dependencies with "
            "`python3 -m pip install -e .`."
        ) from exc
    return pd


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The signal library requires numpy. Install project dependencies with "
            "`python3 -m pip install -e .`."
        ) from exc
    return np


def ensure_panel(
    data: Any,
    required_columns: Iterable[str],
    *,
    asset_col: str,
    date_col: str,
    allow_duplicate_keys: bool = False,
) -> Any:
    """Validate required columns and return a sorted copy of the panel."""

    _require_pandas()
    required = set(required_columns).union({asset_col, date_col})
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")

    frame = data.sort_values([asset_col, date_col]).copy()
    if not allow_duplicate_keys:
        duplicate_mask = frame.duplicated([asset_col, date_col])
        if bool(duplicate_mask.any()):
            sample = frame.loc[duplicate_mask, [asset_col, date_col]].head(5)
            raise ValueError(
                "Panel contains duplicate asset-date rows. Sample duplicates: "
                f"{sample.to_dict(orient='records')}"
            )
    return frame


def key_columns(frame: Any, *, asset_col: str, date_col: str) -> list[str]:
    return [date_col, asset_col]


def select_output(frame: Any, outputs: Iterable[str], *, asset_col: str, date_col: str) -> Any:
    return frame[key_columns(frame, asset_col=asset_col, date_col=date_col) + list(outputs)]


def simple_return(frame: Any, *, price_col: str, periods: int, asset_col: str) -> Any:
    """Compute within-asset simple returns over a positive lag."""

    if periods < 1:
        raise ValueError("Return periods must be positive.")
    return frame.groupby(asset_col, sort=False)[price_col].pct_change(periods=periods)


def rolling_mean(
    series: Any,
    frame: Any,
    *,
    window: int,
    min_periods: int,
    asset_col: str,
) -> Any:
    return series.groupby(frame[asset_col], sort=False).transform(
        lambda values: values.rolling(window=window, min_periods=min_periods).mean()
    )


def rolling_std(
    series: Any,
    frame: Any,
    *,
    window: int,
    min_periods: int,
    asset_col: str,
) -> Any:
    return series.groupby(frame[asset_col], sort=False).transform(
        lambda values: values.rolling(window=window, min_periods=min_periods).std()
    )


def rolling_skew(
    series: Any,
    frame: Any,
    *,
    window: int,
    min_periods: int,
    asset_col: str,
) -> Any:
    return series.groupby(frame[asset_col], sort=False).transform(
        lambda values: values.rolling(window=window, min_periods=min_periods).skew()
    )


def rolling_max(
    series: Any,
    frame: Any,
    *,
    window: int,
    min_periods: int,
    asset_col: str,
) -> Any:
    return series.groupby(frame[asset_col], sort=False).transform(
        lambda values: values.rolling(window=window, min_periods=min_periods).max()
    )


def safe_log1p(series: Any) -> Any:
    """Compute log(1 + x) and return NaN for non-positive domain violations."""

    np = _require_numpy()
    return np.log1p(series.where(series >= 0))
