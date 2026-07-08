"""Leakage-safe supervised-learning targets for volatility and risk models."""

from __future__ import annotations

import math
from typing import Any

from option_hedging.signals.transforms import ensure_panel, select_output


def create_future_return_target(
    data: Any,
    *,
    horizon: int = 1,
    price_col: str = "close",
    asset_col: str = "asset",
    date_col: str = "date",
    output_col: str | None = None,
) -> Any:
    """Create a forward simple-return target aligned to features at time ``t``."""

    if horizon < 1:
        raise ValueError("horizon must be positive.")

    output = output_col or f"ret_fwd_{horizon}d"
    frame = ensure_panel(data, {price_col}, asset_col=asset_col, date_col=date_col)
    future_price = frame.groupby(asset_col, sort=False)[price_col].shift(-horizon)
    frame[output] = future_price / frame[price_col] - 1.0
    return select_output(frame, [output], asset_col=asset_col, date_col=date_col)


def create_future_realized_volatility_target(
    data: Any,
    *,
    horizon: int = 20,
    annualization: int = 252,
    price_col: str = "close",
    asset_col: str = "asset",
    date_col: str = "date",
    output_col: str | None = None,
) -> Any:
    r"""Create annualized future realized volatility.

    At date ``t`` the target uses log returns from ``t+1`` through
    ``t+horizon``:

    .. math::

        \sqrt{\frac{A}{h}\sum_{j=1}^{h} r_{t+j}^{2}},

    where ``A`` is the annualization factor.  The target is intentionally
    forward-looking and must never be included among the model features.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive.")
    if annualization <= 0:
        raise ValueError("annualization must be positive.")

    output = output_col or f"future_realized_vol_{horizon}d"
    frame = ensure_panel(data, {price_col}, asset_col=asset_col, date_col=date_col)

    log_return = frame.groupby(asset_col, sort=False)[price_col].transform(
        lambda values: values.astype(float).map(math.log).diff()
    )
    squared_return = log_return.pow(2)

    # A backward rolling sum at row t+horizon contains returns t+1,...,t+horizon.
    # Shifting it backward aligns that realized variance with information at t.
    future_squared_sum = squared_return.groupby(
        frame[asset_col], sort=False
    ).transform(
        lambda values: values.rolling(
            window=horizon,
            min_periods=horizon,
        ).sum().shift(-horizon)
    )

    frame[output] = ((annualization / horizon) * future_squared_sum).pow(0.5)
    return select_output(frame, [output], asset_col=asset_col, date_col=date_col)


def create_downside_target(
    data: Any,
    *,
    horizon: int = 1,
    threshold: float = 0.0,
    price_col: str = "close",
    asset_col: str = "asset",
    date_col: str = "date",
    output_col: str | None = None,
) -> Any:
    """Create an indicator for whether a future return falls below a threshold."""

    output = output_col or f"downside_fwd_{horizon}d"
    frame = create_future_return_target(
        data,
        horizon=horizon,
        price_col=price_col,
        asset_col=asset_col,
        date_col=date_col,
        output_col="_future_return",
    )
    frame[output] = frame["_future_return"].lt(threshold).astype(float)
    frame.loc[frame["_future_return"].isna(), output] = None
    return select_output(frame, [output], asset_col=asset_col, date_col=date_col)
