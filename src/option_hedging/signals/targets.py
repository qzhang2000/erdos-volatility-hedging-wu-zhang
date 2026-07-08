"""Target construction for return-law learning."""

from __future__ import annotations

from typing import Any

from finance_project.signals.transforms import ensure_panel, select_output


def create_future_return_target(
    data: Any,
    *,
    horizon: int = 1,
    price_col: str = "close",
    asset_col: str = "asset",
    date_col: str = "date",
    output_col: str | None = None,
) -> Any:
    """Create a forward simple-return target aligned to features at time t."""

    if horizon < 1:
        raise ValueError("horizon must be positive.")

    output = output_col or f"ret_fwd_{horizon}d"
    frame = ensure_panel(data, {price_col}, asset_col=asset_col, date_col=date_col)
    future_price = frame.groupby(asset_col, sort=False)[price_col].shift(-horizon)
    frame[output] = future_price / frame[price_col] - 1.0
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

