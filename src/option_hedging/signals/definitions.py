"""Signal definitions selected for volatility forecasting and option hedging."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from option_hedging.signals.base import SignalSpec
from option_hedging.signals.transforms import (
    ensure_panel,
    rolling_max,
    rolling_mean,
    rolling_skew,
    rolling_std,
    safe_log1p,
    select_output,
    simple_return,
)


def _min_periods(window: int) -> int:
    """Require a complete trailing window for an unambiguous signal."""

    return window



def _absolute_return_signal() -> SignalSpec:
    output = "abs_return_1d"

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {"close"}, asset_col=asset_col, date_col=date_col)
        frame[output] = simple_return(
            frame, price_col="close", periods=1, asset_col=asset_col
        ).abs()
        return select_output(frame, [output], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=output,
        family="shock",
        description="Absolute one-period return, a proxy for the latest market shock.",
        required_columns=("close",),
        output_columns=(output,),
        lookback=1,
        tags=("volatility", "short_horizon", "stress"),
        compute=compute,
    )


def _volatility_ratio_signal(
    short_window: int,
    long_window: int,
    annualization: int = 252,
) -> SignalSpec:
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window.")

    output = f"vol_ratio_{short_window}d_{long_window}d"

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {"close"}, asset_col=asset_col, date_col=date_col)
        returns = simple_return(frame, price_col="close", periods=1, asset_col=asset_col)
        short_vol = rolling_std(
            returns,
            frame,
            window=short_window,
            min_periods=_min_periods(short_window),
            asset_col=asset_col,
        ) * math.sqrt(annualization)
        long_vol = rolling_std(
            returns,
            frame,
            window=long_window,
            min_periods=_min_periods(long_window),
            asset_col=asset_col,
        ) * math.sqrt(annualization)
        frame[output] = short_vol / long_vol.where(long_vol != 0)
        return select_output(frame, [output], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=output,
        family="volatility",
        description=(
            f"Ratio of {short_window}-period to {long_window}-period realized "
            "volatility; values above one indicate an accelerating volatility regime."
        ),
        required_columns=("close",),
        output_columns=(output,),
        lookback=long_window,
        tags=("scale", "regime", "stress"),
        params={
            "short_window": short_window,
            "long_window": long_window,
            "annualization": annualization,
        },
        compute=compute,
    )

def _momentum_signal(window: int) -> SignalSpec:
    output = f"mom_{window}d"

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {"close"}, asset_col=asset_col, date_col=date_col)
        frame[output] = simple_return(frame, price_col="close", periods=window, asset_col=asset_col)
        return select_output(frame, [output], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=output,
        family="momentum",
        description=f"Trailing {window}-period simple return.",
        required_columns=("close",),
        output_columns=(output,),
        lookback=window,
        tags=("location", "trend"),
        params={"window": window},
        compute=compute,
    )


def _reversal_signal() -> SignalSpec:
    output = "reversal_1d"

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {"close"}, asset_col=asset_col, date_col=date_col)
        frame[output] = -simple_return(frame, price_col="close", periods=1, asset_col=asset_col)
        return select_output(frame, [output], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=output,
        family="momentum",
        description="Negative of the prior one-period return.",
        required_columns=("close",),
        output_columns=(output,),
        lookback=1,
        tags=("location", "short_horizon"),
        compute=compute,
    )


def _realized_vol_signal(window: int, annualization: int = 252) -> SignalSpec:
    output = f"vol_{window}d"

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {"close"}, asset_col=asset_col, date_col=date_col)
        returns = simple_return(frame, price_col="close", periods=1, asset_col=asset_col)
        frame[output] = rolling_std(
            returns,
            frame,
            window=window,
            min_periods=_min_periods(window),
            asset_col=asset_col,
        ) * math.sqrt(annualization)
        return select_output(frame, [output], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=output,
        family="volatility",
        description=f"Annualized rolling realized volatility over {window} periods.",
        required_columns=("close",),
        output_columns=(output,),
        lookback=window,
        tags=("scale", "tail"),
        params={"window": window, "annualization": annualization},
        compute=compute,
    )


def _downside_vol_signal(window: int, annualization: int = 252) -> SignalSpec:
    output = f"downside_vol_{window}d"

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {"close"}, asset_col=asset_col, date_col=date_col)
        returns = simple_return(frame, price_col="close", periods=1, asset_col=asset_col)
        downside_squared = returns.where(returns < 0, 0.0).pow(2)
        frame[output] = rolling_mean(
            downside_squared,
            frame,
            window=window,
            min_periods=_min_periods(window),
            asset_col=asset_col,
        ).pow(0.5) * math.sqrt(annualization)
        return select_output(frame, [output], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=output,
        family="volatility",
        description=f"Annualized downside semivolatility over {window} periods.",
        required_columns=("close",),
        output_columns=(output,),
        lookback=window,
        tags=("scale", "downside", "tail"),
        params={"window": window, "annualization": annualization},
        compute=compute,
    )


def _drawdown_signal(window: int) -> SignalSpec:
    output = f"drawdown_{window}d"

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {"close"}, asset_col=asset_col, date_col=date_col)
        trailing_high = rolling_max(
            frame["close"],
            frame,
            window=window,
            min_periods=_min_periods(window),
            asset_col=asset_col,
        )
        frame[output] = frame["close"] / trailing_high - 1.0
        return select_output(frame, [output], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=output,
        family="drawdown",
        description=f"Current price distance from the trailing {window}-period high.",
        required_columns=("close",),
        output_columns=(output,),
        lookback=window,
        tags=("downside", "tail", "state"),
        params={"window": window},
        compute=compute,
    )


def _dollar_volume_signal(window: int) -> SignalSpec:
    output = f"log_dollar_volume_{window}d"

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {"close", "volume"}, asset_col=asset_col, date_col=date_col)
        dollar_volume = frame["close"] * frame["volume"]
        smoothed = rolling_mean(
            dollar_volume,
            frame,
            window=window,
            min_periods=_min_periods(window),
            asset_col=asset_col,
        )
        frame[output] = safe_log1p(smoothed)
        return select_output(frame, [output], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=output,
        family="liquidity",
        description=f"Log rolling mean dollar volume over {window} periods.",
        required_columns=("close", "volume"),
        output_columns=(output,),
        lookback=window,
        tags=("liquidity", "tail"),
        params={"window": window},
        compute=compute,
    )


def _amihud_signal(window: int) -> SignalSpec:
    output = f"amihud_{window}d"

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {"close", "volume"}, asset_col=asset_col, date_col=date_col)
        returns = simple_return(frame, price_col="close", periods=1, asset_col=asset_col).abs()
        dollar_volume = (frame["close"] * frame["volume"]).where(lambda values: values > 0)
        illiquidity = returns / dollar_volume
        frame[output] = rolling_mean(
            illiquidity,
            frame,
            window=window,
            min_periods=_min_periods(window),
            asset_col=asset_col,
        )
        return select_output(frame, [output], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=output,
        family="liquidity",
        description=f"Rolling Amihud illiquidity over {window} periods.",
        required_columns=("close", "volume"),
        output_columns=(output,),
        lookback=window,
        tags=("liquidity", "stress", "tail"),
        params={"window": window},
        compute=compute,
    )


def _rolling_skew_signal(window: int) -> SignalSpec:
    output = f"skew_{window}d"

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {"close"}, asset_col=asset_col, date_col=date_col)
        returns = simple_return(frame, price_col="close", periods=1, asset_col=asset_col)
        frame[output] = rolling_skew(
            returns,
            frame,
            window=window,
            min_periods=_min_periods(window),
            asset_col=asset_col,
        )
        return select_output(frame, [output], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=output,
        family="tail_shape",
        description=f"Rolling skewness of one-period returns over {window} periods.",
        required_columns=("close",),
        output_columns=(output,),
        lookback=window,
        tags=("asymmetry", "tail"),
        params={"window": window},
        compute=compute,
    )


def _downside_frequency_signal(window: int) -> SignalSpec:
    output = f"downside_freq_{window}d"

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {"close"}, asset_col=asset_col, date_col=date_col)
        returns = simple_return(frame, price_col="close", periods=1, asset_col=asset_col)
        downside = returns.lt(0).astype(float)
        frame[output] = rolling_mean(
            downside,
            frame,
            window=window,
            min_periods=_min_periods(window),
            asset_col=asset_col,
        )
        return select_output(frame, [output], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=output,
        family="tail_shape",
        description=f"Rolling fraction of negative one-period returns over {window} periods.",
        required_columns=("close",),
        output_columns=(output,),
        lookback=window,
        tags=("downside", "tail", "probability"),
        params={"window": window},
        compute=compute,
    )


def _ratio_signal(
    *,
    name: str,
    numerator: str,
    denominator: str,
    description: str,
    tags: tuple[str, ...],
) -> SignalSpec:
    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {numerator, denominator}, asset_col=asset_col, date_col=date_col)
        frame[name] = frame[numerator] / frame[denominator].where(lambda values: values != 0)
        return select_output(frame, [name], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=name,
        family="valuation",
        description=description,
        required_columns=(numerator, denominator),
        output_columns=(name,),
        lookback=None,
        tags=tags,
        params={"numerator": numerator, "denominator": denominator},
        compute=compute,
    )


def passthrough_signal(
    column: str,
    *,
    family: str = "passthrough",
    tags: Iterable[str] = (),
    description: str | None = None,
) -> SignalSpec:
    """Register an already point-in-time column as a signal."""

    def compute(data: Any, *, asset_col: str, date_col: str) -> Any:
        frame = ensure_panel(data, {column}, asset_col=asset_col, date_col=date_col)
        return select_output(frame, [column], asset_col=asset_col, date_col=date_col)

    return SignalSpec(
        name=column,
        family=family,
        description=description or f"Point-in-time passthrough signal {column}.",
        required_columns=(column,),
        output_columns=(column,),
        lookback=None,
        tags=tuple(tags),
        params={"column": column},
        compute=compute,
    )


def make_hedging_signal_specs() -> list[SignalSpec]:
    """Return the compact signal set used by the option-hedging project."""

    return [
        _reversal_signal(),
        _absolute_return_signal(),
        *[_momentum_signal(window) for window in (5, 21)],
        *[_realized_vol_signal(window) for window in (5, 21, 63)],
        _volatility_ratio_signal(5, 21),
        _volatility_ratio_signal(21, 63),
        _downside_vol_signal(21),
        *[_drawdown_signal(window) for window in (63, 252)],
        _dollar_volume_signal(21),
        _amihud_signal(21),
        _rolling_skew_signal(63),
        _downside_frequency_signal(21),
    ]


def make_default_signal_specs() -> list[SignalSpec]:
    """Return the default option-hedging signal library."""

    return make_hedging_signal_specs()
