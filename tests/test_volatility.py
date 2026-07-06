import math

import pandas as pd
import pytest

from option_hedging.models.volatility import (
    FixedHistoricalVolatility,
    RollingGaussianVolatility,
    annualized_historical_volatility,
)


def test_annualized_historical_volatility() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    expected = returns.std(ddof=1) * math.sqrt(252)

    result = annualized_historical_volatility(returns)

    assert result == pytest.approx(expected)


def test_fixed_model_uses_only_most_recent_lookback() -> None:
    returns = pd.Series([0.50, 0.40, 0.01, -0.01, 0.02, -0.02])
    model = FixedHistoricalVolatility(lookback=4).fit(returns)

    expected = returns.iloc[-4:].std(ddof=1) * math.sqrt(252)

    assert model.predict() == pytest.approx(expected)


def test_fixed_model_requires_fit() -> None:
    model = FixedHistoricalVolatility(lookback=4)

    with pytest.raises(RuntimeError):
        model.predict()


def test_rolling_model_has_expected_alignment() -> None:
    returns = pd.Series(
        [0.01, -0.01, 0.02, -0.02, 0.03],
        index=pd.date_range("2026-01-01", periods=5),
    )
    model = RollingGaussianVolatility(window=3)

    result = model.transform(returns)

    assert result.iloc[:2].isna().all()
    expected_first = returns.iloc[:3].std(ddof=1) * math.sqrt(252)
    assert result.iloc[2] == pytest.approx(expected_first)
    assert result.index.equals(returns.index)


def test_rolling_estimate_does_not_use_future_returns() -> None:
    base = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03])
    altered_future = base.copy()
    altered_future.iloc[-1] = 10.0

    model = RollingGaussianVolatility(window=3)
    base_result = model.transform(base)
    altered_result = model.transform(altered_future)

    # The estimate at index 3 uses indices 1, 2, and 3, so changing index 4
    # must not alter it.
    assert altered_result.iloc[3] == pytest.approx(base_result.iloc[3])


def test_missing_returns_raise_error() -> None:
    returns = pd.Series([0.01, float("nan"), 0.02])

    with pytest.raises(ValueError):
        RollingGaussianVolatility(window=2).transform(returns)


@pytest.mark.parametrize("window", [0, 1])
def test_invalid_window(window: int) -> None:
    with pytest.raises(ValueError):
        RollingGaussianVolatility(window=window)
