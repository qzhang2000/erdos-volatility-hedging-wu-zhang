import math

import pandas as pd
import pytest

from option_hedging.data.market_data import (
    log_returns,
    simple_returns,
    validate_prices,
)


def test_simple_returns() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])
    result = simple_returns(prices)
    assert result.iloc[0] == pytest.approx(0.10)
    assert result.iloc[1] == pytest.approx(-0.10)


def test_log_returns() -> None:
    prices = pd.Series([100.0, 110.0, 121.0])
    result = log_returns(prices)
    assert result.iloc[0] == pytest.approx(math.log(1.10))
    assert result.iloc[1] == pytest.approx(math.log(1.10))


def test_prices_must_be_positive() -> None:
    with pytest.raises(ValueError):
        validate_prices(pd.Series([100.0, 0.0, 101.0]))


def test_prices_must_be_ordered() -> None:
    prices = pd.Series(
        [100.0, 101.0],
        index=pd.to_datetime(["2026-01-02", "2026-01-01"]),
    )
    with pytest.raises(ValueError):
        validate_prices(prices)
