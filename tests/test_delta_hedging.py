import math

import pandas as pd
import pytest

from option_hedging.derivatives.black_scholes import (
    call_delta,
    call_payoff,
    call_price,
)
from option_hedging.strategies.delta_hedging import run_delta_hedge


def test_one_interval_hedge_matches_manual_accounting() -> None:
    prices = pd.Series(
        [100.0, 104.0],
        index=pd.date_range("2026-01-02", periods=2, freq="B"),
    )
    strike = 100.0
    sigma = 0.20
    rate = 0.04
    dt = 1 / 252

    premium = call_price(
        spot=100.0,
        strike=strike,
        maturity=dt,
        rate=rate,
        volatility=sigma,
    )
    delta = call_delta(
        spot=100.0,
        strike=strike,
        maturity=dt,
        rate=rate,
        volatility=sigma,
    )
    initial_cash = premium - delta * 100.0
    terminal_portfolio = initial_cash * math.exp(rate * dt) + delta * 104.0
    expected_error = terminal_portfolio - call_payoff(104.0, strike)

    result = run_delta_hedge(
        prices,
        strike=strike,
        volatility=sigma,
        rate=rate,
    )

    assert result.initial_option_price == pytest.approx(premium)
    assert result.hedging_error == pytest.approx(expected_error)
    assert result.history.iloc[-1]["shares"] == 0.0


def test_transaction_costs_reduce_terminal_error() -> None:
    prices = pd.Series([100.0, 100.0])
    sigma = 0.20
    cost_rate = 0.001

    no_cost = run_delta_hedge(
        prices,
        strike=100.0,
        volatility=sigma,
        transaction_cost_rate=0.0,
    )
    with_cost = run_delta_hedge(
        prices,
        strike=100.0,
        volatility=sigma,
        transaction_cost_rate=cost_rate,
    )

    initial_delta = no_cost.history.iloc[0]["delta"]
    expected_cost = 2 * cost_rate * 100.0 * abs(initial_delta)

    assert with_cost.total_transaction_cost == pytest.approx(expected_cost)
    assert no_cost.hedging_error - with_cost.hedging_error == pytest.approx(
        expected_cost
    )


def test_time_varying_volatility_is_used() -> None:
    index = pd.date_range("2026-01-02", periods=4, freq="B")
    prices = pd.Series([100.0, 101.0, 99.0, 103.0], index=index)
    volatility = pd.Series([0.15, 0.20, 0.35], index=index[:-1])

    result = run_delta_hedge(
        prices,
        strike=100.0,
        volatility=volatility,
    )

    expected = volatility.rename("volatility").copy()
    expected.index.name = result.history.index.name

    pd.testing.assert_series_equal(
        result.history.iloc[:-1]["volatility"],
        expected,
        check_names=True,
    )


def test_missing_volatility_date_raises_error() -> None:
    index = pd.date_range("2026-01-02", periods=3, freq="B")
    prices = pd.Series([100.0, 101.0, 102.0], index=index)
    volatility = pd.Series([0.20], index=index[:1])

    with pytest.raises(ValueError):
        run_delta_hedge(
            prices,
            strike=100.0,
            volatility=volatility,
        )


def test_history_has_one_row_per_price() -> None:
    prices = pd.Series([100.0, 102.0, 101.0, 104.0])
    result = run_delta_hedge(
        prices,
        strike=100.0,
        volatility=0.20,
    )

    assert len(result.history) == len(prices)
    assert result.history.iloc[-1]["option_payoff"] == 4.0
    assert result.history.iloc[-1]["net_value"] == pytest.approx(
        result.hedging_error
    )
