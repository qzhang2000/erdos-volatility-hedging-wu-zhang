import math

import pytest

from option_hedging.derivatives.black_scholes import (
    call_delta,
    call_payoff,
    call_price,
    put_delta,
    put_payoff,
    put_price,
)


COMMON = {
    "spot": 100.0,
    "strike": 100.0,
    "maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.20,
    "dividend_yield": 0.0,
}


def test_call_and_put_payoffs() -> None:
    assert call_payoff(120.0, 100.0) == 20.0
    assert call_payoff(80.0, 100.0) == 0.0
    assert put_payoff(80.0, 100.0) == 20.0
    assert put_payoff(120.0, 100.0) == 0.0


def test_standard_black_scholes_values() -> None:
    call = call_price(**COMMON)
    put = put_price(**COMMON)

    assert call == pytest.approx(10.4506, abs=1e-4)
    assert put == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity() -> None:
    call = call_price(**COMMON)
    put = put_price(**COMMON)

    left = call - put
    right = (
        COMMON["spot"] * math.exp(-COMMON["dividend_yield"] * COMMON["maturity"])
        - COMMON["strike"] * math.exp(-COMMON["rate"] * COMMON["maturity"])
    )
    assert left == pytest.approx(right, abs=1e-10)


def test_delta_parity() -> None:
    call = call_delta(**COMMON)
    put = put_delta(**COMMON)

    expected_difference = math.exp(
        -COMMON["dividend_yield"] * COMMON["maturity"]
    )
    assert call - put == pytest.approx(expected_difference, abs=1e-10)


def test_expiration_values() -> None:
    expiration = {
        "maturity": 0.0,
        "rate": 0.05,
        "volatility": 0.0,
        "dividend_yield": 0.0,
    }

    assert call_price(spot=120.0, strike=100.0, **expiration) == 20.0
    assert put_price(spot=80.0, strike=100.0, **expiration) == 20.0
    assert call_delta(spot=120.0, strike=100.0, **expiration) == 1.0
    assert call_delta(spot=80.0, strike=100.0, **expiration) == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spot", 0.0),
        ("strike", 0.0),
        ("maturity", -1.0),
        ("volatility", 0.0),
    ],
)
def test_invalid_inputs(field: str, value: float) -> None:
    inputs = COMMON.copy()
    inputs[field] = value

    with pytest.raises(ValueError):
        call_price(**inputs)
