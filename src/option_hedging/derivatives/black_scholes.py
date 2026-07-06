"""Black-Scholes prices and deltas for European options.

All rates and volatilities are continuously compounded and annualized.
The maturity argument is measured in years.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _validate_inputs(
    *,
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
) -> None:
    """Validate common Black-Scholes inputs."""
    values = {
        "spot": spot,
        "strike": strike,
        "maturity": maturity,
        "volatility": volatility,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if spot <= 0:
        raise ValueError("spot must be strictly positive.")
    if strike <= 0:
        raise ValueError("strike must be strictly positive.")
    if maturity < 0:
        raise ValueError("maturity cannot be negative.")
    if maturity > 0 and volatility <= 0:
        raise ValueError("volatility must be strictly positive before expiration.")


def call_payoff(spot: float, strike: float) -> float:
    """Return the expiration payoff of a European call option."""
    if spot < 0:
        raise ValueError("spot cannot be negative at expiration.")
    if strike <= 0:
        raise ValueError("strike must be strictly positive.")
    return max(spot - strike, 0.0)


def put_payoff(spot: float, strike: float) -> float:
    """Return the expiration payoff of a European put option."""
    if spot < 0:
        raise ValueError("spot cannot be negative at expiration.")
    if strike <= 0:
        raise ValueError("strike must be strictly positive.")
    return max(strike - spot, 0.0)


def _d1_d2(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> tuple[float, float]:
    """Return the Black-Scholes d1 and d2 quantities."""
    sqrt_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return d1, d2


def call_price(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Return the Black-Scholes price of a European call option."""
    _validate_inputs(
        spot=spot,
        strike=strike,
        maturity=maturity,
        volatility=volatility,
    )
    if maturity == 0:
        return call_payoff(spot, strike)

    d1, d2 = _d1_d2(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)


def put_price(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Return the Black-Scholes price of a European put option."""
    _validate_inputs(
        spot=spot,
        strike=strike,
        maturity=maturity,
        volatility=volatility,
    )
    if maturity == 0:
        return put_payoff(spot, strike)

    d1, d2 = _d1_d2(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)


def call_delta(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Return the Black-Scholes delta of a European call option."""
    _validate_inputs(
        spot=spot,
        strike=strike,
        maturity=maturity,
        volatility=volatility,
    )
    if maturity == 0:
        if spot > strike:
            return 1.0
        if spot < strike:
            return 0.0
        return 0.5

    d1, _ = _d1_d2(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    return math.exp(-dividend_yield * maturity) * norm.cdf(d1)


def put_delta(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Return the Black-Scholes delta of a European put option."""
    _validate_inputs(
        spot=spot,
        strike=strike,
        maturity=maturity,
        volatility=volatility,
    )
    if maturity == 0:
        if spot > strike:
            return 0.0
        if spot < strike:
            return -1.0
        return -0.5

    d1, _ = _d1_d2(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    return math.exp(-dividend_yield * maturity) * (norm.cdf(d1) - 1.0)
