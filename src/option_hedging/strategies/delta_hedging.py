"""Discrete-time Black-Scholes delta hedging for a short European call."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import pandas as pd

from option_hedging.data.market_data import validate_prices
from option_hedging.derivatives.black_scholes import call_delta, call_payoff, call_price


@dataclass(frozen=True)
class DeltaHedgingResult:
    """Results from one completed option-hedging episode."""

    history: pd.DataFrame
    initial_option_price: float
    option_payoff: float
    hedging_error: float
    total_transaction_cost: float


def _prepare_volatility(
    volatility: float | pd.Series,
    prices: pd.Series,
) -> pd.Series:
    """Return volatility aligned with every hedge date."""
    hedge_index = prices.index[:-1]

    if isinstance(volatility, Real):
        sigma = float(volatility)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("volatility must be finite and strictly positive.")
        return pd.Series(sigma, index=hedge_index, name="volatility")

    if not isinstance(volatility, pd.Series):
        raise TypeError("volatility must be a positive number or pandas Series.")

    aligned = volatility.astype(float).reindex(hedge_index)
    if aligned.isna().any():
        raise ValueError(
            "volatility is missing one or more values required on hedge dates."
        )
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("volatility must contain only finite values.")
    if (aligned <= 0).any():
        raise ValueError("all volatility values must be strictly positive.")

    aligned.name = "volatility"
    return aligned


def run_delta_hedge(
    prices: pd.Series,
    *,
    strike: float,
    volatility: float | pd.Series,
    rate: float = 0.0,
    transaction_cost_rate: float = 0.0,
    trading_days_per_year: int = 252,
    initial_option_price: float | None = None,
) -> DeltaHedgingResult:
    """Hedge one short European call over a discrete price path.

    The first price is the option-initiation date and the last price is the
    expiration date. The strategy receives the option premium, buys delta
    shares, rebalances on intermediate dates, liquidates the stock at
    expiration, and pays the call payoff. Remaining cash is hedging error.
    """
    clean_prices = validate_prices(prices, minimum_length=2)

    if not math.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be finite and strictly positive.")
    if not math.isfinite(rate):
        raise ValueError("rate must be finite.")
    if not math.isfinite(transaction_cost_rate) or transaction_cost_rate < 0:
        raise ValueError("transaction_cost_rate must be finite and nonnegative.")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive.")

    sigma_path = _prepare_volatility(volatility, clean_prices)
    n_intervals = len(clean_prices) - 1
    dt = 1.0 / trading_days_per_year
    initial_maturity = n_intervals * dt

    spot0 = float(clean_prices.iloc[0])
    sigma0 = float(sigma_path.iloc[0])

    if initial_option_price is None:
        premium = call_price(
            spot=spot0,
            strike=strike,
            maturity=initial_maturity,
            rate=rate,
            volatility=sigma0,
        )
    else:
        premium = float(initial_option_price)
        if not math.isfinite(premium) or premium < 0:
            raise ValueError("initial_option_price must be finite and nonnegative.")

    delta = call_delta(
        spot=spot0,
        strike=strike,
        maturity=initial_maturity,
        rate=rate,
        volatility=sigma0,
    )

    initial_trade = delta
    initial_cost = transaction_cost_rate * spot0 * abs(initial_trade)
    cash = premium - delta * spot0 - initial_cost
    shares = delta
    total_cost = initial_cost

    rows = [{
        "spot": spot0,
        "time_to_maturity": initial_maturity,
        "volatility": sigma0,
        "delta": delta,
        "shares": shares,
        "cash": cash,
        "trade_size": initial_trade,
        "transaction_cost": initial_cost,
        "portfolio_value": shares * spot0 + cash,
        "option_payoff": 0.0,
        "net_value": shares * spot0 + cash,
    }]

    for step in range(1, len(clean_prices) - 1):
        spot = float(clean_prices.iloc[step])
        sigma = float(sigma_path.iloc[step])
        remaining_maturity = (n_intervals - step) * dt

        cash *= math.exp(rate * dt)

        new_delta = call_delta(
            spot=spot,
            strike=strike,
            maturity=remaining_maturity,
            rate=rate,
            volatility=sigma,
        )
        trade_size = new_delta - shares
        trading_cost = transaction_cost_rate * spot * abs(trade_size)

        cash -= trade_size * spot + trading_cost
        shares = new_delta
        total_cost += trading_cost
        portfolio_value = shares * spot + cash

        rows.append({
            "spot": spot,
            "time_to_maturity": remaining_maturity,
            "volatility": sigma,
            "delta": new_delta,
            "shares": shares,
            "cash": cash,
            "trade_size": trade_size,
            "transaction_cost": trading_cost,
            "portfolio_value": portfolio_value,
            "option_payoff": 0.0,
            "net_value": portfolio_value,
        })

    terminal_spot = float(clean_prices.iloc[-1])
    cash *= math.exp(rate * dt)

    liquidation_trade = -shares
    liquidation_cost = (
        transaction_cost_rate * terminal_spot * abs(liquidation_trade)
    )
    cash += shares * terminal_spot - liquidation_cost
    total_cost += liquidation_cost

    payoff = call_payoff(terminal_spot, strike)
    portfolio_before_payoff = cash
    hedging_error = portfolio_before_payoff - payoff

    rows.append({
        "spot": terminal_spot,
        "time_to_maturity": 0.0,
        "volatility": np.nan,
        "delta": 0.0,
        "shares": 0.0,
        "cash": hedging_error,
        "trade_size": liquidation_trade,
        "transaction_cost": liquidation_cost,
        "portfolio_value": portfolio_before_payoff,
        "option_payoff": payoff,
        "net_value": hedging_error,
    })

    history = pd.DataFrame(rows, index=clean_prices.index)
    history.index.name = clean_prices.index.name or "date"

    return DeltaHedgingResult(
        history=history,
        initial_option_price=float(premium),
        option_payoff=float(payoff),
        hedging_error=float(hedging_error),
        total_transaction_cost=float(total_cost),
    )
