"""Construction and evaluation of repeated historical option episodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from option_hedging.data.market_data import validate_prices
from option_hedging.derivatives.black_scholes import call_price
from option_hedging.strategies.delta_hedging import run_delta_hedge


@dataclass(frozen=True)
class OptionEpisode:
    """One historical European call from initiation through expiration."""

    episode_id: int
    prices: pd.Series
    strike: float

    @property
    def start_date(self):
        return self.prices.index[0]

    @property
    def end_date(self):
        return self.prices.index[-1]

    @property
    def initial_spot(self) -> float:
        return float(self.prices.iloc[0])

    @property
    def terminal_spot(self) -> float:
        return float(self.prices.iloc[-1])


VolatilityProvider = Callable[[OptionEpisode], float | pd.Series]


def generate_option_episodes(
    prices: pd.Series,
    *,
    maturity_days: int = 20,
    start_step: int = 20,
    moneyness: float = 1.0,
    first_start: int = 0,
) -> list[OptionEpisode]:
    """Generate position-based European call episodes from a price history.

    ``maturity_days`` is the number of trading intervals, so every episode
    contains ``maturity_days + 1`` prices including initiation and expiration.
    The strike is ``moneyness * initial_spot``; ``moneyness=1`` is at the money.
    """

    clean = validate_prices(prices, minimum_length=2)
    if maturity_days < 1:
        raise ValueError("maturity_days must be positive.")
    if start_step < 1:
        raise ValueError("start_step must be positive.")
    if moneyness <= 0:
        raise ValueError("moneyness must be positive.")
    if first_start < 0:
        raise ValueError("first_start cannot be negative.")

    episodes: list[OptionEpisode] = []
    episode_id = 0
    final_start = len(clean) - maturity_days - 1
    for start in range(first_start, final_start + 1, start_step):
        path = clean.iloc[start : start + maturity_days + 1].copy()
        strike = float(path.iloc[0]) * moneyness
        episodes.append(
            OptionEpisode(
                episode_id=episode_id,
                prices=path,
                strike=strike,
            )
        )
        episode_id += 1

    return episodes


def _initial_sigma(
    volatility: float | pd.Series,
    episode: OptionEpisode,
) -> float:
    if isinstance(volatility, pd.Series):
        value = volatility.reindex(episode.prices.index[:-1]).iloc[0]
        if pd.isna(value):
            raise ValueError(
                f"Pricing volatility is unavailable on {episode.start_date!r}."
            )
        return float(value)
    return float(volatility)


def run_strategy_comparison(
    episodes: Sequence[OptionEpisode],
    *,
    strategies: Mapping[str, VolatilityProvider],
    pricing_strategy: str,
    rate: float = 0.0,
    transaction_cost_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run multiple volatility-driven hedges on the same option episodes.

    The initial option premium is computed once per episode using
    ``pricing_strategy`` and then held common across all hedges.  This isolates
    hedge performance from differences in the model-implied sale price.

    Returns
    -------
    summary, histories:
        One row per episode-strategy and all date-level hedge paths.
    """

    if not episodes:
        raise ValueError("episodes cannot be empty.")
    if not strategies:
        raise ValueError("strategies cannot be empty.")
    if pricing_strategy not in strategies:
        raise KeyError("pricing_strategy must be a key in strategies.")

    summary_rows: list[dict[str, object]] = []
    history_frames: list[pd.DataFrame] = []

    for episode in episodes:
        supplied = {
            name: provider(episode)
            for name, provider in strategies.items()
        }
        price_sigma = _initial_sigma(supplied[pricing_strategy], episode)
        maturity = (len(episode.prices) - 1) / trading_days_per_year
        premium = call_price(
            spot=episode.initial_spot,
            strike=episode.strike,
            maturity=maturity,
            rate=rate,
            volatility=price_sigma,
        )

        for strategy_name, volatility in supplied.items():
            result = run_delta_hedge(
                episode.prices,
                strike=episode.strike,
                volatility=volatility,
                rate=rate,
                transaction_cost_rate=transaction_cost_rate,
                trading_days_per_year=trading_days_per_year,
                initial_option_price=premium,
            )
            turnover_shares = float(result.history["trade_size"].abs().sum())
            turnover_notional = float(
                (result.history["trade_size"].abs() * result.history["spot"]).sum()
            )

            summary_rows.append(
                {
                    "episode_id": episode.episode_id,
                    "strategy": strategy_name,
                    "start_date": episode.start_date,
                    "end_date": episode.end_date,
                    "initial_spot": episode.initial_spot,
                    "terminal_spot": episode.terminal_spot,
                    "strike": episode.strike,
                    "initial_option_price": premium,
                    "option_payoff": result.option_payoff,
                    "hedging_error": result.hedging_error,
                    "absolute_hedging_error": abs(result.hedging_error),
                    "transaction_cost": result.total_transaction_cost,
                    "turnover_shares": turnover_shares,
                    "turnover_notional": turnover_notional,
                }
            )

            history = result.history.reset_index()
            history.insert(0, "strategy", strategy_name)
            history.insert(0, "episode_id", episode.episode_id)
            history_frames.append(history)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["episode_id", "strategy"]
    ).reset_index(drop=True)
    histories = pd.concat(history_frames, ignore_index=True)
    return summary, histories
