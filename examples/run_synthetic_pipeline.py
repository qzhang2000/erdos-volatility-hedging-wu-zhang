"""Run the complete forecasting and hedging pipeline on synthetic market data.

This script is a smoke test and usage example.  Replace the generated market
panel with real adjusted-close and volume data for the empirical project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from option_hedging.backtesting import (
    generate_option_episodes,
    run_strategy_comparison,
)
from option_hedging.evaluation import (
    summarize_hedging_performance,
    volatility_forecast_metrics,
)
from option_hedging.models import (
    RidgeVolatilityForecaster,
    RollingGaussianVolatility,
)
from option_hedging.signals import (
    SignalLibrary,
    create_future_realized_volatility_target,
)


def make_synthetic_market(periods: int = 900, seed: int = 7) -> pd.DataFrame:
    """Generate a price path with calm and elevated-volatility regimes."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=periods)
    regime = np.where((np.arange(periods) // 120) % 2 == 0, 0.010, 0.025)
    log_returns = 0.0002 + regime * rng.standard_normal(periods)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    volume = 2_000_000.0 * (1.0 + 8.0 * regime) * np.exp(
        0.15 * rng.standard_normal(periods)
    )
    return pd.DataFrame(
        {
            "date": dates,
            "asset": "SYNTH",
            "close": close,
            "volume": volume,
        }
    )


def main() -> None:
    market = make_synthetic_market()

    feature_panel = SignalLibrary.default().compute(market)
    target_panel = create_future_realized_volatility_target(
        market,
        horizon=20,
    )
    modeling = feature_panel.merge(
        target_panel,
        on=["date", "asset"],
        validate="one_to_one",
    ).set_index("date")

    target_column = "future_realized_vol_20d"
    feature_columns = [
        column
        for column in modeling.columns
        if column not in {"asset", target_column}
    ]
    modeling = modeling.dropna(subset=[*feature_columns, target_column])

    split = int(0.70 * len(modeling))
    train = modeling.iloc[:split]
    test = modeling.iloc[split:]

    model = RidgeVolatilityForecaster(alpha=5.0).fit(
        train[feature_columns],
        train[target_column],
    )
    signal_forecast = model.predict(test[feature_columns])

    forecast_metrics = volatility_forecast_metrics(
        test[target_column],
        signal_forecast,
    )

    price_series = market.set_index("date")["close"]
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    rolling_forecast = RollingGaussianVolatility(window=21).transform(log_returns)

    # Use only dates on which the signal model has out-of-sample predictions.
    test_prices = price_series.reindex(test.index)
    episodes = generate_option_episodes(
        test_prices,
        maturity_days=20,
        start_step=20,
    )

    strategies = {
        "fixed_at_start": lambda episode: float(
            rolling_forecast.loc[episode.start_date]
        ),
        "rolling_gaussian": lambda episode: rolling_forecast.reindex(
            episode.prices.index[:-1]
        ),
        "signal_ridge": lambda episode: signal_forecast.reindex(
            episode.prices.index[:-1]
        ),
    }

    episode_results, hedge_paths = run_strategy_comparison(
        episodes,
        strategies=strategies,
        pricing_strategy="fixed_at_start",
        rate=0.03,
        transaction_cost_rate=0.0005,
    )
    hedging_metrics = summarize_hedging_performance(episode_results)

    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_results.to_csv(output_dir / "synthetic_episode_results.csv", index=False)
    hedge_paths.to_csv(output_dir / "synthetic_hedge_paths.csv", index=False)
    hedging_metrics.to_csv(output_dir / "synthetic_hedging_metrics.csv")
    forecast_metrics.to_csv(output_dir / "synthetic_forecast_metrics.csv")

    print("Volatility forecast metrics")
    print(forecast_metrics.round(6))
    print("\nHedging metrics")
    print(hedging_metrics.round(6))


if __name__ == "__main__":
    main()
