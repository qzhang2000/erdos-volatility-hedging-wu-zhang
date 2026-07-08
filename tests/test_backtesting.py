import pandas as pd
import pytest

from option_hedging.backtesting import (
    generate_option_episodes,
    run_strategy_comparison,
)
from option_hedging.evaluation import summarize_hedging_performance


def sample_prices(n: int = 65) -> pd.Series:
    index = pd.bdate_range("2025-01-02", periods=n)
    return pd.Series(
        [100.0 * (1.001 ** i) for i in range(n)],
        index=index,
        name="close",
    )


def test_generate_option_episodes() -> None:
    episodes = generate_option_episodes(
        sample_prices(),
        maturity_days=20,
        start_step=20,
    )

    assert len(episodes) == 3
    assert all(len(episode.prices) == 21 for episode in episodes)
    assert episodes[0].strike == pytest.approx(episodes[0].initial_spot)


def test_strategy_comparison_uses_common_option_premium() -> None:
    episodes = generate_option_episodes(
        sample_prices(),
        maturity_days=10,
        start_step=10,
    )[:2]

    strategies = {
        "fixed": lambda episode: 0.20,
        "adaptive": lambda episode: pd.Series(
            0.25,
            index=episode.prices.index[:-1],
        ),
    }
    summary, histories = run_strategy_comparison(
        episodes,
        strategies=strategies,
        pricing_strategy="fixed",
    )

    premium_counts = summary.groupby("episode_id")["initial_option_price"].nunique()
    assert (premium_counts == 1).all()
    assert set(summary["strategy"]) == {"fixed", "adaptive"}
    assert len(histories) == sum(len(e.prices) for e in episodes) * 2


def test_hedging_metric_summary() -> None:
    results = pd.DataFrame(
        {
            "strategy": ["a", "a", "b", "b"],
            "hedging_error": [-1.0, 1.0, -2.0, 0.0],
            "transaction_cost": [0.1, 0.1, 0.2, 0.2],
            "turnover_notional": [10.0, 12.0, 20.0, 22.0],
        }
    )

    metrics = summarize_hedging_performance(results)

    assert metrics.loc["a", "mae"] == pytest.approx(1.0)
    assert metrics.loc["a", "rmse"] == pytest.approx(1.0)
    assert metrics.loc["b", "mean_transaction_cost"] == pytest.approx(0.2)
