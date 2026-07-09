"""Run the market-signal adjusted hedge on real Yahoo Finance data.

The pipeline uses SPY as the option underlying and VIX as the historical
option-implied volatility signal.  Additional point-in-time signals include
realized-volatility acceleration, volume shocks, drawdown, sector divergence,
and short-rate changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from option_hedging.backtesting import generate_option_episodes, run_strategy_comparison
from option_hedging.evaluation import summarize_hedging_performance
from option_hedging.models import RollingGaussianVolatility
from option_hedging.strategies import (
    ELEVATED,
    NORMAL,
    STRESS,
    MarketSignalRiskModel,
)


@dataclass(frozen=True)
class RealMarketPipelineResult:
    """Outputs from the real-market backtest pipeline."""

    market: pd.DataFrame
    features: pd.DataFrame
    risk_diagnostics: pd.DataFrame
    episode_results: pd.DataFrame
    hedge_paths: pd.DataFrame
    hedging_metrics: pd.DataFrame


def download_market_data(
    *,
    start: str = "2019-01-01",
    end: str = "2025-01-01",
    tickers: tuple[str, ...] = ("SPY", "^VIX", "XLK", "^IRX"),
) -> pd.DataFrame:
    """Download adjusted daily market data from Yahoo Finance."""

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")
    if not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Expected a MultiIndex column frame from yfinance.")

    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    return raw.sort_index()


def _field(raw: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    values = raw[(field, ticker)].astype(float)
    values.name = ticker
    return values


def build_market_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Create a clean daily SPY/VIX/sector/rate market frame."""

    frame = pd.DataFrame(
        {
            "spy_close": _field(raw, "Adj Close", "SPY"),
            "spy_volume": _field(raw, "Volume", "SPY"),
            "vix": _field(raw, "Adj Close", "^VIX") / 100.0,
            "xlk_close": _field(raw, "Adj Close", "XLK"),
            "irx_rate": _field(raw, "Adj Close", "^IRX") / 100.0,
        }
    )
    return frame.ffill().dropna()


def build_signal_features(
    market: pd.DataFrame,
    *,
    annualization: int = 252,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build point-in-time risk features and the base volatility path."""

    log_returns = np.log(market["spy_close"] / market["spy_close"].shift(1)).dropna()
    xlk_returns = np.log(market["xlk_close"] / market["xlk_close"].shift(1))

    rv21 = log_returns.rolling(21, min_periods=21).std() * np.sqrt(annualization)
    rv63 = log_returns.rolling(63, min_periods=63).std() * np.sqrt(annualization)
    base_volatility = RollingGaussianVolatility(window=21).transform(log_returns)
    base_volatility = base_volatility.clip(lower=0.05, upper=1.50)

    vix_min = market["vix"].rolling(126, min_periods=63).min()
    vix_max = market["vix"].rolling(126, min_periods=63).max()
    iv_rank = ((market["vix"] - vix_min) / (vix_max - vix_min)).clip(0.0, 1.0)

    volume_mean = market["spy_volume"].rolling(21, min_periods=21).mean()
    volume_shock = np.log(market["spy_volume"] / volume_mean)

    drawdown_63 = (
        market["spy_close"] / market["spy_close"].rolling(63, min_periods=63).max()
        - 1.0
    ).abs()
    sector_dispersion_21 = (
        xlk_returns.rolling(21, min_periods=21).sum()
        - log_returns.rolling(21, min_periods=21).sum()
    ).abs()
    rate_change_21 = market["irx_rate"].diff(21).abs()

    features = pd.DataFrame(
        {
            "iv_rank": iv_rank,
            "vix_level": market["vix"],
            "vol_ratio": rv21 / rv63,
            "volume_shock": volume_shock,
            "market_drawdown_63": drawdown_63,
            "sector_dispersion_21": sector_dispersion_21,
            "rate_change_21": rate_change_21,
        }
    ).replace([np.inf, -np.inf], np.nan)
    features = features.dropna()
    base_volatility = base_volatility.reindex(features.index).dropna()
    features = features.reindex(base_volatility.index).dropna()
    base_volatility = base_volatility.reindex(features.index)
    return features, base_volatility


def fit_market_signal_model(train_features: pd.DataFrame) -> MarketSignalRiskModel:
    """Fit the transparent MSA-Delta risk-state model."""

    weights = {
        "iv_rank": 0.30,
        "vix_level": 0.20,
        "vol_ratio": 0.20,
        "volume_shock": 0.10,
        "market_drawdown_63": 0.10,
        "sector_dispersion_21": 0.05,
        "rate_change_21": 0.05,
    }
    return MarketSignalRiskModel(
        weights=weights,
        elevated_quantile=0.70,
        stress_quantile=0.90,
        volatility_multipliers={NORMAL: 1.00, ELEVATED: 1.15, STRESS: 1.40},
    ).fit(train_features)


def run_real_market_pipeline(
    *,
    start: str = "2019-01-01",
    end: str = "2025-01-01",
    split_fraction: float = 0.65,
    maturity_days: int = 21,
    start_step: int = 21,
    rate: float = 0.03,
    transaction_cost_rate: float = 0.0005,
) -> RealMarketPipelineResult:
    """Download, process, model, and backtest the real-market dataset."""

    raw = download_market_data(start=start, end=end)
    market = build_market_frame(raw)
    features, base_volatility = build_signal_features(market)

    split = int(split_fraction * len(features))
    train_features = features.iloc[:split]
    test_features = features.iloc[split:]

    risk_model = fit_market_signal_model(train_features)
    risk_diagnostics = risk_model.diagnostic_frame(test_features, base_volatility)

    test_prices = market["spy_close"].reindex(test_features.index).dropna()
    episodes = generate_option_episodes(
        test_prices,
        maturity_days=maturity_days,
        start_step=start_step,
        moneyness=1.0,
    )
    vix_volatility = market["vix"].clip(lower=0.05, upper=1.50)

    strategies = {
        "FV-BS": lambda episode: float(base_volatility.loc[episode.start_date]),
        "Rolling-BS": lambda episode: base_volatility.reindex(
            episode.prices.index[:-1]
        ),
        "VIX-BS": lambda episode: vix_volatility.reindex(episode.prices.index[:-1]),
        "MSA-Delta": lambda episode: risk_diagnostics[
            "msa_delta_volatility"
        ].reindex(episode.prices.index[:-1]),
    }

    episode_results, hedge_paths = run_strategy_comparison(
        episodes,
        strategies=strategies,
        pricing_strategy="FV-BS",
        rate=rate,
        transaction_cost_rate=transaction_cost_rate,
    )
    hedging_metrics = summarize_hedging_performance(episode_results)
    return RealMarketPipelineResult(
        market=market,
        features=features,
        risk_diagnostics=risk_diagnostics,
        episode_results=episode_results,
        hedge_paths=hedge_paths,
        hedging_metrics=hedging_metrics,
    )


def save_outputs(
    result: RealMarketPipelineResult,
    *,
    tables_dir: str | Path = "results/tables",
    figures_dir: str | Path = "docs/figures",
) -> None:
    """Save result tables and figures."""

    tables = Path(tables_dir)
    figures = Path(figures_dir)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    result.episode_results.to_csv(tables / "real_market_episode_results.csv", index=False)
    result.hedge_paths.to_csv(tables / "real_market_hedge_paths.csv", index=False)
    result.hedging_metrics.to_csv(tables / "real_market_hedging_metrics.csv")
    result.risk_diagnostics.to_csv(tables / "real_market_risk_diagnostics.csv")

    _plot_market_and_vix(result, figures / "real_market_spy_vix.png")
    _plot_risk_states(result, figures / "real_market_risk_states.png")
    _plot_hedging_metrics(result, figures / "real_market_hedging_metrics.png")
    _plot_episode_errors(result, figures / "real_market_episode_errors.png")


def _plot_market_and_vix(result: RealMarketPipelineResult, path: Path) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    result.market["spy_close"].plot(ax=ax[0], color="black", lw=1.4)
    ax[0].set_title("SPY adjusted close")
    ax[0].set_ylabel("Price")
    result.market["vix"].mul(100).plot(ax=ax[1], color="tab:red", lw=1.2)
    ax[1].set_title("VIX option-implied volatility signal")
    ax[1].set_ylabel("VIX")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_risk_states(result: RealMarketPipelineResult, path: Path) -> None:
    colors = {NORMAL: "tab:green", ELEVATED: "tab:orange", STRESS: "tab:red"}
    diagnostics = result.risk_diagnostics
    fig, ax = plt.subplots(figsize=(10, 4))
    diagnostics["risk_score"].plot(ax=ax, color="black", lw=1.1)
    for state, frame in diagnostics.groupby("risk_state", sort=False):
        ax.scatter(
            frame.index,
            frame["risk_score"],
            s=18,
            color=colors[state],
            label=state,
            alpha=0.75,
        )
    ax.set_title("Out-of-sample MSA-Delta risk score and states")
    ax.set_ylabel("Risk score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_hedging_metrics(result: RealMarketPipelineResult, path: Path) -> None:
    metrics = result.hedging_metrics[["mae", "rmse", "loss_es_95"]]
    ax = metrics.plot(kind="bar", figsize=(9, 4), rot=0)
    ax.set_title("Out-of-sample hedging error metrics")
    ax.set_ylabel("Dollars per option share")
    ax.legend(["MAE", "RMSE", "95% loss ES"])
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def _plot_episode_errors(result: RealMarketPipelineResult, path: Path) -> None:
    pivot = result.episode_results.pivot(
        index="start_date",
        columns="strategy",
        values="hedging_error",
    )
    ax = pivot.plot(figsize=(10, 4), lw=1.2)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("Episode terminal hedging errors")
    ax.set_ylabel("Terminal hedging error")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def main() -> None:
    result = run_real_market_pipeline()
    save_outputs(result)
    print("Out-of-sample hedging metrics")
    print(result.hedging_metrics.round(6))
    print("\nRisk-state counts")
    print(result.risk_diagnostics["risk_state"].value_counts())


if __name__ == "__main__":
    main()
