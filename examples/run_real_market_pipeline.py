"""Controlled option-payoff replication on observed SPY market paths.

The contracts in this experiment are standardized European call claims, not
historical exchange quotes. Public daily market data determine the underlying
paths and point-in-time hedge signals. Hyperparameters are chosen on a
chronological validation period and evaluated once on the held-out test period.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from option_hedging.backtesting import generate_contract_grid, run_strategy_comparison
from option_hedging.evaluation import paired_block_bootstrap, summarize_hedging_performance
from option_hedging.models import RollingGaussianVolatility
from option_hedging.strategies import ELEVATED, NORMAL, STRESS, MarketSignalRiskModel


WEIGHTS = {
    "iv_rank": 0.30,
    "vix_level": 0.20,
    "vol_ratio": 0.20,
    "volume_shock": 0.10,
    "market_drawdown_63": 0.10,
    "sector_dispersion_21": 0.05,
    "rate_change_21": 0.05,
}


@dataclass(frozen=True)
class RiskConfiguration:
    name: str
    elevated_quantile: float
    stress_quantile: float
    elevated_multiplier: float
    stress_multiplier: float


CANDIDATES = (
    RiskConfiguration("conservative", 0.75, 0.92, 1.10, 1.25),
    RiskConfiguration("moderate", 0.70, 0.90, 1.15, 1.40),
    RiskConfiguration("responsive", 0.60, 0.85, 1.25, 1.60),
)


@dataclass(frozen=True)
class RealMarketPipelineResult:
    market: pd.DataFrame
    features: pd.DataFrame
    risk_diagnostics: pd.DataFrame
    tuning_results: pd.DataFrame
    selected_configuration: RiskConfiguration
    episode_results: pd.DataFrame
    hedge_paths: pd.DataFrame
    hedging_metrics: pd.DataFrame
    robustness_metrics: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    bootstrap_intervals: pd.DataFrame
    split_dates: dict[str, str]


def download_market_data(
    *, start: str = "2016-01-01", end: str = "2025-01-01",
    tickers: tuple[str, ...] = ("SPY", "^VIX", "XLK", "^IRX"),
) -> pd.DataFrame:
    """Download the public daily inputs used by the experiment."""
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=False,
                      progress=False, threads=False)
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Yahoo Finance returned no usable multi-ticker data.")
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    return raw.sort_index()


def _field(raw: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    values = raw[(field, ticker)].astype(float)
    values.name = ticker
    return values


def build_market_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize vendor columns into a stable, auditable daily panel."""
    return pd.DataFrame({
        "spy_close": _field(raw, "Adj Close", "SPY"),
        "spy_volume": _field(raw, "Volume", "SPY"),
        "vix": _field(raw, "Adj Close", "^VIX") / 100.0,
        "xlk_close": _field(raw, "Adj Close", "XLK"),
        "irx_rate": _field(raw, "Adj Close", "^IRX") / 100.0,
    }).ffill().dropna()


def build_signal_features(market: pd.DataFrame, *, annualization: int = 252) -> tuple[pd.DataFrame, pd.Series]:
    """Build trailing-only features and the rolling-volatility path."""
    returns = np.log(market["spy_close"] / market["spy_close"].shift(1)).dropna()
    sector_returns = np.log(market["xlk_close"] / market["xlk_close"].shift(1))
    rv21 = returns.rolling(21, min_periods=21).std() * np.sqrt(annualization)
    rv63 = returns.rolling(63, min_periods=63).std() * np.sqrt(annualization)
    base = RollingGaussianVolatility(window=21).transform(returns).clip(0.05, 1.50)
    vix_min = market["vix"].rolling(126, min_periods=63).min()
    vix_max = market["vix"].rolling(126, min_periods=63).max()
    features = pd.DataFrame({
        "iv_rank": ((market["vix"] - vix_min) / (vix_max - vix_min)).clip(0, 1),
        "vix_level": market["vix"],
        "vol_ratio": rv21 / rv63,
        "volume_shock": np.log(market["spy_volume"] / market["spy_volume"].rolling(21, min_periods=21).mean()),
        "market_drawdown_63": (market["spy_close"] / market["spy_close"].rolling(63, min_periods=63).max() - 1).abs(),
        "sector_dispersion_21": (sector_returns.rolling(21, min_periods=21).sum() - returns.rolling(21, min_periods=21).sum()).abs(),
        "rate_change_21": market["irx_rate"].diff(21).abs(),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    common = features.index.intersection(base.dropna().index)
    return features.loc[common], base.loc[common]


def _fit(features: pd.DataFrame, config: RiskConfiguration) -> MarketSignalRiskModel:
    return MarketSignalRiskModel(
        weights=WEIGHTS,
        elevated_quantile=config.elevated_quantile,
        stress_quantile=config.stress_quantile,
        volatility_multipliers={NORMAL: 1.0, ELEVATED: config.elevated_multiplier, STRESS: config.stress_multiplier},
    ).fit(features)


def _strategies(base: pd.Series, vix: pd.Series, adjusted: pd.Series):
    return {
        "FV-BS": lambda episode: float(base.loc[episode.start_date]),
        "Rolling-BS": lambda episode: base.reindex(episode.prices.index[:-1]),
        "VIX-BS": lambda episode: vix.reindex(episode.prices.index[:-1]),
        "MSA-Delta": lambda episode: adjusted.reindex(episode.prices.index[:-1]),
    }


def _evaluate_period(prices, base, vix, adjusted, *, maturity_days, moneyness,
                     start_step, transaction_cost_rate, keep_paths=True):
    episodes = generate_contract_grid(prices, maturity_days=maturity_days,
                                      moneyness=moneyness, start_step=start_step)
    summary, paths = run_strategy_comparison(
        episodes, strategies=_strategies(base, vix, adjusted), pricing_strategy="FV-BS",
        rate=0.03, transaction_cost_rate=transaction_cost_rate,
    )
    return summary, paths if keep_paths else paths.iloc[:0]


def run_real_market_pipeline(
    *, start: str = "2016-01-01", end: str = "2025-01-01",
    train_fraction: float = 0.50, validation_fraction: float = 0.20,
    maturity_days: tuple[int, ...] = (10, 21, 42),
    moneyness: tuple[float, ...] = (0.95, 1.00, 1.05),
    start_step: int = 5, transaction_cost_rate: float = 0.0005,
) -> RealMarketPipelineResult:
    """Tune on validation data and evaluate a contract grid once on test data."""
    market = build_market_frame(download_market_data(start=start, end=end))
    features, base = build_signal_features(market)
    n = len(features)
    train_end, validation_end = int(n * train_fraction), int(n * (train_fraction + validation_fraction))
    if not 0 < train_end < validation_end < n:
        raise ValueError("split fractions must leave nonempty train, validation, and test periods.")
    train, validation, test = features.iloc[:train_end], features.iloc[train_end:validation_end], features.iloc[validation_end:]
    vix = market["vix"].clip(0.05, 1.50)

    tuning_rows = []
    validation_prices = market["spy_close"].reindex(validation.index)
    for config in CANDIDATES:
        model = _fit(train, config)
        adjusted = model.adjusted_volatility(validation, base)
        results, _ = _evaluate_period(
            validation_prices, base, vix, adjusted, maturity_days=(21,), moneyness=(1.0,),
            start_step=10, transaction_cost_rate=transaction_cost_rate, keep_paths=False,
        )
        msa = results.loc[results["strategy"] == "MSA-Delta"]
        tuning_rows.append({**asdict(config), "validation_episodes": len(msa),
                            "validation_mae": msa["absolute_hedging_error"].mean(),
                            "validation_rmse": np.sqrt(np.mean(msa["hedging_error"] ** 2))})
    tuning = pd.DataFrame(tuning_rows).sort_values(["validation_mae", "validation_rmse"])
    selected = next(c for c in CANDIDATES if c.name == tuning.iloc[0]["name"])

    development = features.iloc[:validation_end]
    final_model = _fit(development, selected)
    diagnostics = final_model.diagnostic_frame(test, base)
    test_prices = market["spy_close"].reindex(test.index)
    results, paths = _evaluate_period(
        test_prices, base, vix, diagnostics["msa_delta_volatility"],
        maturity_days=maturity_days, moneyness=moneyness, start_step=start_step,
        transaction_cost_rate=transaction_cost_rate,
    )
    metrics = summarize_hedging_performance(results, confidence_levels=(0.90, 0.95))
    robustness = pd.concat([
        summarize_hedging_performance(group, confidence_levels=(0.90,))
        .reset_index().assign(maturity_days=mat, moneyness=money)
        for (mat, money), group in results.groupby(["maturity_days", "moneyness"])
    ], ignore_index=True)
    cost_frames = []
    for cost in (0.0, transaction_cost_rate, 0.001):
        if cost == transaction_cost_rate:
            cost_results = results
        else:
            cost_results, _ = _evaluate_period(
                test_prices, base, vix, diagnostics["msa_delta_volatility"],
                maturity_days=maturity_days, moneyness=moneyness,
                start_step=start_step, transaction_cost_rate=cost, keep_paths=False,
            )
        cost_frames.append(
            summarize_hedging_performance(cost_results, confidence_levels=(0.90,))
            .reset_index().assign(transaction_cost_rate=cost)
        )
    cost_sensitivity = pd.concat(cost_frames, ignore_index=True)
    bootstrap = paired_block_bootstrap(results, n_bootstrap=2000, block_days=5)
    split_dates = {
        "train": f"{train.index.min().date()} to {train.index.max().date()}",
        "validation": f"{validation.index.min().date()} to {validation.index.max().date()}",
        "test": f"{test.index.min().date()} to {test.index.max().date()}",
    }
    return RealMarketPipelineResult(market, features, diagnostics, tuning, selected,
                                    results, paths, metrics, robustness, cost_sensitivity,
                                    bootstrap, split_dates)


def save_outputs(result: RealMarketPipelineResult, *, tables_dir="results/tables",
                 figures_dir="docs/figures", processed_dir="data/processed") -> None:
    """Write analysis outputs plus a hashed normalized input snapshot."""
    tables, figures, processed = map(Path, (tables_dir, figures_dir, processed_dir))
    for directory in (tables, figures, processed): directory.mkdir(parents=True, exist_ok=True)
    snapshot = processed / "public_market_daily.csv"
    result.market.to_csv(snapshot, index_label="date")
    digest = sha256(snapshot.read_bytes()).hexdigest()
    metadata = {
        "experiment": "standardized European-call payoff replication; no historical option quotes",
        "source": "Yahoo Finance via yfinance",
        "tickers": ["SPY", "^VIX", "XLK", "^IRX"],
        "snapshot_sha256": digest,
        "rows": len(result.market), "first_date": str(result.market.index.min().date()),
        "last_date": str(result.market.index.max().date()), "splits": result.split_dates,
        "selected_configuration": asdict(result.selected_configuration),
    }
    (processed / "public_market_daily.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    outputs = {
        "replication_episode_results.csv": result.episode_results,
        "replication_hedge_paths.csv": result.hedge_paths,
        "replication_hedging_metrics.csv": result.hedging_metrics,
        "replication_robustness_metrics.csv": result.robustness_metrics,
        "replication_cost_sensitivity.csv": result.cost_sensitivity,
        "replication_bootstrap_intervals.csv": result.bootstrap_intervals,
        "replication_validation_tuning.csv": result.tuning_results,
        "replication_risk_diagnostics.csv": result.risk_diagnostics,
    }
    for name, frame in outputs.items(): frame.to_csv(tables / name)
    _plot_metrics(result, figures / "replication_hedging_metrics.png")
    _plot_improvements(result, figures / "replication_bootstrap_improvements.png")


def _plot_metrics(result, path):
    ax = result.hedging_metrics[["mae", "rmse", "loss_es_95"]].plot.bar(figsize=(9, 4), rot=0)
    ax.set(title="Held-out standardized-contract replication errors", ylabel="Dollars per option share")
    ax.figure.tight_layout(); ax.figure.savefig(path, dpi=160); plt.close(ax.figure)


def _plot_improvements(result, path):
    frame = result.bootstrap_intervals
    yerr = np.vstack([frame["mean_mae_improvement"] - frame["ci_2_5"], frame["ci_97_5"] - frame["mean_mae_improvement"]])
    ax = frame["mean_mae_improvement"].plot.bar(yerr=yerr, capsize=4, figsize=(8, 4), rot=0)
    ax.axhline(0, color="black", lw=.8); ax.set(title="Paired MAE improvement over FV-BS (95% block-bootstrap CI)", ylabel="Positive favors strategy")
    ax.figure.tight_layout(); ax.figure.savefig(path, dpi=160); plt.close(ax.figure)


def main() -> None:
    result = run_real_market_pipeline()
    save_outputs(result)
    print("Splits", result.split_dates)
    print("Selected", result.selected_configuration)
    print(result.hedging_metrics.round(4))
    print(result.bootstrap_intervals.round(4))


if __name__ == "__main__":
    main()
