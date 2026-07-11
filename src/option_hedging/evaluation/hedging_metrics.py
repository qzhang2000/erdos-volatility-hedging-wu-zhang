"""Portfolio-level metrics for comparing option hedging strategies."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {
    "strategy",
    "hedging_error",
    "transaction_cost",
    "turnover_notional",
}


def _expected_shortfall(losses: pd.Series, confidence: float) -> float:
    threshold = float(losses.quantile(confidence))
    tail = losses[losses >= threshold]
    return float(tail.mean()) if len(tail) else threshold


def summarize_hedging_performance(
    results: pd.DataFrame,
    *,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    """Aggregate episode-level hedging outcomes by strategy.

    Hedging loss is defined as ``-hedging_error``: a negative replication error
    is therefore a positive loss to the option seller.
    """

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")
    missing = sorted(_REQUIRED_COLUMNS.difference(results.columns))
    if missing:
        raise KeyError(f"results is missing required columns: {', '.join(missing)}")
    if results.empty:
        raise ValueError("results cannot be empty.")
    if any(not 0 < level < 1 for level in confidence_levels):
        raise ValueError("confidence levels must lie strictly between zero and one.")

    rows: list[dict[str, float | int | str]] = []
    for strategy, frame in results.groupby("strategy", sort=True):
        errors = frame["hedging_error"].astype(float)
        losses = -errors
        absolute_errors = errors.abs()

        row: dict[str, float | int | str] = {
            "strategy": strategy,
            "n_episodes": int(len(frame)),
            "mean_error": float(errors.mean()),
            "mae": float(absolute_errors.mean()),
            "rmse": float(math.sqrt(float((errors.pow(2)).mean()))),
            "error_std": float(errors.std(ddof=1)) if len(errors) > 1 else 0.0,
            "max_absolute_error": float(absolute_errors.max()),
            "mean_transaction_cost": float(frame["transaction_cost"].mean()),
            "mean_turnover_notional": float(frame["turnover_notional"].mean()),
        }
        for level in confidence_levels:
            suffix = str(int(round(level * 100)))
            row[f"absolute_error_q{suffix}"] = float(
                absolute_errors.quantile(level)
            )
            row[f"loss_var_{suffix}"] = float(losses.quantile(level))
            row[f"loss_es_{suffix}"] = _expected_shortfall(losses, level)
        rows.append(row)

    return pd.DataFrame(rows).set_index("strategy").sort_index()


def volatility_forecast_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> pd.Series:
    """Return MAE, RMSE, bias, and correlation for volatility forecasts."""

    if not isinstance(actual, pd.Series) or not isinstance(predicted, pd.Series):
        raise TypeError("actual and predicted must be pandas Series.")
    aligned = pd.concat(
        [actual.rename("actual"), predicted.rename("predicted")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("actual and predicted have no overlapping valid observations.")
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("actual and predicted must contain only finite values.")

    error = aligned["predicted"] - aligned["actual"]
    correlation = aligned["actual"].corr(aligned["predicted"])
    return pd.Series(
        {
            "n_observations": float(len(aligned)),
            "mae": float(error.abs().mean()),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "bias": float(error.mean()),
            "correlation": float(correlation) if pd.notna(correlation) else np.nan,
        },
        name="volatility_forecast_metrics",
    )


def paired_block_bootstrap(
    results: pd.DataFrame,
    *,
    baseline: str = "FV-BS",
    block_days: int = 21,
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Estimate paired MAE improvements with a moving date-block bootstrap.

    Improvement is ``|baseline error| - |strategy error|``; positive values
    favor the strategy. Sampling contiguous start-date blocks preserves much
    of the dependence induced by overlapping option episodes.
    """

    required = {"strategy", "start_date", "maturity_days", "moneyness", "hedging_error"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise KeyError(f"results is missing required columns: {', '.join(missing)}")
    if block_days < 1 or n_bootstrap < 1:
        raise ValueError("block_days and n_bootstrap must be positive.")

    keyed = results.copy()
    keyed["start_date"] = pd.to_datetime(keyed["start_date"])
    pivot = keyed.pivot_table(
        index=["start_date", "maturity_days", "moneyness"],
        columns="strategy",
        values="hedging_error",
        aggfunc="first",
    ).dropna()
    if baseline not in pivot.columns:
        raise KeyError(f"baseline strategy {baseline!r} is unavailable.")

    dates = pd.Index(sorted(pivot.index.get_level_values("start_date").unique()))
    if len(dates) < 2:
        raise ValueError("at least two episode start dates are required.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    for strategy in sorted(set(pivot.columns).difference({baseline})):
        paired = pivot[[baseline, strategy]].dropna()
        improvement = paired[baseline].abs() - paired[strategy].abs()
        date_improvement = improvement.groupby(level="start_date").mean().reindex(dates)
        values = date_improvement.to_numpy(dtype=float)
        observed = float(np.nanmean(values))
        draws = np.empty(n_bootstrap)
        for draw in range(n_bootstrap):
            starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block_days)))
            positions = np.concatenate([
                (start + np.arange(block_days)) % len(values) for start in starts
            ])[: len(values)]
            draws[draw] = float(np.nanmean(values[positions]))
        rows.append({
            "strategy": strategy,
            "mean_mae_improvement": observed,
            "ci_2_5": float(np.quantile(draws, 0.025)),
            "ci_97_5": float(np.quantile(draws, 0.975)),
            "probability_improvement": float(np.mean(draws > 0.0)),
        })
    return pd.DataFrame(rows).set_index("strategy")
