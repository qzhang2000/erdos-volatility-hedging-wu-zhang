import numpy as np
import pandas as pd
import pytest

from option_hedging.signals import (
    SignalLibrary,
    create_future_realized_volatility_target,
)


def sample_panel(periods: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    log_returns = 0.0003 + 0.01 * np.sin(np.arange(periods) / 11.0)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    volume = 1_000_000.0 + 100_000.0 * np.cos(np.arange(periods) / 9.0)
    return pd.DataFrame(
        {
            "date": dates,
            "asset": "TEST",
            "close": close,
            "volume": volume,
        }
    )


def test_default_signal_library_computes_selected_features() -> None:
    panel = sample_panel()
    library = SignalLibrary.default()
    features = library.compute(panel)

    expected = {
        "abs_return_1d",
        "mom_5d",
        "mom_21d",
        "vol_5d",
        "vol_21d",
        "vol_63d",
        "vol_ratio_5d_21d",
        "vol_ratio_21d_63d",
        "downside_vol_21d",
        "drawdown_63d",
        "drawdown_252d",
        "log_dollar_volume_21d",
        "amihud_21d",
        "skew_63d",
        "downside_freq_21d",
    }
    assert expected.issubset(features.columns)
    assert len(features) == len(panel)


def test_signals_do_not_use_future_prices() -> None:
    panel = sample_panel()
    altered = panel.copy()
    altered.loc[altered.index[-1], "close"] *= 5.0

    library = SignalLibrary.default()
    original = library.compute(panel)
    changed = library.compute(altered)

    pd.testing.assert_series_equal(
        original.iloc[-2],
        changed.iloc[-2],
        check_names=False,
    )


def test_future_realized_volatility_alignment() -> None:
    prices = [100.0, 110.0, 121.0, 133.1]
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4),
            "asset": "TEST",
            "close": prices,
        }
    )
    target = create_future_realized_volatility_target(
        panel,
        horizon=2,
        annualization=2,
    )

    expected = np.sqrt(2 * np.log(1.1) ** 2)
    assert target.iloc[0]["future_realized_vol_2d"] == pytest.approx(expected)
    assert target["future_realized_vol_2d"].iloc[-2:].isna().all()


def test_signal_library_skips_missing_volume_features() -> None:
    panel = sample_panel().drop(columns="volume")
    features = SignalLibrary.default().compute(panel, on_missing="skip")

    assert "vol_21d" in features.columns
    assert "amihud_21d" not in features.columns
    skipped = {item["name"] for item in features.attrs["skipped_signals"]}
    assert {"amihud_21d", "log_dollar_volume_21d"}.issubset(skipped)
