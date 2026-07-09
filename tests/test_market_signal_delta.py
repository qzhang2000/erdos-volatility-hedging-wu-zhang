import pandas as pd
import pytest

from option_hedging.strategies import (
    ELEVATED,
    NORMAL,
    STRESS,
    MarketSignalRiskModel,
)


def sample_features() -> pd.DataFrame:
    index = pd.bdate_range("2026-01-02", periods=10)
    return pd.DataFrame(
        {
            "iv_rank": [0.05, 0.10, 0.20, 0.35, 0.45, 0.55, 0.70, 0.82, 0.90, 0.98],
            "vol_ratio": [0.80, 0.90, 1.00, 1.05, 1.10, 1.20, 1.35, 1.55, 1.70, 1.95],
            "volume_shock": [-0.40, -0.20, -0.05, 0.00, 0.10, 0.25, 0.45, 0.70, 0.95, 1.30],
        },
        index=index,
    )


def fitted_model() -> MarketSignalRiskModel:
    model = MarketSignalRiskModel(
        weights={"iv_rank": 0.5, "vol_ratio": 0.3, "volume_shock": 0.2},
        elevated_quantile=0.60,
        stress_quantile=0.85,
        volatility_multipliers={NORMAL: 1.0, ELEVATED: 1.2, STRESS: 1.6},
    )
    return model.fit(sample_features().iloc[:7])


def test_risk_model_classifies_out_of_sample_states() -> None:
    features = sample_features()
    model = fitted_model()

    states = model.classify(features.iloc[7:])

    assert states.index.equals(features.index[7:])
    assert states.iloc[-1] == STRESS
    assert set(states).issubset({NORMAL, ELEVATED, STRESS})


def test_adjusted_volatility_applies_state_multipliers() -> None:
    features = sample_features()
    model = fitted_model()
    base = pd.Series(0.20, index=features.index)

    adjusted = model.adjusted_volatility(features.iloc[7:], base)
    diagnostics = model.diagnostic_frame(features.iloc[7:], base)

    assert adjusted.name == "msa_delta_volatility"
    assert adjusted.index.equals(features.index[7:])
    assert (adjusted >= 0.20).all()
    assert diagnostics.loc[features.index[-1], "risk_state"] == STRESS
    assert diagnostics.loc[features.index[-1], "volatility_multiplier"] == pytest.approx(1.6)
    assert diagnostics.loc[features.index[-1], "msa_delta_volatility"] == pytest.approx(0.32)


def test_thresholds_are_fit_from_training_sample_only() -> None:
    features = sample_features()
    model = fitted_model()
    original_threshold = model.stress_threshold_

    shocked = features.copy()
    shocked.iloc[-1, shocked.columns.get_loc("iv_rank")] = 100.0
    model.classify(shocked.iloc[7:])

    assert model.stress_threshold_ == original_threshold


def test_missing_feature_raises_key_error() -> None:
    model = fitted_model()

    with pytest.raises(KeyError):
        model.classify(sample_features()[["iv_rank", "vol_ratio"]])


def test_invalid_quantiles_raise_value_error() -> None:
    with pytest.raises(ValueError):
        MarketSignalRiskModel(
            weights={"iv_rank": 1.0},
            elevated_quantile=0.90,
            stress_quantile=0.80,
        )
