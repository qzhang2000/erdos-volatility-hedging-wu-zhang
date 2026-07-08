import numpy as np
import pandas as pd
import pytest

from option_hedging.models import (
    RidgeVolatilityForecaster,
    expanding_window_predictions,
)


def sample_regression_data(n: int = 120):
    index = pd.bdate_range("2025-01-02", periods=n)
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.sin(np.arange(n) / 8.0)
    features = pd.DataFrame({"x1": x1, "x2": x2}, index=index)
    target = pd.Series(
        np.exp(-1.7 + 0.25 * x1 - 0.10 * x2),
        index=index,
        name="future_volatility",
    )
    return features, target


def test_ridge_forecaster_returns_positive_predictions() -> None:
    features, target = sample_regression_data()
    model = RidgeVolatilityForecaster(alpha=0.01).fit(features.iloc[:80], target.iloc[:80])

    prediction = model.predict(features.iloc[80:])

    assert prediction.index.equals(features.index[80:])
    assert (prediction > 0).all()
    assert np.mean(np.abs(prediction - target.iloc[80:])) < 0.01


def test_prediction_columns_must_match_training_columns() -> None:
    features, target = sample_regression_data()
    model = RidgeVolatilityForecaster().fit(features, target)

    with pytest.raises(ValueError):
        model.predict(features[["x1"]])


def test_expanding_window_predictions_start_after_training_period() -> None:
    features, target = sample_regression_data()
    prediction = expanding_window_predictions(
        features,
        target,
        initial_train_size=60,
        refit_every=10,
    )

    assert prediction.iloc[:60].isna().all()
    assert prediction.iloc[60:].notna().all()
    assert (prediction.iloc[60:] > 0).all()
