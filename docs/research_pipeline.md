# Research Pipeline

## Estimand

The primary outcome is paired reduction in absolute terminal replication error
relative to FV-BS. Positive improvement means a strategy produced a smaller
absolute error on the same standardized claim and underlying path.

## Data and timing

Daily SPY, VIX, XLK, and 13-week Treasury-bill observations are normalized into
`data/processed/public_market_daily.csv`. Its companion JSON records source,
coverage, split dates, selected validation configuration, and SHA-256 digest.
Every feature at date `t` uses observations dated at or before `t`.

## Chronological workflow

1. Fit feature scaling and score thresholds on training observations.
2. Evaluate three predeclared risk-state configurations on validation data.
3. Select the lowest validation MAE, breaking ties with RMSE.
4. Refit that configuration on training plus validation observations.
5. Generate the test contract grid without inspecting test outcomes.
6. Run all strategies with identical claims, premiums, calendars, and costs.
7. Report aggregate and maturity/moneyness-specific results.
8. Use paired moving date-block bootstrap intervals for overlapping episodes.

## Interpretation boundary

The experiment evaluates replication of model-defined European payoffs on
historical market paths. It is not an exchange-option backtest. Consequently,
historical option pricing error, bid-ask execution, early exercise, and
contract-specific volatility-surface fit are outside the estimand.
