# Research Pipeline

The repository now implements the full computational path needed for the
empirical project:

1. Validate adjusted-close and volume data.
2. Construct leakage-safe market signals.
3. Construct future realized-volatility labels.
4. Split observations chronologically.
5. Fit a positive ridge volatility forecaster.
6. Produce fixed, rolling Gaussian, and signal-based volatility inputs.
7. Generate repeated European-call episodes.
8. Apply the same Black-Scholes delta-hedging engine to every strategy.
9. Compare forecast accuracy, hedging error, tail loss, turnover, and costs.

## Fair comparison convention

Every strategy receives the same option premium within an episode. The premium
is calculated from the strategy selected as `pricing_strategy`. This prevents a
strategy from appearing to hedge better merely because it sold the option at a
different model price.

## Leakage convention

Signals at date `t` use only observations dated at or before `t`. Future
realized volatility is a training label and is never included among the input
features. Model evaluation uses chronological train/test splits.

## Remaining empirical work

The code is complete enough to run once real data are supplied. The remaining
project work is empirical rather than structural:

- choose the underlying and sample period;
- obtain and document the data source;
- select interest-rate and optional external risk signals;
- tune windows and ridge penalty using training/validation data only;
- run robustness checks and interpret results;
- prepare final figures, executive summary, and presentation.
