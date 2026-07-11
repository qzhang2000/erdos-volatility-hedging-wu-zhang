# Market-Signal-Enhanced Delta Hedging on Historical Equity Paths

**Erdos Institute Quantitative Finance Project - Summer 2026**
**Authors:** Yue Wu and Freda Zhang

## Research question

Can publicly available, point-in-time market signals improve replication of
standardized European-call payoffs relative to fixed-volatility Black-Scholes
hedging?

This is a controlled historical-path experiment. SPY prices and all hedge
signals are observed market data, while the option claims are model-defined.

## Experimental design

At each eligible start date the experiment defines European calls over the
observed future SPY path. The held-out contract grid uses:

- 10, 21, and 42 trading-day maturities;
- 0.95, 1.00, and 1.05 strike/spot ratios;
- starts every five trading days;
- daily rebalancing and 5 bps proportional transaction costs;
- one common FV-BS premium within each contract episode.

The common premium isolates payoff-replication performance from model-pricing
differences. Overlapping starts enlarge coverage; moving date-block bootstrap
confidence intervals address the resulting dependence.

Four volatility inputs are compared through the same Black-Scholes delta hedge:

1. **FV-BS:** trailing volatility fixed at initiation.
2. **Rolling-BS:** trailing realized volatility updated daily.
3. **VIX-BS:** VIX used as a forward-looking market-volatility proxy.
4. **MSA-Delta:** rolling volatility multiplied according to a composite market
   risk state using VIX, realized-volatility acceleration, volume, drawdown,
   sector divergence, and short-rate changes.

VIX is derived from S&P 500 index options; it is not contract-level SPY implied
volatility. It is used here as a freely available option-market risk proxy.

## Leakage control and model selection

Observations are split chronologically:

- training: feature scaling and risk thresholds;
- validation: selection among three predeclared MSA configurations;
- test: one final evaluation of the chosen configuration.

After selection, the risk model is refit on training plus validation data. Test
outcomes never determine weights, quantiles, or volatility multipliers.

## Held-out results

The committed run covers 1,149 contracts per strategy.

| Strategy | MAE | RMSE | 95% loss ES | Average cost |
| --- | ---: | ---: | ---: | ---: |
| FV-BS | **1.5523** | 2.5422 | 6.7463 | 0.4631 |
| Rolling-BS | 1.6074 | 2.4385 | 6.4496 | 0.4690 |
| VIX-BS | 1.6956 | **2.3993** | **5.5171** | **0.4580** |
| MSA-Delta | 1.6906 | 2.5368 | 6.5819 | 0.4662 |

The result does not support a blanket claim that more signals improve hedging.
FV-BS has the best average absolute error. VIX-BS reduces squared error and tail
loss, suggesting value in option-market information for large-error episodes.
The validated composite MSA rule does not beat the simpler benchmarks on MAE.
Bootstrap intervals in `results/tables/replication_bootstrap_intervals.csv`
quantify uncertainty in paired improvements.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
python examples/run_real_market_pipeline.py
```

The pipeline downloads public daily data and writes:

- a normalized snapshot and SHA-256 metadata under `data/processed/`;
- validation, test, robustness, and uncertainty tables under `results/tables/`;
- final figures under `docs/figures/`.

The primary report is
[`notebooks/03_real_market_msa_delta_backtest.ipynb`](notebooks/03_real_market_msa_delta_backtest.ipynb).

## Repository map

```text
src/option_hedging/       pricing, signals, strategies, and evaluation
examples/                 reproducible pipeline entry points
notebooks/                demonstrations and final empirical report
data/processed/           frozen normalized public-data snapshot and metadata
results/tables/           committed empirical outputs
docs/figures/             committed result figures
tests/                    automated tests
```

## Limitations

- No historical option bids, asks, spreads, or contract-specific implied vols.
- Standardized European claims omit SPY option early exercise and dividends.
- VIX is an index-level 30-day volatility proxy.
- Overlapping episodes are dependent; bootstrap inference is approximate.
- Results concern this sample, contract grid, and cost model, not live trading.

Historical contract data would be a valuable paid-data extension, but it is not
required for the controlled payoff-replication question studied here.
