# Dynamic Volatility Modeling for European Option Hedging

**Erdos Institute Quantitative Finance Project - Summer 2026**

**Authors:** Yue Wu and Freda Zhang

## Project Overview

This project studies whether dynamic volatility models can improve European
option delta hedging relative to a constant-volatility Black-Scholes benchmark.

The lecture notes motivate three model classes for the project:

1. **Constant-volatility Black-Scholes** as the benchmark.
2. **Markov regime-switching volatility**, including both finite-state
   discrete-time chains and continuous-time two-state chains.
3. **Heston stochastic volatility**, where variance is a mean-reverting
   stochastic process calibrated to option prices.

The project should therefore be framed around Black-Scholes, Markov
regime-switching, and Heston hedging, rather than generic volatility or
risk-signal labels.

## Research Question

> Can Markov regime-switching or Heston stochastic-volatility models reduce
> chronological out-of-sample hedging error relative to a fixed-volatility
> Black-Scholes delta hedge?

## Project Scope

The project will compare three hedging approaches under the same data,
rebalancing, and transaction-cost assumptions.

- **Benchmark hedge:** Black-Scholes delta hedge using one fixed historical
  volatility estimate.
- **Markov regime-switching hedge:** delta hedge using low- and high-volatility
  regimes. The finite-state discrete-time chain supports simulation and regime
  intuition. The continuous-time two-state chain supports Markov-modulated GBM
  option pricing, implied-volatility curves, and model deltas.
- **Heston stochastic-volatility hedge:** delta hedge using Heston prices and
  deltas from a calibrated mean-reverting variance process.

All final comparisons must be chronological and out of sample. Information
available after a hedge date, including future returns, later option quotes, or
future calibration targets, must not be used for that hedge decision.

## Data Inputs

The minimum dataset should include:

- underlying adjusted close prices;
- daily returns and realized volatility;
- option-chain records with quote date, expiration, strike, option type, bid,
  ask, mid price, and implied volatility when available;
- risk-free interest rates or a documented constant-rate assumption;
- calendar fields needed to compute time to maturity.

Raw data belongs in `data/raw/`. Cleaned and aligned datasets belong in
`data/processed/`.

Data cleaning should preserve point-in-time availability. Missing quotes, stale
quotes, crossed markets, and zero-bid contracts should be filtered before
calibration or hedging evaluation.

## Model Plan

### Black-Scholes Baseline

Use the standard Black-Scholes price and delta formulas with fixed historical
volatility. This benchmark isolates the value of moving beyond a single
constant volatility input.

### Markov Regime-Switching Model

Use a two-state volatility model with low- and high-volatility regimes.

The discrete-time finite-state chain describes transitions over a fixed time
grid:

```text
P = [[p_LL, p_LH],
     [p_HL, p_HH]]
```

The continuous-time two-state chain uses transition rates and generator:

```text
Q = [[-alpha, alpha],
     [ beta, -beta]]
```

Under Markov-modulated GBM, the stock follows Black-Scholes dynamics locally,
but the volatility changes with the Markov state. Option prices can be computed
by averaging Black-Scholes prices over the occupation-time distribution of the
high-volatility state. This produces non-flat implied-volatility curves and
model-based deltas.

### Heston Model

Use Heston's stochastic variance process:

```text
dS_t / S_t = r dt + sqrt(v_t) dW_t^S
dv_t       = kappa(theta - v_t) dt + xi sqrt(v_t) dW_t^v
```

with correlation `rho` between the stock and variance shocks. Heston is a
separate stochastic-volatility model, not just a volatility signal. The model
should be calibrated across strikes and maturities, then evaluated through the
same delta-hedging backtester used for the Black-Scholes and Markov approaches.

## Evaluation

Hedging strategies will be evaluated using:

- terminal hedging error;
- mean absolute hedging error;
- root mean squared hedging error;
- 95th and 99th percentile absolute error;
- hedging-loss Value at Risk and Expected Shortfall;
- transaction costs;
- share and notional turnover;
- performance during high-volatility or regime-transition periods.

Model quality will be evaluated using:

- option pricing error;
- implied-volatility fit across strikes and maturities;
- Markov transition-rate and regime-persistence stability;
- Heston calibration stability;
- sensitivity to calibration windows, hedge frequency, and transaction costs.

## Current Implementation

The package currently includes reusable infrastructure for:

- European call and put payoff functions;
- Black-Scholes call and put pricing;
- Black-Scholes call and put delta calculations;
- market-data loading, validation, and return calculations;
- fixed historical and rolling volatility estimators;
- point-in-time signal construction and leakage checks;
- signal-based volatility forecasting;
- discrete-time Black-Scholes delta hedging with transaction costs;
- repeated option episodes and strategy-level hedging metrics;
- unit tests for pricing, data, volatility, signals, backtesting, and metrics.

The Markov and Heston lecture notebooks provide the next model implementations
to promote into the package. The existing signal and rolling-volatility code can
remain useful as supporting infrastructure, but it should not replace the
Markov and Heston model comparison.

## Repository Structure

```text
erdos-volatility-hedging-wu-zhang/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── examples/
├── notebooks/
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   └── option_hedging/
│       ├── backtesting/
│       ├── data/
│       ├── derivatives/
│       ├── evaluation/
│       ├── models/
│       ├── signals/
│       └── strategies/
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Implement Black-Scholes prices and deltas.
- [x] Add market-data validation and return utilities.
- [x] Implement fixed and rolling volatility estimators.
- [x] Build a delta-hedging backtester with transaction costs.
- [x] Add strategy-level backtesting and hedging metrics.
- [ ] Add raw option-chain and underlying market data.
- [ ] Promote Markov regime-switching pricing and delta code into `src/option_hedging/models/`.
- [ ] Promote Heston pricing, delta, and calibration code into `src/option_hedging/models/`.
- [ ] Run chronological out-of-sample hedging comparisons.
- [ ] Generate final tables, figures, notebook, report, and presentation materials.
