# Dynamic Volatility Modeling for European Option Hedging

**Erdős Institute Quantitative Finance Project — Summer 2026**

**Authors:** Yue Wu and Freda Zhang

## Overview

This project studies whether time-varying volatility estimates can improve the delta hedging of European call options.

The Black–Scholes model assumes constant volatility, while real market volatility changes over time. We compare a standard fixed-volatility hedge with adaptive approaches that update volatility using recent returns and market signals.

## Research Question

> Can dynamic volatility estimates reduce out-of-sample hedging error relative to constant-volatility Black–Scholes delta hedging?

## Methods

We plan to compare:

- fixed historical volatility;
- rolling Gaussian volatility;
- signal-based volatility forecasts.

Each volatility estimate is used in the Black–Scholes delta formula to construct and rebalance a stock-and-cash hedging portfolio.

## Evaluation

The strategies will be evaluated using:

- terminal hedging error;
- mean absolute error;
- root mean squared error;
- tail losses;
- turnover;
- transaction costs;
- performance during high-volatility periods.

All experiments will use chronological train/test splits to avoid look-ahead bias.

## Current Status

The repository currently includes:

- European call and put payoff functions;
- Black–Scholes call and put pricing;
- Black–Scholes call and put delta calculations;
- input validation and expiration handling;
- unit tests.

## Repository Structure

```text
erdos-volatility-hedging-wu-zhang/
├── data/
├── notebooks/
├── results/
├── src/
│   └── option_hedging/
│       └── derivatives/
│           └── black_scholes.py
├── tests/
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
git clone https://github.com/qzhang2000/erdos-volatility-hedging-wu-zhang.git
cd erdos-volatility-hedging-wu-zhang

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Run Tests

```bash
pytest
```

## Roadmap

- [x] Implement Black–Scholes prices and deltas.
- [x] Add unit tests.
- [ ] Add historical market data.
- [ ] Implement fixed and rolling volatility models.
- [ ] Build the delta-hedging backtester.
- [ ] Add transaction costs and hedging metrics.
- [ ] Implement signal-based volatility forecasting.
- [ ] Produce final notebooks and project summary.
