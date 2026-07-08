# Dynamic Volatility Modeling for European Option Hedging

**Erdős Institute Quantitative Finance Project — Summer 2026**

**Authors:** Yue Wu and Freda Zhang

## Project Overview

This project studies whether time-varying volatility estimates can improve the
delta hedging of European call options relative to a constant-volatility
Black–Scholes benchmark.

The project compares three volatility inputs:

1. **Fixed historical volatility** — estimated at option initiation and held
   constant through expiration.
2. **Rolling Gaussian volatility** — updated from the most recent return window.
3. **Signal-based volatility** — predicted from market signals using ridge
   regression on future realized volatility.

Every volatility estimate is inserted into the same Black–Scholes delta-hedging
engine. Strategies are evaluated chronologically using hedging error,
transaction costs, turnover, and tail losses.

## Research Question

> Can dynamic and signal-based volatility estimates reduce out-of-sample
> hedging error and tail losses relative to constant-volatility Black–Scholes
> delta hedging?

## Implemented Pipeline

The repository now supports the complete computational pipeline:

```text
market prices and volume
        ↓
leakage-safe market signals
        ↓
future realized-volatility target
        ↓
chronological train/test split
        ↓
ridge volatility forecast
        ↓
fixed / rolling / signal volatility paths
        ↓
repeated European call episodes
        ↓
Black–Scholes delta hedging
        ↓
forecast and hedging metrics
```

### Available signal groups

- recent return shock and absolute return;
- 5-day and 21-day momentum;
- 5-day, 21-day, and 63-day realized volatility;
- short-to-long volatility ratios;
- downside volatility;
- 63-day and 252-day drawdown;
- downside-return frequency and rolling skewness;
- rolling dollar volume and Amihud illiquidity.

All signal calculations use information available at or before the feature date.
The future realized-volatility target is used only as a supervised-learning
label.

## Repository Structure

```text
erdos-volatility-hedging-wu-zhang/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── research_pipeline.md
├── examples/
│   └── run_synthetic_pipeline.py
├── notebooks/
│   └── 01_end_to_end_demo.ipynb
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   └── option_hedging/
│       ├── backtesting/
│       │   └── episodes.py
│       ├── data/
│       │   └── market_data.py
│       ├── derivatives/
│       │   └── black_scholes.py
│       ├── evaluation/
│       │   └── hedging_metrics.py
│       ├── models/
│       │   ├── signal_volatility.py
│       │   └── volatility.py
│       ├── signals/
│       │   ├── base.py
│       │   ├── definitions.py
│       │   ├── library.py
│       │   ├── targets.py
│       │   ├── transforms.py
│       │   └── validation.py
│       └── strategies/
│           └── delta_hedging.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
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

## Run the Tests

```bash
pytest
```

The completed repository contains tests for:

- Black–Scholes prices, payoffs, and deltas;
- market-data validation and return calculations;
- fixed and rolling volatility estimators;
- signal construction and leakage checks;
- future realized-volatility alignment;
- ridge volatility forecasts;
- discrete-time delta hedging and transaction costs;
- repeated option episodes and strategy-level metrics.

## Run the End-to-End Demonstration

```bash
python examples/run_synthetic_pipeline.py
```

This example generates a reproducible synthetic market, trains the signal model,
compares three hedging strategies, and writes output tables to `results/tables/`.
It is a software demonstration, not an empirical project result.

The same workflow is also presented in:

```text
notebooks/01_end_to_end_demo.ipynb
```

## Fair Strategy Comparison

Within each option episode, all hedging strategies receive the same initial
option premium. This separates hedging performance from differences in model
pricing. The premium is calculated using a designated pricing strategy, while
each hedge may use its own volatility path for delta calculations.

## Evaluation Metrics

Volatility forecasts are evaluated using:

- mean absolute error;
- root mean squared error;
- bias;
- forecast correlation.

Hedging strategies are evaluated using:

- mean terminal hedging error;
- mean absolute hedging error;
- root mean squared hedging error;
- error standard deviation;
- 95th and 99th percentile absolute error;
- hedging-loss Value at Risk and Expected Shortfall;
- transaction costs;
- share and notional turnover.

## Remaining Empirical Work

The reusable code is complete enough to run the study. The remaining work is to:

1. choose the underlying asset and historical sample period;
2. obtain and document adjusted-close and volume data;
3. optionally add point-in-time external variables such as VIX or interest rates;
4. select training, validation, and test periods;
5. tune model hyperparameters using only pre-test data;
6. run robustness checks across option maturities, rebalancing frequencies, and
   transaction-cost assumptions;
7. prepare final figures, executive summary, and presentation.
