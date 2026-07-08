# Dynamic Volatility and Risk-Aware Option Hedging

**Erdos Institute Quantitative Finance Project - Summer 2026**

**Authors:** Yue Wu and Freda Zhang

## Overview

This project studies whether quantitative researchers can improve European option hedging by using market and macroeconomic signals beyond historical stock prices. The baseline is a constant-volatility Black-Scholes delta hedge. The planned research extension is a risk-aware hedging system that identifies periods of elevated market risk and adjusts volatility estimates, hedge frequency, or hedge aggressiveness before running an out-of-sample backtest.

The project will combine option pricing, volatility forecasting, and risk-management evaluation. Candidate signals include option-implied volatility, trading volume, interest rates, earnings announcement dates, volatility index levels, macroeconomic releases, sector returns, and news or text-based sentiment measures.

## Research Question

> Can a signal-based risk model reduce out-of-sample hedging error and tail losses relative to a constant-volatility Black-Scholes delta hedge?

## Project Scope

The project will compare three hedging approaches:

- **Benchmark hedge:** Black-Scholes delta hedge using fixed historical volatility.
- **Adaptive volatility hedge:** Black-Scholes delta hedge using rolling historical or Gaussian volatility estimates.
- **Risk-aware hedge:** Black-Scholes delta hedge using a signal-based volatility or risk-state model that reacts to elevated-risk periods.

The final strategy must be evaluated only on chronological out-of-sample periods. Features available after each hedge date must not be used for that hedge decision.

## Data Inputs

The minimum viable dataset will include:

- underlying adjusted close prices;
- daily returns and rolling realized volatility;
- trading volume and volume shocks;
- risk-free interest rates or Treasury yields;
- option-implied volatility or a broad volatility index such as VIX;
- sector or market index returns.

Optional extensions:

- earnings announcement indicators;
- macroeconomic release indicators;
- news or sentiment features;
- option-chain data for implied volatility smiles or term structure.

Raw data will be stored in `data/raw/`. Cleaned, aligned feature panels will be stored in `data/processed/`.

## Modeling Plan

1. Build a daily feature panel aligned by date.
2. Construct volatility labels, such as next-period realized volatility or future absolute return.
3. Train only on past observations using chronological splits.
4. Compare fixed historical volatility, rolling volatility, and signal-based models.
5. Convert model output into a hedging input:
   - predicted volatility for Black-Scholes delta;
   - high-risk indicator for wider volatility buffers;
   - high-risk indicator for more frequent or more conservative rebalancing.
6. Run each strategy through the same delta-hedging backtester.
7. Report accuracy, hedging error, transaction costs, turnover, and tail-risk metrics.

## Planned Executables

The project will expose concrete command-line modules so the full workflow can be reproduced from a clean checkout.

| Executable | Purpose | Primary output |
| --- | --- | --- |
| `python -m option_hedging.cli.fetch_data --config configs/data.yaml` | Download or load raw price, volume, rate, volatility-index, and event data. | `data/raw/*.csv` |
| `python -m option_hedging.cli.build_features --config configs/features.yaml` | Clean, align, and merge market signals into one modeling panel. | `data/processed/feature_panel.parquet` |
| `python -m option_hedging.cli.train_risk_model --config configs/model.yaml` | Train volatility and elevated-risk models using chronological splits. | `results/models/risk_model.pkl` and `results/tables/model_metrics.csv` |
| `python -m option_hedging.cli.run_backtest --config configs/backtest.yaml` | Run baseline, adaptive, and risk-aware hedging strategies. | `results/tables/backtest_metrics.csv` and `results/tables/hedge_paths.csv` |
| `python -m option_hedging.cli.make_report --config configs/report.yaml` | Generate final tables and figures for presentation. | `results/figures/*.png` and `results/report.md` |

The existing package currently implements reusable library functions. These CLI modules are planned deliverables and will call the package code rather than duplicating analysis logic in notebooks.

## Deliverables

The completed project should produce:

- reproducible Python package under `src/option_hedging/`;
- tested Black-Scholes pricing and delta functions;
- tested market-data validation and return calculations;
- tested fixed and rolling volatility estimators;
- tested discrete-time delta-hedging backtester with transaction costs;
- cleaned feature panel at `data/processed/feature_panel.parquet`;
- trained risk or volatility model artifact at `results/models/risk_model.pkl`;
- out-of-sample strategy comparison at `results/tables/backtest_metrics.csv`;
- hedge-path diagnostics at `results/tables/hedge_paths.csv`;
- figures for volatility forecasts, risk regimes, hedging-error distributions, cumulative hedging error, and transaction costs;
- final analysis notebook in `notebooks/final_analysis.ipynb`;
- final written summary in `results/report.md`;
- project presentation slides or figures suitable for the Erdos Institute final presentation.

## Evaluation Metrics

Strategies will be evaluated with:

- terminal hedging error;
- mean absolute hedging error;
- root mean squared hedging error;
- 95th and 99th percentile absolute error;
- expected shortfall of hedging losses;
- total transaction cost;
- turnover and number of rebalances;
- performance during high-volatility or elevated-risk periods.

Model quality will be evaluated with:

- realized-volatility prediction error;
- high-risk classification precision and recall;
- calibration of predicted risk states;
- feature importance or coefficient interpretation when available.

## Current Status

The repository currently includes:

- European call and put payoff functions;
- Black-Scholes call and put pricing;
- Black-Scholes call and put delta calculations;
- market-price loading, validation, and return utilities;
- fixed historical volatility and rolling Gaussian volatility estimators;
- a discrete-time delta-hedging backtester for short European calls;
- transaction-cost handling in the hedge simulator;
- unit tests for pricing, market data, volatility, and hedging modules.

## Repository Structure

Current and planned structure:

```text
erdos-volatility-hedging-wu-zhang/
├── configs/                         # planned reproducible run configs
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── final_analysis.ipynb         # planned final analysis notebook
├── results/
│   ├── figures/
│   ├── models/                      # planned model artifacts
│   ├── tables/
│   └── report.md                    # planned final report
├── src/
│   └── option_hedging/
│       ├── cli/                     # planned executable modules
│       ├── data/
│       │   └── market_data.py
│       ├── derivatives/
│       │   └── black_scholes.py
│       ├── models/
│       │   └── volatility.py
│       └── strategies/
│           └── delta_hedging.py
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

## Reproducible Workflow

Once the planned CLI modules are implemented, the full project should run with:

```bash
python -m option_hedging.cli.fetch_data --config configs/data.yaml
python -m option_hedging.cli.build_features --config configs/features.yaml
python -m option_hedging.cli.train_risk_model --config configs/model.yaml
python -m option_hedging.cli.run_backtest --config configs/backtest.yaml
python -m option_hedging.cli.make_report --config configs/report.yaml
pytest
```

## Roadmap

- [x] Implement Black-Scholes prices and deltas.
- [x] Add unit tests.
- [x] Add market-data validation and return utilities.
- [x] Implement fixed and rolling volatility estimators.
- [x] Build a delta-hedging backtester with transaction costs.
- [ ] Add raw market, rate, volatility-index, and optional event data.
- [ ] Implement feature-panel construction.
- [ ] Implement signal-based volatility or elevated-risk model.
- [ ] Add risk-aware hedge adjustments.
- [ ] Add CLI executables for the full workflow.
- [ ] Generate final tables, figures, notebook, report, and presentation materials.
