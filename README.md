# Market-Signal Risk Management for Option Hedging

**Erdos Institute Quantitative Finance Project - Summer 2026**

**Authors:** Yue Wu and Freda Zhang

## Project Overview

This project studies whether market signals beyond historical stock prices can
improve European option hedging and risk management. The baseline is a
constant-volatility Black-Scholes delta hedge. The main extension is a
signal-driven hedging strategy that identifies elevated-risk periods and adjusts
the hedge accordingly.

The modeling stack covers four levels of volatility and risk modeling:

1. **Constant-volatility Black-Scholes:** the benchmark pricing and delta model.
2. **Finite-state Markov chains:** a discrete low/high volatility state model
   for regime detection and simulation.
3. **Continuous-time Markov chains:** a two-state Markov-modulated GBM model
   with transition rates and occupation-time option pricing.
4. **Heston stochastic volatility:** a calibrated stochastic variance model
   that can fit implied-volatility smiles and term structures.

The research objective is to determine whether a risk-management rule built
from market signals can improve hedging outcomes in a chronological backtest.

## Research Question

> Can a market-signal risk model, using option-implied volatility and other
> point-in-time market variables, reduce out-of-sample hedging error and tail
> losses relative to a constant-volatility Black-Scholes delta hedge?

## Strategy Definitions

The backtest will compare three hedging approaches under the same option
episodes, rebalancing calendar, and transaction-cost assumptions.

- **Fixed-Volatility Black-Scholes (FV-BS) Hedge:** baseline delta hedge using
  one historical volatility estimate fixed at option initiation.
- **Market-Signal Adjusted Delta (MSA-Delta) Hedge:** Black-Scholes delta hedge
  whose volatility input and rebalancing rule are conditioned on a point-in-time
  market risk score. The risk score uses signals such as option-implied
  volatility, realized-volatility acceleration, volume shocks, and market
  drawdowns. In elevated-risk states, the hedge can use a higher volatility
  input, rebalance more frequently, or tighten the no-trade band around the
  target delta.
- **Calibrated Volatility-Model Delta (CVM-Delta) Hedge:** hedge whose delta
  comes from a fitted Markov regime-switching or Heston stochastic-volatility
  model. The Markov version uses low/high volatility states and transition
  probabilities or rates. The Heston version uses parameters calibrated to
  option prices across strikes and maturities. This hedge is model-calibrated
  rather than directly thresholded on the market-signal risk score.

All comparisons must be chronological and out of sample. Information available
after a hedge date, including future returns, later option quotes, or future
calibration targets, must not be used for that hedge decision.

## Market Signals

The risk model should use information beyond the historical stock-price path.
Candidate point-in-time signals include:

- option-implied volatility, implied-volatility rank, skew, and term structure;
- realized volatility over short and medium windows;
- trading volume, dollar volume, and illiquidity;
- VIX or broad volatility-index levels;
- interest rates or Treasury yields;
- earnings announcement indicators;
- macroeconomic release indicators;
- sector or market index returns;
- news or sentiment scores, if available.

The minimum viable strategy should include option-implied volatility plus at
least one non-option signal such as realized volatility, trading volume,
interest rates, or sector-market returns.

## Market-Signal Adjusted Delta Hedge

At each hedge date $t$, construct a point-in-time feature vector $X_t$. The
features are converted into a risk score $R_t$ and a discrete risk state
$Z_t \in \{\mathrm{normal}, \mathrm{elevated}, \mathrm{stress}\}$.

One concrete rule is:

$$
R_t
= w_1 \operatorname{IVRank}_t
+ w_2 \frac{\widehat{\sigma}^{\mathrm{realized}}_{t,21}}
              {\widehat{\sigma}^{\mathrm{realized}}_{t,63}}
+ w_3 \operatorname{VolumeShock}_t
+ w_4 \operatorname{MarketDrawdown}_t .
$$

Then define:

$$
Z_t =
\begin{cases}
\text{normal}, & R_t < q_{70}, \\
\text{elevated}, & q_{70} \le R_t < q_{90}, \\
\text{stress}, & R_t \ge q_{90},
\end{cases}
$$

where $q_{70}$ and $q_{90}$ are thresholds estimated only from the training
period.

The hedge then uses a risk-adjusted volatility input:

$$
\sigma_t^{\mathrm{hedge}}
= m(Z_t)\,\widehat{\sigma}_{t}^{\mathrm{base}},
\qquad
m(\mathrm{normal}) = 1,\quad
m(\mathrm{elevated}) > 1,\quad
m(\mathrm{stress}) > m(\mathrm{elevated}).
$$

Here $\widehat{\sigma}_{t}^{\mathrm{base}}$ can be a fixed historical estimate,
a rolling realized-volatility estimate, or an implied-volatility-based estimate.
The thresholds $q_{70}, q_{90}$ and multipliers $m(Z_t)$ must be selected on
training or validation data only.

The Black-Scholes delta is computed with $\sigma_t^{\mathrm{hedge}}$:

$$
\Delta_t^{\mathrm{signal}}
= \Delta_{\mathrm{BS}}\left(S_t, K, \tau_t, r, \sigma_t^{\mathrm{hedge}}\right).
$$

The strategy can also react to $Z_t$ through the trading rule:

- rebalance more frequently during elevated-risk or stress periods;
- tighten the no-trade band in stress periods when hedge error dominates costs;
- widen the no-trade band in normal markets to reduce unnecessary turnover.

The initial research implementation should prioritize one clear rule, justify
the signal choices, and backtest the rule carefully before adding more signals
or hedge adjustments.

## Volatility Models

### Black-Scholes Baseline

Use the standard Black-Scholes price and delta formulas with fixed historical
volatility. This benchmark isolates the value of using market signals and
dynamic volatility estimates.

### Markov Regime-Switching Model

Use low- and high-volatility regimes to represent volatility clustering.

The finite-state discrete-time chain uses transition matrix

$$
P =
\begin{pmatrix}
p_{LL} & p_{LH} \\
p_{HL} & p_{HH}
\end{pmatrix}.
$$

The continuous-time two-state chain uses generator

$$
Q =
\begin{pmatrix}
-\alpha & \alpha \\
\beta & -\beta
\end{pmatrix}.
$$

Under Markov-modulated GBM, the stock follows Black-Scholes dynamics locally,
but the volatility changes with the Markov state. Option prices can be computed
by averaging Black-Scholes prices over the occupation-time distribution of the
high-volatility state. This gives a principled way to connect regime risk to
option prices, implied-volatility curves, and model deltas.

The corresponding model-based hedge uses the Markov delta:

$$
\Delta_t^{\mathrm{Markov}}
= \frac{\partial C_{\mathrm{Markov}}(S_t, K, \tau_t)}{\partial S_t},
$$

where $C_{\mathrm{Markov}}$ is the regime-switching option price.

### Heston Model

Use Heston's stochastic variance process:

$$
\frac{dS_t}{S_t} = r\,dt + \sqrt{v_t}\,dW_t^S,
$$

$$
dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v,
\qquad
\operatorname{corr}(dW_t^S, dW_t^v) = \rho .
$$

Heston is a stochastic-volatility model, not just a volatility signal. It can be
calibrated to option prices across strikes and maturities, then used to produce
model prices, implied volatilities, and deltas.

The corresponding model-based hedge uses the Heston delta:

$$
\Delta_t^{\mathrm{Heston}}
= \frac{\partial C_{\mathrm{Heston}}(S_t, K, \tau_t; \Theta_t)}{\partial S_t},
$$

where $\Theta_t = (v_0, \kappa, \theta, \xi, \rho)$ is calibrated using only
option data available at or before hedge date $t$.

## Backtesting Experiment

The empirical test evaluates whether the signal-driven risk rule improves
hedging after costs.

1. Choose an underlying asset, sample period, option maturity range, and hedge
   frequency.
2. Split the sample chronologically into training, validation, and test periods.
3. At each date, build signals using only information available at or before
   that date.
4. Estimate thresholds, regression weights, Markov transition parameters, or
   Heston calibration settings using only pre-test data.
5. Run the FV-BS, MSA-Delta, and CVM-Delta hedges on the same option episodes.
6. Charge the same transaction-cost model to every strategy.
7. Compare out-of-sample hedging error, tail losses, turnover, and costs.

Within each option episode, all strategies should receive the same initial
option premium unless the experiment is explicitly testing pricing differences.
This keeps the comparison focused on hedge performance rather than on selling
the option at different model prices.

## Evaluation

Hedging strategies will be evaluated using:

- terminal hedging error;
- mean absolute hedging error;
- root mean squared hedging error;
- 95th and 99th percentile absolute error;
- hedging-loss Value at Risk and Expected Shortfall;
- transaction costs;
- share and notional turnover;
- performance during elevated-risk and stress periods.

Model quality will be evaluated using:

- realized-volatility forecast error, if the risk rule forecasts volatility;
- risk-state precision around high-realized-volatility periods;
- option pricing error;
- implied-volatility fit across strikes and maturities;
- Markov transition-rate and regime-persistence stability;
- Heston calibration stability.

## Current Implementation

The package currently includes reusable infrastructure for:

- European call and put payoff functions;
- Black-Scholes call and put pricing;
- Black-Scholes call and put delta calculations;
- market-data loading, validation, and return calculations;
- fixed historical and rolling volatility estimators;
- point-in-time signal construction and leakage checks;
- signal-based volatility forecasting;
- market-signal risk scoring, risk-state classification, and MSA-Delta
  volatility adjustment;
- discrete-time Black-Scholes delta hedging with transaction costs;
- repeated option episodes and strategy-level hedging metrics;
- synthetic MSA-Delta demonstration notebook;
- unit tests for pricing, data, volatility, signals, backtesting, and metrics.

Markov and Heston components remain planned package extensions. The signal and
backtesting infrastructure already supports the core empirical workflow: design
a point-in-time risk signal, adjust the hedging rule, and test whether the
adjustment improves out-of-sample hedge performance.

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
- [x] Add point-in-time signal construction and leakage checks.
- [x] Add strategy-level backtesting and hedging metrics.
- [x] Build an MSA-Delta risk-state rule using implied volatility and additional market signals.
- [x] Add a synthetic MSA-Delta backtesting demonstration notebook.
- [ ] Add real option-chain and underlying market data.
- [ ] Backtest the MSA-Delta hedge against the FV-BS benchmark on real market data.
- [ ] Promote Markov regime-switching pricing and delta code into `src/option_hedging/models/`.
- [ ] Promote Heston pricing, delta, and calibration code into `src/option_hedging/models/`.
- [ ] Generate final tables, figures, notebook, report, and presentation materials.
