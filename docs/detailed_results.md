# Detailed Empirical Results

## 1. What is being tested

This project tests **payoff replication**, not historical option-price prediction.
For each eligible SPY date, it creates a standardized European call with strike
`K = m S_0`, follows the realized SPY path for `M` trading days, and compares
four daily delta-hedging rules. Public market data supply the underlying path and
hedging signals; the option premium and terminal payoff are model-defined.

The chronological split is:

| Sample | Dates | Purpose |
| --- | --- | --- |
| Training | 2016-04-05 to 2020-08-14 | Fit signal scaling and thresholds |
| Validation | 2020-08-17 to 2022-05-13 | Select one of three predeclared MSA specifications |
| Test | 2022-05-16 to 2024-12-31 | Final held-out comparison |

The test grid contains maturities `M in {10, 21, 42}` trading days and
moneyness values `K/S_0 in {0.95, 1.00, 1.05}`. A new set starts every five
trading days; every open contract is still rebalanced **daily through maturity**.

With `N = 661` test prices, the number of eligible starts for maturity `M` is

```text
n_M = floor((N - M - 1) / 5) + 1.
```

This gives 131, 128, and 124 start dates for 10-, 21-, and 42-day contracts.
Multiplying each by three strikes produces

```text
131*3 + 128*3 + 124*3 = 1,149 contracts per strategy.
```

Thus the primary comparison contains 4,596 strategy-contract outcomes. The
five-day spacing controls how frequently a new episode begins; it is not an
episode length and it is applied identically to all four strategies.

## 2. Strategies and error definitions

Every strategy uses the Black-Scholes call delta

```text
Delta_t = Phi(d1_t),
d1_t = [ln(S_t/K) + (r_t + sigma_t^2/2) tau_t] / [sigma_t sqrt(tau_t)].
```

They differ only in `sigma_t`: FV-BS freezes trailing realized volatility at
initiation; Rolling-BS updates trailing realized volatility daily; VIX-BS uses
the contemporaneously observed VIX; and MSA-Delta adjusts rolling volatility
using the validated composite risk state. FV-BS therefore also requires a
trailing window before an episode may start; it does not estimate volatility
from future test returns.

Let `V_T` be the terminal hedge portfolio after proportional trading costs and
final stock liquidation. For call payoff `H_T = max(S_T-K, 0)`, the signed
replication error is

```text
e_i = V_T,i - H_T,i.
```

Negative error is a hedging loss. For `n` contracts,

```text
MAE  = (1/n) sum_i |e_i|,
RMSE = sqrt[(1/n) sum_i e_i^2].
```

The 95% loss expected shortfall is the average loss `L_i=-e_i` among outcomes
at or above the empirical 95th loss percentile. All dollar results are per one
option share. The primary specification charges 5 basis points times dollar
stock turnover on every rebalance.

## 3. Primary pooled results

| Strategy | MAE | RMSE | 95% loss ES | Mean cost | Mean turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| FV-BS | **1.5523** | 2.5422 | 6.7463 | 0.4631 | 926.18 |
| Rolling-BS | 1.6073 | 2.4385 | 6.4496 | 0.4690 | 937.97 |
| VIX-BS | 1.6956 | **2.3993** | **5.5170** | **0.4580** | **915.97** |
| MSA-Delta | 1.6906 | 2.5368 | 6.5818 | 0.4662 | 932.45 |

These metrics pool all 1,149 test contracts for each strategy. Pooling is valid
as a predeclared overall score, but it can hide heterogeneity. The maturity
weights are close but not identical: 34.2% for 10 days, 33.4% for 21 days, and
32.4% for 42 days. The cell-level tables below therefore accompany the pooled
headline.

The pooled conclusion is deliberately limited: FV-BS minimizes the typical
absolute miss, whereas VIX-BS minimizes squared error and severe downside loss.
Relative to FV-BS, VIX-BS lowers 95% loss ES by `(6.7463-5.5170)/6.7463 = 18.2%`,
but its MAE is 9.2% higher. There is no single winner under every loss function.

## 4. Results by maturity and moneyness

### Mean absolute error

| Maturity | K/S0 | FV-BS | Rolling-BS | VIX-BS | MSA-Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.95 | **0.6427** | 0.6728 | 0.8253 | 0.7610 |
| 10 | 1.00 | 1.5793 | **1.5714** | 1.5999 | 1.5736 |
| 10 | 1.05 | **0.5277** | 0.5347 | 0.6710 | 0.6351 |
| 21 | 0.95 | 1.1386 | **1.1190** | 1.2614 | 1.1912 |
| 21 | 1.00 | 1.8363 | 1.9343 | **1.8027** | 1.9804 |
| 21 | 1.05 | **1.2614** | 1.2686 | 1.6821 | 1.4276 |
| 42 | 0.95 | **1.7784** | 2.1132 | 1.9985 | 2.1737 |
| 42 | 1.00 | 2.9834 | 3.0320 | **2.7299** | 3.0236 |
| 42 | 1.05 | **2.3441** | 2.3516 | 2.8132 | 2.5834 |

FV-BS has the lowest MAE in five of nine cells, Rolling-BS in two, and VIX-BS
in two. MSA-Delta does not win a cell. Errors generally rise with maturity, and
ATM (`K/S0=1`) contracts are usually harder to hedge than the two off-ATM cells.

### Root mean squared error

| Maturity | K/S0 | FV-BS | Rolling-BS | VIX-BS | MSA-Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.95 | **0.9110** | 0.9868 | 1.0764 | 1.0421 |
| 10 | 1.00 | 2.3060 | 2.2778 | **2.1553** | 2.2569 |
| 10 | 1.05 | 1.0638 | **1.0527** | 1.0779 | 1.2512 |
| 21 | 0.95 | 2.0494 | 1.8134 | **1.6416** | 1.7564 |
| 21 | 1.00 | 2.6207 | 2.6674 | **2.3766** | 2.7402 |
| 21 | 1.05 | 1.8954 | **1.7968** | 2.1676 | 2.0372 |
| 42 | 0.95 | 2.7406 | 2.9565 | **2.5140** | 3.0285 |
| 42 | 1.00 | 4.2717 | 3.8487 | **3.5837** | 3.8916 |
| 42 | 1.05 | 3.3617 | **3.1293** | 3.6549 | 3.4512 |

VIX-BS has the lowest RMSE in five cells, Rolling-BS in three, and FV-BS in
one. This shows why VIX-BS can lose on average absolute accuracy while winning
on RMSE: it controls some of the largest errors.

### 90% loss expected shortfall

| Maturity | K/S0 | FV-BS | Rolling-BS | VIX-BS | MSA-Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.95 | **2.0903** | 2.3388 | 2.4173 | 2.4885 |
| 10 | 1.00 | 5.0912 | 5.0347 | **4.5488** | 4.8597 |
| 10 | 1.05 | **1.0492** | 1.1633 | 1.1738 | 1.4125 |
| 21 | 0.95 | 4.5138 | 4.2355 | **3.6221** | 4.0001 |
| 21 | 1.00 | 5.7926 | 5.8311 | **4.9720** | 6.0746 |
| 21 | 1.05 | **1.8227** | 2.0172 | 2.2879 | 2.3960 |
| 42 | 0.95 | 6.2173 | 6.5332 | **5.3546** | 6.5648 |
| 42 | 1.00 | 8.6501 | 7.0880 | **6.6502** | 7.3291 |
| 42 | 1.05 | **3.9678** | 4.5435 | 4.3019 | 5.4428 |

VIX-BS has the lowest 90% loss ES in five cells and FV-BS in four. The VIX
advantage is concentrated in ATM and longer-maturity cells; FV-BS remains
strong for the off-ATM cells shown here.

## 5. Paired uncertainty analysis

The same contract is evaluated under every strategy, so inference uses paired
differences. The statistic is date-balanced MAE improvement relative to FV-BS,
defined so that positive values favor the alternative. Moving date-block
bootstrap intervals preserve local time dependence from overlapping episodes.

| Strategy vs FV-BS | Estimated improvement | 95% interval | Bootstrap P(improvement <= 0) |
| --- | ---: | ---: | ---: |
| MSA-Delta | -0.1379 | [-0.2543, -0.0390] | 0.0020 |
| Rolling-BS | -0.0566 | [-0.1189, 0.0064] | 0.0400 |
| VIX-BS | -0.1481 | [-0.2549, -0.0330] | 0.0065 |

For MAE, none of the alternatives demonstrates improvement over FV-BS. The
negative point estimates mean their date-balanced MAE is worse. These intervals
are not claims about RMSE or expected shortfall; those are distinct objectives.

## 6. Transaction-cost sensitivity

| Cost | Strategy | MAE | RMSE | 90% loss ES |
| ---: | --- | ---: | ---: | ---: |
| 0 bps | FV-BS | **1.4565** | 2.4824 | 4.2379 |
| 0 bps | Rolling-BS | 1.5358 | 2.3943 | 4.3448 |
| 0 bps | VIX-BS | 1.6111 | **2.3939** | **3.8775** |
| 0 bps | MSA-Delta | 1.6201 | 2.5089 | 4.4574 |
| 5 bps | FV-BS | **1.5523** | 2.5422 | 4.9816 |
| 5 bps | Rolling-BS | 1.6073 | 2.4385 | 5.0327 |
| 5 bps | VIX-BS | 1.6956 | **2.3993** | **4.4574** |
| 5 bps | MSA-Delta | 1.6906 | 2.5368 | 5.0937 |
| 10 bps | FV-BS | **1.7454** | 2.7045 | 5.7526 |
| 10 bps | Rolling-BS | 1.7747 | 2.5923 | 5.7441 |
| 10 bps | VIX-BS | 1.8524 | **2.5105** | **5.0510** |
| 10 bps | MSA-Delta | 1.8530 | 2.6695 | 5.7539 |

The qualitative ranking is stable across 0, 5, and 10 bps: FV-BS has the
lowest MAE, while VIX-BS has the lowest RMSE and loss ES. This does not imply
costs are unimportant; every strategy's errors worsen as costs rise.

## 7. Interpretation and limitations

The strongest supported statement is: **in this held-out controlled replication
experiment, a simple fixed-volatility hedge produces the smallest average
absolute error, while VIX-based deltas reduce large and downside errors.** The
composite MSA rule adds no demonstrated advantage over the simpler benchmarks.

The results should not be read as a live-option trading backtest. There are no
historical option bids, asks, contract implied volatilities, dividends, or early
exercise decisions. VIX is a 30-day S&P 500 index option signal rather than a
contract-specific SPY implied volatility. Episodes overlap, and the bootstrap
only approximates their dependence. Conclusions apply to this sample, grid,
premium convention, daily rebalancing rule, and cost model.

## 8. Reproducibility map

- Main pipeline: `examples/run_real_market_pipeline.py`
- Frozen public-data snapshot: `data/processed/`
- Full result tables: `results/tables/`
- Bootstrap output: `results/tables/replication_bootstrap_intervals.csv`
- Final figures: `docs/figures/`
- Executable narrative: `notebooks/03_real_market_msa_delta_backtest.ipynb`

Run `python examples/run_real_market_pipeline.py` from the configured project
environment to regenerate the empirical outputs.
