# Beyond Constant Volatility

**An Empirical Study of Dynamic Delta Hedging**

This project investigates whether time-varying volatility estimates can improve
the delta hedging of European call options relative to a constant-volatility
Black-Scholes benchmark.

## Initial milestone

The repository currently contains:

- Black-Scholes prices for European calls and puts
- Black-Scholes deltas
- expiration-value handling
- input validation
- unit tests for standard identities and boundary cases

## Repository structure

```text
beyond-constant-volatility/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   └── option_hedging/
│       └── derivatives/
│           └── black_scholes.py
├── tests/
│   └── test_black_scholes.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run the tests

```bash
pytest
```

## Example

```python
from option_hedging.derivatives.black_scholes import call_delta, call_price

price = call_price(
    spot=100.0,
    strike=100.0,
    maturity=30 / 365,
    rate=0.04,
    volatility=0.20,
)

delta = call_delta(
    spot=100.0,
    strike=100.0,
    maturity=30 / 365,
    rate=0.04,
    volatility=0.20,
)

print(price, delta)
```

## Planned next steps

1. Add fixed and rolling volatility models.
2. Build a discrete-time delta-hedging engine.
3. Generate repeated historical option episodes.
4. Add out-of-sample hedging metrics.
5. Add a signal-informed volatility model.
