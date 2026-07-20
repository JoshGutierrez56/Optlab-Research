# optlab-research

A systematic-equity research workbench for point-in-time universes, registry-driven signals, long-short backtests, transaction costs, attribution, and portable HTML reports.

The production research path reads CRSP, Compustat, IBES, and Fama-French data from an `optlab` DuckDB/Parquet lake. A deterministic offline demo is included so the package can be verified without proprietary data.

## Verified result

The committed [`results/`](results/README.md) bundle uses a lagged synthetic momentum quintile signal with 20 bps one-way costs. It records an intentionally negative outcome:

- Gross annual return: -2.36%
- Gross Sharpe: -0.293
- Net annual return: -3.49%
- Net Sharpe: -0.433
- Net maximum drawdown: -37.62%
- Average monthly turnover: 48.17%

The result is generated data, not a historical momentum test. Keeping the negative result makes the evidence boundary explicit and demonstrates that the reporting layer does not hide an unfavorable outcome.

## Install and verify

Python 3.12 is recommended for the tested environment.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_synthetic_signal_demo.py --output-dir results --seed 42
```

The current suite contains 74 offline tests.

## Core capabilities

- YAML signal registry with strict schemas
- Point-in-time universe construction
- Value, quality, momentum, size, and risk signal library
- Equal, rank, value, and signal-magnitude weighting
- Gross/net return separation and turnover costs
- Breakeven-cost analysis
- Factor and sector attribution components
- Self-contained HTML backtest reports
- Reproducibility manifests

## Example with the data lake

```python
from optlab_research import workbench as wb

with wb.open() as connection:
    universe = wb.universe("liquid_500", "2023-12-29", con=connection)
    momentum = wb.signal(
        "momentum_12_2", "2023-12-29", universe=universe, con=connection
    )
```

The data-lake path requires authorized WRDS datasets and a configured `optlab` installation. The offline evidence does not.

This software is research infrastructure, not investment advice.
