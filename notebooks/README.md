# Research notebook guide

These notebooks explore factor construction, portfolio evaluation and options-data workflows using the external `optlab` data lake. Saved outputs are historical artifacts, not fresh reproductions or proof of statistically significant alpha.

Read the [root README](../README.md) for current setup, implementation status and data limitations. Some notebooks retain local data paths or older API assumptions; inspect their configuration before execution.

## Start with a cross-section

- [Signal coverage, distributions and correlations](examples/cross_section_summary.ipynb)
- [Saved correlation matrix](examples/cross_section_correlations.png)
- [Saved signal distributions](examples/cross_section_histograms.png)

These artifacts describe a cross-section on 2023-12-29. Sorted signal values and cross-sectional correlation are not forward-return validation.

## Factor studies

| Notebook | Research question |
|---|---|
| [Momentum](factor%20replications/01_momentum_replication.ipynb) | How does a trailing-return sort behave under this universe, sample and portfolio specification? |
| [Low volatility / beta](factor%20replications/02_low_vol_anomaly.ipynb) | How sensitive are defensive-factor conclusions to the signal and available data? |
| [Quality](factor%20replications/03_quality_factor.ipynb) | What do profitability and ROE sorts add, and where are their data gaps? |
| [Momentum transaction costs](portfolio%20analysis/01_momentum_tcost_sensitivity.ipynb) | How do turnover and assumed trading costs affect gross versus net results? |

The current package includes an [idiosyncratic-volatility implementation](../optlab_research/signals/library/idio_vol.py); older notebook notes about its absence describe earlier runs, not the current source. Its use still requires daily CRSP and Fama-French inputs.

The workbench now forwards supplied signal universes and activates its flat-cost model for positive `tcost_bps`. Flat costs debit both long and short returns once, including reductions and exits, using consecutive target weights. Saved notebook net results predate this accounting correction and have not been recomputed; do not treat them as current validated results. Spread/impact backtests are explicitly blocked pending reconciliation. Factor attribution remains unimplemented; a notebook reference to it is not an available result.

## Reporting and data collection

- [HTML report generation](examples/08_report_generation.ipynb)
- [IBKR options snapshot](examples/10_ibkr_snapshot.ipynb)
- [Snapshot monitoring](examples/11_snapshot_monitoring.ipynb)
- [Research project template](templates/research_project_template.ipynb)

Broker examples require a separately configured session and market-data access. Research notebooks require appropriately licensed datasets; the repository does not supply a complete research database.

## Before interpreting a result

Record the data snapshot, sample, universe construction, signal timing, missing-return treatment, transaction costs and metric definitions. Validate temporal availability and benchmark comparability, and distinguish descriptive diagnostics from a held-out test. The historical [audit](../AUDIT.md) is a dated troubleshooting record, not a blanket test-pass label.
