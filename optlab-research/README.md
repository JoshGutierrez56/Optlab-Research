# optlab-research

**Systematic equity research workbench built on CRSP / Compustat / IBES.**

A higher-level research layer on top of the [optlab](https://github.com/joshguti56/optlab) data lake that turns factor research from a 200-line notebook into 10 lines any analyst can write.

```python
from optlab_research import workbench as wb

with wb.open() as con:
    univ = wb.universe("liquid_500", "2023-12-29", con=con)
    mom  = wb.signal("momentum_12_2",  "2023-12-29", universe=univ, con=con)
    bm   = wb.signal("book_to_market", "2023-12-29", universe=univ, con=con)
```

---

## What this is

Most quant research codebases conflate data plumbing with research logic. This project separates them cleanly:

- **`optlab`** (separate repo) — the data lake. Pulls WRDS tables to Parquet, registers DuckDB views, handles incremental refreshes. You write it once and never touch it again.
- **`optlab-research`** (this repo) — the research layer. Registry-driven signals, named universes, a backtester, performance attribution, and a live IBKR options data collector. Members write research, not infrastructure.

Built for the [360 Huntington Fund](https://www.northeastern.edu/360-huntington-fund/) at Northeastern University's D'Amore-McKim School of Business and the Northeastern Options Club.

---

## Architecture

```
optlab data lake (CRSP / Compustat / IBES / OptionMetrics)
    └── Parquet files on disk, DuckDB views on top
            │
            ▼
optlab-research workbench
    ├── Signal registry      config/signals.yaml   → 10 factors, YAML-defined
    ├── Universe registry    config/universes.yaml  → 4 named presets
    ├── Signal library       signals/library/       → complex multi-step signals
    ├── Universe builder     universes/builder.py   → PIT-correct, DuckDB-backed
    ├── Backtest engine      backtest/              → [Week 3, in progress]
    ├── Attribution          attribution/           → [Week 6–7, in progress]
    └── Member API           workbench/api.py       → wb.signal(), wb.universe()
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| Registry-driven (YAML) | Adding a new factor is a one-line edit, not a code change |
| PIT-correct by default | Every join respects `COALESCE(rdq, datadate + 90 days) <= asof` |
| DuckDB views over Parquet | Lake is the source of truth; no data duplication |
| Polars at the edges | Pandas only at the WRDS boundary; backtests run in Polars |
| Pydantic + `extra="forbid"` | Schema violations fail loudly at load time, not silently at runtime |
| Manifests for lineage | Every backtest run records inputs + git commit for reproducibility |

---

## Signal library

10 factors across 5 categories, all PIT-correct on CRSP / Compustat:

| Signal | Category | Source | Sort | Literature |
|---|---|---|---|---|
| `book_to_market` | Value | Compustat | Q5 = long | Fama-French (1992) |
| `gross_profitability` | Quality | Compustat | Q5 = long | Novy-Marx (2013) |
| `roe` | Quality | Compustat | Q5 = long | Asness et al. (2019) |
| `accruals` | Quality | Compustat | Q1 = long | Sloan (1996) |
| `size` | Size | CRSP | Q1 = long | Fama-French (1992) |
| `momentum_12_2` | Momentum | CRSP | Q5 = long | Jegadeesh-Titman (1993) |
| `short_term_reversal` | Momentum | CRSP | Q1 = long | Jegadeesh (1990) |
| `idio_vol_252d` | Risk | CRSP | Q1 = long | Ang et al. (2006) |
| `beta_60m` | Risk | CRSP + FF | Q1 = long | Frazzini-Pedersen (2014) |
| `asset_growth` | Investment | Compustat | Q1 = long | Cooper et al. (2008) |

**Adding a new signal:**

Simple signals (single formula over existing universe columns) are defined entirely in YAML:

```yaml
- name: earnings_yield
  description: EBIT / Enterprise Value
  kind: funda
  formula: "pl.col('ebit') / (pl.col('mcap_musd') + pl.col('dltt') - pl.col('che'))"
  required_columns: [ebit, mcap_musd, dltt, che]
  source_table: comp_funda
```

Complex signals (rolling windows, regressions, multi-table joins) are Python files in `signals/library/` referenced from YAML via `library_fn`.

---

## Named universes

Four presets defined in `config/universes.yaml`:

| Universe | Description | ~Size |
|---|---|---|
| `russell3000` | Top 3000 US common stocks by market cap | ~3000 |
| `russell1000` | Top 1000 US common stocks by market cap | ~1000 |
| `liquid_500` | Top 500 by 3-month avg daily dollar volume | ~500 |
| `tradeable` | Min price $5, min mcap $100M, no size cap | ~2000–2500 |

All universes apply standard CRSP filters (shrcd ∈ {10,11}, exchcd ∈ {1,2,3}) and exclude delisted names. Fundamentals are attached via CCM link with full PIT gating.

---

## Cross-section validation (2023-12-29, `liquid_500`)

All 9 available signals show ≥99% coverage and monotone quintile sorts:

```
signal               total    non_null    coverage    median
book_to_market         500         498       99.6%     0.198
gross_profitability    500         498       99.6%     0.255
roe                    500         498       99.6%     0.155
accruals               500         498       99.6%    -0.033
size                   500         500      100.0%    10.392
momentum_12_2          498         498      100.0%     0.088
short_term_reversal    500         499       99.8%     0.101
beta_60m               500         490       98.0%      1.02
asset_growth           498         491       98.6%     0.089
```

---

## Stack

- **Python 3.11+**, Anaconda
- **DuckDB** — SQL engine over Parquet views
- **Polars** — backtests and signal computation
- **Pydantic v2** — config schema validation
- **CVXPY + Ledoit-Wolf** — portfolio optimization (Week 5+)
- **WRDS** — data source (CRSP, Compustat, IBES, Fama-French)
- **ib_async** — IBKR live options chain snapshotting (Week 10+)

---

## Project roadmap

This is a 12-week build. Current status: **Week 2 of 12 complete.**

- [x] **Week 1** — Signal registry foundation (`SignalSpec`, `compute_signal`, 5 signals)
- [x] **Week 2** — Signal library expansion (10 signals), named universe presets, workbench API
- [ ] **Week 3** — Backtest engine v0 (quintile long-short, `BacktestResult`, manifests)
- [ ] **Week 4** — Replication: Jegadeesh-Titman momentum, low-vol anomaly, quality factor
- [ ] **Week 5** — Transaction costs, value/rank/IC weighting schemes
- [ ] **Week 6** — FF6 factor attribution (Newey-West HAC, rolling exposures)
- [ ] **Week 7** — Brinson-Fachler sector attribution
- [ ] **Week 8** — PDF/HTML report generation
- [ ] **Week 9** — Member API + research project template notebook
- [ ] **Week 10** — IBKR connection + options chain snapshotting
- [ ] **Week 11** — Scheduled snapshots + monitoring dashboard
- [ ] **Week 12** — Streamlit dashboard + full demo

---

## Repository structure

```
optlab-research/
├── config/
│   ├── signals.yaml          # Signal/factor registry (10 factors)
│   ├── universes.yaml        # Named universe presets
│   ├── attribution.yaml      # Attribution model definitions [Week 6]
│   └── ibkr.yaml             # IBKR snapshot schedule [Week 10]
├── optlab_research/
│   ├── signals/
│   │   ├── registry.py       # SignalSpec, SignalRegistry, load_signals()
│   │   ├── compute.py        # compute_signal(name, date, con, universe)
│   │   └── library/          # Complex multi-step signals
│   │       ├── momentum.py
│   │       ├── idio_vol.py
│   │       ├── short_term_reversal.py
│   │       ├── beta_60m.py
│   │       └── asset_growth.py
│   ├── universes/
│   │   └── builder.py        # get_universe(name, date, con)
│   ├── backtest/             # [Week 3]
│   ├── attribution/          # [Week 6–7]
│   ├── ibkr/                 # [Week 10]
│   ├── workbench/
│   │   └── api.py            # wb.signal(), wb.universe(), wb.open()
│   └── db.py                 # DuckDB connection management
├── notebooks/
│   └── examples/
│       └── cross_section_summary.ipynb
└── tests/
    ├── test_signals.py
    └── test_universes.py
```

---

## Setup

```bash
git clone https://github.com/joshguti56/optlab-research
cd optlab-research
pip install -e ".[dev]"
```

Requires a working `optlab` installation with WRDS credentials and refreshed tables (`crsp_msf`, `comp_funda`, `crsp_stocknames`, `crsp_msedelist`, `crsp_ccmxpf_linktable`, `ff_factors_monthly`).

---

## License

MIT
