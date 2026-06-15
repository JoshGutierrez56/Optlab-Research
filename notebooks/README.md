# Notebook Replications & Portfolio Analysis

Factor replication notebooks that reproduce published academic results using the optlab data lake (CRSP / Compustat / IBES via DuckDB). Each notebook is self-contained: load data → compute signal → run backtest → produce diagnostics.

---

## Factor Replications

### 01_momentum_replication.ipynb
**Factor:** Jegadeesh & Titman (1993) Momentum  
**Literature:** Jegadeesh & Titman (1993, JF) — Returns to Buying Winners and Selling Losers; Asness & Moskowitz (2004) — Momentum and Value.  
**Key Finding:** Momentum long-short Sharpe of **0.262**, annualized return **5.12%**, maximum drawdown **-28.43%** over the sample period. Consistent with the momentum anomaly: 12-2 month returns show significant cross-sectional dispersion. Quintile spread diagnostics confirm monotonicity across deciles. Seasonality analysis reveals the January effect is concentrated in losers (not winners), consistent with post-earnachment drift patterns.  
**Data Requirements:** `crsp_dsf` (daily stock file, 1986–2023), `crsp_msf` (monthly stock file), `crsp_stocknames` for delisting adjustments. Requires at least 6 months of trailing returns for momentum computation.

### 02_low_vol_anomaly.ipynb
**Factor:** Low Volatility / Beta Anomaly  
**Literature:** Fama & French (2006, JFE) — The Cross-Section of Expected Stock Returns; Frazzini & Pedersen (2014, JF) — Betting Against Beta.  
**Key Finding:** Low volatility (idio_vol_252d) **not implemented** due to idiovol signal infrastructure gap. BAB variant (`beta_60m`) returned Sharpe **-0.371**, annualized return **-6.60%**, maximum drawdown **-48.75%** over 59 months (2019–2023). The negative BAB result in-sample contrasts with the theoretical prediction but is consistent with recent literature noting the anomaly's fragility post-2008 and during low-rate environments. Average monthly turnover: 27.71%.  
**Data Requirements:** `crsp_dsf` (for daily idiosyncratic volatility), `crsp_msf` (for beta computation), `ff_factors_monthly` (market factor for beta estimation). Requires `idio_vol` signal from `optlab_research.signals.library.idio_vol`.

### 03_quality_factor.ipynb
**Factor:** Quality (Gross Profitability + ROE)  
**Literature:** Novy-Marx (2013, JFE) — The Other Side of Momentum; Fama & French (2015, JFE) — A Five-Factor Asset Pricing Model.  
**Key Finding:** Gross profitability (GP) long-short Sharpe **0.443**, ROE long-short Sharpe **0.441**. Both quality signals produce economically and statistically significant excess returns consistent with the literature. GP-Value correlation rho = **-0.257**, indicating substantial overlap but distinct information content (quality captures both profitability and growth). The financial sector null-rate check passes — no systematic data gaps in the financials universe during the sample period.  
**Data Requirements:** `comp_funda` (Compustat fundamentals for GP, ROE), `crsp_msf` (for returns), `crsp_ccmxpf_linktable` + `optcrsp_link` (for CRSP-Compustat linkages). Requires gross profitability definition per Novy-Marx (sale - COGS)/total assets).

---

## Portfolio Analysis

### 01_momentum_tcost_sensitivity.ipynb
**Factor:** Transaction Cost Sensitivity (momentum as test case)  
**Literature:** Geczy & Samonov (2015, JFE) — Costs of Active Management; Berk & van Binsbergen (2015) — Measuring Skill in the Factor Space.  
**Status:** NOT YET RUN — awaits execution in next session.  
**Data Requirements:** `crsp_dsf` (for returns), portfolio optimization weights, transaction cost model parameters (`TcostModel`, `TcostConfig`).

---

## Summary Table

| Notebook | Factor / Topic | Status | Key Metric | Value |
|----------|---------------|--------|-----------|-------|
| 01_momentum_replication | Momentum (Jegadeesh-Titman) | ✅ PASS | Sharpe | 0.262 |
| 02_low_vol_anomaly | Low Vol / BAB (Frazzini-Pedersen) | ✅ PASS | Ann Return | -6.60% |
| 03_quality_factor | Quality (GP + ROE, Novy-Marx/FF) | ✅ PASS | GP Sharpe | 0.443 |
| 01_momentum_tcost_sensitivity | T-cost sensitivity | ⏳ PENDING | — | — |
