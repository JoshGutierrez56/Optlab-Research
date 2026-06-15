# Audit Report

**Date:** 2026-06-14  
**Auditor:** Merlin (OpenClaw agent)  
**Scope:** `optlab-research` notebook infrastructure, signal library, data lake connectivity

---

## Bugs Found During Audit

### BUG-001: Missing `os.makedirs()` before file saves (NB1, NB3)
**Severity:** High  
**Files:** `nb01.ipynb`, `nb03.ipynb`  
**Symptom:** `FileNotFoundError` when saving outputs to `outputs/` subdirectory that doesn't exist.  
**Fix applied:** ✅ Added `os.makedirs(SAVE_PATH, exist_ok=True)` before every save call in both notebooks.

### BUG-002: Direct dict access causing KeyError in `_show_summary()` (NB1)
**Severity:** High  
**Files:** `nb01.ipynb` (cell 10 — `_show_summary`)  
**Symptom:** `KeyError` when backtest result summary dict contains keys not expected by the display function.  
**Fix applied:** ✅ Replaced direct dict access with `.get('key', 'N/A')` pattern throughout `_show_summary()`.

### BUG-003: Missing `idio_vol` comment block (NB2)
**Severity:** Medium  
**Files:** `nb02.ipynb`  
**Symptom:** Signal registration fails for `idio_vol` because the library import/compute was referenced in `signals.yaml` but no actual implementation existed. The notebook code lacked the bootstrap comment and conditional skip logic for when idiovol data is unavailable.  
**Fix applied:** ✅ Added full `idio_vol` comment block with conditional handling (skip gracefully if IDIOVOL_OK is False).

### BUG-004: Path creation / makedirs for output directories
**Severity:** Medium  
**Files:** `nb02.ipynb`, `nb03.ipynb`, `nb04.ipynb`  
**Symptom:** Output directories (`outputs/`) not created before write.  
**Fix applied:** ✅ Added `os.makedirs()` guards in all notebooks that produce file outputs.

### BUG-005: Dict access pattern inconsistency
**Severity:** Low  
**Files:** Multiple notebooks  
**Symptom:** Mixed use of direct dict access and `.get()` across notebooks, causing intermittent KeyErrors on edge-case backtest results.  
**Fix applied:** ✅ Standardized to `.get('key', 'N/A')` throughout.

### BUG-006: nbconvert subprocess buffering hides progress
**Severity:** Medium (infrastructure)  
**Files:** `_audit.py`, `_execute.py` and all execution harness scripts in `notebooks/`  
**Symptom:** When running notebooks via `nbconvert --execute` through subprocess, stdout is buffered — no intermediate output visible until completion. This makes it impossible to monitor progress for long-running backtests (e.g., 144M row CRSP scans).  
**Remaining effort:** ⚠️ Manual fix required in each notebook: redirect print output to stderr (`sys.stderr.write`) OR use a streaming kernel approach instead of nbconvert subprocess. ~30 min to patch all notebooks.

### BUG-007: Missing `AUDIT.md` and `notebooks/README.md`
**Severity:** Low (documentation)  
**Files:** Repository root, `notebooks/` subfolder  
**Symptom:** No documentation exists in either location despite prior sessions completing the content generation.  
**Fix applied:** ✅ Both files written in this session.

### BUG-008: `optlab.db` empty — real data in `optlab/db/research.duckdb`
**Severity:** Medium  
**Files:** `optlab-research/optlab.db` (empty), `optlab/db/research.duckdb` (populated)  
**Symptom:** The `optlab_research.db.py` module auto-discovers the optlab root and opens `db/research.duckdb`, but the copy of `optlab.db` in the research repo root is empty. This is not a bug per se — the architecture expects optlab to be a separate package with its own data lake. However, it caused confusion during testing.  
**Status:** Documented only. No fix needed if optlab is properly installed via pip install -e.

---

## Data Dependencies

| Source | Location | Coverage | Loaded? |
|--------|----------|----------|---------|
| CRSP DSF (daily stock) | `optlab/db/research.duckdb::crsp_dsf` | 1925-12-31 → 2024-12-31 (144M rows) | ✅ |
| CRSP MSF (monthly stock) | `optlab/db/research.duckdb::crsp_msf` | 1925-12-31 → 2024-12-31 (23M rows) | ✅ |
| Compustat Fundamental | `optlab/db/research.duckdb::comp_funda` | 1950-06-30 → 2026-04-30 (1.8M rows) | ✅ |
| Fama-Factors Monthly | `optlab/db/research.duckdb::ff_factors_monthly` | ~1926 → present (2.4K rows) | ✅ |
| Fama-French Daily | `optlab/db/research.duckdb::ff_factors_daily` | ~1926 → present (26K rows) | ✅ |
| CCM Link Table | `optlab/db/research.duckdb::ccm_link` | All | ✅ |
| CRSP-CCM Exchange Link | `optlab/db/research.duckdb::crsp_ccmxpf_linktable` | All (463K rows) | ✅ |
| IBES EPS (standard) | Referenced in architecture | NOT YET LOADED (not critical for factor reps) | ⏳ |

---

## Bugs Remaining with Effort Estimates

| Bug | Description | Effort | Priority |
|-----|-------------|--------|----------|
| BUG-006 | nbconvert stdout buffering — no visible progress on long backtests | ~30 min | Medium |
| BUG-008 | Empty optlab.db confusion — not actionable unless someone removes the file | <5 min | Low |
| **NEW** | `nb04_clean.py` (momentum tcost) needs to be run and integrated | ~15 min + runtime | High |
| **NEW** | `idio_vol` signal library implementation gap — low vol notebook can't compute idiosyncratic volatility | ~2-4 hours | Medium |
| **NEW** | IBES data not loaded — needed for earnings momentum / surprise factors | ~1 hour (download + register) | Low |

---

## Recommendations for Next Session

1. **Run `nb04_clean.py`** (momentum transaction cost sensitivity backtest) — it's the last unexecuted notebook and completes the replication set.
2. **Implement `idio_vol.compute()`** in `optlab_research.signals.library.idio_vol` to unlock low-volatility factor computations.
3. **Fix BUG-006**: Add stderr streaming to all notebooks so progress is visible during execution (critical for 144M row scans).
4. **Add IBES data** if earnings-based factors are needed for future workbench signals.
5. **Consider pruning the `notebooks/` helper scripts** (`_audit.py`, `_debug_groupby.py`, etc.) — these are technical debt from prior debugging sessions and serve no purpose in production.
6. **Create mm-simulator project** (separate effort) for portfolio construction analytics and scenario modeling tools.

---

*End of audit report.*
