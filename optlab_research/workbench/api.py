"""Member-facing workbench API.

This is the ONLY file members need to import. Everything else is internal.

Usage:
    from optlab_research import workbench as wb

    with wb.open() as con:
        univ  = wb.universe("russell1000", "2023-12-29", con=con)
        sig   = wb.signal("momentum_12_2", "2023-12-29", universe=univ, con=con)
        bt    = wb.backtest("momentum_12_2", "2019-01-01", "2023-12-31",
                            universe="russell1000", con=con)
        attr  = wb.attribution(bt, model="ff6", con=con)
        rpt   = wb.report(bt, title="Momentum Factor")
        rpt.save("outputs/momentum.html")
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from typing import Any, Optional, Union

import polars as pl
import pandas as pd

from optlab_research.data.orats import (
    OratsConfigurationError,
    OratsOptionsLoader,
    load_orats_options,
)


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

@contextmanager
def open(db_path: Optional[str] = None):
 """Open a managed DuckDB connection to the optlab data lake.

 Usage:
  with wb.open() as con:
   df = wb.signal("momentum_12_2", "2023-12-29", con=con)

 Args:
  db_path: Path to research.duckdb. If None, uses optlab default.
 """
 try:
  from optlab.db import connect
  with connect(db_path) as con:
   yield con
 except ImportError:
  raise ImportError(
   "optlab package not found. Install with: pip install -e ../optlab"
  )


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def universe(
 name: str,
 as_of: str,
 con=None,
 min_price: float = 1.0,
 min_mcap: float = 0.0,
) -> pl.DataFrame:
 """Build a point-in-time equity universe.

 Args:
  name: Universe preset name ('russell3000', 'russell1000', 'liquid_500', 'tradeable')
  as_of: Date string 'YYYY-MM-DD'
  con: DuckDB connection (from wb.open())
  min_price: Minimum share price filter
  min_mcap: Minimum market cap filter (in millions USD)

 Returns:
  Polars DataFrame with columns: permno, date, mcap_musd, ...

 Example:
  with wb.open() as con:
   univ = wb.universe("russell1000", "2023-12-29", con=con)
 """
 from optlab_research.universes.builder import get_universe_as_of
 return get_universe_as_of(name, as_of, con, min_price=min_price, min_mcap=min_mcap)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def signal(
 name: str,
 as_of: str,
 universe: Optional[pl.DataFrame] = None,
 con=None,
 n_quantiles: int = 5,
) -> pl.DataFrame:
 """Compute a factor signal for a universe as of a given date.

 Args:
  name: Signal name from signals.yaml registry
        ('momentum_12_2', 'book_to_market', 'gross_profitability',
         'roe', 'size', 'accruals', 'beta_60m', 'asset_growth',
         'short_term_reversal', 'idio_vol_252d')
  as_of: Date string 'YYYY-MM-DD'
  universe: Optional pre-built universe DataFrame. If None, uses default.
  con: DuckDB connection (from wb.open())
  n_quantiles: Number of quantiles for signal ranking (default: 5)

 Returns:
  Polars DataFrame with columns: permno, signal_value, signal_rank,
  signal_quantile, date

 Example:
  with wb.open() as con:
   mom = wb.signal("momentum_12_2", "2023-12-29", con=con)
   print(mom.head())
 """
 from optlab_research.signals.compute import compute_signal
 return compute_signal(name, as_of, con, n_quantiles=n_quantiles)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest(
 signal_name: str,
 start: str,
 end: str,
 universe: str = "russell1000",
 con=None,
 portfolio: str = "quintile_long_short",
 weighting: str = "equal",
 n_quantiles: int = 5,
 long_quantile: int = 5,
 short_quantile: int = 1,
 tcost_bps: float = 0.0,
 options_data_provider=None,
):
 """Run a full backtest for a signal over a date range.

 Args:
  signal_name: Signal name from signals.yaml registry
  start: Start date 'YYYY-MM-DD'
  end: End date 'YYYY-MM-DD'
  universe: Universe preset name
  con: DuckDB connection (from wb.open())
  portfolio: Portfolio type ('quintile_long_short')
  weighting: Weighting scheme ('equal', 'rank', 'value', 'ic')
  n_quantiles: Number of quantiles
  long_quantile: Quantile to go long (default: 5, highest signal)
  short_quantile: Quantile to go short (default: 1, lowest signal)
  tcost_bps: Transaction cost assumption in basis points (default: 0)
  options_data_provider: Optional callable for options-event studies.
                         Defaults to None so ordinary backtests never touch
                         ORATS credentials or external options data.

 Returns:
  BacktestResult object with .summary(), .returns, .plot_cumulative(), etc.

 Example:
  with wb.open() as con:
   bt = wb.backtest("momentum_12_2", "2019-01-01", "2023-12-31",
                   universe="russell1000", tcost_bps=10, con=con)
   print(bt.summary())
 """
 from optlab_research.backtest.engine import Backtest, BacktestConfig
 cfg = BacktestConfig(
  signal=signal_name,
  start=start,
  end=end,
  universe=universe,
  portfolio=portfolio,
  weighting=weighting,
  n_quantiles=n_quantiles,
  long_quantile=long_quantile,
  short_quantile=short_quantile,
  tcost_bps=tcost_bps,
  options_data_provider=options_data_provider,
 )
 return Backtest(cfg).run(con)


# ---------------------------------------------------------------------------
# Options data
# ---------------------------------------------------------------------------

def load_orats_options_data(
 ticker: str,
 start_date: date | str,
 end_date: date | str | None = None,
 fields: tuple[str, ...] | None = None,
 **params: Any,
) -> pd.DataFrame:
 """Load options data from ORATS through the project data adapter.

 The ORATS endpoint adapter is isolated in optlab_research.data.orats. Until
 that adapter is implemented, this function raises the safe NotImplementedError
 from that module without exposing credentials or remote URLs.
 """
 return load_orats_options(
  ticker=ticker,
  start_date=start_date,
  end_date=end_date,
  fields=fields,
  **params,
 )


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def attribution(
 backtest_result,
 model: str = "ff6",
 con=None,
 rolling_window: int = 36,
):
 """Run factor attribution on a backtest result.

 Args:
  backtest_result: BacktestResult from wb.backtest()
  model: Attribution model ('capm', 'ff3', 'ff5', 'ff6', 'carhart4')
  con: DuckDB connection (from wb.open())
  rolling_window: Rolling beta window in months (default: 36)

 Returns:
  Dict with keys: alpha, betas, t_stats, r_squared, rolling_betas

 Example:
  with wb.open() as con:
   bt   = wb.backtest("momentum_12_2", "2019-01-01", "2023-12-31", con=con)
   attr = wb.attribution(bt, model="ff6", con=con)
   print(f"Alpha: {attr['alpha']:.4f} (t={attr['t_stats']['alpha']:.2f})")
 """
 from optlab_research.attribution.factor import factor_attribution
 return factor_attribution(backtest_result, model=model, con=con,
                          rolling_window=rolling_window)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(
 backtest_result,
 title: str = "Factor Backtest Report",
 subtitle: str = "",
 institution: str = "Northeastern University — 360 Huntington Fund",
):
 """Create a BacktestReport ready to save as HTML.

 Args:
  backtest_result: BacktestResult from wb.backtest()
  title: Report title
  subtitle: Report subtitle (signal, universe, period)
  institution: Institution name for the cover page

 Returns:
  BacktestReport object. Call .save("path.html") to write the file.

 Example:
  with wb.open() as con:
   bt  = wb.backtest("momentum_12_2", "2019-01-01", "2023-12-31", con=con)
   rpt = wb.report(bt, title="Momentum Factor",
                  subtitle="Russell 1000 | 2019-2023")
   rpt.save("outputs/momentum.html")
 """
 from optlab_research.reporting import BacktestReport
 from optlab_research.reporting.report import ReportConfig
 cfg = ReportConfig(title=title, subtitle=subtitle, institution=institution)
 return BacktestReport(backtest_result, config=cfg)


# ---------------------------------------------------------------------------
# Convenience: list available signals and universes
# ---------------------------------------------------------------------------

def list_signals() -> list[str]:
 """Return all available signal names from the registry."""
 from optlab_research.signals.registry import load_signals
 registry = load_signals()
 return list(registry.signals.keys())


def list_universes() -> list[str]:
 """Return all available universe preset names."""
 return ["russell3000", "russell1000", "liquid_500", "tradeable"]


def list_attribution_models() -> list[str]:
 """Return all available attribution models."""
 return ["capm", "ff3", "ff5", "ff6", "carhart4"]
