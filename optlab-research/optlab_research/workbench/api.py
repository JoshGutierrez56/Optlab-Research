"""Member-facing workbench API.

This is the only module a club member needs to import. It provides high-level
functions that hide connection management, registry lookups, and SQL plumbing.

Week 2 surface
--------------
    open()                        — context manager yielding a DuckDB connection
    universe(name, date, con)     — build a named universe DataFrame
    signal(name, date, ...)       — compute a signal cross-sectionally

Planned additions (later weeks)
--------------------------------
    backtest(signal, ...)         — Week 3
    attribution(backtest, ...)    — Week 6/7
    report(backtest, ...)         — Week 8

Connection management
---------------------
Two usage patterns are supported:

Pattern A — explicit connection (preferred for multiple calls in one session):
    with wb.open() as con:
        univ = wb.universe("russell1000", date, con=con)
        bm   = wb.signal("book_to_market", date, universe=univ, con=con)
        mom  = wb.signal("momentum_12_2",  date, universe=univ, con=con)

Pattern B — implicit connection (convenient for one-off calls):
    bm = wb.signal("book_to_market", date, universe="russell1000")

Pattern B opens and closes a fresh connection per call. Use Pattern A when
computing multiple signals on the same date — building the universe and
registering views only once is significantly faster.
"""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from typing import Generator

import duckdb
import polars as pl

from optlab_research.logging_setup import get_logger

log = get_logger(__name__)


# ─── Connection helper ────────────────────────────────────────────────────────


@contextmanager
def open(optlab_root=None) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Context manager yielding a fully-initialized DuckDB connection.

    Registers all optlab Parquet views and builds the CCM link views.
    Equivalent to ``optlab_research.open_connection()`` but exposed here
    so members only need to import from ``optlab_research.workbench``.

    Parameters
    ----------
    optlab_root : path-like, optional
        Root directory of the optlab package. If None, resolves via
        OPTLAB_ROOT environment variable or auto-discovery.

    Examples
    --------
    >>> from optlab_research import workbench as wb
    >>> with wb.open() as con:
    ...     univ = wb.universe("liquid_500", "2023-12-29", con=con)
    """
    import optlab_research as olr
    with olr.open_connection(optlab_root=optlab_root) as con:
        yield con


# ─── Universe ─────────────────────────────────────────────────────────────────


def universe(
    name: str,
    date: str | dt.date,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """Build a named universe as of *date*.

    Parameters
    ----------
    name : str
        Universe preset name from config/universes.yaml.
        Available: "russell3000", "russell1000", "liquid_500", "tradeable".
    date : str or datetime.date
        As-of date (PIT). ISO format string or date object.
    con  : DuckDB connection, optional.
        If None, a fresh connection is opened and closed automatically.
        Provide an explicit connection (from ``wb.open()``) when computing
        multiple signals on the same date.

    Returns
    -------
    pl.DataFrame
        Universe with columns: permno, ticker, name, prc, mcap_musd,
        gvkey, ceq, ni, at, oancf, and ~20 other fundamentals columns.

    Examples
    --------
    >>> univ = wb.universe("liquid_500", "2023-12-29")
    >>> print(f"{univ.height} names, {univ['mcap_musd'].is_not_null().sum()} with prices")
    """
    from optlab_research.universes.builder import get_universe

    if con is not None:
        return get_universe(name, date, con)

    # No connection provided — open a managed one for this call only.
    with open() as _con:
        return get_universe(name, date, _con)


# ─── Signal ───────────────────────────────────────────────────────────────────


def signal(
    name: str,
    date: str | dt.date,
    *,
    universe: str | pl.DataFrame | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
    n_quantiles: int = 5,
) -> pl.DataFrame:
    """Compute a registered signal cross-sectionally as of *date*.

    Parameters
    ----------
    name       : Signal name from config/signals.yaml.
                 e.g. "book_to_market", "momentum_12_2", "gross_profitability".
    date       : As-of date (PIT). ISO string or date object.
    universe   : Universe to compute the signal over. Three forms:
                   str          — named preset from universes.yaml; the builder
                                  is called automatically.
                   pl.DataFrame — pre-built universe (use this for speed when
                                  computing multiple signals on the same date).
                   None         — defaults to "russell3000".
    con        : DuckDB connection. If None, a fresh connection is opened.
                 Provide an explicit connection when making multiple calls.
    n_quantiles: Number of quantile buckets (default 5 = quintiles).

    Returns
    -------
    pl.DataFrame with columns:
        permno          int64   CRSP permanent number
        as_of_date      Date    Computation date
        signal_value    Float64 Raw signal value (null = could not compute)
        signal_rank     Float64 Cross-sectional percentile rank [0.0, 1.0]
        signal_quantile Int32   Bucket 1..n_quantiles (null for missing values)

    Examples
    --------
    >>> # One-liner — implicit connection, named universe
    >>> bm = wb.signal("book_to_market", "2023-12-29", universe="russell1000")

    >>> # Efficient multi-signal — one connection, one universe build
    >>> with wb.open() as con:
    ...     univ = wb.universe("liquid_500", "2023-12-29", con=con)
    ...     bm  = wb.signal("book_to_market", "2023-12-29", universe=univ, con=con)
    ...     mom = wb.signal("momentum_12_2",  "2023-12-29", universe=univ, con=con)
    """
    from optlab_research.signals.compute import compute_signal
    from optlab_research.universes.builder import get_universe

    # Resolve the universe name → default if None.
    universe_name: str | None = None
    universe_df: pl.DataFrame | None = None

    if universe is None:
        universe_name = "russell3000"
        log.debug("signal(): no universe specified, defaulting to 'russell3000'")
    elif isinstance(universe, str):
        universe_name = universe
    else:
        universe_df = universe

    if con is not None:
        # Connection already provided — use it directly.
        if universe_df is None:
            # Build the named universe on this connection.
            assert universe_name is not None
            universe_df = get_universe(universe_name, date, con)
        return compute_signal(name, date, con, universe=universe_df, n_quantiles=n_quantiles)

    # No connection provided — open one for this call.
    with open() as _con:
        if universe_df is None:
            assert universe_name is not None
            universe_df = get_universe(universe_name, date, _con)
        return compute_signal(name, date, _con, universe=universe_df, n_quantiles=n_quantiles)
