"""Short-term reversal signal (Jegadeesh 1990).

Signal = prior-month return from crsp_msf.

Formation window
----------------
For as-of date D, the signal is the return of the month-end observation
immediately preceding the month containing D. That is:
  - end_month = last crsp_msf observation with date < first_day_of_month(D)
  - In practice, for D = 2023-12-29, end_month ≈ 2023-11-30.

We do not assume CRSP month-ends fall on the calendar month-end exactly; we
use LATERAL to fetch the most recent observation with date <= last day of the
prior calendar month, which handles non-trading-day month-ends gracefully.

PIT note
--------
Returns are historical by construction. No look-ahead is possible.

Sort direction
--------------
INVERTED. Q1 (most negative prior return = biggest losers) is the long side.
Short-term reversal profits from mean reversion. Stocks that dropped last month
tend to bounce; stocks that rose tend to pull back.
Document this explicitly when reporting factor analysis.

Relationship to momentum_12_2
------------------------------
The two signals are negatively correlated at the 1-month lag. When combining
them, the standard approach is to include short_term_reversal as a control
variable in the OLS attribution rather than as a standalone factor, because
its paper profit vanishes almost entirely after realistic t-costs.
"""
from __future__ import annotations

import calendar
import datetime as dt

import duckdb
import polars as pl

from optlab_research.signals.registry import SignalSpec
from optlab_research.logging_setup import get_logger

log = get_logger(__name__)


def _last_day_of_prior_month(d: dt.date) -> dt.date:
    """Return the last calendar day of the month prior to *d*."""
    # First day of current month, then subtract 1 day.
    first_of_current = d.replace(day=1)
    return first_of_current - dt.timedelta(days=1)


def compute(
    con: duckdb.DuckDBPyConnection,
    date: dt.date,
    spec: SignalSpec,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    """Compute prior-month return for all permnos in *universe*.

    Parameters
    ----------
    con      : DuckDB connection with crsp_msf view registered.
    date     : As-of date. Signal = return in the month ending just before
               the month that contains *date*.
    spec     : SignalSpec (no parameters used; included for API consistency).
    universe : Universe DataFrame. Only the permno column is used.

    Returns
    -------
    pl.DataFrame with columns [permno, signal_value].
        signal_value = monthly return (e.g. 0.05 = +5%).
        Permnos with no crsp_msf observation in the prior month get null
        (handled upstream by compute_signal).
    """
    end_of_prior_month = _last_day_of_prior_month(date)
    # Start of the search window: first day of that same prior month.
    start_of_prior_month = end_of_prior_month.replace(day=1)

    permnos = universe["permno"].cast(pl.Int64).to_list()
    if not permnos:
        return pl.DataFrame(
            {"permno": pl.Series([], dtype=pl.Int64),
             "signal_value": pl.Series([], dtype=pl.Float64)}
        )

    perm_df = pl.DataFrame({"permno": pl.Series(permnos, dtype=pl.Int64)})
    con.register("_str_permnos_tmp", perm_df.to_arrow())

    try:
        # Use LATERAL to fetch the single most recent monthly obs within the
        # prior-calendar-month window. This is safe even if the month-end
        # trading day is the 28th or 30th (not the 31st).
        sql = """
        SELECT
            p.permno,
            m.ret AS signal_value
        FROM _str_permnos_tmp p
        LEFT JOIN LATERAL (
            SELECT ret
            FROM crsp_msf
            WHERE permno = p.permno
              AND date::DATE >= CAST(? AS DATE)
              AND date::DATE <= CAST(? AS DATE)
              AND ret IS NOT NULL
            ORDER BY date DESC
            LIMIT 1
        ) m ON TRUE
        """
        raw = con.execute(
            sql,
            [start_of_prior_month.isoformat(), end_of_prior_month.isoformat()],
        ).pl()
    finally:
        con.unregister("_str_permnos_tmp")

    log.info(
        "short_term_reversal as of %s (prior month window %s–%s): "
        "%d / %d permnos have return",
        date.isoformat(),
        start_of_prior_month.isoformat(),
        end_of_prior_month.isoformat(),
        raw["signal_value"].is_not_null().sum(),
        len(permnos),
    )

    return raw.select([
        pl.col("permno").cast(pl.Int64),
        pl.col("signal_value").cast(pl.Float64),
    ])
