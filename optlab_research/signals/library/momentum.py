"""12-2 momentum signal (Jegadeesh-Titman 1993).

Formation window: the [t-(lookback+skip), t-skip] calendar months of monthly
returns from crsp_msf.

PIT note
--------
This signal uses past returns only. There is no look-ahead risk. The most
recent monthly observation included is the month-end of (asof - skip months),
which is by definition in the past relative to asof.

Skipping the most recent month
-------------------------------
Jegadeesh (1990) documents short-term reversal at the 1-month horizon. The
12-2 specification avoids this by excluding the most recent monthly return
from the formation window. Setting skip_months=0 gives 12-1 momentum (not
recommended for backtests without a microstructure-aware t-cost model).

Compounding
-----------
We use log-sum compounding: EXP(SUM(LOG(1+ret))) - 1. This is algebraically
equivalent to the product of gross returns but numerically more stable in SQL
(avoids multiplying hundreds of small numbers directly).

Missing returns
---------------
CRSP monthly returns can be null for the first/last observation of a security
or around delistings. We exclude null returns from the compound calculation.
The HAVING COUNT(*) >= min_months filter ensures we don't report a momentum
signal for securities with insufficient history.
"""
from __future__ import annotations

import calendar
import datetime as dt

import duckdb
import polars as pl

from optlab_research.signals.registry import SignalSpec
from optlab_research.logging_setup import get_logger

log = get_logger(__name__)


def _add_months(d: dt.date, months: int) -> dt.date:
    """Add *months* (may be negative) to *d*, clamping to month-end."""
    total = d.month + months
    year = d.year + (total - 1) // 12
    month = (total - 1) % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return dt.date(year, month, min(d.day, last_day))


def compute(
    con: duckdb.DuckDBPyConnection,
    date: dt.date,
    spec: SignalSpec,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    """Compute 12-2 momentum for all permnos in *universe* as of *date*.

    Parameters
    ----------
    con      : DuckDB connection with crsp_msf view registered.
    date     : As-of date. The formation window ends at the month-end
               of (date - skip_months).
    spec     : SignalSpec. Reads lookback_months (default 12) and
               skip_months (default 1).
    universe : Universe DataFrame. Only permno column is used.

    Returns
    -------
    pl.DataFrame with columns [permno, signal_value].
        signal_value = compounded gross return - 1 over the formation window.
        Permnos with fewer than (lookback_months - 2) valid monthly returns
        are excluded (they receive a null signal in compute_signal).
    """
    lookback: int = spec.lookback_months or 12
    skip: int = spec.skip_months or 1

    # The most recent monthly observation to include is the end of the month
    # that is `skip` months before `date`.
    end_month: dt.date = _add_months(date, -skip)
    # The oldest monthly observation to include is `lookback` months before that.
    start_month: dt.date = _add_months(end_month, -lookback)

    # Require at least (lookback - 2) valid months: tolerate up to 2 missing.
    min_months: int = max(1, lookback - 2)

    log.debug(
        "momentum window: %s to %s (min %d months)",
        start_month.isoformat(), end_month.isoformat(), min_months,
    )

    permnos = universe["permno"].cast(pl.Int64).to_list()
    if not permnos:
        return pl.DataFrame(
            {"permno": pl.Series([], dtype=pl.Int64),
             "signal_value": pl.Series([], dtype=pl.Float64)}
        )

    # Register the permno list as a temp Arrow relation so DuckDB can join to it.
    # Using UNNEST(?) on a list literal would work for small lists but is slow
    # for thousands of permnos. Arrow registration is O(1) and avoids the limit.
    perm_df = pl.DataFrame({"permno": pl.Series(permnos, dtype=pl.Int64)})
    con.register("_mom_permnos_tmp", perm_df.to_arrow())

    try:
        sql = """
        SELECT
            m.permno,
            -- Compound returns using log-sum for numerical stability.
            -- NULL returns are excluded; COALESCE(ret, 0) would survivorship-bias
            -- the compounding, so we filter them out with the INNER logic in HAVING.
            EXP(SUM(LN(1.0 + m.ret))) - 1.0  AS signal_value,
            COUNT(*)                           AS n_months_used
        FROM  crsp_msf   m
        INNER JOIN _mom_permnos_tmp p ON p.permno = m.permno
        WHERE m.date::DATE >= CAST(? AS DATE)
          AND m.date::DATE <= CAST(? AS DATE)
          AND m.ret IS NOT NULL
          AND m.ret > -1.0   -- guard against log(0) on complete-loss months
        GROUP BY m.permno
        HAVING COUNT(*) >= ?
        """
        raw = con.execute(
            sql,
            [start_month.isoformat(), end_month.isoformat(), min_months],
        ).pl()
    finally:
        con.unregister("_mom_permnos_tmp")

    log.info(
        "momentum_12_2 as of %s: %d / %d permnos have sufficient history",
        date.isoformat(), raw.height, len(permnos),
    )

    return raw.select([
        pl.col("permno").cast(pl.Int64),
        pl.col("signal_value").cast(pl.Float64),
    ])
