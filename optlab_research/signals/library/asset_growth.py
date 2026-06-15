"""Asset growth signal (Cooper, Gulen, Schill 2008).

Signal = (AT_t - AT_{t-1}) / AT_{t-1}

where AT_t is total assets from the most recent PIT-available fiscal year
and AT_{t-1} is total assets from the prior fiscal year.

PIT correctness
---------------
Both observations are gated by COALESCE(rdq, datadate + 90 days) <= asof.
The prior-year observation uses the same gate — this is conservative and correct:
a fiscal-year t-1 report can only be used once it was publicly available.

Why this is a library signal (not a funda formula)
---------------------------------------------------
The universe DataFrame attaches only the single most recent PIT-available
Compustat row per gvkey. Computing year-over-year change requires two rows.
This function queries comp_funda directly for both rows.

Sort direction
--------------
INVERTED. Q1 (lowest / most negative asset growth = asset reduction) is
the long side. Overinvestment is destructive of value; firms that shrink
their asset base tend to outperform. Cooper et al. (2008) show that this
signal has roughly the same predictive power across growth funded by debt,
equity, or internal cash flows.

Fiscal year proximity requirement
----------------------------------
We require the two most recent PIT-available fiscal years to be approximately
one year apart: |datadate_t - datadate_{t-1}| in [180, 550] days. This
filters out stub periods and very stale second observations.

DuckDB notes
------------
1. `at` is a reserved keyword in DuckDB (AT TIME ZONE syntax). Double-quote
   it in all SQL references; alias to `at_val` inside the CTE.

2. comp_funda contains multiple rows per (gvkey, datadate) because the raw
   WRDS pull includes multiple format combinations (indfmt, consol, popsrc,
   datafmt). Even after the standard INDL/STD/D/C WHERE filter, duplicates
   can survive. We deduplicate to one row per (gvkey, datadate) using a
   dedup CTE before applying ROW_NUMBER — otherwise rn=1 and rn=2 both
   land on the same fiscal year and year_gap_days = 0 for every row.
"""
from __future__ import annotations

import datetime as dt

import duckdb
import polars as pl

from optlab_research.signals.registry import SignalSpec
from optlab_research.logging_setup import get_logger

log = get_logger(__name__)

_MIN_YEAR_GAP_DAYS: int = 180
_MAX_YEAR_GAP_DAYS: int = 550


def compute(
    con: duckdb.DuckDBPyConnection,
    date: dt.date,
    spec: SignalSpec,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    """Compute asset growth for all permnos in *universe*.

    Parameters
    ----------
    con      : DuckDB connection with comp_funda and ccm_link views registered.
    date     : As-of date. Both fiscal year observations must be PIT-gated
               COALESCE(rdq, datadate + 90 days) <= date.
    spec     : SignalSpec (no extra parameters used).
    universe : Universe DataFrame. Must contain gvkey and permno columns.

    Returns
    -------
    pl.DataFrame with columns [permno, signal_value].
        signal_value = (AT_recent - AT_prior) / AT_prior.
    """
    if "gvkey" not in universe.columns:
        raise ValueError(
            "asset_growth requires a 'gvkey' column in the universe DataFrame. "
            "Ensure get_universe() was called."
        )

    linked = universe.filter(pl.col("gvkey").is_not_null())
    if linked.is_empty():
        log.warning("asset_growth: no gvkey-linked permnos in universe")
        return pl.DataFrame(
            {"permno": pl.Series([], dtype=pl.Int64),
             "signal_value": pl.Series([], dtype=pl.Float64)}
        )

    gvkeys = linked["gvkey"].unique().to_list()
    gvkey_df = pl.DataFrame({"gvkey": pl.Series(gvkeys, dtype=pl.Utf8)})
    con.register("_ag_gvkeys_tmp", gvkey_df.to_arrow())

    try:
        # Two-step CTE:
        #   deduped    — one row per (gvkey, datadate), taking MAX("at") where
        #                duplicates exist. MAX is arbitrary but stable; the
        #                important thing is exactly one row per fiscal year.
        #   pit_ranked — ROW_NUMBER over distinct fiscal years, descending.
        sql = """
        WITH deduped AS (
            SELECT
                gvkey,
                datadate::DATE  AS datadate,
                MAX("at")       AS at_val
            FROM comp_funda
            WHERE gvkey IN (SELECT gvkey FROM _ag_gvkeys_tmp)
              AND "at" IS NOT NULL
              AND "at" > 0
              AND COALESCE(rdq::DATE, datadate::DATE + INTERVAL '90' DAY)
                  <= CAST(? AS DATE)
            GROUP BY gvkey, datadate::DATE
        ),
        pit_ranked AS (
            SELECT
                gvkey,
                datadate,
                at_val,
                ROW_NUMBER() OVER (
                    PARTITION BY gvkey
                    ORDER BY datadate DESC
                ) AS rn
            FROM deduped
        )
        SELECT
            r1.gvkey,
            r1.datadate  AS datadate_recent,
            r2.datadate  AS datadate_prior,
            r1.at_val    AS at_recent,
            r2.at_val    AS at_prior,
            DATEDIFF('day', r2.datadate, r1.datadate) AS year_gap_days
        FROM pit_ranked r1
        JOIN pit_ranked r2
          ON  r1.gvkey = r2.gvkey
          AND r1.rn = 1
          AND r2.rn = 2
        """
        raw = con.execute(sql, [date.isoformat()]).pl()
    finally:
        con.unregister("_ag_gvkeys_tmp")

    if raw.is_empty():
        log.warning("asset_growth: no gvkeys had two PIT-available fiscal years")
        return pl.DataFrame(
            {"permno": pl.Series([], dtype=pl.Int64),
             "signal_value": pl.Series([], dtype=pl.Float64)}
        )

    before = raw.height
    raw = raw.filter(
        (pl.col("year_gap_days") >= _MIN_YEAR_GAP_DAYS)
        & (pl.col("year_gap_days") <= _MAX_YEAR_GAP_DAYS)
    )
    log.debug(
        "asset_growth: proximity filter kept %d / %d rows "
        "(year_gap in [%d, %d] days)",
        raw.height, before, _MIN_YEAR_GAP_DAYS, _MAX_YEAR_GAP_DAYS,
    )

    gvkey_signals = raw.with_columns([
        ((pl.col("at_recent") - pl.col("at_prior")) / pl.col("at_prior"))
        .alias("signal_value")
    ]).select(["gvkey", "signal_value"])

    permno_map = linked.select(["permno", "gvkey"])
    result = (
        permno_map
        .join(gvkey_signals, on="gvkey", how="left")
        .select(["permno", "signal_value"])
    )

    log.info(
        "asset_growth as of %s: %d / %d linked permnos have signal",
        date.isoformat(),
        result["signal_value"].is_not_null().sum(),
        universe.height,
    )

    return result.select([
        pl.col("permno").cast(pl.Int64),
        pl.col("signal_value").cast(pl.Float64),
    ])