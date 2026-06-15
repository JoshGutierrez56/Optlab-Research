"""Market beta estimated from CAPM OLS over 60 monthly returns.

Frazzini-Pedersen (2014) Betting Against Beta (BAB) anomaly.

Methodology
-----------
For each stock, estimate:

    (r_i,t - rf_t) = alpha_i + beta_i * mktrf_t + epsilon_i,t

over the trailing *lookback_months* monthly observations. Signal = beta_i.

Data sources
------------
  - crsp_msf   : monthly stock returns (ret column)
  - ff_factors_monthly : mktrf (market excess return) and rf (risk-free rate)
    Assumed columns: date, mktrf, rf (at minimum)

PIT note
--------
Uses only historical returns. No look-ahead possible.

Sort direction
--------------
INVERTED. Q1 (lowest beta = most defensive) is the long side of the BAB anomaly.
The intuition: leverage-constrained investors (pension funds, mutual funds) tilt
toward high-beta stocks to amplify returns, bidding them up and making them
overpriced on a risk-adjusted basis. Low-beta stocks are neglected and earn an
anomalous premium.

Performance note
----------------
The regression is run in Python/NumPy per stock. For a ~3000-name universe with
60 months of data: ~3000 × 60 = ~180K rows. This is fast (seconds, not minutes).
If the full Russell 3000 is used for a panel of monthly signals across 30 years,
consider vectorizing with a single matrix solve — see idio_vol.py comments.

ff_factors_monthly view
-----------------------
The view name and column schema mirror ff_factors_daily. If the view is named
differently on your optlab install, either rename it in optlab's tables.yaml
or override by passing the view name via a future spec parameter.
"""
from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np
import polars as pl

from optlab_research.signals.registry import SignalSpec
from optlab_research.logging_setup import get_logger

log = get_logger(__name__)

_FF_MONTHLY_VIEW = "ff_factors_monthly"


def compute(
    con: duckdb.DuckDBPyConnection,
    date: dt.date,
    spec: SignalSpec,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    """Compute CAPM market beta for all permnos in *universe*.

    Parameters
    ----------
    con      : DuckDB connection with crsp_msf and ff_factors_monthly registered.
    date     : As-of date. Window = (date - lookback_months, date].
    spec     : SignalSpec. Reads lookback_months (default 60) and
               min_obs (default 36).
    universe : Universe DataFrame. Only the permno column is used.

    Returns
    -------
    pl.DataFrame with columns [permno, signal_value].
        signal_value = CAPM market beta.
        Permnos with fewer than min_obs valid monthly observations are excluded.

    Raises
    ------
    RuntimeError if ff_factors_monthly is not registered on *con*.
    """
    lookback: int = spec.lookback_months or 60
    min_obs: int = spec.min_obs or 36

    # Verify ff_factors_monthly is available.
    views = con.execute(
        f"SELECT view_name FROM duckdb_views() WHERE view_name = '{_FF_MONTHLY_VIEW}'"
    ).fetchall()
    if not views:
        raise RuntimeError(
            f"'{_FF_MONTHLY_VIEW}' view is not registered on the connection. "
            f"Ensure ff_factors_monthly data exists in the optlab data lake and "
            f"register_all_views() was called. "
            f"Available views: "
            + str([r[0] for r in con.execute("SELECT view_name FROM duckdb_views()").fetchall()])
        )

    # Build the date window. crsp_msf dates are month-ends; we over-fetch
    # slightly with a calendar-day buffer then let min_obs be the hard floor.
    # A safe approximation: lookback months * 31 days.
    start_date = date - dt.timedelta(days=lookback * 31 + 60)

    permnos = universe["permno"].cast(pl.Int64).to_list()
    if not permnos:
        return pl.DataFrame(
            {"permno": pl.Series([], dtype=pl.Int64),
             "signal_value": pl.Series([], dtype=pl.Float64)}
        )

    perm_df = pl.DataFrame({"permno": pl.Series(permnos, dtype=pl.Int64)})
    con.register("_beta_permnos_tmp", perm_df.to_arrow())

    try:
        sql = f"""
        SELECT
            m.permno,
            m.date::DATE  AS date,
            m.ret         AS ret,
            f.mktrf       AS mktrf,
            f.rf          AS rf
        FROM  crsp_msf            m
        INNER JOIN _beta_permnos_tmp p  ON  p.permno = m.permno
        INNER JOIN {_FF_MONTHLY_VIEW}  f  ON  f.date::DATE = m.date::DATE
        WHERE m.date::DATE >= CAST(? AS DATE)
          AND m.date::DATE <= CAST(? AS DATE)
          AND m.ret  IS NOT NULL
          AND f.mktrf IS NOT NULL
          AND f.rf    IS NOT NULL
        ORDER BY m.permno, m.date
        """
        raw = con.execute(sql, [start_date.isoformat(), date.isoformat()]).pl()
    finally:
        con.unregister("_beta_permnos_tmp")

    if raw.is_empty():
        log.warning(
            "beta_60m: no monthly data found for window %s–%s",
            start_date.isoformat(), date.isoformat(),
        )
        return pl.DataFrame(
            {"permno": pl.Series([], dtype=pl.Int64),
             "signal_value": pl.Series([], dtype=pl.Float64)}
        )

    log.info(
        "beta_60m: fetched %d monthly obs for %d permnos",
        raw.height, raw["permno"].n_unique(),
    )

    # ── Per-permno CAPM OLS ───────────────────────────────────────────────────
    results: list[dict] = []

    for permno_val, group in raw.group_by("permno", maintain_order=False):
        # Sort chronologically and take the most recent `lookback` months.
        g = group.sort("date").tail(lookback)
        n = len(g)
        if n < min_obs:
            # Insufficient history — signal will be null in compute_signal output.
            continue

        # Dependent variable: excess return = ret - rf
        y = (g["ret"] - g["rf"]).to_numpy().astype(np.float64)

        # Design matrix: intercept + mktrf
        X = np.column_stack([
            np.ones(n, dtype=np.float64),
            g["mktrf"].to_numpy().astype(np.float64),
        ])

        try:
            betas, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            log.debug("beta_60m: lstsq failed for permno %s — skipping", permno_val)
            continue

        if rank < X.shape[1]:
            # Near-singular: mktrf was essentially constant over the window
            # (e.g. a very short period). Exclude rather than report garbage.
            log.debug("beta_60m: rank-deficient for permno %s — skipping", permno_val)
            continue

        # betas[0] = intercept (alpha), betas[1] = market beta
        market_beta = float(betas[1])
        permno_scalar = permno_val[0] if isinstance(permno_val, tuple) else permno_val
        results.append({"permno": int(permno_scalar), "signal_value": market_beta})

    if not results:
        log.warning(
            "beta_60m: no permnos had sufficient history (min_obs=%d)", min_obs
        )
        return pl.DataFrame(
            {"permno": pl.Series([], dtype=pl.Int64),
             "signal_value": pl.Series([], dtype=pl.Float64)}
        )

    log.info(
        "beta_60m: computed for %d / %d permnos",
        len(results), len(permnos),
    )

    return pl.DataFrame(results).with_columns([
        pl.col("permno").cast(pl.Int64),
        pl.col("signal_value").cast(pl.Float64),
    ])
