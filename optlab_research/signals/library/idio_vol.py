"""Idiosyncratic volatility signal (Ang, Hodrick, Xing, Zhang 2006).

Methodology
-----------
For each stock, estimate the FF3 model:

    r_i,t - rf_t = alpha_i + beta_MKT * (mktrf_t)
                            + beta_SMB * (smb_t)
                            + beta_HML * (hml_t)
                            + epsilon_i,t

over the trailing *lookback_days* trading days. The signal is the annualized
standard deviation of the residuals epsilon:

    idiovol_i = std(epsilon_i,t) * sqrt(252)

PIT note
--------
Uses crsp_dsf daily returns and ff_factors_daily, both of which are historical.
No look-ahead is possible.

Sort direction
--------------
The AHXZ anomaly is that high idiosyncratic vol stocks have LOWER subsequent
returns. So the long side is low idiovol (quintile 1). This is inverted relative
to most factors — the backtest configuration must account for this.

Performance
-----------
The FF3 OLS is run in Python/NumPy per stock. For a ~3000-name universe with
252 days of data, this is ~3000 × 252 = ~750K rows fetched from DuckDB, then
~3000 small regressions. On a modern laptop this takes 15–60 seconds. If
compute time becomes a bottleneck, the regression can be vectorized with
a single matrix solve (all stocks simultaneously) at the cost of memory.
"""
from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np
import polars as pl

from optlab_research.signals.registry import SignalSpec
from optlab_research.logging_setup import get_logger

log = get_logger(__name__)


def compute(
    con: duckdb.DuckDBPyConnection,
    date: dt.date,
    spec: SignalSpec,
    universe: pl.DataFrame,
) -> pl.DataFrame:
    """Compute idiosyncratic volatility for all permnos in *universe*.

    Parameters
    ----------
    con      : DuckDB connection with crsp_dsf and ff_factors_daily registered.
    date     : As-of date. Window = [date - lookback_days*1.6, date] calendar days.
               The 1.6× buffer converts trading days to calendar days approximately
               (252 trading days ≈ 365 calendar days). We over-fetch and the HAVING
               clause enforces the minimum.
    spec     : SignalSpec. Reads lookback_days (default 252) and min_obs (default 120).
    universe : Universe DataFrame. Only permno column is used.

    Returns
    -------
    pl.DataFrame with columns [permno, signal_value].
        signal_value = annualized idiovol = std(FF3 residuals) * sqrt(252).
        Permnos with fewer than min_obs valid daily observations are excluded.

    Raises
    ------
    RuntimeError if ff_factors_daily view is not registered on *con*.
    """
    lookback: int = spec.lookback_days or 252
    min_obs: int = spec.min_obs or 120

    # Generous calendar-day buffer: 252 trading days ≈ 365 calendar days.
    # We fetch a wider window and the per-permno HAVING enforces min_obs.
    cal_buffer = int(lookback * 1.6)
    start_date = date - dt.timedelta(days=cal_buffer)

    permnos = universe["permno"].cast(pl.Int64).to_list()
    if not permnos:
        return pl.DataFrame(
            {"permno": pl.Series([], dtype=pl.Int64),
             "signal_value": pl.Series([], dtype=pl.Float64)}
        )

    perm_df = pl.DataFrame({"permno": pl.Series(permnos, dtype=pl.Int64)})
    con.register("_idvol_permnos_tmp", perm_df.to_arrow())

    try:
        # Verify ff_factors_daily is available before issuing the big query.
        views = con.execute(
            "SELECT view_name FROM duckdb_views() WHERE view_name = 'ff_factors_daily'"
        ).fetchall()
        if not views:
            raise RuntimeError(
                "ff_factors_daily view is not registered on the connection. "
                "Ensure ff_factors_daily data exists in the optlab data lake and "
                "register_all_views() was called."
            )

        sql = """
        SELECT
            d.permno,
            d.date::DATE  AS date,
            d.ret         AS ret,
            f.mktrf       AS mktrf,
            f.smb         AS smb,
            f.hml         AS hml,
            f.rf          AS rf
        FROM  crsp_dsf          d
        INNER JOIN _idvol_permnos_tmp p ON p.permno = d.permno
        INNER JOIN ff_factors_daily   f ON f.date::DATE = d.date::DATE
        WHERE d.date::DATE >= CAST(? AS DATE)
          AND d.date::DATE <= CAST(? AS DATE)
          AND d.ret  IS NOT NULL
          AND f.mktrf IS NOT NULL
        ORDER BY d.permno, d.date
        """
        raw = con.execute(sql, [start_date.isoformat(), date.isoformat()]).pl()
    finally:
        con.unregister("_idvol_permnos_tmp")

    if raw.is_empty():
        log.warning("idio_vol: no daily data found for window %s–%s", start_date, date)
        return pl.DataFrame(
            {"permno": pl.Series([], dtype=pl.Int64),
             "signal_value": pl.Series([], dtype=pl.Float64)}
        )

    log.info(
        "idio_vol: fetched %d daily obs for %d permnos",
        raw.height, raw["permno"].n_unique(),
    )

    # ── Per-permno OLS ────────────────────────────────────────────────────────
    results: list[dict] = []

    for permno_val, group in raw.group_by("permno", maintain_order=False):
        g = group.sort("date")
        n = len(g)
        if n < min_obs:
            # Not enough history — signal will be null in the output.
            continue

        # Excess returns: ret_i - rf
        y: np.ndarray = (g["ret"] - g["rf"]).to_numpy().astype(np.float64)

        # Design matrix: intercept + MKT-RF + SMB + HML
        X: np.ndarray = np.column_stack([
            np.ones(n, dtype=np.float64),
            g["mktrf"].to_numpy().astype(np.float64),
            g["smb"].to_numpy().astype(np.float64),
            g["hml"].to_numpy().astype(np.float64),
        ])

        # OLS: beta = (X'X)^{-1} X'y via least squares.
        # np.linalg.lstsq handles rank-deficient cases gracefully.
        try:
            betas, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            log.debug("idio_vol: lstsq failed for permno %s — skipping", permno_val)
            continue

        if rank < X.shape[1]:
            # Singular or near-singular design matrix (e.g. factor constant
            # during a market halt). Skip rather than report garbage.
            log.debug("idio_vol: rank-deficient design for permno %s — skipping", permno_val)
            continue

        residuals = y - X @ betas

        # ddof=4: 4 estimated parameters (intercept + 3 factor betas).
        # Annualize by sqrt(252).
        idio_vol = float(np.std(residuals, ddof=4)) * np.sqrt(252)
        results.append({"permno": permno_val, "signal_value": idio_vol})

    if not results:
        log.warning("idio_vol: no permnos had sufficient history (min_obs=%d)", min_obs)
        return pl.DataFrame(
            {"permno": pl.Series([], dtype=pl.Int64),
             "signal_value": pl.Series([], dtype=pl.Float64)}
        )

    log.info(
        "idio_vol: computed for %d / %d permnos",
        len(results), len(permnos),
    )

    return pl.DataFrame(results).with_columns([
        pl.col("permno").cast(pl.Int64),
        pl.col("signal_value").cast(pl.Float64),
    ])
