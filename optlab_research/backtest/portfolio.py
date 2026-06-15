"""Portfolio construction: weight assignment and return aggregation.

Conventions
-----------
* Weights: long positions are positive (+), short positions are negative (−).
* Within each leg, |weights| sum to 1.0:
    long leg:  sum(w_i) = +1.0
    short leg: sum(w_i) = −1.0
    combined:  sum(w_i) =  0.0
* Stocks with null signal_quantile are excluded from portfolio construction.
* Stocks with null forward_ret are dropped per period and remaining weights
  are renormalized before return aggregation (rather than forcing a 0 return
  on missing data, which would downward-bias long-short performance).

Weighting schemes (Week 5 additions)
-------------------------------------
equal   — each stock in a leg gets 1/N weight (v0, default)
rank    — weight ∝ |signal_rank − 0.5| × 2; extreme names get more weight
value   — weight ∝ mcap_musd within each leg; requires mcap_musd column in
          signal_df. Falls back to equal if mcap_musd is absent, with a warning.
ic      — weight ∝ |signal_value| within each leg; uses raw signal magnitude.
          Conceptually: names with a stronger signal contribute more to the leg.
          Falls back to equal if signal_value is all-null.

Public API
----------
    assign_weights(signal_df, *, portfolio_type, weighting, long_quantile,
                   short_quantile, n_quantiles) -> pl.DataFrame
    compute_portfolio_returns(holdings) -> pl.DataFrame
    compute_turnover(weights_curr, weights_prev) -> float
"""
from __future__ import annotations

import warnings
from enum import Enum

import polars as pl


# ─── Enums ────────────────────────────────────────────────────────────────────


class PortfolioType(str, Enum):
    quintile_long_short = "quintile_long_short"
    decile_long_short = "decile_long_short"
    long_only = "long_only"
    short_only = "short_only"


class WeightingScheme(str, Enum):
    equal = "equal"
    rank = "rank"
    value = "value"   # Week 5: market-cap weighted
    ic = "ic"         # Week 5: signal-magnitude weighted


# ─── Weight assignment ────────────────────────────────────────────────────────


def assign_weights(
    signal_df: pl.DataFrame,
    *,
    portfolio_type: str | PortfolioType,
    weighting: str | WeightingScheme,
    long_quantile: int,
    short_quantile: int,
    n_quantiles: int,
) -> pl.DataFrame:
    """Assign portfolio weights to a single cross-sectional signal snapshot.

    Parameters
    ----------
    signal_df
        Output of ``compute_signal`` (single-date snapshot):
        ``[permno, as_of_date, signal_value, signal_rank, signal_quantile]``.
        Week 5: if weighting="value", must also contain ``mcap_musd``.
    portfolio_type
        One of ``PortfolioType``.
    weighting
        One of ``WeightingScheme``. See module docstring for semantics.
    long_quantile, short_quantile
        Quantile buckets forming each leg.
    n_quantiles
        Number of buckets (informational).

    Returns
    -------
    pl.DataFrame
        Input columns preserved, plus:
        ``leg``    Utf8   "long" | "short"  (rows NOT in portfolio omitted)
        ``weight`` Float64  Positive for long, negative for short.
    """
    ptype = PortfolioType(portfolio_type)
    wscheme = WeightingScheme(weighting)

    valid = signal_df.filter(pl.col("signal_quantile").is_not_null())

    # ── Assign leg ────────────────────────────────────────────────────────────
    if ptype in (PortfolioType.quintile_long_short, PortfolioType.decile_long_short):
        valid = valid.with_columns(
            pl.when(pl.col("signal_quantile") == long_quantile)
            .then(pl.lit("long"))
            .when(pl.col("signal_quantile") == short_quantile)
            .then(pl.lit("short"))
            .otherwise(pl.lit(None))
            .alias("leg")
        )
    elif ptype == PortfolioType.long_only:
        valid = valid.with_columns(
            pl.when(pl.col("signal_quantile") == long_quantile)
            .then(pl.lit("long"))
            .otherwise(pl.lit(None))
            .alias("leg")
        )
    elif ptype == PortfolioType.short_only:
        valid = valid.with_columns(
            pl.when(pl.col("signal_quantile") == short_quantile)
            .then(pl.lit("short"))
            .otherwise(pl.lit(None))
            .alias("leg")
        )
    else:
        raise ValueError(f"Unhandled portfolio type: {ptype!r}")

    in_port = valid.filter(pl.col("leg").is_not_null())

    if in_port.height == 0:
        return in_port.with_columns(pl.lit(None).cast(pl.Float64).alias("weight"))

    # ── Assign weights ────────────────────────────────────────────────────────
    if wscheme == WeightingScheme.equal:
        in_port = _weight_equal(in_port)

    elif wscheme == WeightingScheme.rank:
        in_port = _weight_rank(in_port)

    elif wscheme == WeightingScheme.value:
        in_port = _weight_value(in_port)

    elif wscheme == WeightingScheme.ic:
        in_port = _weight_ic(in_port)

    else:
        raise ValueError(f"Unhandled weighting scheme: {wscheme!r}")

    return in_port


# ─── Weighting scheme implementations ────────────────────────────────────────


def _weight_equal(in_port: pl.DataFrame) -> pl.DataFrame:
    """Each stock in a leg gets 1/N weight."""
    counts = in_port.group_by("leg").agg(pl.len().alias("_n_leg"))
    return (
        in_port
        .join(counts, on="leg", how="left")
        .with_columns(
            pl.when(pl.col("leg") == "long")
            .then(1.0 / pl.col("_n_leg"))
            .otherwise(-1.0 / pl.col("_n_leg"))
            .alias("weight")
        )
        .drop("_n_leg")
    )


def _weight_rank(in_port: pl.DataFrame) -> pl.DataFrame:
    """Weight ∝ |signal_rank − 0.5| × 2. Extreme names get more weight."""
    in_port = in_port.with_columns(
        ((pl.col("signal_rank") - 0.5).abs() * 2.0).alias("_ext")
    )
    total_ext = (
        in_port
        .group_by("leg")
        .agg(pl.col("_ext").sum().alias("_total_ext"))
    )
    return (
        in_port
        .join(total_ext, on="leg", how="left")
        .with_columns(
            pl.when(pl.col("leg") == "long")
            .then(pl.col("_ext") / pl.col("_total_ext"))
            .otherwise(-(pl.col("_ext") / pl.col("_total_ext")))
            .alias("weight")
        )
        .drop(["_ext", "_total_ext"])
    )


def _weight_value(in_port: pl.DataFrame) -> pl.DataFrame:
    """Weight ∝ mcap_musd within each leg. Falls back to equal if absent.

    Market-cap weighting within a quintile is closer to what a size-aware
    PM would do. The long leg is therefore tilted toward the largest names
    in the top quintile; this reduces small-cap exposure and usually lowers
    both returns and volatility vs. equal-weight.
    """
    if "mcap_musd" not in in_port.columns:
        warnings.warn(
            "weighting='value' requires mcap_musd column in signal_df. "
            "Falling back to equal weighting. Attach universe with mcap_musd "
            "before calling assign_weights.",
            UserWarning,
            stacklevel=3,
        )
        return _weight_equal(in_port)

    # Zero or null mcap → assign a token weight equal to the minimum positive mcap
    # in the leg, so these names are not dropped but also don't dominate.
    min_pos_mcap = in_port.filter(pl.col("mcap_musd") > 0)["mcap_musd"].min()
    if min_pos_mcap is None or min_pos_mcap <= 0:
        min_pos_mcap = 1.0  # last resort: 1M floor

    in_port = in_port.with_columns(
        pl.col("mcap_musd")
        .fill_null(min_pos_mcap)
        .clip(lower_bound=min_pos_mcap)
        .alias("_mcap")
    )

    total_mcap = (
        in_port
        .group_by("leg")
        .agg(pl.col("_mcap").sum().alias("_total_mcap"))
    )
    return (
        in_port
        .join(total_mcap, on="leg", how="left")
        .with_columns(
            pl.when(pl.col("leg") == "long")
            .then(pl.col("_mcap") / pl.col("_total_mcap"))
            .otherwise(-(pl.col("_mcap") / pl.col("_total_mcap")))
            .alias("weight")
        )
        .drop(["_mcap", "_total_mcap"])
    )


def _weight_ic(in_port: pl.DataFrame) -> pl.DataFrame:
    """Weight ∝ |signal_value| within each leg. Falls back to equal if null.

    "IC weighting" is shorthand for weighting by signal conviction (the
    magnitude of the raw signal value), not the realized information
    coefficient. Names where the signal is close to zero contribute little.

    Both legs are normalized independently so each still sums to ±1:
      - Long leg:  weight_i = |signal_i| / Σ|signal_j| for j in long leg
      - Short leg: weight_i = −|signal_i| / Σ|signal_j| for j in short leg

    Note: using |signal_value| rather than signed signal_value ensures that
    the sign of the weight is determined by leg membership alone (already
    correct from quantile assignment), not by the signal's sign within the leg.
    """
    if "signal_value" not in in_port.columns:
        warnings.warn(
            "weighting='ic' requires signal_value column. Falling back to equal.",
            UserWarning,
            stacklevel=3,
        )
        return _weight_equal(in_port)

    # If all signal values are null in a leg, fall back to equal for that leg.
    non_null_count = in_port.filter(pl.col("signal_value").is_not_null()).height
    if non_null_count == 0:
        warnings.warn(
            "weighting='ic': all signal_value entries are null. Falling back to equal.",
            UserWarning,
            stacklevel=3,
        )
        return _weight_equal(in_port)

    # Replace null signal values with a small positive constant so names with
    # missing signals get a token (but small) weight rather than being dropped.
    # Use 1% of the cross-sectional mean absolute signal as the fill value.
    fill_val = float(
        in_port["signal_value"].abs().drop_nulls().mean() or 0.0
    ) * 0.01
    fill_val = max(fill_val, 1e-8)  # prevent div-by-zero edge case

    in_port = in_port.with_columns(
        pl.col("signal_value")
        .fill_null(fill_val)
        .abs()
        .alias("_abs_sig")
    )

    total_sig = (
        in_port
        .group_by("leg")
        .agg(pl.col("_abs_sig").sum().alias("_total_sig"))
    )
    return (
        in_port
        .join(total_sig, on="leg", how="left")
        .with_columns(
            pl.when(pl.col("leg") == "long")
            .then(pl.col("_abs_sig") / pl.col("_total_sig"))
            .otherwise(-(pl.col("_abs_sig") / pl.col("_total_sig")))
            .alias("weight")
        )
        .drop(["_abs_sig", "_total_sig"])
    )


# ─── Portfolio return aggregation ─────────────────────────────────────────────


def compute_portfolio_returns(holdings: pl.DataFrame) -> pl.DataFrame:
    """Aggregate holding-level forward returns into portfolio-level returns.

    Parameters
    ----------
    holdings
        Output of ``assign_weights`` with a ``forward_ret`` column appended.
        Required columns: ``[permno, as_of_date, leg, weight, forward_ret]``.
        (The engine remaps as_of_date to return_date before calling this.)

    Returns
    -------
    pl.DataFrame columns: ``[date, long_ret, short_ret, ls_ret]``

    Notes
    -----
    * Stocks with null ``forward_ret`` are dropped per (date, leg).
      Remaining weights are renormalized so they still sum to ±1.
    * If an entire leg is null for a period, that leg's return is null.
    * ``ls_ret`` is null only if *both* legs are null.
    """
    valid = holdings.filter(pl.col("forward_ret").is_not_null())

    if valid.height == 0:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "long_ret": pl.Float64,
                "short_ret": pl.Float64,
                "ls_ret": pl.Float64,
            }
        )

    # Renormalize |weights| to 1.0 per (date, leg) after null-return drop.
    abs_w = (
        valid
        .group_by(["as_of_date", "leg"])
        .agg(pl.col("weight").abs().sum().alias("_abs_w_sum"))
    )
    valid = (
        valid
        .join(abs_w, on=["as_of_date", "leg"], how="left")
        .with_columns(
            (pl.col("weight") / pl.col("_abs_w_sum")).alias("_w_norm")
        )
        .drop("_abs_w_sum")
    )

    valid = valid.with_columns(
        (pl.col("_w_norm") * pl.col("forward_ret")).alias("_wret")
    )

    leg_rets = (
        valid
        .group_by(["as_of_date", "leg"])
        .agg(pl.col("_wret").sum().alias("leg_ret"))
    )

    wide = (
        leg_rets
        .pivot(index="as_of_date", on="leg", values="leg_ret")
        .sort("as_of_date")
        .rename({"as_of_date": "date"})
    )

    for col in ("long", "short"):
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    wide = wide.rename({"long": "long_ret", "short": "short_ret"})

    wide = wide.with_columns(
        pl.when(pl.col("long_ret").is_null() & pl.col("short_ret").is_null())
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(
            pl.col("long_ret").fill_null(0.0) + pl.col("short_ret").fill_null(0.0)
        )
        .alias("ls_ret")
    )

    return wide.select(["date", "long_ret", "short_ret", "ls_ret"])


# ─── Turnover ─────────────────────────────────────────────────────────────────


def compute_turnover(
    weights_curr: pl.DataFrame,
    weights_prev: pl.DataFrame,
) -> float:
    """Compute one-way portfolio turnover between two consecutive periods.

    Turnover = (1/2) × Σ |w_{i,t} − w_{i,t−1}|
    """
    joined = (
        weights_curr
        .join(
            weights_prev.rename({"weight": "_w_prev"}),
            on="permno",
            how="full",
        )
        .with_columns([
            pl.col("weight").fill_null(0.0),
            pl.col("_w_prev").fill_null(0.0),
        ])
    )
    return float((joined["weight"] - joined["_w_prev"]).abs().sum() / 2.0)
