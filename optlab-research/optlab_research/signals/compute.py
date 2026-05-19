"""Signal computation: dispatch, formula evaluation, ranking.

Public API
----------
    compute_signal(name, date, con, *, universe=None, n_quantiles=5) -> pl.DataFrame

Output schema
-------------
    permno          int     CRSP security identifier
    as_of_date      date    The computation date (PIT)
    signal_value    float   Raw signal value; null = could not compute
    signal_rank     float   Cross-sectional percentile rank [0.0, 1.0]; null for missing
    signal_quantile int     Bucket 1 (bottom) to n_quantiles (top); null for missing

Design notes
------------
Formula evaluation
    funda / crsp_price signals are Python expressions evaluated via
    eval(formula, {"pl": polars}). The formula must return a pl.Expr.
    
    eval() is used intentionally. config/signals.yaml is developer-controlled,
    not user input. Do not expose formula evaluation to untrusted callers.

PIT correctness
    The universe DataFrame from get_universe_as_of() attaches Compustat
    fundamentals using the 90-day lag rule (datadate <= asof - 90 days).
    
    For funda-kind signals, the compute layer does NOT re-query fundamentals —
    it uses the columns already in the universe DataFrame. This means the PIT
    precision matches the universe builder (90-day lag, not rdq). This is a
    known simplification: the universe already enforces no look-ahead, and
    adding a separate lateral join per signal would multiply query cost.
    
    If stricter rdq-based PIT is required for a specific signal, implement it
    as a library signal that does its own DuckDB query.

Null handling
    Signals that cannot be computed (missing fundamentals, insufficient return
    history, etc.) are NULL in the output, NOT silently dropped. The backtest
    decides what to do with NULLs at portfolio construction time.
    
    Null rows are excluded from rank computation so they don't shift the
    percentile of non-null observations.
"""
from __future__ import annotations

import datetime as dt
import importlib
from typing import Callable

import duckdb
import polars as pl

from optlab_research.signals.registry import SignalKind, SignalSpec, load_signals
from optlab_research.logging_setup import get_logger

log = get_logger(__name__)

# Module-level registry cache. Reloaded only if the module is reloaded.
_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        _registry = load_signals()
    return _registry


# ─── Formula evaluation ───────────────────────────────────────────────────────


def _apply_formula(formula: str, universe: pl.DataFrame) -> pl.Series:
    """Evaluate a Polars expression string against *universe* and return the result Series.

    The expression is evaluated with only ``pl`` (polars) in scope. It must
    return a pl.Expr. Non-finite values (inf, -inf) from division-by-zero are
    converted to null downstream by compute_signal().

    Parameters
    ----------
    formula : str
        Python expression, e.g. ``"pl.col('ceq') / pl.col('mcap_musd')"``
    universe : pl.DataFrame
        Universe DataFrame from get_universe_as_of(). Must contain all
        columns referenced in the formula.

    Raises
    ------
    NameError   if the formula references an undefined name.
    ColumnNotFoundError if the formula references a column not in universe.
    """
    # noqa: S307 — intentional eval; see module docstring for justification
    expr: pl.Expr = eval(formula, {"pl": pl})  # type: ignore[arg-type]
    return universe.select(expr.alias("_signal_value"))["_signal_value"]


# ─── Library function loader ──────────────────────────────────────────────────


def _load_library_fn(dotted_path: str) -> Callable:
    """Import a callable from a dotted module path.

    Example: ``"optlab_research.signals.library.momentum.compute"``
    """
    module_path, fn_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise AttributeError(
            f"Module {module_path!r} has no attribute {fn_name!r}. "
            f"Check the library_fn path in signals.yaml."
        )
    return fn


# ─── Ranking ─────────────────────────────────────────────────────────────────


def _rank_and_quantile(df: pl.DataFrame, n_quantiles: int = 5) -> pl.DataFrame:
    """Add signal_rank (percentile [0,1]) and signal_quantile (1..n_quantiles).

    Ranking is cross-sectional: rank 1.0 = highest signal value = top quantile.
    Uses average rank for ties, which is the standard in academic factor research.

    Parameters
    ----------
    df          : DataFrame with a ``signal_value`` column (no nulls).
    n_quantiles : number of buckets (default 5 = quintiles).

    Notes
    -----
    * signal_rank uses average-rank method; tied observations get the same rank.
    * signal_quantile = ceil(signal_rank * n_quantiles), clipped to [1, n_quantiles].
      The clip is needed because rank can equal 1.0 (the top observation), and
      ceil(1.0 * 5) = 5 is already in bounds, but numerical noise could push it to 6.
    """
    n = pl.col("signal_value").count()
    pct_rank = pl.col("signal_value").rank(method="average") / n

    return df.with_columns([
        pct_rank.alias("signal_rank"),
        (pct_rank * n_quantiles)
        .ceil()
        .cast(pl.Int32)
        .clip(1, n_quantiles)
        .alias("signal_quantile"),
    ])


# ─── Main entry point ─────────────────────────────────────────────────────────


def compute_signal(
    name: str,
    date: str | dt.date,
    con: duckdb.DuckDBPyConnection,
    *,
    universe: pl.DataFrame | None = None,
    n_quantiles: int = 5,
) -> pl.DataFrame:
    """Compute a registered signal cross-sectionally as of *date*.

    Parameters
    ----------
    name        : Signal name as defined in config/signals.yaml.
    date        : As-of date (PIT). ``str`` in ISO format or ``datetime.date``.
    con         : Open DuckDB connection with optlab views registered.
                  Use ``optlab_research.open_connection()`` to get one.
    universe    : Pre-built universe DataFrame from
                  ``optlab.universe.get_universe_as_of()``.
                  If None, constructed internally with default parameters
                  (attach_funda=True, min_price=1.0, US common stocks).
                  Providing it explicitly is faster when computing multiple
                  signals on the same date.
    n_quantiles : Number of quantile buckets (default 5 = quintiles).

    Returns
    -------
    pl.DataFrame with columns:
        permno          int64   CRSP permanent number
        as_of_date      Date    Computation date
        signal_value    Float64 Raw signal value (null = could not compute)
        signal_rank     Float64 Cross-sectional percentile rank [0.0, 1.0]
        signal_quantile Int32   Quantile bucket 1 (bottom) to n_quantiles (top)

    Raises
    ------
    KeyError    if *name* is not registered in signals.yaml.
    ValueError  if required universe columns are missing.

    Examples
    --------
    >>> with open_connection() as con:
    ...     bm = compute_signal("book_to_market", "2023-12-29", con)
    ...     print(bm.head(5))
    """
    # ── Resolve date ──────────────────────────────────────────────────────────
    if isinstance(date, str):
        asof = dt.date.fromisoformat(date)
    else:
        asof = date

    # ── Look up spec ──────────────────────────────────────────────────────────
    spec = _get_registry().get(name)
    log.info("computing signal '%s' as of %s", name, asof.isoformat())

    # ── Build universe if not provided ────────────────────────────────────────
    if universe is None:
        # Lazy import: optlab must be on the Python path (pip install -e <path>).
        from optlab.universe import get_universe_as_of  # type: ignore[import]
        universe = get_universe_as_of(
            con,
            asof,
            attach_gvkey=True,
            attach_secid=False,   # optcrsp_link only exists with full OptionMetrics access
            attach_funda=(spec.kind == SignalKind.funda),
        )
        log.info("universe built: %d names", universe.height)

    # ── Validate required columns ─────────────────────────────────────────────
    if spec.required_columns:
        missing = [c for c in spec.required_columns if c not in universe.columns]
        if missing:
            raise ValueError(
                f"Signal '{name}' requires columns {missing} which are absent "
                f"from the universe DataFrame. "
                f"Ensure get_universe_as_of() was called with attach_funda=True "
                f"for funda-kind signals."
            )

    # ── Dispatch by kind ──────────────────────────────────────────────────────
    if spec.kind in (SignalKind.funda, SignalKind.crsp_price):
        values = _apply_formula(spec.formula, universe)  # type: ignore[arg-type]
        raw = pl.DataFrame({
            "permno": universe["permno"].cast(pl.Int64),
            "as_of_date": pl.Series([asof] * len(universe), dtype=pl.Date),
            "signal_value": values.cast(pl.Float64),
        })

    elif spec.kind == SignalKind.library:
        fn = _load_library_fn(spec.library_fn)  # type: ignore[arg-type]
        # Library functions return pl.DataFrame with at least [permno, signal_value].
        raw_lib = fn(con=con, date=asof, spec=spec, universe=universe)
        raw = (
            raw_lib
            .with_columns([
                pl.col("permno").cast(pl.Int64),
                pl.lit(asof).alias("as_of_date"),
                pl.col("signal_value").cast(pl.Float64),
            ])
            .select(["permno", "as_of_date", "signal_value"])
        )

    else:
        raise ValueError(f"Unhandled signal kind: {spec.kind!r}")

    # ── Separate valid vs. null/infinite rows ─────────────────────────────────
    # Nulls and inf are excluded from the rank computation so they don't compress
    # the percentiles of valid observations.
    is_valid = (
        pl.col("signal_value").is_not_null()
        & pl.col("signal_value").is_finite()
    )
    valid = raw.filter(is_valid)
    invalid = raw.filter(~is_valid)

    log.info(
        "signal '%s' as of %s: %d valid, %d null/inf",
        name, asof.isoformat(), valid.height, invalid.height,
    )

    # ── Rank valid rows ───────────────────────────────────────────────────────
    if valid.height > 0:
        ranked_valid = _rank_and_quantile(valid, n_quantiles=n_quantiles)
    else:
        ranked_valid = valid.with_columns([
            pl.lit(None).cast(pl.Float64).alias("signal_rank"),
            pl.lit(None).cast(pl.Int32).alias("signal_quantile"),
        ])

    # Null rows get null rank and quantile — they are NOT assigned to quintile 1.
    ranked_invalid = invalid.with_columns([
        pl.lit(None).cast(pl.Float64).alias("signal_rank"),
        pl.lit(None).cast(pl.Int32).alias("signal_quantile"),
    ])

    return (
        pl.concat([ranked_valid, ranked_invalid])
        .sort("permno")
        .select(["permno", "as_of_date", "signal_value", "signal_rank", "signal_quantile"])
    )