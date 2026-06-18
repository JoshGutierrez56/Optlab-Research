"""Backtest engine: signal computation → portfolio construction → return aggregation.

Usage
-----
    from optlab_research.backtest import Backtest, BacktestConfig

    cfg = BacktestConfig(
        signal="book_to_market",
        start="2010-01-01",
        end="2024-12-31",
        universe="russell1000",
    )
    result = Backtest(cfg).run(con)
    print(result.summary())
    result.plot_cumulative()
    result.save("outputs/bm_russell1000_2010_2024/")

    # With transaction costs (Week 5)
    cfg = BacktestConfig(
        signal="momentum_12_2",
        start="2010-01-01",
        end="2024-12-31",
        universe="russell1000",
        tcost_model="flat_bps",
        tcost_bps=20.0,
    )
    result = Backtest(cfg).run(con)
    print(result.summary())  # now reports both gross and net returns

Design notes
------------
PIT correctness
    Signal computed at month-end t uses only data available by that date,
    enforced by get_universe_as_of (90-day lag) and compute_signal. The
    forward return is the return from t to t+1 (crsp_msf.ret at month-end t+1),
    which represents the return earned by holding positions entered at close of t.

    There is NO look-ahead: the signal on date t never sees returns from t onward.

Delisting adjustment
    crsp_msedelist is left-joined to crsp_msf. When a stock's monthly ret is
    NULL and a delisting occurred in that calendar month, the return is replaced
    with COALESCE(dlret, imputed_value). Imputation follows Shumway (1997):
      - dlstcd 500–584 (performance-related): −30%
      - all other involuntary delistings:     −55%
    This is the standard correction for survivorship bias documented in CRSP
    monthly return backtests. Without it, delistings cause an upward bias in
    equal-weighted strategies, since bankrupt/failed firms simply vanish.

Signal panel construction
    A date-by-date loop is architecturally unavoidable here because
    get_universe_as_of is stateful per date (it respects CRSP namedt/nameenddt
    windows and the PIT lag gate). For ~170 monthly dates over 2010–2024, expect
    ~5–15 minutes depending on WRDS network latency and universe size.

    Future optimization path: vectorize via DuckDB window functions if the
    universe builder is refactored to accept a list of dates. Not needed for v0.

Forward return join strategy
    Adjusted returns are fetched once for the entire date range in a single
    DuckDB query, then indexed into a Python dict for O(1) lookup during the
    per-holding forward_ret attachment. This avoids one DuckDB round-trip per
    stock per period.

Transaction costs (Week 5)
    If tcost_model is not "none", costs are computed BEFORE portfolio return
    aggregation using compute_tcost(). The net_forward_ret = forward_ret - tcost
    column is passed to compute_portfolio_returns so that BacktestResult contains
    both gross and net return series.

    Cost is only incurred on the TRADED portion of each position (|Δw| vs prior
    period), not on the full position size. This requires passing prev_weights
    to compute_tcost(), which the engine handles by tracking the prior period's
    weight DataFrame in the main loop.

Value weighting (Week 5)
    If weighting="value", the universe DataFrame must contain mcap_musd. The
    engine attaches this column to signal_panel before calling assign_weights.
    Capitalization data is PIT-correct because it comes from the universe builder
    which already enforces the 90-day fundamentals lag.

IC weighting (Week 5)
    If weighting="ic", each stock's weight within its leg is proportional to
    |signal_value|, scaled so the leg sums to ±1. This is conceptually similar
    to rank weighting but uses the raw signal rather than the percentile rank.
    The "IC" name reflects the idea that names with higher absolute signal
    contribute more, consistent with the transfer coefficient decomposition:
    IR = IC × sqrt(breadth) × TC. Members should understand that IC weighting
    is NOT the same as weighting by the signal's information coefficient — it's
    a naming convention for "weight by signal magnitude."

Date conventions
    - Rebalance dates ("signal dates"):  month-ends in crsp_msf within [start, end].
    - Return dates:                       the NEXT month-end after each signal date.
    - The last rebalance date has no forward return and is excluded from the
      portfolio return computation (it does contribute its return to the second-to-
      last period).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from pathlib import Path
from typing import Callable, Optional

import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from optlab_research.signals.registry import load_signals, SignalKind
from optlab_research.signals.compute import compute_signal
from optlab_research.backtest.portfolio import (
    PortfolioType,
    WeightingScheme,
    assign_weights,
    compute_portfolio_returns,
    compute_turnover,
)
from optlab_research.backtest.result import BacktestResult
from optlab_research.backtest.tcost import TcostConfig, TcostModel, compute_tcost
from optlab_research.logging_setup import get_logger

log = get_logger(__name__)

# Manifests written here unless BacktestConfig.manifest_dir overrides it.
# Resolved relative to repo root (this file is optlab_research/backtest/engine.py).
_DEFAULT_MANIFEST_DIR = (
    Path(__file__).resolve().parent.parent.parent / "manifests"
)


# ─── Configuration ────────────────────────────────────────────────────────────


class BacktestConfig(BaseModel):
    """Validated configuration for a single backtest run.

    All parameters live here so the manifest can reproduce the run exactly
    without external state. ConfigDict(extra="forbid") catches YAML typos.
    """

    model_config = ConfigDict(extra="forbid")

    signal: str
    """Registered signal name from config/signals.yaml."""

    start: str
    """ISO date string. Inclusive first rebalance date.
    Must correspond to a month-end present in crsp_msf."""

    end: str
    """ISO date string. Inclusive last rebalance date.
    The forward return for this date requires one additional month of CRSP data."""

    universe: str
    """Registered universe name from config/universes.yaml."""

    portfolio: str = "quintile_long_short"
    """Portfolio construction type (PortfolioType enum value)."""

    weighting: str = "equal"
    """Weight assignment scheme within each leg (WeightingScheme enum value).
    Week 5 adds: "value" (market-cap weighted) and "ic" (signal-magnitude weighted).
    """

    n_quantiles: int = Field(default=5, ge=2, le=20)
    """Number of quantile buckets. Default 5 = quintiles."""

    long_quantile: Optional[int] = None
    """Quantile that forms the long leg. Defaults to n_quantiles (top bucket)."""

    short_quantile: int = Field(default=1, ge=1)
    """Quantile that forms the short leg. Default 1 (bottom bucket)."""

    delist_impute_performance: float = -0.30
    """Return imputed for performance delistings (dlstcd 500–584).
    Shumway (1997) recommendation: −30%."""

    delist_impute_other: float = -0.55
    """Return imputed for all other involuntary delistings.
    Shumway (1997) / Beaver et al. (2007) recommendation: −55%."""

    # ── Transaction cost parameters (Week 5) ──────────────────────────────────

    tcost_model: str = "none"
    """Transaction cost model. One of: "none", "flat_bps", "half_spread", "sqrt_adv".
    Default "none" preserves v0 behavior (no t-costs)."""

    tcost_bps: float = Field(default=0.0, ge=0.0, le=500.0)
    """Round-trip cost in basis points. Used by "flat_bps" model and as
    fallback for "half_spread" / "sqrt_adv" when required data is absent."""

    tcost_adv_fraction: float = Field(default=0.01, gt=0.0, le=1.0)
    """Fraction of ADV assumed traded. Used by "sqrt_adv" model only."""

    tcost_eta: float = Field(default=0.01, gt=0.0)
    """Market-impact coefficient. Used by "sqrt_adv" model only."""

    manifest_dir: Optional[str] = None
    """Directory for manifest JSON files. Defaults to <repo_root>/manifests/."""

    options_data_provider: Optional[Callable[..., pl.DataFrame]] = None
    """Optional provider for external options data. Defaults to None so ordinary
    equity backtests never touch credentials or remote data providers."""

    @model_validator(mode="after")
    def _resolve_long_quantile(self) -> "BacktestConfig":
        if self.long_quantile is None:
            self.long_quantile = self.n_quantiles
        return self

    @property
    def start_date(self) -> dt.date:
        return dt.date.fromisoformat(self.start)

    @property
    def end_date(self) -> dt.date:
        return dt.date.fromisoformat(self.end)

    @property
    def tcost_config(self) -> TcostConfig:
        """Build TcostConfig from the flat parameters on this object."""
        return TcostConfig(
            model=TcostModel(self.tcost_model),
            flat_bps=self.tcost_bps,
            adv_fraction=self.tcost_adv_fraction,
            eta=self.tcost_eta,
        )


# ─── Engine ───────────────────────────────────────────────────────────────────


class Backtest:
    """Executes a factor backtest given a BacktestConfig and a DuckDB connection.

    Parameters
    ----------
    config : BacktestConfig or dict of keyword arguments.
    """

    def __init__(self, config: BacktestConfig | dict) -> None:
        if isinstance(config, dict):
            config = BacktestConfig(**config)
        self.config = config
        self.options_data_provider = config.options_data_provider

    @classmethod
    def from_kwargs(cls, **kwargs: object) -> "Backtest":
        """Convenience constructor: ``Backtest.from_kwargs(signal=..., ...)``."""
        return cls(BacktestConfig(**kwargs))

    # ── Public run ─────────────────────────────────────────────────────────────

    def run(self, con: duckdb.DuckDBPyConnection) -> BacktestResult:
        """Execute the full backtest pipeline.

        Pipeline
        --------
        1. Fetch all CRSP month-end dates in [start, end].
        2. Build (date_t, date_t+1) successor pairs for forward return assignment.
        3. Fetch delisting-adjusted monthly returns for the full range (one query).
        4. Loop over signal dates: build universe + compute signal.
        5. Attach forward return to each holding.
        6. [Week 5] Attach transaction costs per holding.
        7. Assign portfolio weights per date.
        8. Aggregate to portfolio-level monthly returns (gross and net).
        9. Compute per-period turnover and holdings summary.
        10. Build and write manifest.

        Parameters
        ----------
        con : Open DuckDB connection with optlab views registered.
              Obtain via ``optlab_research.open_connection()`` or
              ``optlab.db.connect()``.

        Returns
        -------
        BacktestResult
        """
        cfg = self.config
        tcost_cfg = cfg.tcost_config
        log.info(
            "Backtest.run() | signal=%s  universe=%s  %s → %s  portfolio=%s(%s)  tcost=%s@%.1fbps",
            cfg.signal, cfg.universe, cfg.start, cfg.end,
            cfg.portfolio, cfg.weighting,
            cfg.tcost_model, cfg.tcost_bps,
        )

        # ── Step 1: rebalance dates ───────────────────────────────────────────
        rebal_dates = _get_rebalance_dates(con, cfg.start_date, cfg.end_date)
        if len(rebal_dates) < 2:
            raise ValueError(
                f"Need ≥ 2 CRSP month-end dates in [{cfg.start}, {cfg.end}]; "
                f"found {len(rebal_dates)}. Check the date range and CRSP coverage."
            )
        log.info(
            "rebalance dates: %d  (%s … %s)",
            len(rebal_dates), rebal_dates[0], rebal_dates[-1],
        )

        # ── Step 2: (signal_date, return_date) pairs ──────────────────────────
        date_pairs: list[tuple[dt.date, dt.date]] = list(
            zip(rebal_dates[:-1], rebal_dates[1:])
        )
        signal_dates = [d for d, _ in date_pairs]
        return_date_map: dict[dt.date, dt.date] = {d: nxt for d, nxt in date_pairs}

        # ── Step 3: adjusted returns, full range, one query ───────────────────
        adj_rets = _get_adj_returns(
            con,
            start=rebal_dates[0],
            end=rebal_dates[-1],
            delist_perf=cfg.delist_impute_performance,
            delist_other=cfg.delist_impute_other,
        )
        adj_rets_lookup: dict[tuple[int, dt.date], float | None] = {
            (int(r["permno"]), r["date"]): r["adj_ret"]
            for r in adj_rets.to_dicts()
        }
        log.info("adjusted returns loaded: %d rows", adj_rets.height)

        # ── Step 4: signal panel ──────────────────────────────────────────────
        signal_panel, universe_sizes, universe_panels = _build_signal_panel(
            con, signal_dates, cfg
        )

        # ── Step 5: attach forward_ret ────────────────────────────────────────
        signal_panel = signal_panel.with_columns(
            pl.col("as_of_date")
            .map_elements(
                lambda d: return_date_map.get(d),
                return_dtype=pl.Date,
            )
            .alias("return_date")
        )

        signal_panel = signal_panel.with_columns(
            pl.struct(["permno", "return_date"])
            .map_elements(
                lambda row: adj_rets_lookup.get(
                    (int(row["permno"]), row["return_date"])
                ),
                return_dtype=pl.Float64,
            )
            .alias("forward_ret")
        )

        # ── Steps 6/7: weights, tcosts, portfolio returns ─────────────────────
        all_weights: list[pl.DataFrame] = []
        all_weighted_with_costs: list[pl.DataFrame] = []
        prev_weights_df: pl.DataFrame | None = None

        for sig_date in signal_dates:
            date_slice = signal_panel.filter(pl.col("as_of_date") == sig_date)
            if date_slice.height == 0:
                log.warning("no signal data for date %s — skipping", sig_date)
                continue

            # [Week 5] Attach universe price / volume data needed by tcost models.
            # universe_panels stores the raw universe DataFrame per date.
            if tcost_cfg.model != TcostModel.none and sig_date in universe_panels:
                univ = universe_panels[sig_date]
                price_cols = [c for c in ("prc", "bidlo", "askhi", "dvol", "mcap_musd") if c in univ.columns]
                if price_cols:
                    date_slice = date_slice.join(
                        univ.select(["permno"] + price_cols),
                        on="permno",
                        how="left",
                    )

            weighted = assign_weights(
                date_slice,
                portfolio_type=cfg.portfolio,
                weighting=cfg.weighting,
                long_quantile=cfg.long_quantile,  # type: ignore[arg-type]
                short_quantile=cfg.short_quantile,
                n_quantiles=cfg.n_quantiles,
            )
            all_weights.append(weighted)

            # [Week 5] Compute t-costs on the weighted holdings.
            if tcost_cfg.model != TcostModel.none:
                weighted_with_costs = compute_tcost(
                    weighted,
                    cfg=tcost_cfg,
                    prev_weights=prev_weights_df,
                )
                # Check if sqrt_adv fell back to flat (dvol missing).
                if "_tcost_fallback_to_flat" in weighted_with_costs.columns:
                    log.warning(
                        "date=%s: sqrt_adv model fell back to flat_bps (dvol missing).",
                        sig_date,
                    )
                    weighted_with_costs = weighted_with_costs.drop("_tcost_fallback_to_flat")
            else:
                # No tcost model — add zero column for consistent schema.
                weighted_with_costs = weighted.with_columns(
                    pl.lit(0.0).alias("tcost")
                )

            all_weighted_with_costs.append(weighted_with_costs)
            # Track this period's weights for next period's Δw computation.
            prev_weights_df = weighted.select(["permno", "weight"])

        if not all_weights:
            raise ValueError(
                "No portfolio weights produced. "
                "Check that the signal has coverage and the quantile thresholds are valid."
            )

        # Stack all periods into a single holdings DataFrame.
        holdings = pl.concat(all_weighted_with_costs, how="diagonal")

        # Remap as_of_date → return_date before aggregating (so output date
        # reflects when the return was EARNED, not when the signal was formed).
        holdings_for_agg = holdings.with_columns(
            pl.col("return_date").alias("as_of_date")
        )

        # [Week 5] Attach net_forward_ret = forward_ret - tcost.
        # Both gross and net go into BacktestResult.
        holdings_for_agg = holdings_for_agg.with_columns(
            (pl.col("forward_ret") - pl.col("tcost")).alias("net_forward_ret")
        )

        # ── Step 8: aggregate to portfolio returns (gross and net) ────────────
        gross_returns_df = compute_portfolio_returns(
            holdings_for_agg.rename({"forward_ret": "forward_ret"})
        )
        net_returns_df = compute_portfolio_returns(
            holdings_for_agg
            .with_columns(pl.col("net_forward_ret").alias("forward_ret"))
        )
        # Rename net columns to avoid collision.
        net_returns_df = net_returns_df.rename({
            "long_ret":  "net_long_ret",
            "short_ret": "net_short_ret",
            "ls_ret":    "net_ls_ret",
        })

        # Join gross and net into one returns DataFrame for BacktestResult.
        returns_df = gross_returns_df.join(net_returns_df, on="date", how="left")

        # ── Step 9: turnover and holdings summary ─────────────────────────────
        monthly_turnover = _compute_monthly_turnover(all_weights, signal_dates)
        holdings_summary = _build_holdings_summary(
            pl.concat(all_weights, how="diagonal")
        )

        # ── Step 10: manifest ─────────────────────────────────────────────────
        run_id = str(uuid.uuid4())
        manifest = _build_manifest(
            cfg=cfg,
            rebal_dates=rebal_dates,
            universe_sizes=universe_sizes,
            run_id=run_id,
        )

        manifest_dir = Path(cfg.manifest_dir) if cfg.manifest_dir else _DEFAULT_MANIFEST_DIR
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"backtest_{run_id[:8]}.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        log.info("manifest written → %s", manifest_path)

        return BacktestResult(
            returns=returns_df,
            holdings_summary=holdings_summary,
            monthly_turnover=monthly_turnover,
            manifest=manifest,
        )


# ─── Private helpers ─────────────────────────────────────────────────────────


def _get_rebalance_dates(
    con: duckdb.DuckDBPyConnection,
    start: dt.date,
    end: dt.date,
) -> list[dt.date]:
    """Return all distinct CRSP monthly file dates in [start, end], sorted asc."""
    rows = con.execute(
        """
        SELECT DISTINCT CAST(date AS DATE) AS month_end
        FROM crsp_msf
        WHERE CAST(date AS DATE) BETWEEN ? AND ?
        ORDER BY month_end
        """,
        [start, end],
    ).fetchall()
    return [r[0] for r in rows]


def _get_adj_returns(
    con: duckdb.DuckDBPyConnection,
    start: dt.date,
    end: dt.date,
    delist_perf: float,
    delist_other: float,
) -> pl.DataFrame:
    """Fetch delisting-adjusted monthly returns for all permnos in [start, end].

    Survivorship bias correction per Shumway (1997), Beaver et al. (2007),
    and all modern academic factor implementations.
    """
    query = f"""
        WITH monthly_rets AS (
            SELECT
                permno::BIGINT      AS permno,
                CAST(date AS DATE)  AS date,
                ret::DOUBLE         AS ret
            FROM crsp_msf
            WHERE CAST(date AS DATE) BETWEEN '{start}'::DATE AND '{end}'::DATE
        ),
        delistings AS (
            SELECT
                permno::BIGINT                              AS permno,
                DATE_TRUNC('month', CAST(dlstdt AS DATE))   AS dlst_month,
                dlret::DOUBLE                               AS dlret,
                dlstcd::INTEGER                             AS dlstcd
            FROM crsp_msedelist
            WHERE CAST(dlstdt AS DATE) BETWEEN '{start}'::DATE AND '{end}'::DATE
        )
        SELECT
            m.permno,
            m.date,
            CASE
                WHEN m.ret IS NOT NULL                THEN m.ret
                WHEN d.dlret IS NOT NULL               THEN d.dlret
                WHEN d.dlstcd BETWEEN 500 AND 584     THEN {delist_perf}
                WHEN d.dlstcd IS NOT NULL              THEN {delist_other}
                ELSE NULL
            END AS adj_ret
        FROM monthly_rets m
        LEFT JOIN delistings d
            ON  m.permno = d.permno
            AND DATE_TRUNC('month', m.date) = d.dlst_month
    """
    rows = con.execute(query).fetchall()

    if not rows:
        return pl.DataFrame(
            schema={"permno": pl.Int64, "date": pl.Date, "adj_ret": pl.Float64}
        )

    permnos, dates, rets = zip(*rows)
    return pl.DataFrame(
        {
            "permno": list(permnos),
            "date": pl.Series(list(dates), dtype=pl.Date),
            "adj_ret": list(rets),
        },
        schema={"permno": pl.Int64, "date": pl.Date, "adj_ret": pl.Float64},
    )


def _build_signal_panel(
    con: duckdb.DuckDBPyConnection,
    dates: list[dt.date],
    cfg: BacktestConfig,
) -> tuple[pl.DataFrame, dict[dt.date, int], dict[dt.date, pl.DataFrame]]:
    """Compute the signal cross-sectionally for each date in *dates*.

    Returns
    -------
    panel : pl.DataFrame
        Stacked signal output for all dates.
    universe_sizes : dict[date, int]
        Number of stocks in the universe on each date (for manifest).
    universe_panels : dict[date, pl.DataFrame]
        Full universe DataFrame per date (needed by tcost models for price/volume).
        Only populated when tcost_model != "none" to avoid memory overhead in
        vanilla backtests.
    """
    registry = load_signals()
    spec = registry.get(cfg.signal)
    attach_funda = spec.kind == SignalKind.funda
    tcost_needs_price = TcostModel(cfg.tcost_model) != TcostModel.none

    try:
        from optlab_research.universes.builder import get_universe  # type: ignore[import]

        def _get_univ(d: dt.date) -> pl.DataFrame:
            return get_universe(cfg.universe, d, con)

    except ImportError:
        from optlab.universe import get_universe_as_of  # type: ignore[import]

        log.warning(
            "optlab_research.universes.builder not importable; "
            "falling back to optlab.universe.get_universe_as_of with default filters. "
            "Universe preset '%s' will not be applied.",
            cfg.universe,
        )

        def _get_univ(d: dt.date) -> pl.DataFrame:  # type: ignore[misc]
            return get_universe_as_of(
                con, d,
                attach_gvkey=True,
                attach_secid=False,
                attach_funda=attach_funda,
            )

    panels: list[pl.DataFrame] = []
    universe_sizes: dict[dt.date, int] = {}
    universe_panels: dict[dt.date, pl.DataFrame] = {}

    for i, d in enumerate(dates):
        log.info(
            "signal panel %d/%d  date=%s  signal=%s",
            i + 1, len(dates), d, cfg.signal,
        )
        universe_df = _get_univ(d)
        universe_sizes[d] = universe_df.height

        # Store universe for tcost price/volume lookup.
        if tcost_needs_price:
            universe_panels[d] = universe_df

        sig_df = compute_signal(
            cfg.signal,
            d,
            con,
            universe=universe_df,
            n_quantiles=cfg.n_quantiles,
        )
        panels.append(sig_df)

    if not panels:
        raise ValueError("Signal panel is empty — no rebalance dates produced data.")

    return pl.concat(panels, how="vertical_relaxed"), universe_sizes, universe_panels


def _compute_monthly_turnover(
    all_weights: list[pl.DataFrame],
    signal_dates: list[dt.date],
) -> pl.DataFrame:
    """Compute one-way turnover for each period relative to the prior period."""
    records: list[dict] = []
    for i, sig_date in enumerate(signal_dates):
        if i == 0:
            to = float("nan")
        else:
            curr = all_weights[i].select(["permno", "weight"])
            prev = all_weights[i - 1].select(["permno", "weight"])
            to = compute_turnover(curr, prev)
        records.append({"date": sig_date, "turnover": to})

    return pl.DataFrame(
        {
            "date": pl.Series([r["date"] for r in records], dtype=pl.Date),
            "turnover": [r["turnover"] for r in records],
        },
        schema={"date": pl.Date, "turnover": pl.Float64},
    )


def _build_holdings_summary(holdings: pl.DataFrame) -> pl.DataFrame:
    """Per-date portfolio statistics: leg counts and average weights."""
    return (
        holdings
        .group_by("as_of_date")
        .agg([
            (pl.col("leg") == "long").sum().alias("n_long"),
            (pl.col("leg") == "short").sum().alias("n_short"),
            pl.col("weight").filter(pl.col("leg") == "long").mean().alias("avg_long_weight"),
            pl.col("weight").filter(pl.col("leg") == "short").mean().alias("avg_short_weight"),
        ])
        .sort("as_of_date")
        .rename({"as_of_date": "date"})
    )


def _build_manifest(
    cfg: BacktestConfig,
    rebal_dates: list[dt.date],
    universe_sizes: dict[dt.date, int],
    run_id: str,
) -> dict:
    """Build the full reproducibility manifest dict."""
    registry = load_signals()
    spec = registry.get(cfg.signal)
    signal_hash = hashlib.sha256(spec.model_dump_json().encode()).hexdigest()[:16]

    avg_univ_size = (
        round(sum(universe_sizes.values()) / len(universe_sizes), 1)
        if universe_sizes else 0
    )

    return {
        "run_id": run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        # Signal
        "signal": cfg.signal,
        "signal_spec_hash": signal_hash,
        # Universe
        "universe": cfg.universe,
        # Date range
        "start": cfg.start,
        "end": cfg.end,
        "n_rebalance_dates": len(rebal_dates),
        "first_rebal_date": str(rebal_dates[0]) if rebal_dates else None,
        "last_rebal_date": str(rebal_dates[-1]) if rebal_dates else None,
        # Portfolio construction
        "portfolio": cfg.portfolio,
        "weighting": cfg.weighting,
        "n_quantiles": cfg.n_quantiles,
        "long_quantile": cfg.long_quantile,
        "short_quantile": cfg.short_quantile,
        # Delisting imputation
        "delist_impute_performance": cfg.delist_impute_performance,
        "delist_impute_other": cfg.delist_impute_other,
        # Transaction costs
        "tcost_model": cfg.tcost_model,
        "tcost_bps": cfg.tcost_bps,
        "tcost_adv_fraction": cfg.tcost_adv_fraction,
        "tcost_eta": cfg.tcost_eta,
        # Universe coverage stats
        "avg_universe_size": avg_univ_size,
        # Package version
        "optlab_research_version": "0.1.0",
    }
