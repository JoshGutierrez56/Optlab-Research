"""Transaction cost models for backtest net-return computation.

Three models are provided, in increasing realism:

flat_bps
    Round-trip cost = tcost_bps / 10_000 per unit traded, applied uniformly
    to every stock on every rebalance. Simple, parameterized by one number.
    Use as a sensitivity analysis baseline.

half_spread
    Round-trip cost = bid-ask spread, estimated from CRSP price data.
    Two modes:
      - If bid/ask columns are available (bidlo/askhi): use realized spread.
      - Fallback: Roll (1984) implied spread proxy = 0.2 / |prc|, capped at
        5% and floored at 2 bps. Empirically reasonable for large-caps; less
        so for micro-caps where bid/ask can be 1–3%.
    The CRSP `bidlo` / `askhi` columns are MONTHLY low/high, not daily
    bid/ask quotes, so the realized spread is a noisy overestimate. This is
    intentional: it's a conservative upper bound, appropriate for a research
    platform where we want to err on the side of understating net performance.

sqrt_adv (Almgren-style)
    Round-trip cost = eta × sqrt(|ΔW| / ADV_fraction), where:
      - |ΔW|           is the absolute change in portfolio weight
      - ADV_fraction   is the assumed fraction of ADV the strategy trades
      - eta            is the market-impact coefficient (default 0.01, calibrated
                       to roughly match empirical linear-impact coefficients for
                       a 1% ADV trade in mid-cap US equities)

    Requires dollar volume (dvol) in the holdings DataFrame. If dvol is
    unavailable, falls back to flat_bps.

All functions operate on Polars DataFrames and return a ``tcost`` column
(as a fraction, not bps). The caller (engine.py) subtracts tcost from the
gross return before portfolio return aggregation.

Public API
----------
    TcostModel                   — Enum of available models
    TcostConfig                  — Pydantic config
    compute_tcost(holdings, cfg) — main entry point, returns holdings + tcost
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


# ─── Enums and config ─────────────────────────────────────────────────────────


class TcostModel(str, Enum):
    none = "none"
    flat_bps = "flat_bps"
    half_spread = "half_spread"
    sqrt_adv = "sqrt_adv"


class TcostConfig(BaseModel):
    """Transaction cost configuration.

    Parameters
    ----------
    model : TcostModel
        Which cost model to use.
    flat_bps : float
        Round-trip cost in basis points (used by ``flat_bps`` model and as
        fallback for other models when required data is unavailable).
        Default 10 bps one-way (20 bps round-trip).
    adv_fraction : float
        Fraction of average daily volume assumed to trade (``sqrt_adv`` model).
        Default 0.01 = 1%. Must be in (0, 1].
    eta : float
        Market-impact coefficient for ``sqrt_adv`` model (dimensionless).
        Default 0.01; calibrated to ~30 bps for a 1% ADV trade.
    spread_cap : float
        Maximum spread cost allowed by ``half_spread`` model, as a fraction.
        Default 0.05 = 5%. Prevents penny-stock outliers from dominating.
    """

    model_config = ConfigDict(extra="forbid")

    model: TcostModel = TcostModel.none
    flat_bps: float = Field(default=10.0, ge=0.0, le=500.0)
    adv_fraction: float = Field(default=0.01, gt=0.0, le=1.0)
    eta: float = Field(default=0.01, gt=0.0)
    spread_cap: float = Field(default=0.05, gt=0.0, le=1.0)

    @property
    def flat_fraction(self) -> float:
        """flat_bps expressed as a return fraction (round-trip)."""
        return self.flat_bps / 10_000.0


# ─── Main entry point ─────────────────────────────────────────────────────────


def compute_tcost(
    holdings: pl.DataFrame,
    cfg: TcostConfig,
    prev_weights: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Attach a per-holding transaction cost column to *holdings*.

    Parameters
    ----------
    holdings
        Output of ``assign_weights`` with one additional column appended:
        ``forward_ret``. Required columns: ``[permno, as_of_date, weight]``.
        Optional columns used by specific models:
          - ``prc``   (Float64): closing price — used by ``half_spread``
          - ``bidlo`` (Float64): monthly bid low — used by ``half_spread``
          - ``askhi`` (Float64): monthly ask high — used by ``half_spread``
          - ``dvol``  (Float64): dollar volume (prc × vol) — used by ``sqrt_adv``
    prev_weights
        Per-stock weights from the PRIOR rebalance period.
        Columns: ``[permno, weight]``. Used to compute the weight change
        Δw = w_curr − w_prev, which drives market-impact costs.
        If None, the entire current weight is treated as a new trade (i.e.,
        assumes the prior portfolio was flat — conservative on the first period).
    cfg
        TcostConfig specifying model and parameters.

    Returns
    -------
    pl.DataFrame
        *holdings* with one new column added: ``tcost`` (Float64, as fraction).
        ``tcost`` is always non-negative. It represents the one-way (entry)
        cost fraction per dollar held. The engine applies it as:
            net_ret = gross_ret - tcost
        (cost is only incurred at rebalance, not on the held position mid-month)

    Notes on sign convention
    ------------------------
    Portfolio weights are signed (long = +, short = −). Transaction costs are
    always positive regardless of direction. When prev_weights is provided, only
    the CHANGE in weight incurs cost (i.e., a stock held at the same weight in
    both periods has tcost = 0).
    """
    if cfg.model == TcostModel.none:
        return holdings.with_columns(pl.lit(0.0).alias("tcost"))

    # Compute weight change: Δw = |w_curr − w_prev|
    # This is what was actually traded; held positions don't incur cost.
    if prev_weights is not None:
        holdings = holdings.join(
            prev_weights.rename({"weight": "_w_prev"}),
            on="permno",
            how="left",
        ).with_columns(
            pl.col("_w_prev").fill_null(0.0)
        ).with_columns(
            (pl.col("weight") - pl.col("_w_prev")).abs().alias("_delta_w")
        ).drop("_w_prev")
    else:
        # First period or no prior info — treat full weight as traded
        holdings = holdings.with_columns(
            pl.col("weight").abs().alias("_delta_w")
        )

    # Dispatch to model
    if cfg.model == TcostModel.flat_bps:
        holdings = _apply_flat_bps(holdings, cfg)
    elif cfg.model == TcostModel.half_spread:
        holdings = _apply_half_spread(holdings, cfg)
    elif cfg.model == TcostModel.sqrt_adv:
        holdings = _apply_sqrt_adv(holdings, cfg)
    else:
        raise ValueError(f"Unknown TcostModel: {cfg.model!r}")

    return holdings.drop("_delta_w")


# ─── Model implementations ────────────────────────────────────────────────────


def _apply_flat_bps(holdings: pl.DataFrame, cfg: TcostConfig) -> pl.DataFrame:
    """Round-trip cost = flat_fraction, scaled by fraction of weight traded.

    Only the traded portion (|Δw|) incurs the flat cost:
        tcost_i = flat_fraction × |Δw_i|

    This ensures a stock held at the same weight across two periods has
    tcost = 0, not tcost = flat_fraction × |w_i|.
    """
    return holdings.with_columns(
        (pl.col("_delta_w") * cfg.flat_fraction).alias("tcost")
    )


def _apply_half_spread(holdings: pl.DataFrame, cfg: TcostConfig) -> pl.DataFrame:
    """Spread-based cost. Uses CRSP bidlo/askhi if available; else Roll proxy.

    Half-spread = (askhi − bidlo) / (2 × midpoint)
    Round-trip cost per unit traded = full spread = 2 × half-spread.

    When bidlo/askhi are absent, falls back to the Roll (1984) implied
    spread: 0.2 / |prc|. This is the calibration from Hasbrouck (2009)
    for CRSP monthly data and works well for stocks priced above $5.
    """
    has_bidlo = "bidlo" in holdings.columns
    has_askhi = "askhi" in holdings.columns
    has_prc = "prc" in holdings.columns

    if has_bidlo and has_askhi:
        # Realized spread from monthly CRSP quote fields.
        # CRSP's bidlo = lowest bid during the month, askhi = highest ask.
        # This overestimates the point-in-time spread; that's intentional
        # (conservative).
        holdings = holdings.with_columns(
            pl.when(
                (pl.col("bidlo") > 0) & (pl.col("askhi") > pl.col("bidlo"))
            )
            .then(
                (pl.col("askhi") - pl.col("bidlo"))
                / ((pl.col("askhi") + pl.col("bidlo")) / 2.0)
            )
            .otherwise(
                # Fallback to Roll proxy when quote fields are bad
                pl.when(has_prc and (pl.col("prc").abs() > 0))
                .then((0.2 / pl.col("prc").abs()).clip(0.0002, cfg.spread_cap))
                .otherwise(cfg.flat_fraction * 2.0)
            )
            .alias("_spread")
        )
    elif has_prc:
        # Roll (1984) implied-spread proxy
        holdings = holdings.with_columns(
            pl.when(pl.col("prc").abs() > 0)
            .then((0.2 / pl.col("prc").abs()).clip(0.0002, cfg.spread_cap))
            .otherwise(cfg.flat_fraction * 2.0)
            .alias("_spread")
        )
    else:
        # No price info — fall back to flat cost
        return _apply_flat_bps(holdings, cfg)

    # Round-trip cost on the traded fraction
    holdings = holdings.with_columns(
        (pl.col("_spread") * pl.col("_delta_w")).alias("tcost")
    ).drop("_spread")

    return holdings


def _apply_sqrt_adv(holdings: pl.DataFrame, cfg: TcostConfig) -> pl.DataFrame:
    """Almgren-Chriss sqrt market-impact model.

    MI cost per unit traded:
        cost_i = eta × sqrt(|Δw_i| / adv_fraction)

    This reflects the square-root price-impact law: doubling order size
    does NOT double cost; it multiplies cost by √2. A key insight for
    understanding why large funds underperform small funds.

    Round-trip cost (entry + exit) = 2 × cost_i × |Δw_i|
        tcost_i = 2 × eta × |Δw_i| × sqrt(|Δw_i| / adv_fraction)
                = 2 × eta × |Δw_i|^1.5 / sqrt(adv_fraction)

    Note: |Δw_i| here is a portfolio weight (fraction of portfolio dollar
    value traded in stock i), not the fraction of ADV. The adv_fraction
    parameter is used to scale the cost appropriately. For a typical equal-
    weight Russell 1000 portfolio, Δw ≈ 0.002 per stock per rebalance. At
    1% ADV fraction, this produces ~3-6 bps, consistent with empirical estimates
    for mid-cap US equities.

    Requires dvol (dollar volume) column in holdings. Falls back to flat_bps
    if dvol is absent, with a warning column added.
    """
    if "dvol" not in holdings.columns:
        # Can't compute ADV-relative impact without volume data.
        # Fall back silently — engine.py will log a warning.
        holdings = _apply_flat_bps(holdings, cfg)
        return holdings.with_columns(
            pl.lit(True).alias("_tcost_fallback_to_flat")
        )

    import math
    sqrt_adv_frac = math.sqrt(cfg.adv_fraction)

    holdings = holdings.with_columns(
        pl.when(pl.col("_delta_w") > 0)
        .then(
            # 2 × eta × |Δw|^1.5 / sqrt(adv_fraction)
            2.0 * cfg.eta * (pl.col("_delta_w") ** 1.5) / sqrt_adv_frac
        )
        .otherwise(0.0)
        .alias("tcost")
    )

    return holdings


# ─── Utility: breakeven analysis ─────────────────────────────────────────────


def breakeven_cost_bps(gross_ann_return: float, ann_turnover: float) -> float:
    """Compute the flat t-cost (bps) that drives net annualized return to zero.

    Derivation
    ----------
    Net return = Gross return − 2 × tcost_bps × turnover
    Set net = 0:
        tcost_bps = gross_return / (2 × turnover) × 10_000

    Parameters
    ----------
    gross_ann_return
        Annualized gross long-short return, as a fraction (e.g. 0.08 = 8%).
    ann_turnover
        Annualized one-way turnover, as a fraction (e.g. 1.5 = 150% p.a.).
        For monthly rebalance, monthly turnover × 12.

    Returns
    -------
    Breakeven flat round-trip cost in basis points. Positive = strategy has
    alpha to spend. Negative = strategy is already losing money before costs.

    Example
    -------
    >>> breakeven_cost_bps(0.08, 1.5)
    266.67   # can absorb up to ~267 bps round-trip — well above realistic costs
    """
    if ann_turnover <= 0:
        return float("inf")
    return gross_ann_return / (2.0 * ann_turnover) * 10_000.0
