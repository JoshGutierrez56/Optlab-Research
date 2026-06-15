"""BacktestResult: container for backtest output.

Week 5 additions
----------------
* ``.gross_returns`` / ``.net_returns`` — convenience accessors.
* ``.summary()`` includes net_ann_return_ls, net_sharpe_ls, net_max_drawdown_ls,
  breakeven_cost_bps alongside gross equivalents.
* ``.breakeven_cost_bps()`` — standalone method, does NOT call summary().
* ``.plot_cumulative()`` — overlays gross and net L/S when t-costs are present.

Backward compatibility
----------------------
tcost_model="none" (default) → gross and net are identical, no net keys in summary.
v0 saved results (no net columns) load and work correctly.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import polars as pl


@dataclass
class BacktestResult:
    """Container for all output from a single backtest run.

    Attributes
    ----------
    returns
        Monthly portfolio returns DataFrame.
        Gross columns:  [date, long_ret, short_ret, ls_ret]
        Net columns:    [net_long_ret, net_short_ret, net_ls_ret]
        (net columns present only when tcost_model != "none")
    holdings_summary
        Per-period portfolio statistics.
    monthly_turnover
        Per-period one-way turnover.
    manifest
        Full reproducibility record.
    """

    returns: pl.DataFrame
    holdings_summary: pl.DataFrame
    monthly_turnover: pl.DataFrame
    manifest: dict

    # ─── Gross / net accessors ────────────────────────────────────────────────

    @property
    def gross_returns(self) -> pl.DataFrame:
        """Returns DataFrame with gross [date, long_ret, short_ret, ls_ret]."""
        cols = ["date"]
        for c in ("long_ret", "short_ret", "ls_ret"):
            if c in self.returns.columns:
                cols.append(c)
        return self.returns.select(cols)

    @property
    def net_returns(self) -> pl.DataFrame:
        """Returns DataFrame with net [date, long_ret, short_ret, ls_ret].

        Falls back to gross when no t-cost model was applied.
        """
        df = self.returns
        if "net_ls_ret" not in df.columns:
            return self.gross_returns

        rename_map = {
            "net_long_ret":  "long_ret",
            "net_short_ret": "short_ret",
            "net_ls_ret":    "ls_ret",
        }
        cols = ["date"] + [c for c in rename_map if c in df.columns]
        return df.select(cols).rename({k: v for k, v in rename_map.items() if k in df.columns})

    def _has_tcost(self) -> bool:
        return self.manifest.get("tcost_model", "none") not in ("none", None, "")

    # ─── Breakeven cost ───────────────────────────────────────────────────────

    def breakeven_cost_bps(self) -> float:
        """Flat round-trip cost (bps) that drives gross L/S net return to zero.

        Computed directly from returns and turnover — does NOT call summary()
        to avoid circular recursion.
        """
        from optlab_research.backtest.tcost import breakeven_cost_bps as _be

        gross_df = self.gross_returns.drop_nulls(subset=["ls_ret"])
        ls = gross_df["ls_ret"]
        if ls.len() == 0:
            return float("nan")

        gross_ann_ret = float((1.0 + ls).product()) ** (12.0 / ls.len()) - 1.0

        to_df = self.monthly_turnover.filter(
            pl.col("turnover").is_not_null() & pl.col("turnover").is_finite()
        )
        avg_monthly_to = float(to_df["turnover"].mean()) if to_df.height > 0 else 0.0
        ann_turnover = avg_monthly_to * 12.0

        return _be(gross_ann_ret, ann_turnover)

    # ─── Summary statistics ───────────────────────────────────────────────────

    def summary(self) -> dict:
        """Compute standard backtest performance statistics.

        Keys always present
        -------------------
        n_months, start, end,
        ann_return_ls, ann_vol_ls, sharpe_ls, max_drawdown_ls, win_rate_ls,
        ann_return_long, ann_vol_long, sharpe_long,
        ann_return_short, ann_vol_short,
        avg_monthly_turnover, avg_n_long, avg_n_short,
        breakeven_cost_bps

        Keys present only when tcost_model != "none"
        --------------------------------------------
        net_ann_return_ls, net_ann_vol_ls, net_sharpe_ls, net_max_drawdown_ls
        """
        gross_df = self.gross_returns.drop_nulls(subset=["ls_ret"])
        net_df   = self.net_returns.drop_nulls(subset=["ls_ret"])
        n = len(gross_df)

        def _ann_ret(series: pl.Series) -> float:
            s = series.drop_nulls()
            if s.len() == 0:
                return float("nan")
            return float((1.0 + s).product()) ** (12.0 / s.len()) - 1.0

        def _ann_vol(series: pl.Series) -> float:
            s = series.drop_nulls()
            if s.len() < 2:
                return float("nan")
            return float(s.std() * math.sqrt(12.0))

        def _sharpe(series: pl.Series) -> float:
            r = _ann_ret(series)
            v = _ann_vol(series)
            if not math.isfinite(v) or v == 0:
                return float("nan")
            return r / v

        def _max_drawdown(series: pl.Series) -> float:
            s = series.drop_nulls().fill_null(0.0).to_numpy()
            if len(s) == 0:
                return float("nan")
            cum  = np.cumprod(1.0 + s)
            peak = np.maximum.accumulate(cum)
            return float(((cum - peak) / peak).min())

        gross_ls = gross_df["ls_ret"]
        net_ls   = net_df["ls_ret"]
        long_  = gross_df["long_ret"].drop_nulls()  if "long_ret"  in gross_df.columns else pl.Series([], dtype=pl.Float64)
        short_ = gross_df["short_ret"].drop_nulls() if "short_ret" in gross_df.columns else pl.Series([], dtype=pl.Float64)

        to_df  = self.monthly_turnover.filter(
            pl.col("turnover").is_not_null() & pl.col("turnover").is_finite()
        )
        avg_to = float(to_df["turnover"].mean()) if to_df.height > 0 else float("nan")

        hs = self.holdings_summary
        avg_n_long  = float(hs["n_long"].mean())  if "n_long"  in hs.columns and hs.height > 0 else float("nan")
        avg_n_short = float(hs["n_short"].mean()) if "n_short" in hs.columns and hs.height > 0 else float("nan")

        win_rate = float((gross_ls > 0).sum() / n) if n > 0 else float("nan")

        def _fmt(x: float, decimals: int = 4) -> float | None:
            return round(x, decimals) if math.isfinite(x) else None

        # Breakeven — computed inline to avoid circular call with breakeven_cost_bps()
        _gross_ann = _ann_ret(gross_ls)
        _ann_to    = (avg_to * 12.0) if math.isfinite(avg_to) else 0.0
        if _ann_to > 0 and math.isfinite(_gross_ann):
            _be_bps = _gross_ann / (2.0 * _ann_to) * 10_000.0
        else:
            _be_bps = float("nan")

        result = {
            # Metadata
            "n_months": n,
            "start":    str(gross_df["date"].min()),
            "end":      str(gross_df["date"].max()),
            # Gross L/S
            "ann_return_ls":   _fmt(_ann_ret(gross_ls)),
            "ann_vol_ls":      _fmt(_ann_vol(gross_ls)),
            "sharpe_ls":       _fmt(_sharpe(gross_ls), 3),
            "max_drawdown_ls": _fmt(_max_drawdown(gross_ls)),
            "win_rate_ls":     _fmt(win_rate, 3),
            # Gross long leg
            "ann_return_long": _fmt(_ann_ret(long_)),
            "ann_vol_long":    _fmt(_ann_vol(long_)),
            "sharpe_long":     _fmt(_sharpe(long_), 3),
            # Gross short leg
            "ann_return_short": _fmt(_ann_ret(short_)),
            "ann_vol_short":    _fmt(_ann_vol(short_)),
            # Turnover and holdings
            "avg_monthly_turnover": _fmt(avg_to),
            "avg_n_long":           _fmt(avg_n_long, 1),
            "avg_n_short":          _fmt(avg_n_short, 1),
            # Breakeven (always present, computed inline)
            "breakeven_cost_bps":   _fmt(_be_bps, 1),
        }

        # Net statistics — only when a t-cost model was applied
        if self._has_tcost():
            result["net_ann_return_ls"]   = _fmt(_ann_ret(net_ls))
            result["net_ann_vol_ls"]      = _fmt(_ann_vol(net_ls))
            result["net_sharpe_ls"]       = _fmt(_sharpe(net_ls), 3)
            result["net_max_drawdown_ls"] = _fmt(_max_drawdown(net_ls))

        return result

    # ─── Plots ────────────────────────────────────────────────────────────────

    def plot_cumulative(
        self,
        figsize: tuple[int, int] = (12, 5),
        log_scale: bool = False,
        show_net: bool = True,
    ) -> matplotlib.figure.Figure:
        """Plot cumulative returns for long, short, and long-short legs.

        When t-costs are present, overlays a dashed net L/S line.
        """
        gross_df = self.gross_returns.sort("date")
        dates    = gross_df["date"].to_list()

        fig, ax = plt.subplots(figsize=figsize)

        for col, label, color, lw, alpha in [
            ("ls_ret",    "Long-Short (gross)", "steelblue", 2.2, 1.0),
            ("long_ret",  "Long only",          "seagreen",  1.3, 0.85),
            ("short_ret", "Short only",         "firebrick", 1.3, 0.85),
        ]:
            if col not in gross_df.columns:
                continue
            series = gross_df[col].fill_null(0.0).to_numpy()
            cum = np.log1p(series).cumsum() if log_scale else np.cumprod(1.0 + series) - 1.0
            ax.plot(dates, cum, label=label, color=color, linewidth=lw, alpha=alpha)

        if show_net and self._has_tcost():
            net_df = self.net_returns.sort("date")
            if "ls_ret" in net_df.columns:
                net_series = net_df["ls_ret"].fill_null(0.0).to_numpy()
                net_cum = np.log1p(net_series).cumsum() if log_scale else np.cumprod(1.0 + net_series) - 1.0
                tcost_bps   = self.manifest.get("tcost_bps", 0)
                tcost_model = self.manifest.get("tcost_model", "")
                ax.plot(
                    net_df["date"].to_list(), net_cum,
                    label=f"Long-Short (net, {tcost_model} {tcost_bps:.0f}bps)",
                    color="darkorange", linewidth=2.0, linestyle="--", alpha=0.9,
                )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Log Cumulative Return" if log_scale else "Cumulative Return", fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{self.manifest.get('signal','')}  |  {self.manifest.get('universe','')}  |  "
            f"{self.manifest.get('portfolio','')} ({self.manifest.get('weighting','equal')})  |  "
            f"{self.manifest.get('start','')}–{self.manifest.get('end','')}",
            fontsize=11,
        )
        fig.tight_layout()
        return fig

    def plot_drawdown(
        self,
        figsize: tuple[int, int] = (12, 3),
        show_net: bool = True,
    ) -> matplotlib.figure.Figure:
        """Plot drawdown of the L/S cumulative return series."""
        gross_df  = self.gross_returns.sort("date").drop_nulls(subset=["ls_ret"])
        dates     = gross_df["date"].to_list()
        gross_rets = gross_df["ls_ret"].to_numpy()
        gross_cum  = np.cumprod(1.0 + gross_rets)
        gross_peak = np.maximum.accumulate(gross_cum)
        gross_dd   = (gross_cum - gross_peak) / gross_peak

        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(dates, gross_dd, 0.0, color="steelblue", alpha=0.25, label="Gross DD")
        ax.plot(dates, gross_dd, color="steelblue", linewidth=0.9)

        if show_net and self._has_tcost():
            net_df   = self.net_returns.sort("date").drop_nulls(subset=["ls_ret"])
            net_rets = net_df["ls_ret"].to_numpy()
            net_cum  = np.cumprod(1.0 + net_rets)
            net_peak = np.maximum.accumulate(net_cum)
            net_dd   = (net_cum - net_peak) / net_peak
            ax.fill_between(net_df["date"].to_list(), net_dd, 0.0, color="darkorange", alpha=0.25, label="Net DD")
            ax.plot(net_df["date"].to_list(), net_dd, color="darkorange", linewidth=0.9, linestyle="--")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Drawdown", fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
        max_dd = float(gross_dd.min()) if len(gross_dd) > 0 else float("nan")
        ax.set_title(
            f"{self.manifest.get('signal','')} — Drawdown  (max gross = {max_dd:.1%})",
            fontsize=11,
        )
        fig.tight_layout()
        return fig

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save all result data to *path* (directory).

        Writes: returns.parquet, holdings_summary.parquet,
        monthly_turnover.parquet, manifest.json.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.returns.write_parquet(out / "returns.parquet")
        self.holdings_summary.write_parquet(out / "holdings_summary.parquet")
        self.monthly_turnover.write_parquet(out / "monthly_turnover.parquet")
        with (out / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, default=str)
