"""Self-contained HTML reports for :class:`BacktestResult` objects."""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


@dataclass
class ReportConfig:
    """Display settings for a backtest report."""

    title: str = "Factor Backtest Report"
    subtitle: str = ""
    author: str = "optlab-research"
    institution: str = "Northeastern University - 360 Huntington Fund"
    benchmark_name: str = "Russell 1000"
    include_methodology: bool = True
    include_top_bottom: bool = True
    n_top_bottom: int = 10
    primary_color: str = "#1a365d"
    accent_color: str = "#c7923e"


class BacktestReport:
    """Render a professional, portable HTML report from a backtest result."""

    def __init__(self, result, config: ReportConfig | None = None):
        self.result = result
        self.config = config or ReportConfig()

    @staticmethod
    def _fig_to_b64(fig) -> str:
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def _returns_series(self):
        frame = self.result.returns.to_pandas()
        frame["date"] = frame["date"].astype("datetime64[ns]")
        frame = frame.set_index("date").sort_index()
        column = "net_ls_ret" if "net_ls_ret" in frame.columns else "ls_ret"
        return frame[column].fillna(0.0)

    def _chart_cumulative_returns(self) -> str:
        returns = self._returns_series()
        cumulative = (1.0 + returns).cumprod() - 1.0
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(cumulative.index, cumulative, color=self.config.primary_color, lw=1.8)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.fill_between(cumulative.index, cumulative, 0, alpha=0.1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title("Cumulative Long-Short Returns")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_drawdown(self) -> str:
        cumulative = (1.0 + self._returns_series()).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1.0
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.fill_between(drawdown.index, drawdown, 0, color="firebrick", alpha=0.35)
        ax.plot(drawdown.index, drawdown, color="firebrick", lw=1.0)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title("Drawdown")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_monthly_heatmap(self) -> str:
        returns = self._returns_series()
        monthly = returns.groupby([returns.index.year, returns.index.month]).sum()
        pivot = monthly.unstack(level=1).reindex(columns=range(1, 13))
        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig, ax = plt.subplots(figsize=(14, max(3.5, len(pivot) * 0.45)))
        values = pivot.to_numpy() * 100.0
        image = ax.imshow(values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
        ax.set_xticks(range(12), labels=labels)
        ax.set_yticks(range(len(pivot)), labels=pivot.index)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isfinite(value):
                    ax.text(col, row, f"{value:.1f}%", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, format="%.0f%%", shrink=0.8)
        ax.set_title("Monthly Returns Heatmap (%)")
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _summary_table_html(self) -> str:
        summary = self.result.summary()

        def pct(key: str) -> str:
            value = summary.get(key)
            return "N/A" if value is None else f"{value:.2%}"

        def number(key: str, digits: int = 3) -> str:
            value = summary.get(key)
            return "N/A" if value is None else f"{value:.{digits}f}"

        rows = [
            ("Period", f"{summary.get('start', 'N/A')} to {summary.get('end', 'N/A')}"),
            ("Annual return (L/S)", pct("ann_return_ls")),
            ("Annual volatility (L/S)", pct("ann_vol_ls")),
            ("Sharpe ratio (L/S)", number("sharpe_ls")),
            ("Maximum drawdown (L/S)", pct("max_drawdown_ls")),
            ("Monthly win rate", pct("win_rate_ls")),
            ("Average monthly turnover", pct("avg_monthly_turnover")),
            ("Average N (long / short)", f"{number('avg_n_long', 1)} / {number('avg_n_short', 1)}"),
        ]
        if summary.get("net_ann_return_ls") is not None:
            rows.insert(2, ("Annual return after costs (L/S)", pct("net_ann_return_ls")))
            rows.insert(5, ("Sharpe after costs (L/S)", number("net_sharpe_ls")))
        body = "".join(
            f'<tr><td class="label">{html.escape(label)}</td><td class="value">{html.escape(value)}</td></tr>'
            for label, value in rows
        )
        return f'<table class="summary-table">{body}</table>'

    def _build_html(self) -> str:
        cfg = self.config
        generated = datetime.now().strftime("%B %d, %Y")
        charts = "".join(
            f'<div class="chart-block"><img src="{chart}" alt="Backtest chart"></div>'
            for chart in (
                self._chart_cumulative_returns(),
                self._chart_drawdown(),
                self._chart_monthly_heatmap(),
            )
        )
        methodology = ""
        if cfg.include_methodology:
            methodology = """
            <section><h2>Methodology and limitations</h2><ul>
            <li>Signals must be dated before their associated forward returns.</li>
            <li>Portfolio formation and transaction-cost assumptions come from the saved manifest.</li>
            <li>Results are research evidence, not a forecast or investment recommendation.</li>
            </ul></section>"""
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(cfg.title)}</title><style>
body{{font-family:Georgia,serif;color:#2d3748;max-width:1000px;margin:0 auto;padding:40px 20px}}
.cover{{border-bottom:3px solid {cfg.primary_color};padding-bottom:24px;margin-bottom:32px}}
h1,h2{{color:{cfg.primary_color}}}h1{{margin-bottom:6px}}h2{{font-size:18px;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
.subtitle{{color:#718096}}.meta,.footer{{font-size:12px;color:#718096}}section{{margin-bottom:36px}}
.summary-table{{width:100%;border-collapse:collapse}}.summary-table tr:nth-child(even){{background:#f7fafc}}
.summary-table td{{padding:8px 12px;border-bottom:1px solid #e2e8f0}}.value{{text-align:right;font-weight:600;color:{cfg.primary_color}}}
.chart-block{{margin-bottom:24px;border:1px solid #e2e8f0}}.chart-block img{{display:block;width:100%}}
li{{margin-bottom:8px;line-height:1.5}}.footer{{border-top:1px solid #e2e8f0;padding-top:16px;text-align:center}}
</style></head><body>
<div class="cover"><h1>{html.escape(cfg.title)}</h1><div class="subtitle">{html.escape(cfg.subtitle)}</div>
<div class="meta">{html.escape(cfg.institution)} | Generated {generated} | {html.escape(cfg.author)}</div></div>
<section><h2>Executive summary</h2>{self._summary_table_html()}</section>
<section><h2>Performance charts</h2>{charts}</section>{methodology}
<div class="footer">Generated by optlab-research | {generated}</div></body></html>"""

    def save(self, path: str | Path) -> str:
        """Write the portable report to *path* and return its absolute path."""
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._build_html(), encoding="utf-8")
        return str(target)

    def _repr_html_(self) -> str:
        return self._build_html()
