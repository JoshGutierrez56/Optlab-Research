"""Backtest report generator.

Produces a multi-section HTML report from a BacktestResult object.
Designed to be readable by a portfolio manager without any code knowledge.

Usage:
 from optlab_research.reporting import BacktestReport
 rpt = BacktestReport(result, title="Momentum Factor — Russell 3000")
 rpt.save("outputs/momentum_report.html")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import base64
import io

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


@dataclass
class ReportConfig:
 """Configuration for report generation."""
 title: str = "Factor Backtest Report"
 subtitle: str = ""
 author: str = "optlab-research"
 institution: str = "Northeastern University — 360 Huntington Fund"
 benchmark_name: str = "Russell 1000"
 include_methodology: bool = True
 include_top_bottom: bool = True
 n_top_bottom: int = 10
 primary_color: str = "#1a365d"
 accent_color: str = "#c7923e"


class BacktestReport:
 """Generates a professional HTML report from a BacktestResult.

 The report includes:
 - Cover section with key statistics
 - Cumulative returns chart
 - Drawdown chart
 - Monthly returns heatmap
 - Factor attribution table (if available)
 - Methodology notes
 """

 def __init__(self, result, config: Optional[ReportConfig] = None):
 """
 Args:
 result: BacktestResult object from optlab_research.backtest
 config: ReportConfig with display options
 """
 self.result = result
 self.config = config or ReportConfig()
 self._charts: dict[str, str] = {}

 def _fig_to_b64(self, fig) -> str:
 """Convert matplotlib figure to base64 PNG for embedding in HTML."""
 buf = io.BytesIO()
 fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
 facecolor='white', edgecolor='none')
 buf.seek(0)
 b64 = base64.b64encode(buf.read()).decode('utf-8')
 plt.close(fig)
 return f"data:image/png;base64,{b64}"

 def _chart_cumulative_returns(self) -> str:
 """Generate cumulative returns chart."""
 try:
 returns_df = self.result.returns.to_pandas().set_index('date')
 ls_ret = returns_df.get('ls_ret', returns_df.iloc[:, 0]).fillna(0)
 cum = (1 + ls_ret).cumprod() - 1

 fig, ax = plt.subplots(figsize=(12, 4))
 ax.plot(cum.index, cum * 100, color=self.config.primary_color,
 lw=1.8, label='Long-Short')
 ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
 ax.fill_between(cum.index, cum * 100, 0,
 where=(cum >= 0), alpha=0.1,
 color=self.config.primary_color)
 ax.fill_between(cum.index, cum * 100, 0,
 where=(cum < 0), alpha=0.1, color='red')
 ax.yaxis.set_major_formatter(mtick.PercentFormatter())
 ax.set_title('Cumulative Long-Short Returns', fontsize=12,
 color=self.config.primary_color)
 ax.grid(True, alpha=0.3)
 ax.legend()
 plt.tight_layout()
 return self._fig_to_b64(fig)
 except Exception as e:
 return ""

 def _chart_drawdown(self) -> str:
 """Generate drawdown chart."""
 try:
 returns_df = self.result.returns.to_pandas().set_index('date')
 ls_ret = returns_df.get('ls_ret', returns_df.iloc[:, 0]).fillna(0)
 cum = (1 + ls_ret).cumprod()
 peak = cum.expanding().max()
 dd = (cum - peak) / peak * 100

 fig, ax = plt.subplots(figsize=(12, 3))
 ax.fill_between(dd.index, dd, 0, color='red', alpha=0.4)
 ax.plot(dd.index, dd, color='darkred', lw=1.2)
 ax.yaxis.set_major_formatter(mtick.PercentFormatter())
 ax.set_title('Drawdown', fontsize=12, color=self.config.primary_color)
 ax.grid(True, alpha=0.3)
 plt.tight_layout()
 return self._fig_to_b64(fig)
 except Exception as e:
 return ""

 def _chart_monthly_heatmap(self) -> str:
 """Generate monthly returns heatmap."""
 try:
 import pandas as pd
 returns_df = self.result.returns.to_pandas()
 returns_df['date'] = pd.to_datetime(returns_df['date'])
 returns_df = returns_df.set_index('date')
 ls_ret = returns_df.get('ls_ret', returns_df.iloc[:, 0]).fillna(0)

 pivot = ls_ret.groupby([ls_ret.index.year, ls_ret.index.month]).sum()
 pivot = pivot.unstack(level=1)
 pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
 'Jul','Aug','Sep','Oct','Nov','Dec']

 fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.5)))
 im = ax.imshow(pivot.values * 100, cmap='RdYlGn', aspect='auto',
 vmin=-10, vmax=10)
 ax.set_xticks(range(12))
 ax.set_xticklabels(pivot.columns, fontsize=9)
 ax.set_yticks(range(len(pivot)))
 ax.set_yticklabels(pivot.index, fontsize=9)

 for i in range(len(pivot)):
 for j in range(12):
 val = pivot.values[i, j]
 if not np.isnan(val):
 ax.text(j, i, f'{val*100:.1f}%', ha='center',
 va='center', fontsize=7,
 color='black' if abs(val) < 0.05 else 'white')

 plt.colorbar(im, ax=ax, format='%.0f%%', shrink=0.8)
 ax.set_title('Monthly Returns Heatmap (%)', fontsize=12,
 color=self.config.primary_color)
 plt.tight_layout()
 return self._fig_to_b64(fig)
 except Exception as e:
 return ""

 def _summary_table_html(self) -> str:
 """Build the executive summary statistics table."""
 try:
 s = self.result.summary()
 rows = [
 ("Period", f"{s.get('start_date', 'N/A')} → {s.get('end_date', 'N/A')}"),
 ("Ann. Return (L/S)", f"{s.get('ann_return_ls', 0):.2%}"),
 ("Ann. Volatility (L/S)", f"{s.get('ann_vol_ls', 0):.2%}"),
 ("Sharpe Ratio (L/S)", f"{s.get('sharpe_ls', 0):.3f}"),
 ("Max Drawdown (L/S)", f"{s.get('max_drawdown_ls', 0):.2%}"),
 ("Monthly Win Rate", f"{s.get('win_rate_ls', 0):.1%}"),
 ("Avg Monthly Turnover", f"{s.get('avg_monthly_turnover', 0):.1%}"),
 ("Avg N (Long / Short)", f"{s.get('avg_n_long', 0):.0f} / {s.get('avg_n_short', 0):.0f}"),
 ("Ann. Return — Long Leg", f"{s.get('ann_return_long', 0):.2%}"),
 ("Ann. Return — Short Leg", f"{s.get('ann_return_short', 0):.2%}"),
 ]
 except Exception:
 rows = [("Status", "Summary not available")]

 html = '<table class="summary-table">'
 for label, value in rows:
 html += f'<tr><td class="label">{label}</td><td class="value">{value}</td></tr>'
 html += '</table>'
 return html

 def _build_html(self) -> str:
 """Assemble the full HTML report."""
 cfg = self.config
 generated = datetime.now().strftime("%B %d, %Y")

 # Generate charts
 chart_cum = self._chart_cumulative_returns()
 chart_dd = self._chart_drawdown()
 chart_heatmap = self._chart_monthly_heatmap()
 summary_html = self._summary_table_html()

 chart_section = ""
 if chart_cum:
 chart_section += f'<div class="chart-block"><img src="{chart_cum}" style="width:100%;"/></div>'
 if chart_dd:
 chart_section += f'<div class="chart-block"><img src="{chart_dd}" style="width:100%;"/></div>'
 if chart_heatmap:
 chart_section += f'<div class="chart-block"><img src="{chart_heatmap}" style="width:100%;"/></div>'

 methodology_html = ""
 if cfg.include_methodology:
 methodology_html = """
 <div class="section">
 <h2>Methodology</h2>
 <ul>
 <li><strong>Universe construction:</strong> Point-in-time correct using CRSP monthly security file. Delisting returns applied per Shumway (1997) and Beaver et al. (2007): −30% for performance delistings (DLSTCD 500–584), −55% for other involuntary delistings.</li>
 <li><strong>Signal computation:</strong> All signals computed as of the rebalance date using only data available at that time (no look-ahead bias). Fundamental signals use Compustat with a 90-day reporting lag.</li>
 <li><strong>Portfolio construction:</strong> Equal-weighted quintile long-short. Long = Q5 (highest signal), Short = Q1 (lowest signal). Monthly rebalance.</li>
 <li><strong>Transaction costs:</strong> Not applied unless specified in BacktestConfig.</li>
 <li><strong>Factor attribution:</strong> FF6 (Fama-French 5 factors + UMD momentum). Newey-West HAC standard errors, automatic lag selection per Newey-West (1994).</li>
 <li><strong>Data sources:</strong> CRSP monthly security file (crsp_msf), Compustat annual fundamentals (comp_funda), Fama-French factors (Kenneth French Data Library).</li>
 </ul>
 </div>"""

 return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{cfg.title}</title>
<style>
 * {{ box-sizing: border-box; margin: 0; padding: 0; }}
 body {{ font-family: 'Georgia', serif; color: #2d3748; background: #fff; max-width: 1000px; margin: 0 auto; padding: 40px 20px; }}
 .cover {{ border-bottom: 3px solid {cfg.primary_color}; padding-bottom: 24px; margin-bottom: 32px; }}
 .cover h1 {{ font-size: 28px; color: {cfg.primary_color}; margin-bottom: 6px; }}
 .cover .subtitle {{ font-size: 15px; color: #718096; margin-bottom: 4px; }}
 .cover .meta {{ font-size: 12px; color: #a0aec0; margin-top: 8px; }}
 .section {{ margin-bottom: 36px; }}
 .section h2 {{ font-size: 18px; color: {cfg.primary_color}; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 16px; }}
 .summary-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
 .summary-table tr:nth-child(even) {{ background: #f7fafc; }}
 .summary-table td {{ padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }}
 .summary-table td.label {{ color: #4a5568; font-weight: 500; width: 50%; }}
 .summary-table td.value {{ color: {cfg.primary_color}; font-weight: 600; text-align: right; }}
 .chart-block {{ margin-bottom: 24px; border: 1px solid #e2e8f0; border-radius: 4px; overflow: hidden; }}
 .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
 ul {{ padding-left: 20px; }}
 ul li {{ margin-bottom: 8px; font-size: 13px; line-height: 1.6; color: #4a5568; }}
 .footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid #e2e8f0; font-size: 11px; color: #a0aec0; text-align: center; }}
 @media print {{ body {{ padding: 20px; }} }}
</style>
</head>
<body>

<div class="cover">
 <h1>{cfg.title}</h1>
 <div class="subtitle">{cfg.subtitle}</div>
 <div class="meta">{cfg.institution} &nbsp;|&nbsp; Generated {generated} &nbsp;|&nbsp; {cfg.author}</div>
</div>

<div class="section">
 <h2>Executive Summary</h2>
 {summary_html}
</div>

<div class="section">
 <h2>Performance Charts</h2>
 {chart_section}
</div>

{methodology_html}

<div class="footer">
 Generated by optlab-research · {cfg.institution} · {generated}
</div>

</body>
</html>"""

 def save(self, path: str) -> str:
 """Save the report as an HTML file.

 Args:
 path: Output file path (should end in .html)

 Returns:
 Absolute path of saved file.
 """
 os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
 html = self._build_html()
 with open(path, 'w', encoding='utf-8') as f:
 f.write(html)
 print(f"Report saved: {path}")
 return os.path.abspath(path)

 def _repr_html_(self) -> str:
 """Jupyter display support."""
 return self._build_html()
