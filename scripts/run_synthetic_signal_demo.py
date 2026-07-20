"""Generate a deterministic, offline signal-backtest evidence bundle.

This demo proves that the portfolio and reporting layers execute without WRDS.
Its simulated returns are not evidence of market alpha.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from optlab_research.backtest.result import BacktestResult
from optlab_research.reporting.report import BacktestReport, ReportConfig


def synthetic_monthly_returns(
    *, months: int = 144, assets: int = 60, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic month-end dates and simulated asset returns."""
    if months < 24 or assets < 10:
        raise ValueError("months must be >= 24 and assets must be >= 10")
    rng = np.random.default_rng(seed)
    market = rng.normal(0.005, 0.035, size=months)
    innovations = rng.normal(0.0, 0.055, size=(months, assets))
    returns = np.empty_like(innovations)
    returns[0] = market[0] + innovations[0]
    for month in range(1, months):
        # Mild persistence makes the demo informative while keeping it synthetic.
        returns[month] = market[month] + 0.08 * returns[month - 1] + innovations[month]
    dates = np.arange("2013-01", "2025-01", dtype="datetime64[M]")[:months]
    return dates, np.clip(returns, -0.95, 0.95)


def momentum_backtest(
    dates: np.ndarray,
    asset_returns: np.ndarray,
    *,
    lookback: int = 11,
    cost_bps: float = 20.0,
) -> tuple[BacktestResult, list[np.ndarray]]:
    """Run a lagged 12-to-2-style momentum quintile sort."""
    if len(dates) != asset_returns.shape[0]:
        raise ValueError("dates and returns must have matching rows")
    if lookback < 2 or asset_returns.shape[0] <= lookback + 1:
        raise ValueError("insufficient history for the requested lookback")

    rows: list[dict] = []
    turnover_rows: list[dict] = []
    holdings_rows: list[dict] = []
    signals: list[np.ndarray] = []
    previous_weights = np.zeros(asset_returns.shape[1])

    for month in range(lookback + 1, asset_returns.shape[0]):
        # Exclude month t-1 and use only returns known by the signal date.
        history = asset_returns[month - lookback - 1 : month - 1]
        signal = np.prod(1.0 + history, axis=0) - 1.0
        signals.append(signal.copy())
        order = np.argsort(signal)
        leg_size = max(1, asset_returns.shape[1] // 5)
        short_assets = order[:leg_size]
        long_assets = order[-leg_size:]
        weights = np.zeros(asset_returns.shape[1])
        weights[long_assets] = 1.0 / leg_size
        weights[short_assets] = -1.0 / leg_size
        turnover = float(np.abs(weights - previous_weights).sum() / 2.0)
        next_returns = asset_returns[month]
        long_return = float(next_returns[long_assets].mean())
        short_return = float(-next_returns[short_assets].mean())
        gross = long_return + short_return
        net = gross - turnover * cost_bps / 10_000.0
        date = dates[month].astype("datetime64[D]").astype(object)
        rows.append(
            {
                "date": date,
                "long_ret": long_return,
                "short_ret": short_return,
                "ls_ret": gross,
                "net_long_ret": long_return,
                "net_short_ret": short_return,
                "net_ls_ret": net,
            }
        )
        turnover_rows.append({"date": date, "turnover": turnover})
        holdings_rows.append({"date": date, "n_long": leg_size, "n_short": leg_size})
        previous_weights = weights

    result = BacktestResult(
        returns=pl.DataFrame(rows),
        holdings_summary=pl.DataFrame(holdings_rows),
        monthly_turnover=pl.DataFrame(turnover_rows),
        manifest={
            "signal": "synthetic_momentum_12_2",
            "data": "deterministic synthetic monthly returns",
            "lookback_months": lookback,
            "skip_months": 1,
            "portfolio": "equal-weight top/bottom quintile",
            "tcost_model": "flat_one_way_bps",
            "cost_bps": cost_bps,
            "limitations": "Execution evidence only; no historical-market alpha claim.",
        },
    )
    return result, signals


def run_demo(output_dir: str | Path = "results", *, seed: int = 42) -> dict:
    """Run the demo and write JSON, CSV, PNG, and HTML evidence."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dates, returns = synthetic_monthly_returns(seed=seed)
    result, _ = momentum_backtest(dates, returns)
    summary = result.summary()
    evidence = {
        "evidence_type": "deterministic synthetic execution demo",
        "claim_boundary": "Not a historical backtest and not evidence of investable alpha.",
        "seed": seed,
        "summary": summary,
        "manifest": result.manifest,
    }
    (output / "synthetic_signal_summary.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    result.returns.write_csv(output / "synthetic_signal_returns.csv")
    figure = result.plot_cumulative()
    figure.savefig(output / "synthetic_signal_cumulative.png", dpi=150, bbox_inches="tight")
    report = BacktestReport(
        result,
        ReportConfig(
            title="Synthetic Momentum Signal Execution Demo",
            subtitle="Deterministic software evidence - not an empirical performance claim",
        ),
    )
    report.save(output / "synthetic_signal_report.html")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evidence = run_demo(args.output_dir, seed=args.seed)
    print(json.dumps(evidence["summary"], indent=2))


if __name__ == "__main__":
    main()
