"""optlab_research.backtest — portfolio construction and backtest execution.

Public API
----------
    from optlab_research.backtest import Backtest, BacktestConfig, BacktestResult

Usage
-----
    cfg = BacktestConfig(
        signal="book_to_market",
        start="2010-01-01",
        end="2024-12-31",
        universe="russell1000",
    )
    result = Backtest(cfg).run(con)
    print(result.summary())
    result.plot_cumulative()
    result.save("outputs/bm_backtest/")
"""
from __future__ import annotations

from optlab_research.backtest.engine import Backtest, BacktestConfig
from optlab_research.backtest.portfolio import PortfolioType, WeightingScheme
from optlab_research.backtest.result import BacktestResult

__all__ = [
    "Backtest",
    "BacktestConfig",
    "BacktestResult",
    "PortfolioType",
    "WeightingScheme",
]
