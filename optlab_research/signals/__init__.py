"""Signal computation package.

Public API:
    compute_signal(name, date, con, *, universe=None, n_quantiles=5) -> pl.DataFrame
    load_signals(path=None) -> SignalRegistry
"""
from __future__ import annotations

from optlab_research.signals.registry import SignalSpec, SignalRegistry, load_signals
from optlab_research.signals.compute import compute_signal

__all__ = ["SignalSpec", "SignalRegistry", "load_signals", "compute_signal"]
