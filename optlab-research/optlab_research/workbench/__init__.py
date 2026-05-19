"""Member-facing workbench API.

The workbench is the only surface a club member needs to import. Internals
(signal registry, universe builder, DuckDB connection management) are hidden
behind this module.

Quick start
-----------
    from optlab_research import workbench as wb

    # Build a named universe
    with wb.open() as con:
        univ = wb.universe("liquid_500", "2023-12-29", con=con)
        bm   = wb.signal("book_to_market", "2023-12-29", universe=univ, con=con)

    # Or let the workbench manage the connection internally (one call = one connection)
    bm = wb.signal("book_to_market", "2023-12-29", universe="liquid_500")

See optlab_research.workbench.api for full documentation.
"""
from __future__ import annotations

from optlab_research.workbench.api import open, signal, universe

__all__ = ["open", "signal", "universe"]
