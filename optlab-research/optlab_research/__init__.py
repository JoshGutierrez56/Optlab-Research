"""optlab_research — Equity research workbench and live options lab.

Built on top of the optlab data lake (CRSP / Compustat / IBES / OptionMetrics).

Quick start
-----------
    import optlab_research as olr

    with olr.open_connection() as con:
        universe = olr.universe(con, "2023-12-29")
        bm = olr.compute_signal("book_to_market", "2023-12-29", con, universe=universe)
        print(bm.head())

Member-facing API is in :mod:`optlab_research.workbench.api`.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__", "open_connection", "workbench"]


def open_connection(optlab_root=None, read_only: bool = False):
    """Return a context manager yielding a fully-initialized DuckDB connection.

    Registers all optlab Parquet views and builds the CCM / optcrsp link views.
    This is the recommended way to get a connection in notebooks.

    Parameters
    ----------
    optlab_root : path-like, optional
        Root directory of the optlab package (the folder that contains
        ``config/``, ``data/``, ``db/``). If None, reads the
        ``OPTLAB_ROOT`` environment variable, then falls back to auto-
        discovery via ``import optlab; optlab.__file__``.
    read_only : bool
        Open the DuckDB file in read-only mode (safe for concurrent readers).

    Examples
    --------
    >>> import optlab_research as olr
    >>> with olr.open_connection() as con:
    ...     df = con.execute("SHOW TABLES").pl()
    """
    from optlab_research.db import _ManagedConnection
    return _ManagedConnection(optlab_root=optlab_root, read_only=read_only)


import optlab_research.workbench as workbench  # noqa: E402