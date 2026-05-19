"""DuckDB connection management for optlab_research.

optlab owns the data lake and the DuckDB file. This module provides a
convenience wrapper that:
  1. Resolves the optlab root (from argument → env var → auto-discovery).
  2. Opens the DuckDB connection via optlab.db.connect().
  3. Registers all Parquet views (optlab.db.register_all_views).
  4. Builds the CCM and optcrsp link views (optlab.links.build_link_views).

You should use open_connection() (exposed from optlab_research.__init__)
rather than calling optlab.db.connect() directly — it ensures the views
exist before any signal code runs.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

from optlab_research.logging_setup import get_logger

log = get_logger(__name__)


def _resolve_optlab_root(optlab_root=None) -> Path:
    """Determine the optlab project root.

    Resolution order:
      1. Explicit ``optlab_root`` argument.
      2. ``OPTLAB_ROOT`` environment variable.
      3. Auto-discovery: import optlab and walk up from optlab.__file__.
    """
    if optlab_root is not None:
        return Path(optlab_root)

    env = os.environ.get("OPTLAB_ROOT")
    if env:
        return Path(env)

    # Auto-discover: optlab package lives at <root>/optlab/__init__.py
    try:
        import optlab as _optlab  # noqa: F401 (import for side-effect of locating it)
        pkg_file = Path(_optlab.__file__).resolve()
        # __file__ is <root>/optlab/__init__.py → parent is <root>/optlab → parent.parent is <root>
        root = pkg_file.parent.parent
        log.debug("auto-discovered optlab root: %s", root)
        return root
    except ImportError as exc:
        raise ImportError(
            "optlab is not installed and OPTLAB_ROOT is not set. "
            "Either install optlab (pip install -e <path>) or set the "
            "OPTLAB_ROOT environment variable to the optlab project root."
        ) from exc


class _ManagedConnection:
    """Context manager returned by open_connection().

    Not intended to be instantiated directly — use open_connection().
    """

    def __init__(self, optlab_root=None, read_only: bool = False):
        self._root = _resolve_optlab_root(optlab_root)
        self._read_only = read_only
        self._con: duckdb.DuckDBPyConnection | None = None

    def __enter__(self) -> duckdb.DuckDBPyConnection:
        from optlab.config import load_registry
        from optlab.db import connect, register_all_views
        from optlab.links import build_link_views

        # Load the optlab registry from the resolved root.
        registry_path = self._root / "config" / "tables.yaml"
        registry = load_registry(registry_path)

        # Open the DuckDB file from the resolved root.
        db_path = self._root / "db" / "research.duckdb"

        # connect() is a context manager; we enter it manually so we can
        # keep the connection open for the lifetime of _ManagedConnection.
        self._optlab_ctx = connect(db_path=db_path, read_only=self._read_only)
        self._con = self._optlab_ctx.__enter__()

        # Register all Parquet-backed table views.
        data_dir = self._root / "data"
        n = register_all_views(self._con, registry, data_dir=data_dir)
        log.info("registered %d optlab views", n)

        # Build the CCM and optcrsp identifier link views.
        built = build_link_views(self._con)
        log.info("built link views: %s", built)

        return self._con

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._optlab_ctx is not None:
            self._optlab_ctx.__exit__(exc_type, exc_val, exc_tb)
        return False  # do not suppress exceptions
