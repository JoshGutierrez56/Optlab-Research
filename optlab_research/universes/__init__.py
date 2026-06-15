"""Universe construction package.

Public API:
    UniverseSpec      — Pydantic model for one universe definition
    UniverseRegistry  — container with .get() and .names()
    load_universes()  — load + validate from config/universes.yaml
    get_universe()    — build a named universe as a Polars DataFrame
"""
from __future__ import annotations

from optlab_research.universes.builder import (
    UniverseSpec,
    UniverseRegistry,
    load_universes,
    get_universe,
)

__all__ = ["UniverseSpec", "UniverseRegistry", "load_universes", "get_universe"]
