"""Signal registry: schema, validation, and loader.

The registry is a YAML file (config/signals.yaml). This module provides:
  - SignalKind   — enum of computation strategies
  - SignalSpec   — Pydantic model for one signal definition
  - SignalRegistry — container with .get() and .names() helpers
  - load_signals()  — load + validate from YAML

Design notes
------------
* ConfigDict(extra="forbid") on every model: unknown YAML keys fail loudly
  rather than silently being ignored. This catches typos.
* SignalSpec stores all kind-specific parameters (lookback_months, etc.) at
  the top level with Optional typing. The alternative — a discriminated union —
  is cleaner in theory but adds friction when writing YAML. The model_validator
  ensures each kind has exactly the fields it needs.
* The formula field contains a Python expression string evaluated by compute.py
  via eval(formula, {"pl": polars}). This is safe only because signals.yaml is
  a developer-controlled file, not user input. See compute.py for details.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ─── Enums ────────────────────────────────────────────────────────────────────


class SignalKind(str, Enum):
    funda = "funda"
    """Polars expression over Compustat columns already attached to the universe."""

    crsp_price = "crsp_price"
    """Polars expression over CRSP price / market-cap columns in the universe."""

    library = "library"
    """Python callable for signals too complex for a single expression."""


# ─── Signal spec ──────────────────────────────────────────────────────────────


class SignalSpec(BaseModel):
    """Schema for one signal entry in config/signals.yaml."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    kind: SignalKind

    # ── For funda / crsp_price kinds ─────────────────────────────────────────
    # Python expression evaluated as eval(formula, {"pl": polars}).
    # Must produce a pl.Expr. Example: "pl.col('ceq') / pl.col('mcap_musd')"
    formula: str | None = None

    # ── For library kind ─────────────────────────────────────────────────────
    # Fully-qualified dotted path to a callable.
    # Signature: (con, date, spec, universe) -> pl.DataFrame[permno, signal_value]
    library_fn: str | None = None

    # Universe columns that must be present before computation.
    # Missing columns raise ValueError at runtime, not at import time.
    required_columns: list[str] = Field(default_factory=list)

    # Extra PIT buffer added on top of the rdq / datadate gate.
    # 0 = use COALESCE(rdq, datadate + 90 days) as-is; rarely needs changing.
    pit_lag_days: int = 0

    # Documentation fields
    source_table: str | None = None
    notes: str | None = None

    # ── Library-signal parameters ─────────────────────────────────────────────
    # Library functions read these directly from spec rather than accepting
    # them as function arguments, which keeps the callable signature stable.
    lookback_months: int | None = None
    skip_months: int | None = None
    lookback_days: int | None = None
    min_obs: int | None = None

    # ── Validation ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> "SignalSpec":
        if self.kind in (SignalKind.funda, SignalKind.crsp_price):
            if not self.formula:
                raise ValueError(
                    f"Signal '{self.name}': kind='{self.kind}' requires a `formula` field."
                )
        elif self.kind == SignalKind.library:
            if not self.library_fn:
                raise ValueError(
                    f"Signal '{self.name}': kind='library' requires a `library_fn` field."
                )
        return self


# ─── Registry container ───────────────────────────────────────────────────────


class SignalRegistry(BaseModel):
    """Container for the full set of registered signals."""

    model_config = ConfigDict(extra="forbid")

    signals: list[SignalSpec]

    @model_validator(mode="after")
    def _validate_unique_names(self) -> "SignalRegistry":
        names = [s.name for s in self.signals]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate signal names in registry: {sorted(dupes)}")
        return self

    def get(self, name: str) -> SignalSpec:
        """Return the spec for signal *name*. Raises KeyError if not found."""
        for s in self.signals:
            if s.name == name:
                return s
        raise KeyError(
            f"Unknown signal {name!r}. "
            f"Registered signals: {self.names()}"
        )

    def names(self) -> list[str]:
        """Return all registered signal names."""
        return [s.name for s in self.signals]

    def __len__(self) -> int:
        return len(self.signals)

    def __iter__(self):
        return iter(self.signals)


# ─── Loader ───────────────────────────────────────────────────────────────────

# Default path: <repo_root>/config/signals.yaml, resolved relative to this file.
# This file lives at <repo_root>/optlab_research/signals/registry.py,
# so parent.parent.parent is the repo root.
_DEFAULT_SIGNALS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "signals.yaml"


def load_signals(path: Path | str | None = None) -> SignalRegistry:
    """Load and validate the signal registry from YAML.

    Parameters
    ----------
    path : Path or str, optional
        Path to signals.yaml. Defaults to config/signals.yaml in the repo root.

    Returns
    -------
    SignalRegistry
        Validated registry. Raises on schema violations or duplicate names.
    """
    p = Path(path) if path is not None else _DEFAULT_SIGNALS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Signal registry not found at {p}. "
            "Ensure config/signals.yaml exists and OPTLAB_ROOT is set if needed."
        )
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return SignalRegistry.model_validate(raw)
