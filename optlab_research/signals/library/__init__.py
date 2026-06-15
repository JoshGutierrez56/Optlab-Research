"""Complex multi-step signals that don't fit in a YAML formula.

Each module exposes a ``compute(con, date, spec, universe) -> pl.DataFrame``
function returning ``[permno, signal_value]``.
"""
