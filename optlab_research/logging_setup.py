"""Logging setup for optlab_research.

Mirrors optlab.logging_setup so the two packages emit consistent log lines.
"""
from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
