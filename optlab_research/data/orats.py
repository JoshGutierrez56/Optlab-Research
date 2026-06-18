"""Minimal ORATS options data loader.

This module keeps ORATS credentials out of notebooks, logs, configs, and source
files. The token is read only from the ORATS_TOKEN environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

import pandas as pd


class OratsConfigurationError(RuntimeError):
    """Raised when ORATS is requested but local credentials are not configured."""


def get_orats_token() -> str:
    """Return the ORATS token from the environment.

    Raises:
        OratsConfigurationError: If ORATS_TOKEN is not set or is empty.
    """

    token = os.environ.get("ORATS_TOKEN", "").strip()
    if not token:
        raise OratsConfigurationError(
            "ORATS_TOKEN is not set. Set ORATS_TOKEN in the environment before "
            "requesting ORATS options data."
        )
    return token


@dataclass(frozen=True)
class OratsOptionsRequest:
    """Parameters for a small ORATS options data request."""

    ticker: str
    start_date: date | str
    end_date: date | str | None = None
    fields: tuple[str, ...] | None = None


class OratsOptionsLoader:
    """Load options data from ORATS using an environment-provided token.

    The public API is intentionally stable before the endpoint adapter is
    completed. Callers can depend on this interface while the request/response
    mapping remains isolated to this module.
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token or get_orats_token()

    def load_options(
        self,
        ticker: str,
        start_date: date | str,
        end_date: date | str | None = None,
        fields: tuple[str, ...] | None = None,
        **params: Any,
    ) -> pd.DataFrame:
        """Return ORATS options data for one ticker as a DataFrame.

        Network-specific endpoint wiring is intentionally isolated here so the
        workbench and backtest layers do not need to know about credentials.

        Raises:
            NotImplementedError: Until the project selects the exact ORATS
                endpoint and response schema to support.
        """

        request = OratsOptionsRequest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )
        return self._load_request(request, params)

    def _load_request(
        self,
        request: OratsOptionsRequest,
        params: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Load a prepared ORATS request.

        The token is deliberately not included in exception messages or printed
        output. Keep endpoint URLs out of logs and member-facing notebooks.
        """

        _ = (request, params, self._token)
        raise NotImplementedError(
            "ORATS options data loading is available through this placeholder "
            "API, but the endpoint adapter has not been implemented yet. Add "
            "the request/response mapping inside optlab_research.data.orats "
            "without logging credentials or remote URLs."
        )


def load_orats_options(
    ticker: str,
    start_date: date | str,
    end_date: date | str | None = None,
    fields: tuple[str, ...] | None = None,
    **params: Any,
) -> pd.DataFrame:
    """Convenience wrapper for loading ORATS options data."""

    return OratsOptionsLoader().load_options(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        **params,
    )
