"""IBKR options chain snapshotting via ib_async.

Connects to Interactive Brokers TWS or IB Gateway (paper or live),
fetches the full option chain for a list of underlyings, and returns
structured DataFrames ready for Parquet storage.

Requires:
 pip install ib_async

IBKR paper trading must be running with API enabled:
 TWS -> Edit -> Global Configuration -> API -> Settings
 Check "Enable ActiveX and Socket Clients"
 Port: 7497 (paper), 7496 (live)
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import time

import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SnapshotConfig:
    """Configuration for a single snapshot run."""
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 = paper, 7496 = live
    client_id: int = 1
    timeout: int = 30
    max_expirations: int = 6
    max_strikes_atm: int = 20
    rate_limit_seconds: float = 0.5
    retry_attempts: int = 3
    retry_delay_seconds: float = 5.0
    fields: List[str] = field(default_factory=lambda: [
        "BID", "ASK", "LAST", "VOLUME", "OPEN_INT", "MODEL_OPTION"
    ])


@dataclass
class OptionQuote:
    """A single option contract quote."""
    underlying: str
    expiration: str  # YYYYMMDD
    strike: float
    right: str  # 'C' or 'P'
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    volume: Optional[int]
    open_interest: Optional[int]
    implied_vol: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    underlying_price: Optional[float]
    snapshot_ts: str  # ISO timestamp


class IBKRSnapshot:
    """Fetches full option chains from IBKR.

    Usage:
        config = SnapshotConfig(port=7497)  # paper trading
        snap = IBKRSnapshot(config)
        quotes = snap.fetch("SPY")
        df = snap.to_polars(quotes)
    """

    def __init__(self, config: SnapshotConfig):
        self.config = config
        self._ib = None

    def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway.

        Returns True if connection successful.
        Raises ImportError if ib_async not installed.
        Raises ConnectionError if TWS not running.
        """
        try:
            from ib_async import IB
        except ImportError:
            raise ImportError(
                "ib_async not installed. Run: pip install ib_async\n"
                "Also ensure TWS or IB Gateway is running with API enabled."
            )

        self._ib = IB()
        try:
            self._ib.connect(
                self.config.host,
                self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=True,
            )
            logger.info(f"Connected to IBKR at {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to IBKR at {self.config.host}:{self.config.port}.\n"
                f"Is TWS/Gateway running with API enabled? Error: {e}"
            )

    def disconnect(self):
        """Disconnect from IBKR."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR")

    def _get_underlying_price(self, symbol: str) -> Optional[float]:
        """Get current price of the underlying."""
        try:
            from ib_async import Stock
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            ticker = self._ib.reqMktData(contract, "", True, False)
            self._ib.sleep(1)
            price = ticker.last or ticker.close or ticker.marketPrice()
            self._ib.cancelMktData(contract)
            return float(price) if price and price == price else None  # NaN check
        except Exception as e:
            logger.warning(f"Could not get price for {symbol}: {e}")
            return None

    def _get_expirations(self, symbol: str) -> List[str]:
        """Get available option expiration dates."""
        try:
            from ib_async import Stock, Option
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            chains = self._ib.reqSecDefOptParams(
                contract.symbol, "", contract.secType, contract.conId
            )
            if not chains:
                return []
            # Take the chain with most strikes (usually SMART)
            chain = max(chains, key=lambda c: len(c.strikes))
            expirations = sorted(chain.expirations)
            return expirations[:self.config.max_expirations]
        except Exception as e:
            logger.error(f"Could not get expirations for {symbol}: {e}")
            return []

    def _get_strikes_for_expiry(
        self, symbol: str, expiry: str, underlying_price: float
    ) -> List[float]:
        """Get strikes near ATM for a given expiry."""
        try:
            from ib_async import Stock
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            chains = self._ib.reqSecDefOptParams(
                contract.symbol, "", contract.secType, contract.conId
            )
            if not chains:
                return []
            chain = max(chains, key=lambda c: len(c.strikes))
            all_strikes = sorted(chain.strikes)
            # Filter to ATM +/- n strikes
            atm_idx = min(
                range(len(all_strikes)),
                key=lambda i: abs(all_strikes[i] - underlying_price)
            )
            n = self.config.max_strikes_atm
            return all_strikes[max(0, atm_idx - n): atm_idx + n + 1]
        except Exception as e:
            logger.warning(f"Could not get strikes for {symbol} {expiry}: {e}")
            return []

    def _fetch_contract_quote(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        underlying_price: Optional[float],
        snap_ts: str,
    ) -> Optional[OptionQuote]:
        """Fetch quote for a single option contract."""
        try:
            from ib_async import Option
            contract = Option(symbol, expiry, strike, right, "SMART")
            qualified = self._ib.qualifyContracts(contract)
            if not qualified:
                return None

            ticker = self._ib.reqMktData(contract, "106", True, False)
            self._ib.sleep(self.config.rate_limit_seconds)

            greeks = ticker.modelGreeks or ticker.lastGreeks

            quote = OptionQuote(
                underlying=symbol,
                expiration=expiry,
                strike=strike,
                right=right,
                bid=ticker.bid if ticker.bid == ticker.bid else None,
                ask=ticker.ask if ticker.ask == ticker.ask else None,
                last=ticker.last if ticker.last == ticker.last else None,
                volume=int(ticker.volume) if ticker.volume == ticker.volume else None,
                open_interest=int(ticker.callOpenInterest or ticker.putOpenInterest or 0) or None,
                implied_vol=float(greeks.impliedVol) if greeks and greeks.impliedVol == greeks.impliedVol else None,
                delta=float(greeks.delta) if greeks and greeks.delta == greeks.delta else None,
                gamma=float(greeks.gamma) if greeks and greeks.gamma == greeks.gamma else None,
                theta=float(greeks.theta) if greeks and greeks.theta == greeks.theta else None,
                vega=float(greeks.vega) if greeks and greeks.vega == greeks.vega else None,
                underlying_price=underlying_price,
                snapshot_ts=snap_ts,
            )
            self._ib.cancelMktData(contract)
            return quote
        except Exception as e:
            logger.debug(f"Failed {symbol} {expiry} {strike} {right}: {e}")
            return None

    def fetch(self, symbol: str) -> List[OptionQuote]:
        """Fetch full option chain for one underlying.

        Args:
            symbol: Ticker symbol (e.g. 'SPY')

        Returns:
            List of OptionQuote objects (may be partial if some contracts fail)
        """
        if not self._ib or not self._ib.isConnected():
            raise RuntimeError("Not connected. Call connect() first.")

        snap_ts = datetime.utcnow().isoformat()
        logger.info(f"Fetching chain for {symbol}")

        underlying_price = self._get_underlying_price(symbol)
        if underlying_price is None:
            logger.warning(f"No underlying price for {symbol}, using strikes without ATM filter")
            underlying_price = 0.0

        expirations = self._get_expirations(symbol)
        if not expirations:
            logger.error(f"No expirations found for {symbol}")
            return []

        quotes = []
        for expiry in expirations:
            strikes = self._get_strikes_for_expiry(symbol, expiry, underlying_price)
            for strike in strikes:
                for right in ["C", "P"]:
                    for attempt in range(self.config.retry_attempts):
                        quote = self._fetch_contract_quote(
                            symbol, expiry, strike, right,
                            underlying_price, snap_ts
                        )
                        if quote is not None:
                            quotes.append(quote)
                            break
                        if attempt < self.config.retry_attempts - 1:
                            time.sleep(self.config.retry_delay_seconds)

            logger.info(f" {symbol} {expiry}: {len([q for q in quotes if q.expiration == expiry])} contracts")

        logger.info(f"Fetched {len(quotes)} quotes for {symbol}")
        return quotes

    def fetch_all(self, symbols: List[str]) -> Dict[str, List[OptionQuote]]:
        """Fetch option chains for multiple underlyings.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol -> list of OptionQuote
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch(symbol)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = []
        return results

    @staticmethod
    def to_polars(quotes: List[OptionQuote]) -> pl.DataFrame:
        """Convert list of OptionQuote to Polars DataFrame."""
        if not quotes:
            return pl.DataFrame()

        records = [
            {
                "underlying": q.underlying,
                "expiration": q.expiration,
                "strike": q.strike,
                "right": q.right,
                "bid": q.bid,
                "ask": q.ask,
                "last": q.last,
                "volume": q.volume,
                "open_interest": q.open_interest,
                "implied_vol": q.implied_vol,
                "delta": q.delta,
                "gamma": q.gamma,
                "theta": q.theta,
                "vega": q.vega,
                "underlying_price": q.underlying_price,
                "snapshot_ts": q.snapshot_ts,
                "mid": (q.bid + q.ask) / 2 if q.bid and q.ask else None,
                "spread_pct": (q.ask - q.bid) / q.mid if q.bid and q.ask and (q.bid + q.ask) > 0 else None,
            }
            for q in quotes
        ]
        return pl.DataFrame(records)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
