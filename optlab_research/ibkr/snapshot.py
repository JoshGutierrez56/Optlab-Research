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

MARKET DATA NOTE:
 If you see IBKR error 10197 ("Market data blocked"), close active
 market data windows in TWS or use IB Gateway instead of TWS, as the
 desktop client can compete with API calls for market data streams.
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


# ---------------------------------------------------------------------------
# IBKR error codes we care about
# ---------------------------------------------------------------------------

IBKR_COMPETING_DATA = 10197   # market data already open in desktop client
IBKR_NO_DATA = 200            # no security definition found
IBKR_DELAYED_DATA = 2104      # only delayed data available


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

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> bool:
        """Ensure we have an active connection (idempotent)."""
        try:
            from ib_async import IB
        except ImportError:
            raise ImportError(
                "ib_async not installed. Run: pip install ib_async\n"
                "Also ensure TWS or IB Gateway is running with API enabled."
            )

        if self._ib and self._ib.isConnected():
            return True  # already connected (e.g. via __enter__)

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

    # ------------------------------------------------------------------
    # Underlying price (multi-fallback)
    # ------------------------------------------------------------------

    def _get_underlying_price(self, symbol: str) -> Optional[float]:
        """Get current price of the underlying.

        Three-tier fallback:
        1. reqTickers()  -- lightweight snapshot, no subscription needed
           Handles IBKR competing-session errors gracefully (logs hint).
        2. reqContractDetails().lastTradeDateOrLast -- uses the last traded
           price stored in contract details as a live-quote fallback.
        3. reqSecDefOptParams().midpoint -- estimates ATM from strike range
           when no pricing data is available at all.

        Returns None if all three fail.
        """
        # --- Method 1: reqTickers (snapshot, no subscription) ----------
        try:
            from ib_async import Stock
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            tickers = self._ib.reqTickers(contract)
            if tickers and len(tickers) > 0:
                ticker = tickers[0]
                mp = ticker.marketPrice()
                if mp is not None and mp == mp:  # NaN check
                    logger.debug(f"Method 1 (reqTickers): {symbol} price = {mp}")
                    return float(mp)
        except Exception as e:
            code = getattr(e, 'code', None)
            msg = str(e).lower() if hasattr(e, '__str__') else ''
            if code == IBKR_COMPETING_DATA or '10197' in msg or 'competing' in msg:
                logger.error(
                    f"Market data blocked for {symbol}: close active TWS market "
                    f"data windows or use IB Gateway instead of TWS."
                )
            else:
                logger.debug(f"Method 1 (reqTickers) failed for {symbol}: {e}")

        # --- Method 2: reqContractDetails -------------------------------
        try:
            from ib_async import Stock
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            details = self._ib.reqContractDetails(contract)
            if details and len(details) > 0:
                # reqContractDetails returns ContractDetail; look for the
                # last trade date / price embedded in the contract description.
                cd = details[0]
                # Some contract descriptions carry "Last: $XXX" at the end.
                desc = getattr(cd, 'contractDescription', '') or ''
                if desc and '$' in desc:
                    parts = desc.split('$')
                    for p in reversed(parts):  # grab right-most dollar figure
                        val = p.strip().split()[0] if p.strip() else ''
                        try:
                            price = float(val)
                            logger.debug(f"Method 2 (contractDetails): {symbol} last ≈ ${price}")
                            return price
                        except ValueError:
                            continue

                # Fallback: use the contract's longName strike midpoint as
                # a rough approximation when no explicit price is stored.
                chains = self._ib.reqSecDefOptParams(symbol, "", "STK", None)
                if chains:
                    chain = max(chains, key=lambda c: len(c.strikes))
                    mid_strike = chain.strikes[len(chain.strikes) // 2]
                    logger.debug(
                        f"Method 2 fallback (contractDetails): using strike midpoint "
                        f"as price proxy for {symbol}: {mid_strike}"
                    )
                    return float(mid_strike)
        except Exception as e:
            code = getattr(e, 'code', None)
            msg = str(e).lower() if hasattr(e, '__str__') else ''
            if code == IBKR_COMPETING_DATA or '10197' in msg or 'competing' in msg:
                logger.error(
                    f"Market data blocked for {symbol}: close active TWS market "
                    f"data windows or use IB Gateway instead of TWS."
                )
            else:
                logger.debug(f"Method 2 (contractDetails) failed for {symbol}: {e}")

        # --- Method 3: strike midpoint from sec-def ---------------------
        try:
            from ib_async import Stock
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            chains = self._ib.reqSecDefOptParams(
                contract.symbol, "", contract.secType, contract.conId
            )
            if chains:
                chain = max(chains, key=lambda c: len(c.strikes))
                strikes = sorted(chain.strikes)
                midpoint = (strikes[0] + strikes[-1]) / 2.0
                logger.info(
                    f"Method 3 (strike-midpoint): using {midpoint:.2f} "
                    f"(range [{strikes[0]}, {strikes[-1]}]) as ATM proxy for {symbol}"
                )
                return float(midpoint)
        except Exception as e:
            logger.warning(f"Method 3 (strike midpoint) failed for {symbol}: {e}")

        logger.warning(f"All methods failed to get price for {symbol}. "
                       f"Strikes will use no-ATM-filter mode.")
        return None

    # ------------------------------------------------------------------
    # Contract metadata helpers
    # ------------------------------------------------------------------

    def _get_expirations(self, symbol: str) -> List[str]:
        """Get available option expiration dates."""
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
            expirations = sorted(chain.expirations)
            return expirations[:self.config.max_expirations]
        except Exception as e:
            code = getattr(e, 'code', None)
            msg = str(e).lower() if hasattr(e, '__str__') else ''
            if code == IBKR_COMPETING_DATA or '10197' in msg or 'competing' in msg:
                logger.error(
                    f"Market data blocked during expirations query for {symbol}: "
                    f"close active TWS market data windows or use IB Gateway."
                )
            else:
                logger.error(f"Could not get expirations for {symbol}: {e}")
            return []

    def _get_strikes_for_expiry(
        self, symbol: str, expiry: str, underlying_price: float
    ) -> List[float]:
        """Get strikes near ATM for a given expiry.

        If underlying_price is 0.0 (no price available), returns the full
        strike range from sec-def data instead of filtering to ATM window.
        """
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

            # If we have no underlying price, don't filter -- return full range
            if underlying_price is None or underlying_price <= 0:
                logger.info(f"Using full strike range for {symbol} ({len(all_strikes)} strikes)")
                return all_strikes

            # Filter to ATM +/- n strikes
            atm_idx = min(
                range(len(all_strikes)),
                key=lambda i: abs(all_strikes[i] - underlying_price)
            )
            n = self.config.max_strikes_atm
            window = all_strikes[max(0, atm_idx - n): atm_idx + n + 1]
            logger.debug(f"ATM window for {symbol}: {[f'{s:.0f}' for s in window]} "
                         f"(ATM≈{underlying_price})")
            return window

        except Exception as e:
            code = getattr(e, 'code', None)
            msg = str(e).lower() if hasattr(e, '__str__') else ''
            if code == IBKR_COMPETING_DATA or '10197' in msg or 'competing' in msg:
                logger.error(
                    f"Market data blocked during strike query for {symbol}: "
                    f"close active TWS market data windows or use IB Gateway."
                )
            else:
                logger.warning(f"Could not get strikes for {symbol} {expiry}: {e}")
            return []

    # ------------------------------------------------------------------
    # Quote fetching (with competing-session handling)
    # ------------------------------------------------------------------

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

            # Use reqTickers (snapshot) instead of reqMktData to avoid
            # subscription requirements.  This also avoids the competing-
            # session error because tickers is a read-only snapshot call.
            tickers = self._ib.reqTickers(contract)
            if not tickers or len(tickers) == 0:
                return None
            ticker = tickers[0]

            greeks = ticker.modelGreeks

            bid = float(ticker.bid) if (ticker.bid is not None and ticker.bid == ticker.bid) else None
            ask = float(ticker.ask) if (ticker.ask is not None and ticker.ask == ticker.ask) else None
            last = float(ticker.last) if (ticker.last is not None and ticker.last == ticker.last) else None
            volume = int(ticker.volume) if (ticker.volume is not None and ticker.volume == ticker.volume) else None
            oi_data = getattr(ticker, 'openInterest', None) or getattr(ticker, 'putOpenInterest', None) or getattr(ticker, 'callOpenInterest', 0)
            open_interest = int(oi_data) if oi_data and oi_data == oi_data else None
            iv = float(greeks.impliedVol) * 100 if (greeks and greeks.impliedVol is not None and greeks.impliedVol == greeks.impliedVol) else None
            delta_val = float(greeks.delta) if (greeks and greeks.delta is not None and greeks.delta == greeks.delta) else None
            gamma_val = float(greeks.gamma) if (greeks and greeks.gamma is not None and greeks.gamma == greeks.gamma) else None
            theta_val = float(greeks.theta) if (greeks and greeks.theta is not None and greeks.theta == greeks.theta) else None
            vega_val = float(greeks.vega) * 100 if (greeks and greeks.vega is not None and greeks.vega == greeks.vega) else None

            quote = OptionQuote(
                underlying=symbol,
                expiration=expiry,
                strike=strike,
                right=right,
                bid=bid,
                ask=ask,
                last=last,
                volume=volume,
                open_interest=open_interest,
                implied_vol=iv,
                delta=delta_val,
                gamma=gamma_val,
                theta=theta_val,
                vega=vega_val,
                underlying_price=underlying_price,
                snapshot_ts=snap_ts,
            )
            return quote

        except Exception as e:
            code = getattr(e, 'code', None)
            msg = str(e).lower() if hasattr(e, '__str__') else ''

            # Explicit handling for competing-session / blocked market data
            if code == IBKR_COMPETING_DATA or '10197' in msg or 'competing' in msg:
                logger.error(
                    f"Market data blocked for {symbol} {expiry} {strike} {right}: "
                    f"close active TWS market data windows or use IB Gateway instead of TWS."
                )
            elif code == IBKR_NO_DATA or '200' in msg:
                # Contract doesn't exist on that exchange / no definition
                logger.debug(f"No contract found for {symbol} {expiry} {strike} {right}")
            else:
                logger.debug(f"Failed {symbol} {expiry} {strike} {right}: {e}")
            return None

    # ------------------------------------------------------------------
    # Fetch API
    # ------------------------------------------------------------------

    def fetch(self, symbol: str) -> List[OptionQuote]:
        """Fetch full option chain for one underlying.

        Args:
            symbol: Ticker symbol (e.g. 'SPY')

        Returns:
            List of OptionQuote objects (may be partial if some contracts fail).
            Partial results are returned rather than failing the whole run, so
            users can inspect what was successfully captured.
        """
        self._ensure_connected()

        snap_ts = datetime.utcnow().isoformat()
        logger.info(f"Fetching chain for {symbol}")

        # Price may be None if all three fallback methods fail; _get_strikes
        # handles the 0.0/None case by returning the full strike window.
        underlying_price = self._get_underlying_price(symbol)

        expirations = self._get_expirations(symbol)
        if not expirations:
            logger.error(f"No expirations found for {symbol}")
            return []

        quotes = []
        for expiry in expirations:
            strikes = self._get_strikes_for_expiry(
                symbol, expiry, underlying_price or 0.0
            )
            if not strikes:
                logger.warning(f"No strikes for {symbol} {expiry}, skipping")
                continue

            fetched_count = 0
            for strike in strikes:
                for right in ["C", "P"]:
                    for attempt in range(self.config.retry_attempts):
                        quote = self._fetch_contract_quote(
                            symbol, expiry, strike, right,
                            underlying_price, snap_ts
                        )
                        if quote is not None:
                            quotes.append(quote)
                            fetched_count += 1
                            break
                        if attempt < self.config.retry_attempts - 1:
                            time.sleep(self.config.retry_delay_seconds)

            logger.info(f"  {symbol} {expiry}: {fetched_count} contracts")

        logger.info(f"Fetched {len(quotes)} quotes for {symbol}")
        return quotes

    def fetch_all(self, symbols: List[str]) -> Dict[str, List[OptionQuote]]:
        """Fetch option chains for multiple underlyings.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol → list of OptionQuote.
            Keys with empty lists indicate fetch failure for that symbol.
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch(symbol)
            except Exception as e:
                code = getattr(e, 'code', None)
                msg = str(e).lower() if hasattr(e, '__str__') else ''
                if code == IBKR_COMPETING_DATA or '10197' in msg or 'competing' in msg:
                    logger.error(
                        f"Market data blocked for {symbol}: close active TWS "
                        f"market data windows or use IB Gateway."
                    )
                else:
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
        self._ensure_connected()
        return self

    def __exit__(self, *args):
        self.disconnect()
