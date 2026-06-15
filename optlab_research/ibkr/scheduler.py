"""Scheduled snapshot runner for IBKR options chains.

Runs snapshots every N minutes during market hours (9:30 AM – 4:00 PM ET,
Monday–Friday, excluding US market holidays).

Usage:
 from optlab_research.ibkr.scheduler import SnapshotScheduler
 scheduler = SnapshotScheduler(interval_minutes=5)
 scheduler.run() # blocks until market close
"""
from __future__ import annotations

import logging
import time
import signal
import sys
from datetime import datetime, date, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from .snapshot import IBKRSnapshot, SnapshotConfig
from .store import ChainStore

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# US market holidays 2026 (add more years as needed)
US_MARKET_HOLIDAYS_2026 = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 11, 27), # Black Friday (early close — treat as holiday)
    date(2026, 12, 24), # Christmas Eve (early close)
    date(2026, 12, 25), # Christmas Day
}

MARKET_OPEN = (9, 30)  # 9:30 AM ET
MARKET_CLOSE = (16, 0)  # 4:00 PM ET


def is_market_open(now: Optional[datetime] = None) -> bool:
    """Return True if US equity market is currently open."""
    if now is None:
        now = datetime.now(ET)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if now.date() in US_MARKET_HOLIDAYS_2026:
        return False
    market_open = now.replace(
        hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0
    )
    market_close = now.replace(
        hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0
    )
    return market_open <= now < market_close


def next_market_open(now: Optional[datetime] = None) -> datetime:
    """Return the next market open datetime in ET."""
    if now is None:
        now = datetime.now(ET)
    candidate = now.replace(
        hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0
    )
    if candidate <= now:
        candidate += timedelta(days=1)
    # Skip weekends and holidays
    while candidate.weekday() >= 5 or candidate.date() in US_MARKET_HOLIDAYS_2026:
        candidate += timedelta(days=1)
    return candidate


def seconds_until_market_open(now: Optional[datetime] = None) -> float:
    """Return seconds until next market open."""
    if now is None:
        now = datetime.now(ET)
    nxt = next_market_open(now)
    return (nxt - now).total_seconds()


class SnapshotScheduler:
    """Runs IBKR option chain snapshots on a schedule during market hours.

    Usage:
        scheduler = SnapshotScheduler(
            interval_minutes=5,
            symbols=["SPY", "QQQ", "AAPL"],
            config_path="config/ibkr.yaml",
        )
        scheduler.run()
    """

    def __init__(
        self,
        interval_minutes: int = 5,
        symbols: Optional[List[str]] = None,
        config_path: str = "config/ibkr.yaml",
        store_path: str = "data/ibkr_chains",
        ibkr_host: str = "127.0.0.1",
        ibkr_port: int = 7497,
        max_errors_before_pause: int = 5,
        pause_after_errors_seconds: int = 300,
    ):
        self.interval_minutes = interval_minutes
        self.config_path = config_path
        self.store = ChainStore(store_path)
        self.ibkr_host = ibkr_host
        self.ibkr_port = ibkr_port
        self.max_errors_before_pause = max_errors_before_pause
        self.pause_after_errors_seconds = pause_after_errors_seconds
        self._running = False
        self._consecutive_errors = 0
        self._stats = {
            "snapshots_attempted": 0,
            "snapshots_succeeded": 0,
            "snapshots_failed": 0,
            "total_quotes": 0,
            "session_start": None,
        }

        # Load symbols from config or use provided list
        if symbols:
            self.symbols = symbols
        else:
            self.symbols = self._load_symbols_from_config()

    def _load_symbols_from_config(self) -> List[str]:
        """Load symbol list from ibkr.yaml."""
        if not _HAS_YAML:
            logger.warning("PyYAML not installed. Using default symbols.")
            return ["SPY", "QQQ", "IWM"]
        try:
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f)
            symbols = [u["symbol"] for u in cfg.get("underlyings", [])]
            logger.info(f"Loaded {len(symbols)} symbols from {self.config_path}")
            return symbols
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using default symbols.")
            return ["SPY", "QQQ", "IWM"]

    def _snapshot_config(self) -> SnapshotConfig:
        """Build SnapshotConfig from scheduler settings."""
        if not _HAS_YAML:
            return SnapshotConfig(host=self.ibkr_host, port=self.ibkr_port)
        try:
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f)
            snap_cfg = cfg.get("snapshot", {})
            conn_cfg = cfg.get("connection", {})
            return SnapshotConfig(
                host=self.ibkr_host,
                port=self.ibkr_port,
                client_id=conn_cfg.get("client_id", 1),
                timeout=conn_cfg.get("timeout", 30),
                max_expirations=snap_cfg.get("max_expirations", 6),
                max_strikes_atm=snap_cfg.get("max_strikes_atm", 20),
                rate_limit_seconds=snap_cfg.get("rate_limit_seconds", 0.5),
            )
        except Exception:
            return SnapshotConfig(host=self.ibkr_host, port=self.ibkr_port)

    def _run_one_snapshot_cycle(self) -> dict:
        """Run one full snapshot cycle across all symbols.

        Returns dict with cycle statistics.
        """
        cycle_start = datetime.now(ET)
        cycle_stats = {
            "ts": cycle_start.isoformat(),
            "symbols_attempted": len(self.symbols),
            "symbols_succeeded": 0,
            "symbols_failed": 0,
            "total_quotes": 0,
            "files_written": [],
        }

        snap_config = self._snapshot_config()
        try:
            with IBKRSnapshot(snap_config) as snap:
                for symbol in self.symbols:
                    try:
                        quotes = snap.fetch(symbol)
                        if quotes:
                            df = IBKRSnapshot.to_polars(quotes)
                            path = self.store.write(symbol, df)
                            cycle_stats["symbols_succeeded"] += 1
                            cycle_stats["total_quotes"] += len(quotes)
                            cycle_stats["files_written"].append(path)
                            self._consecutive_errors = 0
                        else:
                            logger.warning(f"No quotes for {symbol}")
                            cycle_stats["symbols_failed"] += 1
                    except Exception as e:
                        logger.error(f"Failed snapshot for {symbol}: {e}")
                        cycle_stats["symbols_failed"] += 1
                        self._consecutive_errors += 1
        except Exception as e:
            logger.error(f"Connection failed for cycle: {e}")
            cycle_stats["symbols_failed"] = len(self.symbols)
            self._consecutive_errors += 1

        elapsed = (datetime.now(ET) - cycle_start).total_seconds()
        cycle_stats["elapsed_seconds"] = round(elapsed, 1)

        self._stats["snapshots_attempted"] += 1
        if cycle_stats["symbols_succeeded"] > 0:
            self._stats["snapshots_succeeded"] += 1
        else:
            self._stats["snapshots_failed"] += 1
        self._stats["total_quotes"] += cycle_stats["total_quotes"]

        logger.info(
            f"Cycle complete: {cycle_stats['symbols_succeeded']}/{len(self.symbols)} "
            f"symbols, {cycle_stats['total_quotes']} quotes, {elapsed:.1f}s"
        )
        return cycle_stats

    def run(self):
        """Run the scheduler. Blocks until market close or KeyboardInterrupt."""
        self._running = True
        self._stats["session_start"] = datetime.now(ET).isoformat()

        def _handle_signal(sig, frame):
            logger.info("Shutdown signal received. Stopping scheduler...")
            self._running = False
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        logger.info(
            f"Scheduler started | interval={self.interval_minutes}min | "
            f"symbols={len(self.symbols)} | port={self.ibkr_port}"
        )

        while self._running:
            now = datetime.now(ET)

            if not is_market_open(now):
                wait_seconds = seconds_until_market_open(now)
                next_open = next_market_open(now)
                logger.info(
                    f"Market closed. Next open: {next_open.strftime('%Y-%m-%d %H:%M ET')}. "
                    f"Sleeping {wait_seconds/3600:.1f} hours."
                )
                # Sleep in chunks so we can respond to shutdown signals
                for _ in range(int(wait_seconds // 60)):
                    if not self._running:
                        break
                    time.sleep(60)
                continue

            # Check error threshold
            if self._consecutive_errors >= self.max_errors_before_pause:
                logger.warning(
                    f"{self._consecutive_errors} consecutive errors. "
                    f"Pausing {self.pause_after_errors_seconds}s before retry."
                )
                time.sleep(self.pause_after_errors_seconds)
                self._consecutive_errors = 0
                continue

            # Run snapshot cycle
            cycle_stats = self._run_one_snapshot_cycle()

            # Sleep until next interval
            elapsed = cycle_stats.get("elapsed_seconds", 0)
            sleep_seconds = max(0, self.interval_minutes * 60 - elapsed)
            if sleep_seconds > 0 and self._running:
                logger.debug(f"Sleeping {sleep_seconds:.0f}s until next cycle")
                time.sleep(sleep_seconds)

        logger.info(f"Scheduler stopped. Session stats: {self._stats}")

    def status(self) -> dict:
        """Return current scheduler status."""
        now = datetime.now(ET)
        return {
            "running": self._running,
            "market_open": is_market_open(now),
            "current_time_et": now.strftime("%Y-%m-%d %H:%M:%S ET"),
            "next_open": next_market_open(now).strftime(
                "%Y-%m-%d %H:%M ET"
            )
            if not is_market_open(now)
            else "NOW",
            "interval_minutes": self.interval_minutes,
            "symbols": self.symbols,
            **self._stats,
        }
