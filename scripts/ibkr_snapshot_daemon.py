"""IBKR snapshot daemon — runs as a background process.

Snapshots all configured underlyings every 5 minutes during market hours.
Runs indefinitely, sleeping between market sessions.

Usage:
 # Start the daemon
 python scripts/ibkr_snapshot_daemon.py

 # Start with custom settings
 python scripts/ibkr_snapshot_daemon.py --interval 1 --symbols SPY QQQ AAPL

 # Paper trading (default port 7497)
 python scripts/ibkr_snapshot_daemon.py --port 7497

 # Windows Task Scheduler: set to run at 9:00 AM on weekdays.
 # The daemon will wait for market open, then start snapshotting.

 # To run in background (PowerShell):
 # Start-Process python -ArgumentList "scripts/ibkr_snapshot_daemon.py" -WindowStyle Hidden

Logs to: logs/ibkr_daemon_YYYYMMDD.log
"""
import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optlab_research.ibkr.scheduler import SnapshotScheduler


def setup_logging(log_dir: str = "logs") -> None:
    """Set up file + console logging."""
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f"ibkr_daemon_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("ib_async").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="IBKR options chain snapshot daemon"
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Snapshot interval in minutes (default: 5)"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to snapshot (default: from config/ibkr.yaml)"
    )
    parser.add_argument(
        "--port", type=int, default=7497,
        help="IBKR port: 7497=paper, 7496=live (default: 7497)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="IBKR host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--config", default="config/ibkr.yaml",
        help="Path to ibkr.yaml config (default: config/ibkr.yaml)"
    )
    parser.add_argument(
        "--store", default="data/ibkr_chains",
        help="Path to Parquet store (default: data/ibkr_chains)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("optlab-research IBKR Snapshot Daemon")
    logger.info(f" Interval: {args.interval} minutes")
    logger.info(f" Port: {args.port} ({'paper' if args.port == 7497 else 'live'})")
    logger.info(f" Symbols: {args.symbols or 'from config'}")
    logger.info("=" * 60)

    scheduler = SnapshotScheduler(
        interval_minutes=args.interval,
        symbols=args.symbols,
        config_path=args.config,
        store_path=args.store,
        ibkr_host=args.host,
        ibkr_port=args.port,
    )

    logger.info("Starting scheduler. Press Ctrl+C to stop.")
    scheduler.run()


if __name__ == "__main__":
    main()
