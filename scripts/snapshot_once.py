"""One-shot snapshot script — acceptance criterion for Week 10.

Run this script to verify your IBKR connection and data pipeline:

    python scripts/snapshot_once.py

What it does:
1. Connects to IBKR paper trading (port 7497)
2. Fetches SPY's full option chain
3. Writes to data/ibkr_chains/SPY/year=.../month=.../snap_....parquet
4. Queries the file back via DuckDB to confirm it's readable
5. Prints a summary

Prerequisites:
- TWS or IB Gateway running with paper trading account
- API enabled (TWS -> Edit -> Global Configuration -> API -> Enable)
- Port 7497 open
- pip install ib_async
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from optlab_research.ibkr.snapshot import IBKRSnapshot, SnapshotConfig
from optlab_research.ibkr.store import ChainStore
import duckdb


def main():
    print("=" * 60)
    print("optlab-research Week 10 — IBKR Snapshot Acceptance Test")
    print("=" * 60)

    config = SnapshotConfig(
        host="127.0.0.1",
        port=7497,  # paper trading
        client_id=1,
        timeout=30,
        max_expirations=2,   # just 2 expirations for the test
        max_strikes_atm=5,   # just 5 strikes ATM for the test
        rate_limit_seconds=0.3,
    )

    store = ChainStore("data/ibkr_chains")

    print("\nStep 1: Connecting to IBKR paper trading...")
    try:
        with IBKRSnapshot(config) as snap:
            print(" Connected successfully")

        print("\nStep 2: Fetching SPY option chain (2 expirations, 5 strikes ATM)...")
        quotes = snap.fetch("SPY")
        print(f" Fetched {len(quotes)} option quotes")

        if not quotes:
            print(" WARNING: No quotes returned. Check TWS connection and market hours.")
            return

        df = snap.to_polars(quotes)
        print(f" DataFrame shape: {df.shape}")
        print(f" Columns: {df.columns}")

    except ImportError as e:
        print(f"\nERROR: {e}")
        return
    except ConnectionError as e:
        print(f"\nERROR: {e}")
        return

    print("\nStep 3: Writing to Parquet...")
    filepath = store.write("SPY", df)
    print(f" Written to: {filepath}")

    print("\nStep 4: Querying back via DuckDB...")
    con = duckdb.connect()
    result = con.execute(
        f"SELECT COUNT(*) as n, MIN(strike) as min_strike, MAX(strike) as max_strike "
        f"FROM read_parquet('{filepath}')"
    ).fetchone()
    print(f" Rows: {result[0]}, Strike range: {result[1]:.1f} \u2192 {result[2]:.1f}")

    print("\nStep 5: Manifest check...")
    summary = store.summary()
    for k, v in summary.items():
        print(f" {k}: {v}")

    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERION: PASS")
    print("SPY chain fetched, written to Parquet, queryable via DuckDB.")
    print("=" * 60)


if __name__ == "__main__":
    main()
