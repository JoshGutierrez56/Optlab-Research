"""Parquet storage layer for IBKR option chain snapshots.

Partitions by underlying/year/month for efficient querying.
Maintains a manifest of all snapshots for monitoring.

Storage layout:
 data/ibkr_chains/
 ├── SPY/
 │   ├── year=2026/
 │   │   └── month=06/
 │   │       ├── snap_20260614_143022.parquet
 │   │       └── snap_20260614_143122.parquet
 └── manifest.parquet
"""
from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import polars as pl

logger = logging.getLogger(__name__)


class ChainStore:
    """Manages Parquet storage for option chain snapshots.

    Usage:
        store = ChainStore("data/ibkr_chains")
        store.write("SPY", df)
        df = store.read("SPY", year=2026, month=6)
    """

    def __init__(self, base_path: str = "data/ibkr_chains"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.base_path / "manifest.parquet"

    def _partition_path(self, underlying: str, ts: datetime) -> Path:
        """Compute the partition directory for a snapshot."""
        return (
            self.base_path
            / underlying
            / f"year={ts.year}"
            / f"month={ts.month:02d}"
        )

    def _snap_filename(self, ts: datetime) -> str:
        """Generate snapshot filename from timestamp."""
        return f"snap_{ts.strftime('%Y%m%d_%H%M%S')}.parquet"

    def write(self, underlying: str, df: pl.DataFrame) -> str:
        """Write a snapshot DataFrame to Parquet.

        Args:
            underlying: Ticker symbol
            df: Polars DataFrame from IBKRSnapshot.to_polars()

        Returns:
            Path of written file.
        """
        if df.is_empty():
            logger.warning(f"Empty DataFrame for {underlying}, skipping write")
            return ""

        ts = datetime.utcnow()
        partition_dir = self._partition_path(underlying, ts)
        partition_dir.mkdir(parents=True, exist_ok=True)

        filepath = partition_dir / self._snap_filename(ts)
        df.write_parquet(str(filepath))

        n_rows = len(df)
        logger.info(f"Wrote {n_rows} rows to {filepath}")

        self._update_manifest(underlying, ts, str(filepath), n_rows)
        return str(filepath)

    def write_many(self, results: dict[str, pl.DataFrame]) -> dict[str, str]:
        """Write snapshots for multiple underlyings.

        Args:
            results: Dict mapping symbol -> DataFrame

        Returns:
            Dict mapping symbol -> file path
        """
        paths = {}
        for symbol, df in results.items():
            if not df.is_empty():
                paths[symbol] = self.write(symbol, df)
        return paths

    def read(
        self,
        underlying: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
        latest_only: bool = False,
    ) -> pl.DataFrame:
        """Read snapshots for an underlying.

        Args:
            underlying: Ticker symbol
            year: Filter to specific year (optional)
            month: Filter to specific month (optional)
            latest_only: If True, return only the most recent snapshot

        Returns:
            Polars DataFrame with all matching snapshots concatenated.
        """
        base = self.base_path / underlying
        if not base.exists():
            logger.warning(f"No data found for {underlying}")
            return pl.DataFrame()

        files = []
        for root, dirs, fnames in os.walk(base):
            for fname in fnames:
                if not fname.endswith('.parquet') or fname == 'manifest.parquet':
                    continue
                path = Path(root) / fname
                # Apply year/month filters
                parts = str(path).split(os.sep)
                if year and f"year={year}" not in parts:
                    continue
                if month and f"month={month:02d}" not in parts:
                    continue
                files.append(str(path))

        if not files:
            return pl.DataFrame()

        if latest_only:
            files = [sorted(files)[-1]]

        dfs = [pl.read_parquet(f) for f in files]
        return pl.concat(dfs) if len(dfs) > 1 else dfs[0]

    def query_duckdb(self, underlying: str, year: int, month: int) -> str:
        """Return a DuckDB-compatible glob pattern for querying snapshots.

        Usage:
            import duckdb
            pattern = store.query_duckdb("SPY", 2026, 6)
            con = duckdb.connect()
            df = con.execute(f"SELECT * FROM read_parquet('{pattern}')").df()
        """
        pattern = str(
            self.base_path / underlying / f"year={year}" / f"month={month:02d}" / "*.parquet"
        )
        return pattern

    def _update_manifest(self, underlying: str, ts: datetime, filepath: str, n_rows: int):
        """Append a record to the manifest."""
        new_row = pl.DataFrame([{
            "underlying": underlying,
            "snapshot_ts": ts.isoformat(),
            "filepath": filepath,
            "n_rows": n_rows,
            "date": ts.date().isoformat(),
            "hour": ts.hour,
        }])

        if self._manifest_path.exists():
            existing = pl.read_parquet(str(self._manifest_path))
            manifest = pl.concat([existing, new_row])
        else:
            manifest = new_row

        manifest.write_parquet(str(self._manifest_path))

    def manifest(self) -> pl.DataFrame:
        """Return the full manifest of all snapshots."""
        if not self._manifest_path.exists():
            return pl.DataFrame()
        return pl.read_parquet(str(self._manifest_path))

    def summary(self) -> dict:
        """Return a summary of stored data."""
        mf = self.manifest()
        if mf.is_empty():
            return {"status": "empty", "total_snapshots": 0}

        return {
            "total_snapshots": len(mf),
            "underlyings": mf["underlying"].n_unique(),
            "date_range": f"{mf['date'].min()} \u2192 {mf['date'].max()}",
            "total_rows": int(mf["n_rows"].sum()),
            "avg_rows_per_snap": round(float(mf["n_rows"].mean()), 0),
        }
