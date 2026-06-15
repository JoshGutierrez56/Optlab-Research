"""Verify ff_factors_monthly fix + check optlab architecture."""
import sys, duckdb, os

sys.stdout.reconfigure(encoding='utf-8')

# Check existing tables/views in optlab.db
db = r"C:\Users\Owner\OneDrive\Desktop\School Work\Grad School Work\Options Club\optlab-research\optlab.db"
con = duckdb.connect(db)

print("=== Existing tables/views in optlab.db ===")
tables = con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name").fetchall()
for t in tables:
    print(f"  {t[0]}")

# Check duckdb_views
views = con.execute("SELECT view_name FROM duckdb_views() ORDER BY view_name").fetchall()
print(f"\n=== DuckDB views ({len(views)} total) ===")
for v in views[:30]:
    print(f"  {v[0]}")

con.close()

# Now check the optlab root architecture
import optlab
from pathlib import Path

pkg_file = Path(optlab.__file__).resolve()
root = pkg_file.parent.parent
print(f"\n=== Optlab root: {root} ===")

# Check config/tables.yaml for ff_factors_monthly source
with open(str(root / "config" / "tables.yaml"), encoding="utf-8", errors="replace") as f:
    content = f.read()
lines = content.split("\n")
for i, line in enumerate(lines):
    if "ff_factor" in line.lower():
        start = max(0, i-2)
        end = min(len(lines), i+15)
        for j in range(start, end):
            marker = " >>> " if j == i else "     "
            print(f"{marker}{j}: {lines[j]}")

# Check data directory
data_dir = root / "data"
print(f"\n=== Data dir: {data_dir} ===")
if data_dir.exists():
    for item in sorted(data_dir.glob("**/*.parquet"))[:20]:
        print(f"  parquet: {item.relative_to(root)} ({item.stat().st_size/1e6:.1f} MB)")
else:
    print("  not found")

# Check for CSV data lake instead
csv_lake = root / "data_lake"
if csv_lake.exists():
    print(f"\n=== Data lake CSV dir: {csv_lake} ===")
    for item in sorted(csv_lake.glob("*.csv"))[:20]:
        print(f"  csv: {item.name} ({item.stat().st_size/1e6:.1f} MB)")
