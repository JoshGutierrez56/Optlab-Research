"""Convert FF3 monthly CSV to Parquet for optlab data lake."""
import sys, csv, calendar, os
sys.stdout.reconfigure(encoding='utf-8')

csv_path = r"C:\Users\Owner\AppData\Local\Temp\FF_extracted\F-F_Research_Data_Factors.csv"
optlab_root = r"C:\Users\Owner\OneDrive\Desktop\School Work\Grad School Work\Options Club\optlab"
output_dir = os.path.join(optlab_root, "data", "ff_factors_monthly")

os.makedirs(output_dir, exist_ok=True)
print(f"Output dir: {output_dir}")

# Parse FF3 CSV (5-factor source has more columns but this is 3-factor)
rows = []
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    for _ in range(5):  # skip comments + header
        next(reader)
    
    for row in reader:
        if len(row) != 5:
            continue
        date_str = row[0].strip()
        if not date_str or len(date_str) != 6 or not date_str.isdigit():
            continue
        
        try:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            if year < 1925 or year > 2030 or month < 1 or month > 12:
                continue
        except ValueError:
            continue
        
        mkt_rf = float(row[1]) / 100.0 if row[1].strip() else None
        smb = float(row[2]) / 100.0 if row[2].strip() else None
        hml = float(row[3]) / 100.0 if row[3].strip() else None
        rf = float(row[4]) / 100.0 if row[4].strip() else None
        
        day = calendar.monthrange(year, month)[1]
        date_val = f"{year}-{month:02d}-{day:02d}"
        
        rows.append([date_val, mkt_rf, smb, hml, rf])

print(f"Parsed {len(rows)} rows")

# Write as a single Parquet file — optlab uses hive partitioning for year subdirs
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

df = pl.DataFrame({
    "date": [r[0] for r in rows],
    "mktrf": [r[1] for r in rows],
    "smb": [r[2] for r in rows],
    "hml": [r[3] for r in rows],
    "rf": [r[4] for r in rows],
})

# Write as partitioned by year (to match optlab's hive partitioning pattern)
parquet_path = os.path.join(output_dir, "data.parquet")
pq.write_table(pa.Table.from_pandas(df.to_pandas()), parquet_path)
file_size = os.path.getsize(parquet_path) / 1e3

print(f"Wrote Parquet to {parquet_path} ({file_size:.1f} KB)")

# Verify
verify_df = pl.read_parquet(parquet_path)
print(f"\nVerification:")
print(f"  Rows: {len(verify_df)}")
print(f"  Cols: {list(verify_df.columns)}")
print(f"  Types: {verify_df.dtypes}")
print(f"  Date range: {verify_df['date'].min()} to {verify_df['date'].max()}")
print(f"  First row: date={verify_df['date'][0]}, mkt_rf={verify_df['mktrf'][0]:.4f}, smb={verify_df['smb'][0]:.4f}, hml={verify_df['hml'][0]:.4f}, rf={verify_df['rf'][0]:.4f}")
print(f"  Last row:  date={verify_df['date'][-1]}, mkt_rf={verify_df['mktrf'][-1]:.4f}, smb={verify_df['smb'][-1]:.4f}, hml={verify_df['hml'][-1]:.4f}, rf={verify_df['rf'][-1]:.4f}")

# Check if the path matches what optlab expects
import duckdb
con = duckdb.connect()
con.execute(f"CREATE VIEW test_ff AS SELECT * FROM read_parquet('{parquet_path}')")
schema = con.execute("DESCRIBE SELECT * FROM test_ff").fetchall()
print(f"\nDuckDB schema:")
for col in schema:
    print(f"  {col[0]:12s} {col[1]}")
con.close()

# List the output directory
files = list(Path(output_dir).rglob("*"))
print(f"\nFiles in optlab/data/ff_factors_monthly/:")
for f in files:
    print(f"  {f.name}")
