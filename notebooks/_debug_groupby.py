"""Debug what group_by actually returns."""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore", message=".*optionm.*")

repo_root = r"C:\Users\Owner\OneDrive\Desktop\School Work\Grad School Work\Options Club\optlab-research"
os.chdir(repo_root)
sys.path.insert(0, repo_root)

import optlab_research.workbench as wb
import polars as pl
import duckdb

ASOF = "2023-12-29"

with wb.open() as con:
    univ = wb.universe("liquid_500", ASOF, con=con)
    
    # Run the same query beta_60m uses
    start_date = "1999-04-27"  # ~60 months before ASOF + buffer
    permnos = univ["permno"].cast(pl.Int64).to_list()[:3]  # just 3 for debug
    
    perm_df = pl.DataFrame({"permno": pl.Series(permnos, dtype=pl.Int64)})
    con.register("_debug_permnos_tmp", perm_df.to_arrow())
    
    sql = """
    SELECT m.permno, m.date::DATE AS date, m.ret AS ret,
           f.mktrf AS mktrf, f.rf AS rf
    FROM crsp_msf m
    INNER JOIN _debug_permnos_tmp p ON p.permno = m.permno
    INNER JOIN ff_factors_monthly f ON f.date::DATE = m.date::DATE
    WHERE m.date::DATE >= CAST(? AS DATE)
      AND m.date::DATE <= CAST(? AS DATE)
      AND m.ret IS NOT NULL
      AND f.mktrf IS NOT NULL
      AND f.rf IS NOT NULL
    ORDER BY m.permno, m.date
    """
    
    raw = con.execute(sql, [start_date, ASOF]).pl()
    print(f"Raw query returned: {raw.height} rows, {raw['permno'].n_unique()} unique permnos")
    print(f"Column types: {raw.dtypes}")
    print(f"First few permno values: {raw['permno'].head().to_list()}")
    
    # Now group_by and check type
    print("\n--- group_by test ---")
    for idx, (key, group) in enumerate(raw.group_by("permno", maintain_order=False)):
        if idx >= 1:
            break
        print(f"Key type: {type(key)}")
        print(f"Key value: {key}")
        print(f"Group type: {type(group)}, rows: {len(group)}")
        
        # If key is a tuple/list, unpack it
        if isinstance(key, (tuple, list)):
            for i, elem in enumerate(key):
                print(f"  key[{i}] type={type(elem)}, value={elem}")
    
    con.unregister("_debug_permnos_tmp")
