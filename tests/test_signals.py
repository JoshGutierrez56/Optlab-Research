"""Week 1 tests: signal registry validation and PIT correctness.

All tests run without a live WRDS connection. They use:
  - In-memory DuckDB with synthetic tables.
  - Synthetic universe DataFrames that mimic get_universe_as_of() output.
  - Minimal signals.yaml written to tmp_path fixtures.

The PIT correctness tests are the most critical. A failing PIT test means
backtest results are contaminated with look-ahead bias — invisible in-sample
but catastrophic if traded.

Test structure
--------------
TestSignalRegistry       — YAML loading, schema validation, error paths
TestPITCorrectness       — the SQL logic that enforces no look-ahead
TestFormulaEvaluation    — _apply_formula on synthetic data
TestRankAndQuantile      — _rank_and_quantile: shape, bounds, null handling
TestComputeSignalUnit    — compute_signal() end-to-end on synthetic data
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import polars as pl
import pytest

from optlab_research.signals.registry import (
    SignalKind,
    SignalSpec,
    SignalRegistry,
    load_signals,
)
from optlab_research.signals.compute import (
    _apply_formula,
    _rank_and_quantile,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _minimal_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "signals.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_registry(tmp_path) -> SignalRegistry:
    """A valid two-signal registry for testing."""
    yaml_text = """\
signals:
  - name: test_bm
    description: Test book-to-market
    kind: funda
    formula: "pl.col('ceq') / pl.col('mcap_musd')"
    required_columns: [ceq, mcap_musd]
    source_table: comp_funda

  - name: test_size
    description: Test log market cap
    kind: crsp_price
    formula: "pl.col('mcap_musd').log()"
    required_columns: [mcap_musd]
    source_table: crsp_msf
"""
    return load_signals(_minimal_yaml(tmp_path, yaml_text))


@pytest.fixture
def synthetic_universe() -> pl.DataFrame:
    """Minimal DataFrame that mimics get_universe_as_of() output.

    Five firms with varying B/M and market cap for testing sort logic.
    """
    return pl.DataFrame({
        "permno":   pl.Series([10001, 10002, 10003, 10004, 10005], dtype=pl.Int64),
        "gvkey":    ["001000", "002000", "003000", "004000", "005000"],
        "ticker":   ["AAA", "BBB", "CCC", "DDD", "EEE"],
        "prc":      [10.0,  20.0,   50.0,  100.0,  5.0],
        "shrout":   [1000.0, 500.0, 200.0, 100.0, 2000.0],
        # mcap_musd = prc * shrout / 1000  (all = 10.0 $M deliberately, except EEE)
        # EEE: 5 * 2000 / 1000 = 10.0
        # All firms have mcap_musd = 10.0 for simplicity.
        "mcap_musd": [10.0, 10.0, 10.0, 10.0, 10.0],
        # Book equity in $M. B/M will vary: 0.8, 0.5, 1.5, 0.4, 1.2
        "ceq": [8.0, 5.0, 15.0, 4.0, 12.0],
        "funda_datadate": [dt.date(2023, 9, 30)] * 5,
    })


# ─── TestSignalRegistry ────────────────────────────────────────────────────────

class TestSignalRegistry:
    def test_load_canonical_registry(self):
        """The production config/signals.yaml loads without errors."""
        canonical = Path(__file__).resolve().parent.parent / "config" / "signals.yaml"
        if not canonical.exists():
            pytest.skip("config/signals.yaml not found; run tests from repo root")
        registry = load_signals(canonical)
        assert len(registry) >= 5
        # Spot-check names from Week 1 deliverables
        for expected in ["book_to_market", "gross_profitability", "size",
                         "momentum_12_2", "idio_vol_252d"]:
            assert registry.get(expected) is not None, f"{expected} missing from registry"

    def test_registry_names(self, sample_registry):
        assert set(sample_registry.names()) == {"test_bm", "test_size"}

    def test_get_known_signal(self, sample_registry):
        spec = sample_registry.get("test_bm")
        assert spec.kind == SignalKind.funda
        assert spec.formula is not None

    def test_get_unknown_raises_keyerror(self, sample_registry):
        with pytest.raises(KeyError, match="nonexistent"):
            sample_registry.get("nonexistent")

    def test_duplicate_names_rejected(self, tmp_path):
        yaml_text = """\
signals:
  - name: dup
    description: First
    kind: crsp_price
    formula: "pl.col('prc')"
    source_table: crsp_dsf
  - name: dup
    description: Second
    kind: crsp_price
    formula: "pl.col('prc')"
    source_table: crsp_dsf
"""
        with pytest.raises(Exception, match="[Dd]uplicate"):
            load_signals(_minimal_yaml(tmp_path, yaml_text))

    def test_funda_kind_requires_formula(self, tmp_path):
        yaml_text = """\
signals:
  - name: bad_funda
    description: Missing formula
    kind: funda
    source_table: comp_funda
"""
        with pytest.raises(Exception):
            load_signals(_minimal_yaml(tmp_path, yaml_text))

    def test_library_kind_requires_library_fn(self, tmp_path):
        yaml_text = """\
signals:
  - name: bad_lib
    description: Missing library_fn
    kind: library
    source_table: crsp_msf
"""
        with pytest.raises(Exception):
            load_signals(_minimal_yaml(tmp_path, yaml_text))

    def test_extra_fields_rejected(self, tmp_path):
        """ConfigDict(extra='forbid') means unknown YAML keys raise an error."""
        yaml_text = """\
signals:
  - name: bad_extra
    description: Unknown field
    kind: crsp_price
    formula: "pl.col('prc')"
    source_table: crsp_dsf
    unknown_field: 42
"""
        with pytest.raises(Exception):
            load_signals(_minimal_yaml(tmp_path, yaml_text))

    def test_library_signal_params_roundtrip(self, tmp_path):
        """lookback_months and skip_months survive the YAML round-trip."""
        yaml_text = """\
signals:
  - name: mom
    description: Test momentum
    kind: library
    library_fn: optlab_research.signals.library.momentum.compute
    lookback_months: 12
    skip_months: 1
    source_table: crsp_msf
"""
        registry = load_signals(_minimal_yaml(tmp_path, yaml_text))
        spec = registry.get("mom")
        assert spec.lookback_months == 12
        assert spec.skip_months == 1


# ─── TestPITCorrectness ────────────────────────────────────────────────────────

class TestPITCorrectness:
    """
    THE most important tests in this repo.

    A backtest at date D must NEVER use information published after D.
    We verify the SQL logic that enforces this for the two main cases:
      1. Compustat fundamentals gated by rdq (earnings announcement date).
      2. CRSP stocknames gated by namedt..nameenddt.
    """

    def test_funda_pit_excludes_future_announcements(self):
        """
        Setup:
          Firm A: fiscal year ended 2023-09-30, reported 2023-11-15 (before asof).
          Firm B: fiscal year ended 2023-09-30, reported 2024-01-10 (AFTER asof).

        As-of date: 2023-12-29.

        Expected:
          Firm A → use the 2023-09-30 observation (ceq = 100).
          Firm B → fall back to its 2022-09-30 observation (ceq = 180).
                   The 2023-09-30 obs with rdq=2024-01-10 MUST be excluded.

        If Firm B returns ceq=200, we have look-ahead bias.
        """
        con = duckdb.connect(":memory:")
        con.execute("""
            CREATE TABLE comp_funda AS
            SELECT * FROM (VALUES
                ('001', DATE '2023-09-30', DATE '2023-11-15', 100.0),
                ('001', DATE '2022-09-30', DATE '2022-11-12',  90.0),
                ('002', DATE '2023-09-30', DATE '2024-01-10', 200.0),
                ('002', DATE '2022-09-30', DATE '2022-11-08', 180.0)
            ) AS t(gvkey, datadate, rdq, ceq)
        """)

        asof = dt.date(2023, 12, 29)

        result = con.execute("""
            SELECT g.gvkey, f.datadate, f.rdq, f.ceq
            FROM (VALUES ('001'), ('002')) AS g(gvkey)
            LEFT JOIN LATERAL (
                SELECT datadate, rdq, ceq
                FROM comp_funda
                WHERE gvkey = g.gvkey
                  AND COALESCE(rdq::DATE, datadate::DATE + INTERVAL '90' DAY)
                      <= CAST(? AS DATE)
                ORDER BY datadate DESC
                LIMIT 1
            ) f ON TRUE
        """, [asof.isoformat()]).pl()

        firm_a = result.filter(pl.col("gvkey") == "001")
        firm_b = result.filter(pl.col("gvkey") == "002")

        # Firm A: the 2023-09-30 report (rdq 2023-11-15 < asof) should be used.
        assert firm_a["ceq"][0] == pytest.approx(100.0), (
            "Firm A: should use the report announced before as-of date (rdq <= asof)."
        )

        # Firm B: the 2023-09-30 report has rdq=2024-01-10 > asof → must be excluded.
        # The 2022-09-30 report (rdq 2022-11-08 < asof) is the correct fallback.
        assert firm_b["ceq"][0] == pytest.approx(180.0), (
            "Firm B: future-announced report (rdq=2024-01-10 > asof=2023-12-29) "
            "must NOT be used. If this returns 200.0, there is look-ahead bias."
        )

        con.close()

    def test_funda_pit_fallback_when_rdq_is_null(self):
        """When rdq is null, COALESCE falls back to datadate + 90 days."""
        con = duckdb.connect(":memory:")
        con.execute("""
            CREATE TABLE comp_funda AS
            SELECT * FROM (VALUES
                ('001', DATE '2023-09-30', NULL, 100.0),
                ('001', DATE '2022-09-30', NULL,  90.0)
            ) AS t(gvkey, datadate, rdq, ceq)
        """)

        # As-of 2023-12-29. datadate 2023-09-30 + 90 days = 2023-12-29 → included.
        asof = dt.date(2023, 12, 29)
        result = con.execute("""
            SELECT f.ceq FROM (VALUES ('001')) AS g(gvkey)
            LEFT JOIN LATERAL (
                SELECT ceq FROM comp_funda
                WHERE gvkey = g.gvkey
                  AND COALESCE(rdq::DATE, datadate::DATE + INTERVAL '90' DAY)
                      <= CAST(? AS DATE)
                ORDER BY datadate DESC LIMIT 1
            ) f ON TRUE
        """, [asof.isoformat()]).pl()

        # The 2023-09-30 observation: datadate + 90 = Dec 29 <= Dec 29 ✓
        assert result["ceq"][0] == pytest.approx(100.0)

    def test_stocknames_pit_window(self):
        """
        Permnos whose nameenddt < asof must NOT appear in the universe.
        """
        con = duckdb.connect(":memory:")
        con.execute("""
            CREATE TABLE crsp_stocknames AS
            SELECT * FROM (VALUES
                (10001, DATE '2010-01-01', DATE '2023-06-30', 11, 1),
                (10002, DATE '2010-01-01', DATE '9999-12-31', 11, 1)
            ) AS t(permno, namedt, nameenddt, shrcd, exchcd)
        """)

        asof = dt.date(2023, 12, 29)
        result = con.execute("""
            SELECT permno FROM crsp_stocknames
            WHERE CAST(? AS DATE)
                  BETWEEN namedt AND COALESCE(nameenddt, DATE '9999-12-31')
              AND shrcd IN (10, 11)
              AND exchcd IN (1, 2, 3)
        """, [asof.isoformat()]).pl()

        permnos = set(result["permno"].to_list())
        assert 10001 not in permnos, (
            "Expired security (nameenddt=2023-06-30 < asof=2023-12-29) "
            "must not appear in the universe."
        )
        assert 10002 in permnos, "Active security must appear in the universe."

        con.close()

    def test_stocknames_namedt_boundary(self):
        """A permno whose namedt > asof (future listing) must not appear."""
        con = duckdb.connect(":memory:")
        con.execute("""
            CREATE TABLE crsp_stocknames AS
            SELECT * FROM (VALUES
                (20001, DATE '2024-03-01', DATE '9999-12-31', 11, 1)
            ) AS t(permno, namedt, nameenddt, shrcd, exchcd)
        """)

        asof = dt.date(2023, 12, 29)
        result = con.execute("""
            SELECT permno FROM crsp_stocknames
            WHERE CAST(? AS DATE)
                  BETWEEN namedt AND COALESCE(nameenddt, DATE '9999-12-31')
        """, [asof.isoformat()]).pl()

        assert result.is_empty(), (
            "A security listed in 2024 must not appear in a 2023-12-29 universe."
        )
        con.close()


# ─── TestFormulaEvaluation ────────────────────────────────────────────────────

class TestFormulaEvaluation:
    def test_book_to_market_formula(self, synthetic_universe):
        """B/M = ceq / mcap_musd. Verify against known values."""
        formula = "pl.col('ceq') / pl.col('mcap_musd')"
        values = _apply_formula(formula, synthetic_universe)

        expected = [8.0/10.0, 5.0/10.0, 15.0/10.0, 4.0/10.0, 12.0/10.0]
        assert values.to_list() == pytest.approx(expected)

    def test_gross_profitability_formula(self, synthetic_universe):
        """GP/assets formula with synthetic revt, cogs, at columns."""
        universe = synthetic_universe.with_columns([
            pl.lit(100.0).alias("revt"),
            pl.lit(60.0).alias("cogs"),
            pl.lit(500.0).alias("at"),
        ])
        formula = "(pl.col('revt') - pl.col('cogs')) / pl.col('at')"
        values = _apply_formula(formula, universe)
        assert all(v == pytest.approx(40.0 / 500.0) for v in values.to_list())

    def test_size_formula(self, synthetic_universe):
        """Log of mcap_musd. All firms have mcap_musd=10 → log(10) ≈ 2.303."""
        formula = "pl.col('mcap_musd').log()"
        values = _apply_formula(formula, synthetic_universe)
        import math
        assert all(v == pytest.approx(math.log(10.0)) for v in values.to_list())

    def test_division_by_zero_produces_inf(self, synthetic_universe):
        """Division by zero should produce inf/null, not raise. compute_signal handles it."""
        universe = synthetic_universe.with_columns(pl.lit(0.0).alias("denom"))
        formula = "pl.col('ceq') / pl.col('denom')"
        values = _apply_formula(formula, universe)
        # Polars returns inf for x/0 (for float x). compute_signal filters these out.
        assert values.is_infinite().all() or values.is_null().all()


# ─── TestRankAndQuantile ──────────────────────────────────────────────────────

class TestRankAndQuantile:
    def test_output_columns_present(self):
        df = pl.DataFrame({"permno": [1, 2, 3], "signal_value": [1.0, 2.0, 3.0]})
        result = _rank_and_quantile(df, n_quantiles=5)
        assert "signal_rank" in result.columns
        assert "signal_quantile" in result.columns

    def test_rank_bounds(self):
        df = pl.DataFrame({
            "permno": list(range(100)),
            "signal_value": [float(i) for i in range(100)],
        })
        result = _rank_and_quantile(df, n_quantiles=5)
        assert result["signal_rank"].min() > 0.0
        assert result["signal_rank"].max() <= 1.0

    def test_quantile_bounds(self):
        df = pl.DataFrame({
            "permno": list(range(100)),
            "signal_value": [float(i) for i in range(100)],
        })
        result = _rank_and_quantile(df, n_quantiles=5)
        assert result["signal_quantile"].min() == 1
        assert result["signal_quantile"].max() == 5

    def test_highest_value_in_top_quantile(self):
        """The observation with the highest signal_value must be in quintile 5."""
        df = pl.DataFrame({
            "permno": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "signal_value": [float(i) for i in range(10)],
        })
        result = _rank_and_quantile(df, n_quantiles=5)
        top = result.filter(pl.col("permno") == 10)["signal_quantile"][0]
        assert top == 5

    def test_lowest_value_in_bottom_quantile(self):
        df = pl.DataFrame({
            "permno": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "signal_value": [float(i) for i in range(10)],
        })
        result = _rank_and_quantile(df, n_quantiles=5)
        bottom = result.filter(pl.col("permno") == 1)["signal_quantile"][0]
        assert bottom == 1

    def test_row_count_preserved(self):
        """_rank_and_quantile must not drop any rows."""
        df = pl.DataFrame({
            "permno": list(range(50)),
            "signal_value": [float(i) for i in range(50)],
        })
        result = _rank_and_quantile(df, n_quantiles=5)
        assert len(result) == 50

    def test_tied_values_get_same_rank(self):
        """Average-rank method: ties produce the same signal_rank."""
        df = pl.DataFrame({
            "permno": [1, 2, 3],
            "signal_value": [1.0, 1.0, 2.0],  # permnos 1 and 2 are tied
        })
        result = _rank_and_quantile(df).sort("permno")
        rank1 = result.filter(pl.col("permno") == 1)["signal_rank"][0]
        rank2 = result.filter(pl.col("permno") == 2)["signal_rank"][0]
        assert rank1 == pytest.approx(rank2)


# ─── TestComputeSignalUnit ────────────────────────────────────────────────────

class TestComputeSignalUnit:
    """
    Integration-level tests for compute_signal() using synthetic data.
    These tests do NOT require a WRDS connection.
    """

    @pytest.fixture
    def registry_path(self, tmp_path) -> Path:
        yaml_text = """\
signals:
  - name: test_bm
    description: Test B/M
    kind: funda
    formula: "pl.col('ceq') / pl.col('mcap_musd')"
    required_columns: [ceq, mcap_musd]
    source_table: comp_funda
"""
        p = tmp_path / "signals.yaml"
        p.write_text(yaml_text)
        return p

    def test_output_schema(self, registry_path, synthetic_universe):
        """compute_signal() returns the expected column schema."""
        from optlab_research.signals.compute import compute_signal
        from unittest.mock import patch
        import optlab_research.signals.compute as compute_mod

        # Override the registry to use our test YAML.
        compute_mod._registry = load_signals(registry_path)

        # Provide a synthetic DuckDB connection (unused for funda kind).
        con = duckdb.connect(":memory:")
        result = compute_signal(
            "test_bm",
            "2023-12-29",
            con,
            universe=synthetic_universe,
        )

        assert set(result.columns) == {"permno", "as_of_date", "signal_value",
                                       "signal_rank", "signal_quantile"}
        # Reset module-level cache so it doesn't bleed into other tests.
        compute_mod._registry = None

    def test_null_signal_gets_null_rank(self, registry_path, synthetic_universe):
        """Firms with null signal_value must have null rank/quantile."""
        from optlab_research.signals.compute import compute_signal
        import optlab_research.signals.compute as compute_mod

        # Introduce a null ceq for one firm.
        univ = synthetic_universe.with_columns(
            pl.when(pl.col("permno") == 10003)
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("ceq"))
            .alias("ceq")
        )
        compute_mod._registry = load_signals(registry_path)
        con = duckdb.connect(":memory:")

        result = compute_signal("test_bm", "2023-12-29", con, universe=univ)
        null_row = result.filter(pl.col("permno") == 10003)

        assert null_row["signal_rank"][0] is None
        assert null_row["signal_quantile"][0] is None

        # Non-null rows must have valid rank/quantile.
        non_null = result.filter(pl.col("permno") != 10003)
        assert non_null["signal_rank"].is_not_null().all()
        assert non_null["signal_quantile"].is_not_null().all()

        compute_mod._registry = None

    def test_as_of_date_column(self, registry_path, synthetic_universe):
        """The as_of_date column must match the input date exactly."""
        from optlab_research.signals.compute import compute_signal
        import optlab_research.signals.compute as compute_mod

        compute_mod._registry = load_signals(registry_path)
        con = duckdb.connect(":memory:")
        result = compute_signal("test_bm", "2023-12-29", con, universe=synthetic_universe)

        expected_date = dt.date(2023, 12, 29)
        assert result["as_of_date"].to_list() == [expected_date] * len(result)

        compute_mod._registry = None

    def test_missing_required_columns_raises(self, registry_path, synthetic_universe):
        """Passing a universe without required_columns raises ValueError."""
        from optlab_research.signals.compute import compute_signal
        import optlab_research.signals.compute as compute_mod

        compute_mod._registry = load_signals(registry_path)
        con = duckdb.connect(":memory:")

        # Drop ceq from the universe.
        bad_universe = synthetic_universe.drop("ceq")
        with pytest.raises(ValueError, match="ceq"):
            compute_signal("test_bm", "2023-12-29", con, universe=bad_universe)

        compute_mod._registry = None
