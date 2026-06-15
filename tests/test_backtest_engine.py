"""Tests for backtest engine, portfolio construction, and result.

All tests run on synthetic data — no WRDS connection or optlab install needed.
This makes the suite runnable in CI and on any machine.

Structure
---------
TestPortfolioWeights      — assign_weights() correctness and edge cases
TestPortfolioReturns      — compute_portfolio_returns() correctness
TestTurnover              — compute_turnover() math
TestBacktestResult        — BacktestResult.summary(), plots, and .save()
TestBacktestConfig        — Pydantic validation
TestPrivateHelpers        — _get_adj_returns() query logic (via DuckDB in-memory)
"""
from __future__ import annotations

import datetime as dt
import json
import math

import duckdb
import polars as pl
import pytest

from optlab_research.backtest.portfolio import (
    PortfolioType,
    WeightingScheme,
    assign_weights,
    compute_portfolio_returns,
    compute_turnover,
)
from optlab_research.backtest.result import BacktestResult
from optlab_research.backtest.engine import (
    BacktestConfig,
    _get_rebalance_dates,
    _get_adj_returns,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def quintile_signal_10stocks() -> pl.DataFrame:
    """10 stocks, 2 per quintile, all valid, single date."""
    return pl.DataFrame(
        {
            "permno": list(range(1, 11)),
            "as_of_date": [dt.date(2023, 12, 29)] * 10,
            "signal_value": [float(i) / 10 for i in range(1, 11)],
            "signal_rank":  [float(i) / 10 for i in range(1, 11)],
            "signal_quantile": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        }
    )


@pytest.fixture()
def synthetic_monthly_returns() -> pl.DataFrame:
    """36 months of deterministic synthetic L/S returns (seeded)."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 36
    # Build realistic month-end dates (approximate; good enough for tests)
    start = dt.date(2021, 1, 31)
    dates = [
        dt.date(start.year + (start.month + i - 1) // 12,
                (start.month + i - 1) % 12 + 1, 28)
        for i in range(n)
    ]
    long_rets  = rng.normal(0.010, 0.040, n).tolist()
    short_rets = rng.normal(-0.004, 0.040, n).tolist()
    ls_rets    = [l + s for l, s in zip(long_rets, short_rets)]

    return pl.DataFrame(
        {
            "date":      pl.Series(dates, dtype=pl.Date),
            "long_ret":  long_rets,
            "short_ret": short_rets,
            "ls_ret":    ls_rets,
        }
    )


@pytest.fixture()
def minimal_result(synthetic_monthly_returns) -> BacktestResult:
    n = synthetic_monthly_returns.height
    return BacktestResult(
        returns=synthetic_monthly_returns,
        holdings_summary=pl.DataFrame(
            {
                "date":    synthetic_monthly_returns["date"],
                "n_long":  [50] * n,
                "n_short": [50] * n,
            }
        ),
        monthly_turnover=pl.DataFrame(
            {
                "date":     synthetic_monthly_returns["date"],
                "turnover": [0.15] * n,
            }
        ),
        manifest={
            "run_id":    "test-run-uuid",
            "signal":    "book_to_market",
            "universe":  "russell1000",
            "portfolio": "quintile_long_short",
            "weighting": "equal",
            "start":     "2021-01-31",
            "end":       "2023-12-31",
        },
    )


# ─── In-memory DuckDB fixture for _get_rebalance_dates / _get_adj_returns ─────


@pytest.fixture()
def crsp_con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with minimal crsp_msf and crsp_msedelist views."""
    con = duckdb.connect(":memory:")

    # crsp_msf: 3 stocks × 4 months (2023-09 through 2023-12)
    # permno 1 has a NULL return in 2023-12 (delisted that month)
    con.execute("""
        CREATE TABLE crsp_msf AS
        SELECT * FROM (VALUES
            (1, TIMESTAMP '2023-09-29', 0.02),
            (1, TIMESTAMP '2023-10-31', 0.01),
            (1, TIMESTAMP '2023-11-30', -0.03),
            (1, TIMESTAMP '2023-12-29', NULL),   -- delisted
            (2, TIMESTAMP '2023-09-29', 0.05),
            (2, TIMESTAMP '2023-10-31', 0.03),
            (2, TIMESTAMP '2023-11-30', 0.01),
            (2, TIMESTAMP '2023-12-29', 0.04),
            (3, TIMESTAMP '2023-09-29', -0.01),
            (3, TIMESTAMP '2023-10-31', 0.02),
            (3, TIMESTAMP '2023-11-30', 0.00),
            (3, TIMESTAMP '2023-12-29', 0.01)
        ) t(permno, date, ret)
    """)

    # crsp_msedelist: permno 1 delisted 2023-12-15, performance delisting (dlstcd=520)
    con.execute("""
        CREATE TABLE crsp_msedelist AS
        SELECT * FROM (VALUES
            (1, TIMESTAMP '2023-12-15', -0.25, 520)
        ) t(permno, dlstdt, dlret, dlstcd)
    """)

    # Register as views (matching how optlab registers its parquet views)
    con.execute("CREATE VIEW crsp_msf_v AS SELECT * FROM crsp_msf")
    con.execute("CREATE VIEW crsp_msedelist_v AS SELECT * FROM crsp_msedelist")

    return con


# ─── Portfolio weight tests ────────────────────────────────────────────────────


class TestPortfolioWeights:

    def test_equal_weight_long_sums_to_one(self, quintile_signal_10stocks):
        result = assign_weights(
            quintile_signal_10stocks,
            portfolio_type="quintile_long_short",
            weighting="equal",
            long_quantile=5,
            short_quantile=1,
            n_quantiles=5,
        )
        long_w = result.filter(pl.col("leg") == "long")["weight"].sum()
        assert abs(long_w - 1.0) < 1e-10, f"Long weights sum = {long_w}, expected 1.0"

    def test_equal_weight_short_sums_to_neg_one(self, quintile_signal_10stocks):
        result = assign_weights(
            quintile_signal_10stocks,
            portfolio_type="quintile_long_short",
            weighting="equal",
            long_quantile=5,
            short_quantile=1,
            n_quantiles=5,
        )
        short_w = result.filter(pl.col("leg") == "short")["weight"].sum()
        assert abs(short_w - (-1.0)) < 1e-10, f"Short weights sum = {short_w}, expected -1.0"

    def test_middle_quintiles_excluded(self, quintile_signal_10stocks):
        result = assign_weights(
            quintile_signal_10stocks,
            portfolio_type="quintile_long_short",
            weighting="equal",
            long_quantile=5,
            short_quantile=1,
            n_quantiles=5,
        )
        # 2 stocks in Q1 + 2 stocks in Q5 = 4 total; Q2/Q3/Q4 excluded
        assert result.height == 4, f"Expected 4 portfolio stocks, got {result.height}"
        q2_to_q4_permnos = {3, 4, 5, 6, 7, 8}
        assert not q2_to_q4_permnos.intersection(set(result["permno"].to_list()))

    def test_null_signal_quantile_excluded(self):
        df = pl.DataFrame(
            {
                "permno":          [1, 2, 3],
                "as_of_date":      [dt.date(2023, 1, 31)] * 3,
                "signal_value":    [1.0, None, 3.0],
                "signal_rank":     [0.2, None, 0.8],
                "signal_quantile": pl.Series([1, None, 5], dtype=pl.Int32),
            }
        )
        result = assign_weights(
            df,
            portfolio_type="quintile_long_short",
            weighting="equal",
            long_quantile=5,
            short_quantile=1,
            n_quantiles=5,
        )
        assert 2 not in result["permno"].to_list(), "Null-quantile stock should be excluded"

    def test_rank_weighted_long_sums_to_one(self, quintile_signal_10stocks):
        result = assign_weights(
            quintile_signal_10stocks,
            portfolio_type="quintile_long_short",
            weighting="rank",
            long_quantile=5,
            short_quantile=1,
            n_quantiles=5,
        )
        long_w = result.filter(pl.col("leg") == "long")["weight"].sum()
        assert abs(long_w - 1.0) < 1e-10

    def test_rank_weighted_short_sums_to_neg_one(self, quintile_signal_10stocks):
        result = assign_weights(
            quintile_signal_10stocks,
            portfolio_type="quintile_long_short",
            weighting="rank",
            long_quantile=5,
            short_quantile=1,
            n_quantiles=5,
        )
        short_w = result.filter(pl.col("leg") == "short")["weight"].sum()
        assert abs(short_w - (-1.0)) < 1e-10

    def test_rank_weighted_more_extreme_gets_higher_weight(self, quintile_signal_10stocks):
        # Q5 stocks have signal_rank 0.85 and 0.95; permno 10 (rank 0.95) > permno 9 (rank 0.85)
        result = assign_weights(
            quintile_signal_10stocks,
            portfolio_type="quintile_long_short",
            weighting="rank",
            long_quantile=5,
            short_quantile=1,
            n_quantiles=5,
        )
        long_df = result.filter(pl.col("leg") == "long").sort("permno")
        # permno 10 (signal_rank=0.95, extremity=0.9) > permno 9 (signal_rank=0.85, extremity=0.7)
        assert long_df[1]["weight"][0] > long_df[0]["weight"][0]

    def test_long_only_no_short_leg(self, quintile_signal_10stocks):
        result = assign_weights(
            quintile_signal_10stocks,
            portfolio_type="long_only",
            weighting="equal",
            long_quantile=5,
            short_quantile=1,
            n_quantiles=5,
        )
        assert "short" not in result["leg"].to_list()
        assert result.height == 2

    def test_empty_input_returns_correct_schema(self):
        empty = pl.DataFrame(
            schema={
                "permno": pl.Int64, "as_of_date": pl.Date,
                "signal_value": pl.Float64, "signal_rank": pl.Float64,
                "signal_quantile": pl.Int32,
            }
        )
        result = assign_weights(
            empty,
            portfolio_type="quintile_long_short",
            weighting="equal",
            long_quantile=5,
            short_quantile=1,
            n_quantiles=5,
        )
        assert result.height == 0
        assert "weight" in result.columns


# ─── Portfolio return tests ────────────────────────────────────────────────────


class TestPortfolioReturns:

    def _holdings(
        self,
        permnos: list[int],
        weights: list[float],
        legs: list[str],
        fwd_rets: list[float | None],
        as_of_date: dt.date = dt.date(2023, 11, 30),
    ) -> pl.DataFrame:
        n = len(permnos)
        return pl.DataFrame(
            {
                "permno":          permnos,
                "as_of_date":      [as_of_date] * n,
                "signal_quantile": [5 if l == "long" else 1 for l in legs],
                "leg":             legs,
                "weight":          weights,
                "forward_ret":     pl.Series(fwd_rets, dtype=pl.Float64),
                "return_date":     [as_of_date] * n,  # not used inside function
            }
        )

    def test_simple_long_short_return(self):
        # Long: stock 1 earns +10%, weight +1.0 → contributes +0.10
        # Short: stock 2 earns +5%,  weight −1.0 → contributes −0.05
        # ls_ret = 0.10 + (−0.05) = 0.05
        h = self._holdings([1, 2], [1.0, -1.0], ["long", "short"], [0.10, 0.05])
        r = compute_portfolio_returns(h)
        assert r.height == 1
        assert abs(r["long_ret"][0] - 0.10) < 1e-10
        assert abs(r["short_ret"][0] - (-0.05)) < 1e-10
        assert abs(r["ls_ret"][0] - 0.05) < 1e-10

    def test_null_fwd_ret_dropped_and_renormalized(self):
        # Long leg: stocks 1 and 2 at equal weight 0.5 each.
        # Stock 2 has null return; only stock 1 survives. Its weight renormalizes to 1.0.
        # Expected long_ret = 1.0 * 0.10 = 0.10 (not 0.5 * 0.10 = 0.05)
        h = self._holdings(
            [1, 2, 3],
            [0.5, 0.5, -1.0],
            ["long", "long", "short"],
            [0.10, None, 0.02],
        )
        r = compute_portfolio_returns(h)
        assert abs(r["long_ret"][0] - 0.10) < 1e-10

    def test_all_null_returns_ls_is_null(self):
        h = self._holdings([1, 2], [1.0, -1.0], ["long", "short"], [None, None])
        r = compute_portfolio_returns(h)
        assert r["ls_ret"][0] is None

    def test_output_date_matches_as_of_date(self):
        """date output column should equal as_of_date in the input."""
        target = dt.date(2023, 11, 30)
        h = self._holdings([1], [1.0], ["long"], [0.05], as_of_date=target)
        r = compute_portfolio_returns(h)
        assert r["date"][0] == target

    def test_empty_holdings_returns_correct_schema(self):
        empty = pl.DataFrame(
            schema={
                "permno": pl.Int64, "as_of_date": pl.Date,
                "signal_quantile": pl.Int32, "leg": pl.Utf8,
                "weight": pl.Float64, "forward_ret": pl.Float64,
                "return_date": pl.Date,
            }
        )
        r = compute_portfolio_returns(empty)
        assert r.height == 0
        assert set(r.columns) == {"date", "long_ret", "short_ret", "ls_ret"}

    def test_two_period_stacking(self):
        """Holdings from two signal dates should produce two return rows."""
        h1 = self._holdings([1, 2], [1.0, -1.0], ["long", "short"],
                             [0.10, 0.05], as_of_date=dt.date(2023, 10, 31))
        h2 = self._holdings([1, 2], [1.0, -1.0], ["long", "short"],
                             [0.02, -0.01], as_of_date=dt.date(2023, 11, 30))
        combined = pl.concat([h1, h2], how="diagonal")
        r = compute_portfolio_returns(combined)
        assert r.height == 2
        assert abs(r.filter(pl.col("date") == dt.date(2023, 10, 31))["ls_ret"][0] - 0.05) < 1e-10


# ─── Turnover tests ───────────────────────────────────────────────────────────


class TestTurnover:

    def test_identical_portfolio_zero_turnover(self):
        w = pl.DataFrame({"permno": [1, 2, 3], "weight": [0.5, 0.3, 0.2]})
        assert abs(compute_turnover(w, w)) < 1e-10

    def test_fully_replaced_portfolio_turnover_one(self):
        # No overlap between periods: old stocks 1,2 → new stocks 3,4
        old = pl.DataFrame({"permno": [1, 2], "weight": [0.5, 0.5]})
        new = pl.DataFrame({"permno": [3, 4], "weight": [0.5, 0.5]})
        # Δ = |0.5−0| + |0.5−0| + |0−0.5| + |0−0.5| = 2.0; one-way = 2.0/2 = 1.0
        assert abs(compute_turnover(new, old) - 1.0) < 1e-10

    def test_half_replacement(self):
        # Old: {1: 0.5, 2: 0.5}  →  New: {1: 0.5, 3: 0.5}
        # Δ: stock 1 unchanged (0), stock 2 exits (0.5), stock 3 enters (0.5)
        # one-way turnover = (0.5 + 0.5) / 2 = 0.5
        old = pl.DataFrame({"permno": [1, 2], "weight": [0.5, 0.5]})
        new = pl.DataFrame({"permno": [1, 3], "weight": [0.5, 0.5]})
        assert abs(compute_turnover(new, old) - 0.5) < 1e-10

    def test_sign_flip_counts_as_large_turnover(self):
        # Stock goes from long (+0.5) to short (−0.5): Δ = |−0.5 − 0.5| = 1.0
        old = pl.DataFrame({"permno": [1], "weight": [0.5]})
        new = pl.DataFrame({"permno": [1], "weight": [-0.5]})
        # one-way = 1.0 / 2 = 0.5
        assert abs(compute_turnover(new, old) - 0.5) < 1e-10


# ─── BacktestResult tests ─────────────────────────────────────────────────────


class TestBacktestResult:

    def test_summary_required_keys(self, minimal_result):
        s = minimal_result.summary()
        required = {
            "n_months", "start", "end",
            "ann_return_ls", "ann_vol_ls", "sharpe_ls",
            "max_drawdown_ls", "win_rate_ls",
            "avg_monthly_turnover", "avg_n_long", "avg_n_short",
        }
        missing = required - set(s.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_summary_n_months(self, minimal_result):
        assert minimal_result.summary()["n_months"] == 36

    def test_max_drawdown_non_positive(self, minimal_result):
        assert minimal_result.summary()["max_drawdown_ls"] <= 0.0

    def test_sharpe_is_finite(self, minimal_result):
        s = minimal_result.summary()
        assert s["sharpe_ls"] is None or math.isfinite(s["sharpe_ls"])

    def test_win_rate_in_zero_one(self, minimal_result):
        wr = minimal_result.summary()["win_rate_ls"]
        assert 0.0 <= wr <= 1.0

    def test_known_return_annualization(self):
        """With flat +1% monthly return, annualized = (1.01^12)−1 ≈ 12.68%."""
        n = 24
        dates = [dt.date(2022, 1, 31 if i % 2 == 0 else 28) for i in range(n)]  # approximate
        returns = pl.DataFrame(
            {
                "date":      pl.Series(dates, dtype=pl.Date),
                "long_ret":  [0.01] * n,
                "short_ret": [0.0] * n,
                "ls_ret":    [0.01] * n,
            }
        )
        result = BacktestResult(
            returns=returns,
            holdings_summary=pl.DataFrame({"date": pl.Series(dates, dtype=pl.Date), "n_long": [50]*n, "n_short": [0]*n}),
            monthly_turnover=pl.DataFrame({"date": pl.Series(dates, dtype=pl.Date), "turnover": [0.1]*n}),
            manifest={},
        )
        s = result.summary()
        expected = (1.01 ** 12) - 1
        assert abs(s["ann_return_ls"] - round(expected, 4)) < 0.001

    def test_plot_cumulative_returns_figure(self, minimal_result):
        import matplotlib.figure
        fig = minimal_result.plot_cumulative()
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_drawdown_returns_figure(self, minimal_result):
        import matplotlib.figure
        fig = minimal_result.plot_drawdown()
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_save_creates_all_files(self, minimal_result, tmp_path):
        out = tmp_path / "test_bt"
        minimal_result.save(out)
        assert (out / "returns.parquet").exists()
        assert (out / "holdings_summary.parquet").exists()
        assert (out / "monthly_turnover.parquet").exists()
        assert (out / "manifest.json").exists()

    def test_save_returns_parquet_reloadable(self, minimal_result, tmp_path):
        out = tmp_path / "test_bt2"
        minimal_result.save(out)
        reloaded = pl.read_parquet(out / "returns.parquet")
        assert reloaded.height == 36
        assert "ls_ret" in reloaded.columns

    def test_save_manifest_json_valid(self, minimal_result, tmp_path):
        out = tmp_path / "test_bt3"
        minimal_result.save(out)
        with (out / "manifest.json").open() as f:
            m = json.load(f)
        assert m["signal"] == "book_to_market"
        assert m["universe"] == "russell1000"


# ─── BacktestConfig validation ────────────────────────────────────────────────


class TestBacktestConfig:

    def test_valid_defaults(self):
        cfg = BacktestConfig(
            signal="book_to_market",
            start="2010-01-29",
            end="2024-12-31",
            universe="russell1000",
        )
        assert cfg.n_quantiles == 5
        assert cfg.long_quantile == 5   # auto-set to n_quantiles
        assert cfg.short_quantile == 1
        assert cfg.weighting == "equal"

    def test_long_quantile_auto_resolves_to_n_quantiles(self):
        cfg = BacktestConfig(
            signal="momentum_12_2",
            start="2010-01-29",
            end="2020-12-31",
            universe="russell3000",
            n_quantiles=10,
        )
        assert cfg.long_quantile == 10

    def test_explicit_long_quantile_respected(self):
        cfg = BacktestConfig(
            signal="size",
            start="2010-01-29",
            end="2024-12-31",
            universe="russell1000",
            long_quantile=2,  # inverted signal: small-cap = long side
        )
        assert cfg.long_quantile == 2

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            BacktestConfig(
                signal="roe",
                start="2010-01-29",
                end="2024-12-31",
                universe="tradeable",
                typo_field="oops",
            )

    def test_n_quantiles_minimum_bound(self):
        with pytest.raises(Exception):
            BacktestConfig(
                signal="size",
                start="2010-01-29",
                end="2024-12-31",
                universe="russell1000",
                n_quantiles=1,  # must be >= 2
            )

    def test_start_date_property(self):
        cfg = BacktestConfig(
            signal="roe",
            start="2015-06-30",
            end="2024-12-31",
            universe="liquid_500",
        )
        assert cfg.start_date == dt.date(2015, 6, 30)

    def test_end_date_property(self):
        cfg = BacktestConfig(
            signal="roe",
            start="2015-06-30",
            end="2024-11-29",
            universe="liquid_500",
        )
        assert cfg.end_date == dt.date(2024, 11, 29)


# ─── Private helper tests (DuckDB) ────────────────────────────────────────────


class TestPrivateHelpers:

    def test_get_rebalance_dates_count(self, crsp_con):
        dates = _get_rebalance_dates(
            crsp_con,
            dt.date(2023, 9, 1),
            dt.date(2023, 12, 31),
        )
        assert len(dates) == 4, f"Expected 4 month-ends, got {len(dates)}: {dates}"

    def test_get_rebalance_dates_sorted(self, crsp_con):
        dates = _get_rebalance_dates(
            crsp_con,
            dt.date(2023, 9, 1),
            dt.date(2023, 12, 31),
        )
        assert dates == sorted(dates)

    def test_get_rebalance_dates_restricted_range(self, crsp_con):
        dates = _get_rebalance_dates(
            crsp_con,
            dt.date(2023, 11, 1),
            dt.date(2023, 12, 31),
        )
        assert len(dates) == 2

    def test_adj_returns_uses_dlret_when_ret_null(self, crsp_con):
        """permno 1 Dec: ret=NULL, dlret=−0.25 → adj_ret should be −0.25."""
        adj = _get_adj_returns(
            crsp_con,
            dt.date(2023, 12, 1),
            dt.date(2023, 12, 31),
            delist_perf=-0.30,
            delist_other=-0.55,
        )
        row = adj.filter(
            (pl.col("permno") == 1) & (pl.col("date") == dt.date(2023, 12, 29))
        )
        assert row.height == 1
        assert abs(row["adj_ret"][0] - (-0.25)) < 1e-10

    def test_adj_returns_non_null_ret_unchanged(self, crsp_con):
        """permno 2 Dec: ret=0.04 → adj_ret should be 0.04 (delisting has no effect)."""
        adj = _get_adj_returns(
            crsp_con,
            dt.date(2023, 12, 1),
            dt.date(2023, 12, 31),
            delist_perf=-0.30,
            delist_other=-0.55,
        )
        row = adj.filter(
            (pl.col("permno") == 2) & (pl.col("date") == dt.date(2023, 12, 29))
        )
        assert abs(row["adj_ret"][0] - 0.04) < 1e-10

    def test_adj_returns_output_schema(self, crsp_con):
        adj = _get_adj_returns(
            crsp_con,
            dt.date(2023, 9, 1),
            dt.date(2023, 12, 31),
            delist_perf=-0.30,
            delist_other=-0.55,
        )
        assert adj.schema["permno"] == pl.Int64
        assert adj.schema["date"] == pl.Date
        assert adj.schema["adj_ret"] == pl.Float64
