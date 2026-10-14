"""Public API regression checks without a licensed data lake or network."""
import datetime as dt

import polars as pl
import pytest

from optlab_research import workbench as wb


def test_supplied_universe_controls_signal_membership_and_values():
    # No connection exists: falling back to a different universe must fail.
    universe = pl.DataFrame({
        "permno": [101, 202, 303],
        "ceq": [10.0, 40.0, 30.0],
        "mcap_musd": [100.0, 100.0, 100.0],
    })
    result = wb.signal("book_to_market", "2023-12-29", universe=universe)
    assert result["permno"].to_list() == [101, 202, 303]
    assert result["signal_value"].to_list() == pytest.approx([0.1, 0.4, 0.3])


def test_named_universe_preserves_preset_then_applies_extra_screens(monkeypatch):
    from optlab_research.universes import builder

    connection = object()

    def prepared_preset(name, as_of, con):
        assert (name, as_of, con) == ("liquid_500", "2023-12-29", connection)
        return pl.DataFrame({
            "permno": [101, 202, 303, 404],
            "prc": [4.0, -10.0, 20.0, 30.0],
            "mcap_musd": [200.0, 200.0, 50.0, 300.0],
        })

    monkeypatch.setattr(builder, "get_universe", prepared_preset)
    result = wb.universe("liquid_500", "2023-12-29", con=connection, min_price=5, min_mcap=100)
    assert result["permno"].to_list() == [202, 404]


def test_public_registry_and_unimplemented_attribution_are_honest():
    assert len(wb.list_signals()) == 10
    assert "book_to_market" in wb.list_signals()
    assert wb.list_attribution_models() == []
    with pytest.raises(NotImplementedError, match="not implemented"):
        wb.attribution(None)


def test_backtest_positive_costs_reduce_long_only_net_returns(monkeypatch, tmp_path):
    from optlab_research.backtest import engine

    # Stub the data boundary only; exercise the actual engine, weights, cost
    # model, return aggregation and manifest through the public wrapper.
    dates = [dt.date(2023, 9, 29), dt.date(2023, 10, 31),
             dt.date(2023, 11, 30), dt.date(2023, 12, 29)]
    panel = pl.DataFrame({
        "permno": [101, 101, 202],
        "as_of_date": dates[:-1],
        "signal_value": [1.0, 1.0, 1.0],
        "signal_rank": [1.0, 1.0, 1.0],
        "signal_quantile": [5, 5, 5],
    })
    returns = pl.DataFrame({
        "permno": [101, 101, 202], "date": dates[1:],
        "adj_ret": [0.10, 0.02, 0.03],
    })
    monkeypatch.setattr(engine, "_get_rebalance_dates", lambda *args: dates)
    monkeypatch.setattr(engine, "_get_adj_returns", lambda *args, **kwargs: returns)
    monkeypatch.setattr(engine, "_build_signal_panel", lambda *args: (
        panel, dict.fromkeys(dates[:-1], 1), {},
    ))
    monkeypatch.setattr(engine, "_DEFAULT_MANIFEST_DIR", tmp_path)

    baseline = wb.backtest("book_to_market", "2023-09-29", "2023-12-29",
                           portfolio="long_only")
    costed = wb.backtest("book_to_market", "2023-09-29", "2023-12-29",
                        portfolio="long_only", tcost_bps=10)
    assert baseline.manifest["tcost_model"] == "none"
    assert costed.manifest["tcost_model"] == "flat_bps"
    gross = costed.returns["ls_ret"].to_list()
    net = costed.returns["net_ls_ret"].to_list()
    assert gross == pytest.approx([0.10, 0.02, 0.03])
    assert baseline.returns["net_ls_ret"].to_list() == pytest.approx(gross)
    assert net[0] < gross[0]  # initial entry
    assert net[1] == pytest.approx(gross[1])  # unchanged holding
    assert net[2] < gross[2]  # replacement holding


def test_flat_costs_debit_both_legs_once_including_reductions_and_exits(monkeypatch, tmp_path):
    from optlab_research.backtest import engine

    dates = [dt.date(2023, 8, 31), dt.date(2023, 9, 29),
             dt.date(2023, 10, 31), dt.date(2023, 11, 30), dt.date(2023, 12, 29)]
    # Equal weighting gives 0.5 per name in a two-name leg, 1.0 in a one-name
    # leg. These transitions exercise retention, reduction, exit and entry.
    memberships = [([101, 102], [201, 202]), ([101, 103], [201]),
                   ([101], [201, 203]), ([101], [201, 203])]
    signal_rows, return_rows = [], []
    for date, next_date, (longs, shorts) in zip(dates[:-1], dates[1:], memberships):
        for permno in longs + shorts:
            is_long = permno in longs
            signal_rows.append({
                "permno": permno, "as_of_date": date,
                "signal_value": 1.0 if is_long else -1.0,
                "signal_rank": 1.0 if is_long else 0.1,
                "signal_quantile": 5 if is_long else 1,
            })
            return_rows.append({"permno": permno, "date": next_date,
                                "adj_ret": 0.10 if is_long else 0.05})
    panel, returns = pl.DataFrame(signal_rows), pl.DataFrame(return_rows)
    monkeypatch.setattr(engine, "_get_rebalance_dates", lambda *args: dates)
    monkeypatch.setattr(engine, "_get_adj_returns", lambda *args, **kwargs: returns)
    monkeypatch.setattr(engine, "_build_signal_panel", lambda *args: (
        panel, {d: len(l) + len(s) for d, (l, s) in zip(dates, memberships)}, {},
    ))
    monkeypatch.setattr(engine, "_DEFAULT_MANIFEST_DIR", tmp_path)

    result = wb.backtest("book_to_market", "2023-08-31", "2023-12-29", tcost_bps=10)
    zero = wb.backtest("book_to_market", "2023-08-31", "2023-12-29")
    assert result.returns["ls_ret"].to_list() == pytest.approx([0.05] * 4)
    assert zero.returns["ls_ret"].to_list() == pytest.approx([0.05] * 4)
    assert zero.returns["net_ls_ret"].to_list() == pytest.approx([0.05] * 4)
    # First three dates each trade 1.0 long and 1.0 short exposure; final
    # holdings are unchanged. Each 10bp leg cost reduces, never credits, P&L.
    assert result.returns["net_long_ret"].to_list() == pytest.approx([0.099] * 3 + [0.10])
    assert result.returns["net_short_ret"].to_list() == pytest.approx([-0.051] * 3 + [-0.05])
    assert result.returns["net_ls_ret"].to_list() == pytest.approx([0.048] * 3 + [0.05])


@pytest.mark.parametrize("model", ["half_spread", "sqrt_adv"])
def test_unreconciled_cost_proxies_cannot_publish_net_backtest_results(model):
    from optlab_research.backtest import Backtest, BacktestConfig

    config = BacktestConfig(signal="book_to_market", start="2023-01-01",
                            end="2023-12-31", universe="liquid_500", tcost_model=model)
    with pytest.raises(NotImplementedError, match="cost-unit and exit reconciliation"):
        Backtest(config).run(None)
