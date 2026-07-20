import numpy as np

from scripts.run_synthetic_signal_demo import momentum_backtest, synthetic_monthly_returns


def test_synthetic_inputs_are_deterministic():
    dates_a, returns_a = synthetic_monthly_returns(seed=7)
    dates_b, returns_b = synthetic_monthly_returns(seed=7)
    assert np.array_equal(dates_a, dates_b)
    assert np.array_equal(returns_a, returns_b)


def test_signal_does_not_use_future_returns():
    dates, returns = synthetic_monthly_returns(seed=11)
    _, signals_before = momentum_backtest(dates, returns)
    changed = returns.copy()
    changed[-1] = 0.90
    _, signals_after = momentum_backtest(dates, changed)
    assert np.array_equal(signals_before[-1], signals_after[-1])


def test_costs_reduce_long_short_returns():
    dates, returns = synthetic_monthly_returns(seed=13)
    result, _ = momentum_backtest(dates, returns, cost_bps=20)
    assert (result.returns["net_ls_ret"] <= result.returns["ls_ret"] + 1e-12).all()
