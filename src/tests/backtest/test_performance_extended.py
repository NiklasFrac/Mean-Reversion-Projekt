import numpy as np
import pandas as pd

from backtest.simulators import performance


def test_daily_pnl_and_costs() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    signals = pd.Series([0, 1, 1, -1, 0], index=idx)
    price_y = pd.Series([10, 10.5, 11, 10.8, 11.2], index=idx)
    price_x = pd.Series([20, 19.8, 19.5, 19.7, 19.4], index=idx)

    pnl = performance.calculate_daily_pnl(signals, price_y)
    pair_pnl = performance.calculate_pair_daily_pnl(signals, price_y, price_x)
    assert len(pnl) == len(pair_pnl) == len(idx)

    costs_series = performance.apply_costs(
        signals,
        price_y,
        price_x,
        per_trade_cost=0.1,
        slippage_pct=0.001,
        adv_t1=1_000.0,
        adv_t2=2_000.0,
    )
    assert costs_series.abs().sum() > 0

    size_ts = pd.Series([0, 10, 10, 10, 0], index=idx)
    costs_size = performance.apply_costs_with_size(
        signals,
        price_y,
        price_x,
        size_ts,
        per_trade_cost=0.1,
        slippage_pct=0.001,
        adv_t1=1_000.0,
        adv_t2=2_000.0,
        min_fee_per_lot=0.01,
    )
    assert costs_size.loc[idx[1]] > 0


def test_borrow_and_performance_metrics() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    short_shares = pd.Series([0, 10, 10, 0], index=idx)
    price = pd.Series([10.0, 10.5, 10.2, 10.3], index=idx)

    borrow = performance.accrue_borrow_series(
        short_shares, price, annual_rate=0.05, day_basis=252.0
    )
    assert (borrow <= 0.0).all()

    signals = pd.Series([0, 1, 1, -1], index=idx)
    size_ts = pd.Series([0, 10, 10, 10], index=idx)
    borrow_pair = performance.accrue_borrow_pair(
        signals,
        size_ts,
        price_y=price,
        price_x=price * 2.0,
        annual_rate_y=0.02,
        annual_rate_x=0.03,
    )
    assert (borrow_pair <= 0.0).all()

    pnl = pd.Series([1.0, -0.5, 0.2, 0.3], index=idx)
    trades = pd.DataFrame({"net_pnl": [1.0, -0.5, 0.2]})
    perf = performance.compute_performance(pnl, initial_capital=100.0, trades_df=trades)
    assert perf["equity_final"] > 0

    mean_sharpe, lower, upper = performance.bootstrap_sharpe_ci(
        pnl, initial_capital=100.0, n_boot=50, rng=1
    )
    assert np.isfinite(mean_sharpe) and lower <= upper


def test_pnl_explain_and_bucket_reports() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": ["2024-01-02", "2024-01-10", "2024-02-05"],
            "exit_date": ["2024-01-05", "2024-01-20", "2024-02-10"],
            "net_pnl": [10.0, -5.0, 3.0],
            "fees": [-1.0, -1.2, -0.8],
            "slippage_cost": [-0.5, -0.4, -0.3],
            "impact_cost": [-0.2, -0.1, -0.05],
            "borrow_cost": [-0.1, -0.05, -0.02],
            "maker_fills": [1, 0, 2],
            "taker_fills": [0, 1, 1],
            "gross_notional": [1_000.0, 2_000.0, 3_000.0],
            "pair": ["AAA-BBB", "CCC-DDD", "AAA-BBB"],
        }
    )

    explained = performance.pnl_explain(trades, by=["month"])
    assert "net_pnl" in explained.columns

    reports = performance.make_bucket_reports(trades, size_quantiles=3, top_n_pairs=1)
    assert {"by_month", "by_holding", "by_size_q", "by_liquidity", "by_pair"}.issubset(
        reports.keys()
    )
