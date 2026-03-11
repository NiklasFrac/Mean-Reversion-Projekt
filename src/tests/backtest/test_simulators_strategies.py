import numpy as np
import pandas as pd
import pytest

from backtest.simulators import (
    common,
    costs,
    fill_model,
    liquidity_model,
    orderbook_sim,
    performance,
)
from backtest.strat import baseline


def test_common_helpers() -> None:
    row = pd.Series({"units_y": -5, "entry_price_y": 10.0})
    assert common.infer_units(row, "y") == 5
    assert common.infer_side(row, "y") == "sell"
    assert common.opposite_side("buy") == "sell"
    assert common.cap_cross_price(
        side="buy", submit_mid=10.0, cap_bps=5.0
    ) == pytest.approx(10.005)


def test_costs_and_performance_helpers() -> None:
    params = costs.exec_params_from_cfg(
        {"execution": {"heuristic": {"impact_model": "linear"}}}
    )
    slip = costs.calc_adv_slippage(1000.0, adv=1_000_000, vol=0.2, params=params)
    assert slip > 0.0

    trade_cost = costs.calc_trade_cost(
        10, 100.0, 50.0, per_trade_fixed=1.0, slippage_pct=0.001
    )
    assert trade_cost > 0.0

    sim = costs.simulate_execution(
        size=10,
        price_y=100.0,
        price_x=50.0,
        per_trade_fixed=1.0,
        adv=1_000_000.0,
        vol=0.2,
        params=params,
    )
    assert sim.total_cost > 0.0

    slip_pct = costs.calc_pair_slippage_pct(
        10, 100.0, 50.0, 1_000_000.0, 2_000_000.0, 0.2, 0.2, params=params
    )
    assert slip_pct >= 0.0

    pfill = costs.partial_fill_probability(1000.0, 1_000_000.0, params=params)
    assert 0.0 < pfill <= 1.0

    idx = pd.bdate_range("2024-01-02", periods=5)
    signals = pd.Series([0, 1, 1, 0, 0], index=idx)
    py = pd.Series([10, 10.2, 10.3, 10.1, 10.0], index=idx)
    px = pd.Series([5, 5.1, 5.0, 5.05, 5.0], index=idx)
    pnl = performance.calculate_pair_daily_pnl(signals, py, px)
    costs_s = performance.apply_costs(
        signals, py, px, per_trade_cost=1.0, slippage_pct=0.001
    )
    perf = performance.compute_performance(pnl - costs_s, initial_capital=100000.0)
    assert perf["equity_final"] > 0.0


def test_performance_borrow_and_bootstrap() -> None:
    idx = pd.bdate_range("2024-01-02", periods=6)
    signals = pd.Series([0, 1, 1, 0, -1, 0], index=idx)
    py = pd.Series([10, 10.2, 10.1, 10.0, 9.9, 9.8], index=idx)
    px = pd.Series([5, 5.1, 5.0, 4.9, 5.0, 5.1], index=idx)
    size_ts = pd.Series([0, 10, 10, 0, 8, 0], index=idx)
    costs_sz = performance.apply_costs_with_size(signals, py, px, size_ts, 1.0, 0.001)
    assert costs_sz.abs().sum() >= 0.0

    borrow = performance.accrue_borrow_series(
        pd.Series([0, 10, 10, 0, 0, 0], index=idx), py, 0.05
    )
    assert borrow.min() <= 0.0

    borrow_pair = performance.accrue_borrow_pair(signals, size_ts, py, px, 0.05)
    assert borrow_pair.min() <= 0.0

    pnl = performance.calculate_pair_daily_pnl(signals, py, px)
    ci = performance.bootstrap_sharpe_ci(pnl, 100000.0, n_boot=50)
    assert len(ci) == 3


def test_fill_and_liquidity_models() -> None:
    cfg = fill_model.FillModelCfg(enabled=True)
    frac, diag = fill_model.sample_package_fill_fraction(
        cfg=cfg,
        seed=42,
        shard_id=0,
        depth_total_shares_pair=1000.0,
        qty_pair_shares=100.0,
        adv_usd_pair=1_000_000.0,
        adv_ref_usd=1_000_000.0,
        participation_usd=1000.0,
        sigma_pair=0.02,
    )
    assert 0.0 <= frac <= 1.0
    assert diag["expected"] >= 0.0

    idx = pd.bdate_range("2024-01-02", periods=10, tz="America/New_York")
    panel = pd.DataFrame(
        {
            ("AAA", "close"): np.linspace(10, 11, len(idx)),
            ("AAA", "high"): np.linspace(10.1, 11.1, len(idx)),
            ("AAA", "low"): np.linspace(9.9, 10.9, len(idx)),
            ("AAA", "volume"): np.full(len(idx), 1000.0),
        },
        index=idx,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    lm = liquidity_model.LiquidityModel(panel, cfg=liquidity_model.LiquidityModelCfg())
    params = lm.book_params(
        "AAA", idx[-1], base={"levels": 3, "tick": 0.01, "steps_per_day": 1}
    )
    assert params["levels"] == 3


def test_orderbook_flow() -> None:
    ob = orderbook_sim.OrderBook(
        mid_price=100.0, tick=0.01, levels=3, size_per_level=50, min_spread_ticks=1
    )

    def latency(_ctx: dict) -> dict:
        return {"latency_ms": 5}

    def auction(order: dict, _ctx: dict) -> dict | None:
        if order.get("order_type") == "market" and order.get("qty") == 1:
            return {"handled": True, "report": {"filled_size": 1, "avg_price": 100.0}}
        return None

    ob.set_hooks(latency_fn=latency, auction_fn=auction)

    rep1 = ob.process_market_order(size=1, side="buy")
    assert rep1["filled_size"] == 1

    rep2 = ob.process_limit_order(
        "buy", 100.0, 10, tif="gtc", post_only=True, po_action="slide"
    )
    assert rep2["role"] in {"maker", "taker", "maker_posted", "po_rejected"}

    oid = rep2.get("oid")
    if oid:
        pos = ob.order_position(oid)
        assert pos is not None
        filled = ob.collect_fills_for_oid(oid)
        assert isinstance(filled, list)
        ob.cancel_order(oid)

    ob.step()
    snap = ob.snapshot()
    assert "bids" in snap and "asks" in snap


def _make_pairs_data(idx: pd.DatetimeIndex) -> dict[str, dict[str, object]]:
    y = pd.Series(np.linspace(10, 11, len(idx)), index=idx)
    x = pd.Series(np.linspace(9, 10, len(idx)), index=idx)
    prices = pd.DataFrame({"y": y, "x": x})
    return {"AAA-BBB": {"prices": prices, "meta": {"t1": "AAA", "t2": "BBB"}}}


def test_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.bdate_range("2024-01-02", periods=30)

    cfg_base = {
        "backtest": {
            "initial_capital": 100000.0,
            "splits": {
                "train": {"start": str(idx[0].date()), "end": str(idx[10].date())},
                "test": {"start": str(idx[11].date()), "end": str(idx[-1].date())},
            },
        },
        "signal": {"entry_z": 0.5, "exit_z": 0.1, "stop_z": 2.0, "max_hold_days": 5},
        "spread_zscore": {"z_window": 5},
    }
    strat_base = baseline.BaselineZScoreStrategy(cfg_base, borrow_ctx=None)
    out_base = strat_base(_make_pairs_data(idx))
    assert isinstance(out_base, dict)
