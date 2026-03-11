import pytest

from backtest.simulators import orderbook_sim


def test_round_to_tick_and_limit_level() -> None:
    assert orderbook_sim._round_to_tick(10.005, 0.01) == 10.01
    with pytest.raises(ValueError):
        orderbook_sim._round_to_tick(10.0, 0.0)

    lvl = orderbook_sim.LimitLevel(price=10.0, public=5)
    lvl.add_queue(orderbook_sim.RestingOrder(oid=1, qty=3))
    event_log: list[tuple[int, float, int]] = []
    fills = lvl.take(6, event_log=event_log)
    assert sum(q for _, q in fills) == 6
    assert event_log == [(1, 10.0, 1)]

    reduced = lvl.cancel_any(10, prefer_public=True)
    assert reduced >= 0


def test_orderbook_orders_hooks_and_step() -> None:
    def latency_fn(ctx: dict) -> dict:
        return {"latency_ms": 5}

    def auction_fn(order: dict, _ctx: dict) -> dict | None:
        if order.get("symbol") == "AUC" and order.get("order_type") == "limit":
            return {
                "handled": True,
                "report": {
                    "filled_size": 1,
                    "avg_price": order.get("limit_price"),
                    "fills": [(order.get("limit_price"), 1)],
                },
            }
        return None

    ob = orderbook_sim.OrderBook(
        mid_price=100.0,
        levels=2,
        size_per_level=10,
        tick=0.01,
        seed=1,
        latency_fn=latency_fn,
        auction_fn=auction_fn,
    )

    rep_auc = ob.process_limit_order("buy", price=99.99, size=1, symbol="AUC")
    assert rep_auc["role"] == "auction"
    assert "latency" in rep_auc

    rep_po = ob.process_limit_order(
        "buy",
        price=float(ob.best_ask()) + 0.05,
        size=1,
        post_only=True,
        po_action="reject",
    )
    assert rep_po["role"] == "po_rejected"

    rep_slide = ob.process_limit_order(
        "buy",
        price=float(ob.best_ask()) + 0.05,
        size=2,
        post_only=True,
        po_action="slide",
        owner="me",
    )
    assert rep_slide["role"] == "maker_posted"
    oid = int(rep_slide["resting_oid"])

    canceled = ob.cancel_order(oid, qty=1)
    assert canceled == 1

    rep_fok = ob.process_market_order(size=1_000, side="buy", tif="fok")
    assert rep_fok["filled_size"] == 0

    rep_mkt = ob.process_limit_order(
        "sell", price=float(ob.best_bid()), size=25, tif="gtc", is_short=False
    )
    assert rep_mkt["role"] == "taker"

    rep_rest = ob.process_limit_order(
        "buy", price=float(ob.best_bid()), size=2, tif="gtc", owner="me"
    )
    rest_oid = int(rep_rest["resting_oid"])

    ob.set_public_level_sizes([0, 0])
    _ = ob.process_market_order(size=1, side="sell", tif="ioc", is_short=False)
    fills = ob.collect_fills_for_oid(rest_oid)
    assert fills

    ob.step(lam=1.0, max_add=5, cancel_prob=1.0, aggr_prob=1.0, aggr_max=2)


def test_orderbook_misc_helpers() -> None:
    ob = orderbook_sim.OrderBook(
        mid_price=50.0, levels=3, size_per_level=5, tick=0.05, seed=2
    )
    snap = ob.snapshot()
    assert snap["best_bid"] is not None

    ob.set_min_spread_ticks(2)
    ob.set_public_level_sizes([2, 2, 2])
    depth = ob.depth("both", n_levels=1)
    assert depth["bids"] and depth["asks"]

    assert ob.total_liquidity("bids") > 0
    assert -1.0 <= ob.imbalance() <= 1.0

    clone = ob.clone(copy_rng_state=True)
    assert clone.best_bid() == ob.best_bid()

    ob.recenter(51.0, preserve_spread=False)
    ob.set_seed(3)
    ob.set_shard(1)
