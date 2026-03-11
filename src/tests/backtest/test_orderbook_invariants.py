import pytest

from backtest.simulators.orderbook_sim import OrderBook


def test_orderbook_invariants_hold_under_ops() -> None:
    ob = OrderBook(
        mid_price=100.0,
        levels=3,
        size_per_level=10,
        tick=0.01,
        seed=1,
        check_invariants=True,
    )
    _ = ob.process_market_order(size=5, side="buy", tif="ioc")
    _ = ob.process_limit_order(
        side="sell", price=float(ob.best_ask()), size=3, tif="gtc"
    )
    _ = ob.process_limit_order(
        side="buy", price=float(ob.best_bid()), size=2, tif="gtc"
    )
    ob.step(lam=1.0, max_add=5, cancel_prob=0.5, aggr_prob=0.5, aggr_max=2)


def test_orderbook_invariants_detect_crossed() -> None:
    ob = OrderBook(
        mid_price=50.0,
        levels=2,
        size_per_level=5,
        tick=0.01,
        seed=2,
        check_invariants=False,
    )
    # Force a crossed book and ensure invariant check fails.
    if ob.asks and ob.bids:
        ob.asks[0].price = float(ob.bids[0].price) - ob.tick
    with pytest.raises(AssertionError):
        ob.assert_invariants()
