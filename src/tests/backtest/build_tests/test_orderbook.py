from src.orderbook_sim import OrderBook


def test_orderbook_basic():
    ob = OrderBook(mid_price=100.0, levels=3, size_per_level=100, tick=0.1)
    filled, avgp, slip = ob.process_market_order(150, "buy")
    assert filled == 150
    assert avgp is not None
