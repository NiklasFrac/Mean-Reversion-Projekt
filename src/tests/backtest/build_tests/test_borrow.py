from borrow_manager import BorrowManager


def test_borrow_manager_basics():
    bm = BorrowManager(borrow_rates={"A": 0.05}, availability={"A": True})
    assert bm.is_short_available("A")
    cost = bm.daily_borrow_cost("A", size=10, price=100.0)
    assert cost > 0
    bm.register_open_short("A", 10, price=100.0)
    acc = bm.accrue_daily()
    assert acc > 0
