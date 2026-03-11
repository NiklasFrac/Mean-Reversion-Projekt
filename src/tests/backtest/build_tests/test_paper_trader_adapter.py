from src.paper_trader_adapter import CCXTAdapter, IBAdapter, PaperBroker


def test_paper_and_adapters():
    pb = PaperBroker()
    r = pb.place_order("X", 10, price=100.0)
    assert r["status"] == "filled"
    cc = CCXTAdapter(dry_run=True)
    assert cc.place_order("X", 5)["status"] == "simulated"
    ib = IBAdapter(dry_run=True)
    assert ib.place_order("X", 1)["status"] == "simulated"
