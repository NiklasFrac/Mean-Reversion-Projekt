from pathlib import Path

from backtest.borrow import context as bctx
from backtest.borrow import events as bevents


def test_parse_borrow_inputs() -> None:
    rates = bctx._parse_per_asset_rates_any({"AAA": 0.1, "bbb": "0.2"})
    assert rates["AAA"] == 0.1
    assert rates["BBB"] == 0.2

    series_map = bctx._parse_rate_series_by_symbol_any(
        {"AAA": {"2024-01-01": 0.05, "2024-01-03": 0.06}}
    )
    assert "AAA" in series_map

    long = bctx._parse_rate_series_by_symbol_any(
        [{"date": "2024-01-01", "symbol": "AAA", "rate_annual": 0.1}]
    )
    assert "AAA" in long

    avail = bctx._parse_availability_long_any(
        [{"date": "2024-01-01", "symbol": "AAA", "available": "yes"}]
    )
    assert avail is not None and not avail.empty


def test_build_borrow_context_and_resolve() -> None:
    cfg = {
        "borrow": {
            "enabled": True,
            "default_rate_annual": 0.1,
            "per_asset_rate_annual": {"AAA": 0.2},
            "rate_series_by_symbol": {"BBB": {"2024-01-01": 0.3}},
        }
    }
    ctx = bctx.build_borrow_context(cfg)
    assert ctx is not None
    assert ctx.resolve_borrow_rate("AAA", "2024-01-02") == 0.2
    assert ctx.resolve_borrow_rate("BBB", "2024-01-02") == 0.3


def test_generate_borrow_events(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "borrow:\n  enabled: true\n  default_rate_annual: 0.1\n", encoding="utf-8"
    )
    out = bevents.generate_borrow_events(
        universe=["AAA", "BBB"],
        day="2024-01-03",
        cfg_path=cfg_path,
    )
    assert not out.empty
