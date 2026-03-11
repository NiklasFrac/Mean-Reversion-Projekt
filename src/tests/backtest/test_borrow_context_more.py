from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.borrow import accrual, context, events


def test_borrow_context_parsing_variants() -> None:
    rates = context._parse_per_asset_rates_any(
        [
            {"symbol": "AAA", "rate_annual": 0.1},
            {"ticker": "BBB", "rate": 0.2},
            {"symbol": "", "rate": 0.3},
        ]
    )
    assert rates == {"AAA": 0.1, "BBB": 0.2}

    series_map = context._parse_rate_series_by_symbol_any(
        {"AAA": [{"date": "2024-01-02", "rate": 0.1}, ["2024-01-03", 0.2]]}
    )
    assert "AAA" in series_map and not series_map["AAA"].empty

    avail_df = context._parse_availability_long_any(
        pd.DataFrame({"dt": ["2024-01-02"], "ticker": ["AAA"], "borrowable": ["yes"]})
    )
    assert avail_df is not None
    assert float(avail_df["available"].iloc[0]) == 1.0


def test_borrow_context_events_for_range() -> None:
    cfg = {"borrow": {"enabled": True, "default_rate_annual": 0.1}}
    ctx = context.build_borrow_context(cfg)
    assert ctx is not None
    out = ctx.events_for_range(["AAA"], "2024-01-01", "2024-01-03")
    assert not out.empty
    assert set(out["symbol"]) == {"AAA"}


def test_generate_borrow_events_with_jitter_and_fees(tmp_path) -> None:
    class DummyBorrow:
        enabled = True
        default_rate_annual = 0.05
        day_basis = 252

        def resolve_borrow_rate(self, symbol: str, day: pd.Timestamp) -> float:
            return 0.2 if symbol == "AAA" else 0.0

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "borrow:\n  enabled: true\n  synthetic_jitter_sigma: 0.2\n", encoding="utf-8"
    )
    out = events.generate_borrow_events(
        universe=["AAA"],
        day="2024-01-02",
        cfg_path=cfg_path,
        lead_days=2,
        locate_fee_bps=5.0,
        borrow_ctx=DummyBorrow(),
    )
    assert float(out["lead_days"].iloc[0]) == 2.0
    assert float(out["locate_fee_bps"].iloc[0]) == 5.0
    assert float(out["rate_annual"].iloc[0]) > 0.0


def test_borrow_meta_single_asset() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=3)
    ctx = context.BorrowContext(enabled=True, default_rate_annual=0.1)
    row = pd.Series(
        {
            "entry_date": calendar[0],
            "exit_date": calendar[2],
            "symbol": "AAA",
            "size": -10,
            "entry_price": 10.0,
        }
    )
    meta = accrual.compute_borrow_meta_for_trade_row(
        row, calendar=calendar, price_data={}, borrow_ctx=ctx
    )
    assert meta["short_symbol"] == "AAA"

    cost = accrual.compute_borrow_cost_for_trade_row(
        row, calendar=calendar, price_data={}, borrow_ctx=ctx
    )
    assert np.isfinite(cost)


def test_borrow_context_rate_series_and_per_asset() -> None:
    series = pd.Series([0.1, 0.2], index=pd.to_datetime(["2024-01-02", "2024-01-05"]))
    ctx = context.BorrowContext(
        enabled=True,
        default_rate_annual=0.05,
        per_asset_rate_annual={"BBB": 0.3},
        rate_series_by_symbol={"AAA": series},
    )
    assert ctx.resolve_borrow_rate("AAA", "2024-01-04") == 0.1
    assert ctx.resolve_borrow_rate("BBB", "2024-01-04") == 0.3
    assert ctx.resolve_borrow_rate("CCC", "2024-01-04") == 0.05


def test_parse_rate_series_long_list_and_mapping() -> None:
    rows = [
        {"date": "2024-01-02", "symbol": "AAA", "rate_annual": 0.1},
        {"date": "2024-01-03", "symbol": "AAA", "rate_annual": 0.2},
        {"date": "2024-01-02", "symbol": "BBB", "rate_annual": 0.05},
    ]
    long_map = context._parse_rate_series_by_symbol_any(rows)
    assert set(long_map.keys()) == {"AAA", "BBB"}
    assert float(long_map["AAA"].iloc[-1]) == 0.2

    per_asset = context._parse_per_asset_rates_any({"AAA": 0.1, "": 0.2, "BBB": "bad"})
    assert per_asset == {"AAA": 0.1}


def test_parse_availability_long_any_bool_strings() -> None:
    avail = context._parse_availability_long_any(
        [
            {"date": "2024-01-02", "symbol": "AAA", "available": "no"},
            {"date": "2024-01-03", "symbol": "BBB", "available": "yes"},
        ]
    )
    assert avail is not None
    assert float(avail.loc[avail["symbol"] == "AAA", "available"].iloc[0]) == 0.0
    assert float(avail.loc[avail["symbol"] == "BBB", "available"].iloc[0]) == 1.0


def test_events_generate_from_env_cfg(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "borrow:\n  enabled: true\n  default_rate_annual: 0.05\n", encoding="utf-8"
    )
    monkeypatch.setenv("WF_CFG", str(cfg_path))
    out = events.generate_borrow_events(
        universe=["aaa"], day="2024-01-02", cfg_path=None
    )
    assert not out.empty
    assert float(out["rate_annual"].iloc[0]) == 0.05
    monkeypatch.delenv("WF_CFG", raising=False)


def test_generate_borrow_events_disabled_empty(tmp_path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("borrow:\n  enabled: false\n", encoding="utf-8")
    out = events.generate_borrow_events(
        universe=["AAA"], day="2024-01-02", cfg_path=cfg_path
    )
    assert out.empty

    class DisabledBorrow:
        enabled = False

    out2 = events.generate_borrow_events(
        universe=["AAA"], day="2024-01-02", borrow_ctx=DisabledBorrow()
    )
    assert out2.empty


def test_borrow_events_helpers_and_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    assert events._get({"a": {"b": 1}}, "a.missing", default="x") == "x"

    monkeypatch.setattr(events.Path, "exists", lambda _self: False)
    assert events._discover_config_path() is None

    assert events._load_yaml(None) == {}
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("{bad: [", encoding="utf-8")
    out = events._load_yaml(bad_yaml)
    assert out == {}

    class BadBorrow:
        class _BadBool:
            def __bool__(self):
                raise ValueError("boom")

        enabled = _BadBool()

        @property
        def default_rate_annual(self):
            raise ValueError("boom")

        @property
        def day_basis(self):
            raise ValueError("boom")

        def resolve_borrow_rate(self, _sym, _day):
            raise ValueError("boom")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "borrow:\n  enabled: true\n  default_rate_annual: 0.1\n  day_basis: 252\n  synthetic_jitter_sigma: bad\n",
        encoding="utf-8",
    )
    out2 = events.generate_borrow_events(
        universe=["AAA"],
        day="2024-01-02",
        cfg_path=cfg_path,
        locate_fee_bps="bad",
        borrow_ctx=BadBorrow(),
    )
    assert not out2.empty

    out3 = events.generate_borrow_events(
        universe=[], day="2024-01-02", cfg_path=cfg_path
    )
    assert out3.empty


def test_generate_borrow_events_invalid_day_returns_empty(tmp_path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("borrow:\n  enabled: true\n", encoding="utf-8")
    out = events.generate_borrow_events(
        universe=["AAA"], day="not-a-date", cfg_path=cfg_path
    )
    assert out.empty
