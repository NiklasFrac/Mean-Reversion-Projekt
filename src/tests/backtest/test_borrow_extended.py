from pathlib import Path

import numpy as np
import pandas as pd

from backtest.borrow import accrual, context, events


def test_generate_borrow_events_with_ctx(tmp_path: Path) -> None:
    class DummyBorrow:
        enabled = True
        default_rate_annual = 0.05
        day_basis = 252

        def resolve_borrow_rate(self, symbol: str, day: pd.Timestamp) -> float:
            return 0.1 if symbol == "AAA" else 0.0

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "borrow:\n  enabled: true\n  synthetic_jitter_sigma: 0.0\n", encoding="utf-8"
    )

    avail = pd.DataFrame(
        {"symbol": ["AAA"], "date": [pd.Timestamp("2024-01-02")], "available": [1]}
    )
    out = events.generate_borrow_events(
        universe=["AAA", "BBB"],
        day="2024-01-02",
        cfg_path=cfg_path,
        availability_df=avail,
        borrow_ctx=DummyBorrow(),
    )
    assert set(out["symbol"]) == {"AAA", "BBB"}
    assert "availability_hint" in out["notes"].iloc[0]


def test_borrow_context_and_accruals() -> None:
    cfg = {
        "borrow": {
            "enabled": True,
            "default_rate_annual": 0.1,
            "day_basis": 252,
            "accrual_mode": "entry_notional",
            "day_count": "busdays",
            "include_exit_day": False,
            "min_days": 1,
            "per_asset_rate_annual": {"BBB": 0.2},
            "rate_series_by_symbol": {
                "BBB": [{"date": "2024-01-02", "rate_annual": 0.25}]
            },
            "availability_long": [
                {"date": "2024-01-02", "symbol": "BBB", "available": 1}
            ],
        }
    }
    borrow_ctx = context.build_borrow_context(cfg)
    assert borrow_ctx is not None and borrow_ctx.enabled is True

    calendar = pd.bdate_range("2024-01-02", periods=5)
    price_data = {"BBB": pd.Series([10, 10.5, 10.2, 10.3, 10.1], index=calendar)}
    row = pd.Series(
        {
            "entry_date": calendar[0],
            "exit_date": calendar[3],
            "y_symbol": "AAA",
            "x_symbol": "BBB",
            "notional_y": 1_000.0,
            "notional_x": 1_000.0,
            "signal": 1,
        }
    )
    cost = accrual.compute_borrow_cost_for_trade_row(
        row, calendar=calendar, price_data=price_data, borrow_ctx=borrow_ctx
    )
    assert cost <= 0.0

    cfg_mtm = {**cfg}
    cfg_mtm["borrow"] = {**cfg["borrow"], "accrual_mode": "mtm_daily"}
    borrow_ctx_mtm = context.build_borrow_context(cfg_mtm)
    row_mtm = row.copy()
    row_mtm["y_units"] = 10
    row_mtm["x_units"] = -10
    cost_mtm = accrual.compute_borrow_cost_for_trade_row(
        row_mtm,
        calendar=calendar,
        price_data=price_data,
        borrow_ctx=borrow_ctx_mtm,
    )
    assert cost_mtm <= 0.0

    df = pd.DataFrame([row_mtm])
    series = accrual.compute_borrow_cost_for_trades_df(
        df, calendar=calendar, price_data=price_data, borrow_ctx=borrow_ctx_mtm
    )
    assert np.isfinite(series.iloc[0])
