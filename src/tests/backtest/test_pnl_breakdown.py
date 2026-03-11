from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.reporting.pnl_breakdown import generate_pnl_breakdown


def _write_sample_trades(path: Path) -> None:
    trades = pd.DataFrame(
        {
            "entry_date": ["2024-01-02 14:30:00+00:00", "2024-01-03 14:30:00+00:00"],
            "exit_date": ["2024-01-02 20:00:00+00:00", "2024-01-03 20:00:00+00:00"],
            "pair": ["AAA-BBB", "AAA-BBB"],
            "side": ["BUY", "SELL"],
            "qty": [10, 10],
            "exec_px": [100.0, 101.0],
            "arrival_px": [99.5, 101.5],
            "decision_px": [99.0, 102.0],
            "gross_pnl": [5.0, -2.0],
            "net_pnl": [4.0, -3.0],
            "fees": [-0.5, -0.5],
            "slippage_cost": [-0.2, -0.2],
            "impact_cost": [-0.1, -0.1],
            "borrow_cost": [0.0, -0.1],
        }
    )
    trades.to_csv(path, index=False)


def test_generate_pnl_breakdown_writes_reports(tmp_path: Path) -> None:
    out = tmp_path / "performance"
    out.mkdir(parents=True, exist_ok=True)
    _write_sample_trades(out / "trades.csv")

    mid_idx = pd.date_range(
        "2024-01-02 09:30", periods=2_000, freq="min", tz="America/New_York"
    )
    mid = pd.Series(100.0, index=mid_idx)
    cfg = {}

    generate_pnl_breakdown(cfg, out, mid_prices=mid)

    expected_csv = [
        "pnl_breakdown_daily.csv",
        "pnl_breakdown_symbol.csv",
        "pnl_components_daily.csv",
        "pnl_components_symbol.csv",
        "pnl_tca_daily.csv",
        "pnl_tca_symbol.csv",
    ]
    for name in expected_csv:
        assert (out / name).exists(), name

    daily = pd.read_csv(out / "pnl_breakdown_daily.csv")
    assert not daily.empty
    assert "total_costs" in daily.columns


def test_generate_pnl_breakdown_runs_when_called_even_without_cfg_flags(
    tmp_path: Path,
) -> None:
    out = tmp_path / "performance"
    out.mkdir(parents=True, exist_ok=True)
    _write_sample_trades(out / "trades.csv")

    cfg = {}
    generate_pnl_breakdown(cfg, out)

    assert (out / "pnl_breakdown_daily.csv").exists()
