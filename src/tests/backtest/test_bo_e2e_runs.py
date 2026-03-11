from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import runner_backtest as rb


def _make_panel(idx: pd.DatetimeIndex) -> pd.DataFrame:
    base = np.linspace(100.0, 106.0, len(idx))
    panel = pd.DataFrame(
        {
            ("AAA", "open"): base + 0.1,
            ("AAA", "high"): base + 0.6,
            ("AAA", "low"): base - 0.5,
            ("AAA", "close"): base + 0.2,
            ("AAA", "volume"): np.full(len(idx), 1_000.0, dtype=float),
            ("BBB", "open"): base + 1.1,
            ("BBB", "high"): base + 1.6,
            ("BBB", "low"): base + 0.5,
            ("BBB", "close"): base + 1.2,
            ("BBB", "volume"): np.full(len(idx), 1_400.0, dtype=float),
        },
        index=idx,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["symbol", "field"])
    return panel


def _write_pairs_csv(path: Path) -> None:
    pd.DataFrame([{"t1": "AAA", "t2": "BBB"}]).to_csv(path, index=False)


def _install_fake_paper_bo(monkeypatch, calls: dict[str, int]) -> None:
    fake_bo_mod = types.ModuleType("backtest.optimize.paper_bo")

    def _run_paper_bo_conservative(**kwargs):
        calls["run"] = calls.get("run", 0) + 1
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        bo_res = {
            "theta_sig_hat": {
                "entry_z": 1.4,
                "exit_z": 0.6,
                "stop_z": 2.5,
            },
        }
        (out_dir / "bo_best.json").write_text(
            json.dumps(bo_res, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        pd.DataFrame([{"score": 1.0, "stage": "mock"}]).to_csv(
            out_dir / "bo_trials.csv", index=False
        )
        return bo_res

    def _apply_bo_params_to_cfg(cfg: dict, bo_best: dict) -> dict:
        calls["apply"] = calls.get("apply", 0) + 1
        out = dict(cfg)
        theta = dict((bo_best.get("theta_sig_hat") or {}))
        if theta:
            signal = dict(out.get("signal") or {})
            signal.update(theta)
            out["signal"] = signal
        return out

    fake_bo_mod.run_paper_bo_conservative = _run_paper_bo_conservative
    fake_bo_mod.apply_bo_params_to_cfg = _apply_bo_params_to_cfg
    monkeypatch.setitem(sys.modules, "backtest.optimize.paper_bo", fake_bo_mod)


def _write_backtest_cfg(
    path: Path, *, prices_path: Path, pairs_path: Path, bo_out: Path
) -> None:
    path.write_text(
        "\n".join(
            [
                "seed: 101",
                "data:",
                f"  prices_path: {prices_path}",
                f"  pairs_path: {pairs_path}",
                "  prefer_col: close",
                "  calendar_name: XNYS",
                "pair_prefilter:",
                "  prefilter_active: false",
                "backtest:",
                "  initial_capital: 100000.0",
                "  splits:",
                "    train:",
                "      start: 2024-01-02",
                "      end: 2024-02-15",
                "    test:",
                "      start: 2024-02-16",
                "      end: 2024-03-15",
                "strategy:",
                "  name: baseline",
                "signal:",
                "  entry_z: 0.5",
                "  exit_z: 0.1",
                "  stop_z: 2.0",
                "  max_hold_days: 5",
                "spread_zscore:",
                "  z_window: 5",
                "  z_min_periods: 3",
                "execution:",
                "  mode: lob",
                "  lob:",
                "    tick: 0.01",
                "    levels: 2",
                "    size_per_level: 1000",
                "    min_spread_ticks: 1",
                "    lam: 1.0",
                "    max_add: 10",
                "    bias_top: 0.6",
                "    cancel_prob: 0.1",
                "    max_cancel: 10",
                "    steps_per_day: 1",
                "    post_costs: {}",
                "borrow:",
                "  enabled: false",
                "reporting:",
                "  mode: core",
                "  test_tearsheet:",
                "    enabled: false",
                "bo:",
                "  enabled: true",
                f"  out_dir: {bo_out}",
                "  entry_z_range: [1.0, 2.5]",
                "  exit_z_range: [0.3, 0.9]",
                "  stop_z_range: [2.6, 3.5]",
                "  init_points: 16",
                "  n_iter: 24",
            ]
        ),
        encoding="utf-8",
    )


def test_backtest_main_bo_e2e_mock_data(tmp_path: Path, monkeypatch) -> None:
    idx = pd.bdate_range("2024-01-02", periods=60)
    panel = _make_panel(idx)
    prices_path = tmp_path / "panel.pkl"
    panel.to_pickle(prices_path)
    pairs_path = tmp_path / "pairs.csv"
    _write_pairs_csv(pairs_path)

    cfg_path = tmp_path / "cfg_backtest.yaml"
    bo_out = tmp_path / "bo_runs"
    _write_backtest_cfg(
        cfg_path, prices_path=prices_path, pairs_path=pairs_path, bo_out=bo_out
    )

    calls: dict[str, int] = {}
    _install_fake_paper_bo(monkeypatch, calls)

    out_dir = tmp_path / "out_backtest"
    code = rb.main(["--cfg", str(cfg_path), "--out", str(out_dir)])

    assert code == 0
    assert calls["run"] == 1
    assert (out_dir / "report" / "test_summary.json").exists()
    assert (out_dir / "report" / "test_trades.csv").exists()
    assert (out_dir / "report" / "train_selection_summary.json").exists()
    bo_run_dir = next(bo_out.glob("BO-*"))
    assert (bo_run_dir / "bo_best.json").exists()
    assert (bo_run_dir / "bo_trials.csv").exists()
