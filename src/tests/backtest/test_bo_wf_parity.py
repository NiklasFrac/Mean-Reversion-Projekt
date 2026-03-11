from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest import runner_backtest as rb
from backtest.calibration import runner_calibration as rc
from backtest.run.bo_runner import run_bo_if_enabled
from backtest.utils.run.bo import _bo_key_payload


def _make_panel(idx: pd.DatetimeIndex) -> pd.DataFrame:
    base = np.linspace(100.0, 105.0, len(idx))
    panel = pd.DataFrame(
        {
            ("AAA", "open"): base + 0.1,
            ("AAA", "high"): base + 0.5,
            ("AAA", "low"): base - 0.4,
            ("AAA", "close"): base + 0.2,
            ("AAA", "volume"): np.full(len(idx), 1_000.0, dtype=float),
            ("BBB", "open"): base + 1.1,
            ("BBB", "high"): base + 1.5,
            ("BBB", "low"): base + 0.6,
            ("BBB", "close"): base + 1.2,
            ("BBB", "volume"): np.full(len(idx), 1_500.0, dtype=float),
        },
        index=idx,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["symbol", "field"])
    return panel


def _write_analysis_meta_cfg(tmp_path: Path) -> tuple[Path, Path, Path]:
    idx = pd.bdate_range("2024-01-02", periods=60)
    panel = _make_panel(idx)
    prices_path = tmp_path / "panel.pkl"
    panel.to_pickle(prices_path)

    pairs_path = tmp_path / "pairs.pkl"
    pd.to_pickle({"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}, pairs_path)

    resolved_cfg_path = tmp_path / "resolved.json"
    resolved_cfg_path.write_text(
        json.dumps({"data": {"panel_prices_path": str(prices_path)}}),
        encoding="utf-8",
    )

    meta_path = pairs_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "run": "ANL-PARITY",
                "outputs": {"run_scoped_pairs_path": str(pairs_path)},
                "config": {"resolved_config_json": str(resolved_cfg_path)},
            }
        ),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "seed: 17",
                "data:",
                "  input_mode: analysis_meta",
                f"  analysis_meta_path: {meta_path}",
                f"  pairs_path: {pairs_path}",
                "  prefer_col: close",
                "pair_prefilter:",
                "  prefilter_active: false",
                "backtest:",
                "  initial_capital: 100000.0",
                "  splits:",
                "    analysis:",
                "      start: 2024-01-02",
                "      end: 2024-01-10",
                "    train:",
                "      start: 2024-01-11",
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
                "calibration:",
                "  trials: 1",
            ]
        ),
        encoding="utf-8",
    )
    return cfg_path, prices_path, pairs_path


def _load_effective_cfg(out_dir: Path) -> dict:
    return json.loads((out_dir / "config_effective.json").read_text(encoding="utf-8"))


def test_analysis_meta_runtime_parity_across_runners(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path, prices_path, original_pairs_path = _write_analysis_meta_cfg(tmp_path)

    bt_out = tmp_path / "bt"
    assert rb.main(["--cfg", str(cfg_path), "--out", str(bt_out)]) == 0

    monkeypatch.setattr(
        rc,
        "prepare_pairs_data",
        lambda *_a, **_k: {
            "AAA-BBB": {
                "prices": pd.DataFrame({"y": [1.0], "x": [1.0]}),
                "meta": {"t1": "AAA", "t2": "BBB"},
            }
        },
    )
    monkeypatch.setattr(
        rc, "_build_strategy", lambda *_a, **_k: lambda _pairs: {"AAA-BBB": {}}
    )
    monkeypatch.setattr(rc, "_iter_grid", lambda _rng, *, n: [{"vol_window": 12}])
    monkeypatch.setattr(
        rc,
        "backtest_portfolio_with_yaml_cfg",
        lambda **_k: (
            pd.DataFrame(
                {"equity": [100.0]},
                index=pd.DatetimeIndex([pd.Timestamp("2024-01-11")]),
            ),
            pd.DataFrame({"trade": [1], "impact_bps": [1.0]}),
        ),
    )
    monkeypatch.setattr(
        rc, "derive_exec_metrics", lambda *_a, **_k: pd.DataFrame({"impact_bps": [1.0]})
    )
    monkeypatch.setattr(
        rc,
        "score_trades",
        lambda *_a, **_k: rc.ScoreBreakdown(
            score=1.0, penalties={"stub": 1.0}, notes={"ok": True}
        ),
    )
    monkeypatch.setattr(
        rc,
        "summarize_metrics",
        lambda *_a, **_k: SimpleNamespace(n_trades=1, impact_bps_mean=1.0),
    )
    monkeypatch.setattr(
        rc, "report_by_adv_decile", lambda *_a, **_k: pd.DataFrame({"bucket": [1]})
    )
    monkeypatch.setattr(
        rc, "report_by_participation", lambda *_a, **_k: pd.DataFrame({"bucket": [1]})
    )

    cal_out = tmp_path / "cal"
    assert (
        rc.main(["--cfg", str(cfg_path), "--out", str(cal_out), "--trials", "1"]) == 0
    )

    expected_runtime = {
        "prices_path": str(prices_path),
        "prefer_col": "close",
        "input_mode": "analysis_meta",
        "analysis_meta_path": str(original_pairs_path.with_suffix(".meta.json")),
    }

    for out_dir in (bt_out, cal_out):
        eff = _load_effective_cfg(out_dir)
        assert (out_dir / "inputs_provenance.json").exists()
        assert eff["data"]["prices_path"] == expected_runtime["prices_path"]
        assert eff["data"]["prefer_col"] == expected_runtime["prefer_col"]
        assert eff["data"]["input_mode"] == expected_runtime["input_mode"]
        assert (
            eff["data"]["analysis_meta_path"] == expected_runtime["analysis_meta_path"]
        )
        assert (
            Path(eff["data"]["pairs_path"]).read_bytes()
            == original_pairs_path.read_bytes()
        )


def _install_fake_paper_bo(monkeypatch, captured: dict[str, object]) -> None:
    fake_bo_mod = types.ModuleType("backtest.optimize.paper_bo")

    def _run_paper_bo_conservative(**kwargs):
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        captured["cfg"] = kwargs["cfg"]
        bo_best = {"theta_sig_hat": {"entry_z": 1.5}}
        (out_dir / "bo_best.json").write_text(json.dumps(bo_best), encoding="utf-8")
        pd.DataFrame(
            [
                {
                    "component": "theta_sig",
                    "fold_id": 0,
                    "oos_score": 1.0,
                    "params_json": json.dumps({"entry_z": 1.5}),
                }
            ]
        ).to_csv(out_dir / "bo_trials.csv", index=False)
        return bo_best

    def _apply_bo_params_to_cfg(cfg, bo_res):
        out = dict(cfg)
        out["_bo_entry_z"] = float(
            (bo_res.get("theta_sig_hat") or {}).get("entry_z", -1.0)
        )
        return out

    fake_bo_mod.run_paper_bo_conservative = _run_paper_bo_conservative
    fake_bo_mod.apply_bo_params_to_cfg = _apply_bo_params_to_cfg
    monkeypatch.setitem(sys.modules, "backtest.optimize.paper_bo", fake_bo_mod)


def _bo_cfg(tmp_path: Path) -> dict:
    return {
        "seed": 23,
        "data": {
            "prices_path": str(tmp_path / "prices.pkl"),
            "pairs_path": str(tmp_path / "pairs.pkl"),
        },
        "pair_prefilter": {"prefilter_active": False},
        "execution": {"mode": "lob"},
        "backtest": {
            "splits": {
                "train": {"start": "2024-01-02", "end": "2024-01-05"},
                "test": {"start": "2024-01-08", "end": "2024-01-12"},
            }
        },
        "bo": {
            "enabled": True,
            "out_dir": str(tmp_path / "bo"),
            "init_points": 16,
            "n_iter": 24,
            "entry_z_range": [1.0, 2.5],
        },
    }


def _bo_prices_and_pairs() -> tuple[
    pd.DataFrame, dict[str, dict[str, str]], dict[str, dict[str, object]]
]:
    idx = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.DataFrame(
        {
            "AAA": np.linspace(10.0, 11.0, len(idx)),
            "BBB": np.linspace(20.0, 19.0, len(idx)),
        },
        index=idx,
    )
    pairs = {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}
    pairs_data = {
        "AAA-BBB": {
            "prices": pd.DataFrame({"y": prices["AAA"], "x": prices["BBB"]}),
            "meta": {"t1": "AAA", "t2": "BBB"},
        }
    }
    return prices, pairs, pairs_data


def test_run_bo_if_enabled_quick_writes_expected_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    prices, pairs, pairs_data = _bo_prices_and_pairs()
    cfg_eff = _bo_cfg(tmp_path)
    captured: dict[str, object] = {}
    _install_fake_paper_bo(monkeypatch, captured)

    result = run_bo_if_enabled(
        cfg_eff=cfg_eff,
        prices=prices,
        prices_panel=None,
        pairs=pairs,
        adv_map=None,
        out_dir=tmp_path / "out",
        quick=True,
        pairs_data=pairs_data,
        persist_quick_budget=False,
    )

    expected_id = hashlib.sha256(
        json.dumps(_bo_key_payload(cfg_eff), sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    bo_cfg_seen = captured["cfg"]
    assert result.bo_id == expected_id
    assert isinstance(bo_cfg_seen, dict)
    assert bo_cfg_seen["bo"]["init_points"] == 3
    assert bo_cfg_seen["bo"]["n_iter"] == 6
    assert result.cfg_eff["bo"]["init_points"] == 16
    assert result.cfg_eff["bo"]["n_iter"] == 24
    assert result.cfg_eff["_bo_entry_z"] == 1.5

    assert result.bo_out is not None
    assert (result.bo_out / "bo_best.json").exists()
    assert (result.bo_out / "bo_trials.csv").exists()
    assert not result.selected_cv_scores.empty
    assert result.selected_cv_scores.iloc[0]["score"] == 1.0


def test_run_bo_if_enabled_can_persist_quick_budget(
    tmp_path: Path, monkeypatch
) -> None:
    prices, pairs, pairs_data = _bo_prices_and_pairs()
    cfg_eff = _bo_cfg(tmp_path)
    captured: dict[str, object] = {}
    _install_fake_paper_bo(monkeypatch, captured)

    result = run_bo_if_enabled(
        cfg_eff=cfg_eff,
        prices=prices,
        prices_panel=None,
        pairs=pairs,
        adv_map=None,
        out_dir=tmp_path / "out-persist",
        quick=True,
        pairs_data=pairs_data,
        persist_quick_budget=True,
    )

    assert result.cfg_eff["bo"]["init_points"] == 3
    assert result.cfg_eff["bo"]["n_iter"] == 6
