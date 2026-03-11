from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from backtest.calibration import runner_calibration as rc


def test_patch_liq_model_updates_execution_lob_path() -> None:
    cfg = {"execution": {"lob": {"liq_model": {"vol_window": 10}}}}
    out = rc._patch_liq_model(cfg, {"vol_window": 22})
    assert int(out["execution"]["lob"]["liq_model"]["vol_window"]) == 22


def test_extract_base_liq_model_cfg_reads_execution_lob_only() -> None:
    cfg = {"execution": {"lob": {"liq_model": {"vol_window": 17, "adv_window": 33}}}}
    base = rc._extract_base_liq_model_cfg(cfg)
    assert int(base["vol_window"]) == 17
    assert int(base["adv_window"]) == 33


def test_runner_calibration_main_analysis_meta_walkforward_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=80)
    panel = pd.DataFrame(
        {
            ("AAA", "close"): pd.Series(range(len(idx)), index=idx, dtype=float) + 10.0,
            ("AAA", "volume"): 1000.0,
            ("BBB", "close"): pd.Series(range(len(idx)), index=idx, dtype=float) + 20.0,
            ("BBB", "volume"): 1200.0,
        },
        index=idx,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["symbol", "field"])

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
                "run": "ANL-TEST",
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
                "seed: 11",
                "data:",
                "  input_mode: analysis_meta",
                f"  analysis_meta_path: {meta_path}",
                f"  pairs_path: {pairs_path}",
                "  prefer_col: close",
                "execution:",
                "  mode: lob",
                "  lob:",
                "    liq_model:",
                "      vol_window: 10",
                "      adv_window: 20",
                "backtest:",
                "  initial_capital: 100000.0",
                "  range:",
                "    start: 2024-02-01",
                "    end: 2024-04-15",
                "  walkforward:",
                "    enabled: true",
                "    train_mode: expanding",
                "    initial_train_months: 1",
                "    test_months: 1",
                "    step_months: 1",
                "pair_prefilter:",
                "  prefilter_active: false",
                "calibration:",
                "  wf_window: 0",
                "strategy:",
                "  name: baseline",
                "borrow:",
                "  enabled: false",
                "reporting:",
                "  mode: core",
                "  test_tearsheet:",
                "    enabled: false",
            ]
        ),
        encoding="utf-8",
    )

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
            pd.DataFrame({"equity": [100.0]}, index=pd.DatetimeIndex([idx[0]])),
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

    out_dir = tmp_path / "cal"
    code = rc.main(["--cfg", str(cfg_path), "--out", str(out_dir), "--trials", "1"])

    assert code == 0
    assert (out_dir / "config_effective.json").exists()
    assert (out_dir / "calibration_splits.json").exists()
    assert (out_dir / "summary_test.json").exists()
