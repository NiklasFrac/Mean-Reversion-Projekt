import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest import runner_backtest as rb
from backtest.utils.run.data import _resolve_data_inputs


def _make_prices(idx: pd.DatetimeIndex) -> pd.DataFrame:
    base = np.linspace(50.0, 55.0, len(idx))
    y = base + np.linspace(0.0, 1.5, len(idx))
    x = base + 0.2 * np.sin(np.linspace(0.0, 6.0, len(idx)))
    return pd.DataFrame({"AAA": y, "BBB": x}, index=idx)


def _make_pairs() -> dict[str, dict[str, str]]:
    return {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}


def test_resolve_data_inputs_analysis_meta(tmp_path: Path) -> None:
    prices_path = tmp_path / "panel.pkl"
    prices = _make_prices(pd.bdate_range("2024-01-02", periods=10))
    prices.to_pickle(prices_path)

    pairs_path = tmp_path / "pairs.json"
    pairs_path.write_text('{"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}', encoding="utf-8")

    adv_path = prices_path.parent / "adv_map.pkl"
    pd.to_pickle({"AAA": 1000.0, "BBB": 2000.0}, adv_path)

    resolved_cfg_path = tmp_path / "resolved.json"
    resolved_cfg_path.write_text(
        json.dumps({"data": {"panel_prices_path": str(prices_path)}}),
        encoding="utf-8",
    )

    meta_path = pairs_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "run": "TEST",
                "outputs": {"run_scoped_pairs_path": str(pairs_path)},
                "config": {"resolved_config_json": str(resolved_cfg_path)},
            }
        ),
        encoding="utf-8",
    )

    cfg = {
        "data": {
            "pairs_path": str(pairs_path),
            "input_mode": "analysis_meta",
        }
    }
    out_dir = tmp_path / "out"
    out = _resolve_data_inputs(cfg, out_dir=out_dir)
    frozen_pairs = Path(out["data"]["pairs_path"])
    assert frozen_pairs.exists()
    assert out["data"]["prices_path"] == str(prices_path)
    assert out["data"]["adv_map_path"] == str(adv_path)
    assert (out_dir / "inputs_provenance.json").exists()


def test_run_once_with_bo_release_overfit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=40)
    prices = _make_prices(idx)
    pairs = _make_pairs()

    prices_path = tmp_path / "prices.pkl"
    pairs_path = tmp_path / "pairs.json"
    prices.to_pickle(prices_path)
    pairs_path.write_text(json.dumps(pairs), encoding="utf-8")

    cfg = {
        "seed": 3,
        "data": {"prices_path": str(prices_path), "pairs_path": str(pairs_path)},
        "pair_prefilter": {"prefilter_active": False},
        "backtest": {
            "initial_capital": 100000.0,
            "splits": {
                "train": {"start": "2024-01-02", "end": "2024-01-20"},
                "test": {"start": "2024-01-21", "end": "2024-02-20"},
            },
        },
        "strategy": {"name": "baseline"},
        "signal": {"entry_z": 0.5, "exit_z": 0.1, "stop_z": 2.0, "max_hold_days": 5},
        "spread_zscore": {"z_window": 5, "z_min_periods": 3},
        "execution": {
            "mode": "lob",
            "lob": {
                "tick": 0.01,
                "levels": 2,
                "size_per_level": 1000,
                "min_spread_ticks": 1,
                "lam": 1.0,
                "max_add": 10,
                "bias_top": 0.6,
                "cancel_prob": 0.1,
                "max_cancel": 10,
                "steps_per_day": 1,
                "post_costs": {},
            },
        },
        "borrow": {"enabled": False},
        "bo": {
            "enabled": True,
            "entry_z_range": [1.0, 1.0],
            "exit_z_range": [0.5, 0.5],
            "stop_z_range": [2.0, 2.0],
        },
        "overfit": {"enabled": True, "trials_paths": []},
        "reporting": {"mode": "core", "test_tearsheet": {"enabled": False}},
    }

    def fake_bo(**kwargs) -> dict[str, object]:
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "component": "theta_sig",
                    "fold_id": 0,
                    "oos_score": 1.0,
                    "params_json": json.dumps(
                        {
                            "entry_z": 1.0,
                            "exit_z": 0.5,
                            "stop_z": 2.0,
                        }
                    ),
                },
                {
                    "component": "theta_markov",
                    "fold_id": 0,
                    "oos_score": 2.0,
                    "params_json": json.dumps(
                        {
                            "min_revert_prob": 0.7,
                            "horizon_days": 8,
                        }
                    ),
                },
            ]
        ).to_csv(out_dir / "bo_trials.csv", index=False)
        bo_best = {
            "theta_sig_hat": {"entry_z": 1.0, "exit_z": 0.5, "stop_z": 2.0},
            "theta_markov_hat": {"min_revert_prob": 0.7, "horizon_days": 8},
        }
        (out_dir / "bo_best.json").write_text(json.dumps(bo_best), encoding="utf-8")
        return bo_best

    def fake_apply(cfg_in: dict, _bo: dict) -> dict:
        return cfg_in

    overfit_seen: dict[str, object] = {}

    def fake_overfit(*_args, **_kwargs) -> Path:
        overfit_seen["component_filter"] = _kwargs.get("component_filter")
        p = Path(_kwargs["out_path"])
        p.write_text("{}", encoding="utf-8")
        return p

    import sys
    import types

    paper_bo_stub = types.ModuleType("backtest.optimize.paper_bo")
    paper_bo_stub.run_paper_bo_conservative = fake_bo
    paper_bo_stub.apply_bo_params_to_cfg = fake_apply
    monkeypatch.setitem(sys.modules, "backtest.optimize.paper_bo", paper_bo_stub)

    overfit_stub = types.ModuleType("backtest.overfit")
    overfit_stub.analyze_bo_trials = fake_overfit
    monkeypatch.setitem(sys.modules, "backtest.overfit", overfit_stub)

    out_dir = tmp_path / "run"
    res = rb._run_once(
        cfg_eff=dict(cfg),
        out_dir=out_dir,
        prices=prices,
        prices_panel=None,
        pairs=pairs,
        adv_map=None,
        quick=False,
    )
    assert res.n_pairs >= 1
    assert res.bo_run.bo_out is not None
    assert not res.bo_run.selected_cv_scores.empty
    assert str(res.bo_run.selected_cv_scores.iloc[0]["component"]) == "theta_markov"
    assert (res.bo_run.bo_out / "bo_best.json").exists()
    assert (out_dir / "overfit_summary.json").exists()
    assert overfit_seen["component_filter"] == "theta_markov"


def test_run_once_borrow_report_and_monitoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=40)
    prices = _make_prices(idx)
    pairs = _make_pairs()

    prices_path = tmp_path / "prices.pkl"
    pairs_path = tmp_path / "pairs.json"
    prices.to_pickle(prices_path)
    pairs_path.write_text(json.dumps(pairs), encoding="utf-8")

    cfg = {
        "seed": 4,
        "data": {"prices_path": str(prices_path), "pairs_path": str(pairs_path)},
        "pair_prefilter": {"prefilter_active": False},
        "backtest": {
            "initial_capital": 100000.0,
            "splits": {
                "train": {"start": "2024-01-02", "end": "2024-01-20"},
                "test": {"start": "2024-01-21", "end": "2024-02-20"},
            },
        },
        "strategy": {"name": "baseline"},
        "signal": {"entry_z": 0.5, "exit_z": 0.1, "stop_z": 2.0, "max_hold_days": 5},
        "spread_zscore": {"z_window": 5, "z_min_periods": 3},
        "execution": {
            "mode": "lob",
            "lob": {
                "tick": 0.01,
                "levels": 2,
                "size_per_level": 1000,
                "min_spread_ticks": 1,
                "lam": 1.0,
                "max_add": 10,
                "bias_top": 0.6,
                "cancel_prob": 0.1,
                "max_cancel": 10,
                "steps_per_day": 1,
                "post_costs": {},
            },
        },
        "borrow": {"enabled": True, "default_rate_annual": 0.1},
        "reporting": {"mode": "debug", "test_tearsheet": {"enabled": False}},
    }

    import sys
    import types

    sys.modules.pop("backtest.reporting.pnl_breakdown", None)
    pnl_stub = types.ModuleType("backtest.reporting.pnl_breakdown")
    pnl_stub.generate_pnl_breakdown = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "backtest.reporting.pnl_breakdown", pnl_stub)

    out_dir = tmp_path / "borrow"
    res = rb._run_once(
        cfg_eff=dict(cfg),
        out_dir=out_dir,
        prices=prices,
        prices_panel=None,
        pairs=pairs,
        adv_map=None,
        quick=False,
        debug_out=out_dir / "debug",
    )
    assert res.n_trades >= 0
    assert (out_dir / "debug" / "config_effective.json").exists()
