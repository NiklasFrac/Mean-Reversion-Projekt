import hashlib
import importlib
import json
import logging
import sys
import types
from pathlib import Path

import pytest

from backtest import runner_backtest as runner
from backtest.utils.run.bo import _bo_key_payload, _sanitize_bo_cfg_for_key
from backtest.utils.run.data import _resolve_data_inputs
from backtest.utils.run.io import (
    _deep_merge,
    _file_fingerprint,
    _load_json,
    _load_yaml,
    _sha256_file,
)


def test_runner_backtest_file_helpers(tmp_path: Path) -> None:
    file_path = tmp_path / "a.txt"
    file_path.write_text("abc", encoding="utf-8")
    assert _sha256_file(file_path) == hashlib.sha256(b"abc").hexdigest()

    json_path = tmp_path / "a.json"
    json_path.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert _load_json(json_path) == {}

    yaml_path = tmp_path / "a.yaml"
    yaml_path.write_text("- a\n- b\n", encoding="utf-8")
    assert _load_yaml(yaml_path) == {}

    assert _file_fingerprint(None) is None
    missing_fp = _file_fingerprint(tmp_path / "missing.txt")
    assert "path" in missing_fp and "size" not in missing_fp

    ok_fp = _file_fingerprint(file_path)
    assert ok_fp["path"] == str(file_path)
    assert ok_fp["size"] == file_path.stat().st_size


def test_runner_backtest_module_reload_with_existing_logger_handler() -> None:
    logger = logging.getLogger("backtest.runner")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    try:
        sys.modules.pop("backtest.runner_backtest", None)
        import backtest.runner_backtest as mod

        importlib.reload(mod)
    finally:
        logger.removeHandler(handler)


def test_runner_backtest_merge_and_bo_payload(tmp_path: Path) -> None:
    merged = _deep_merge({"a": {"b": 1}}, {"a": {"c": 2}})
    assert merged["a"]["b"] == 1 and merged["a"]["c"] == 2

    bo_cfg = {"out_dir": "x", "init_points": 2, "n_iter": 3, "patience": 4, "keep": 1}
    sanitized = _sanitize_bo_cfg_for_key(bo_cfg)
    assert "out_dir" not in sanitized
    assert "init_points" not in sanitized
    assert sanitized["keep"] == 1

    prices = tmp_path / "prices.parquet"
    pairs = tmp_path / "pairs.csv"
    prices.write_text("x", encoding="utf-8")
    pairs.write_text("y", encoding="utf-8")
    cfg = {
        "seed": 7,
        "data": {"prices_path": str(prices), "pairs_path": str(pairs)},
        "backtest": {
            "initial_capital": 123.0,
            "splits": {"train": {"start": "2024-01-01", "end": "2024-01-31"}},
        },
        "bo": bo_cfg,
    }
    payload = _bo_key_payload(cfg)
    assert payload["seed"] == 7
    assert payload["data"]["prices"]["path"] == str(prices)
    assert payload["data"]["pairs"]["path"] == str(pairs)


def test_runner_backtest_bo_helpers_reject_legacy_stage_cfg() -> None:
    with pytest.raises(ValueError, match="Legacy BO config keys"):
        _sanitize_bo_cfg_for_key({"stage2": {"entry_z_range": [1.0, 2.0]}})


def test_runner_backtest_strategy_and_cfg_validation(tmp_path: Path) -> None:
    cfg_eff = {"strategy": {"name": "baseline", "params": {"extra": 1}}}
    with pytest.raises(TypeError):
        runner._build_strategy(cfg_eff, borrow_ctx=None)

    with pytest.raises(KeyError):
        runner._validate_cfg_strict({})

    prices = tmp_path / "prices.parquet"
    prices.write_text("x", encoding="utf-8")
    cfg_bad_mode = {
        "data": {"prices_path": str(prices), "pairs_path": str(prices)},
        "execution": {"mode": "sim"},
    }
    with pytest.raises(ValueError):
        runner._validate_cfg_strict(cfg_bad_mode)


def test_runner_backtest_resolve_data_inputs_errors(tmp_path: Path) -> None:
    cfg = {"data": {"input_mode": "unknown"}}
    assert _resolve_data_inputs(cfg, out_dir=tmp_path) is cfg

    cfg_missing_pairs = {"data": {"input_mode": "analysis_meta"}}
    with pytest.raises(KeyError):
        _resolve_data_inputs(cfg_missing_pairs, out_dir=tmp_path)


def test_run_once_smoke_with_stubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prices_path = tmp_path / "prices.parquet"
    pairs_path = tmp_path / "pairs.csv"
    prices_path.write_text("x", encoding="utf-8")
    pairs_path.write_text("y", encoding="utf-8")

    cfg_eff = {
        "data": {"prices_path": str(prices_path), "pairs_path": str(pairs_path)},
        "execution": {"mode": "lob"},
        "backtest": {
            "initial_capital": 1000.0,
            "splits": {
                "train": {"start": "2024-01-01", "end": "2024-01-03"},
                "test": {"start": "2024-01-04", "end": "2024-01-05"},
            },
        },
        "pair_prefilter": {"prefilter_active": False},
        "borrow": {"enabled": True},
        "reporting": {"mode": "core", "test_tearsheet": {"enabled": False}},
    }

    prices = runner.pd.DataFrame(
        {"AAA": [10.0, 10.1, 10.2]},
        index=runner.pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    pairs = {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}

    pairs_data = {
        "AAA-BBB": {
            "prices": runner.pd.DataFrame(
                {"y": [10.0, 10.1, 10.2], "x": [20.0, 20.1, 20.2]},
                index=prices.index,
            ),
            "meta": {"t1": "AAA", "t2": "BBB"},
        }
    }

    def _fake_prepare_pairs_data_for_cfg(**_kwargs):
        return pairs_data

    def _fake_execute_window_backtest(**_kwargs):
        stats = runner.pd.DataFrame(
            {
                "date": runner.pd.date_range("2024-01-01", periods=2, freq="D"),
                "equity": [1000.0, 1010.0],
                "returns": [0.0, 0.01],
                "drawdown": [0.0, -1.0],
                "drawdown_pct": [0.0, -0.001],
            }
        )
        stats.attrs["exec_rejected_count"] = 0
        stats.attrs["exec_reject_reasons"] = {}
        trades = runner.pd.DataFrame(
            {
                "borrow_cost": [-1.0, -2.0],
                "gross_pnl": [10.0, -5.0],
                "net_pnl": [9.0, -7.0],
            }
        )
        orders = runner.pd.DataFrame(
            {
                "dt": [runner.pd.Timestamp("2024-01-02")],
                "symbol": ["AAA"],
                "side": ["BUY"],
                "qty": [1],
                "price": [10.0],
            }
        )
        return types.SimpleNamespace(
            borrow_ctx=type("BC", (), {"enabled": True})(),
            pairs_data=pairs_data,
            stats=stats,
            trades=trades,
            raw_trades=runner.pd.DataFrame({"pair": ["AAA-BBB"]}),
            orders=orders,
        )

    def _fake_summary(_eq, trades_df=None, **_kwargs):
        return runner.pd.DataFrame([{"sharpe": 1.0, "cagr": 0.1}])

    monkeypatch.setattr(
        runner, "prepare_pairs_data_for_cfg", _fake_prepare_pairs_data_for_cfg
    )
    monkeypatch.setattr(
        runner,
        "run_bo_if_enabled",
        lambda **kwargs: types.SimpleNamespace(
            cfg_eff=kwargs["cfg_eff"],
            pairs_data=kwargs.get("pairs_data"),
            bo_out=None,
            bo_res=None,
            bo_meta=None,
        ),
    )
    monkeypatch.setattr(
        runner, "execute_window_backtest", _fake_execute_window_backtest
    )
    monkeypatch.setattr(runner, "summarize_stats", _fake_summary)

    out = runner._run_once(
        cfg_eff=cfg_eff,
        out_dir=tmp_path / "out",
        prices=prices,
        prices_panel=None,
        pairs=pairs,
        adv_map=None,
        quick=False,
    )
    assert isinstance(out, runner.SingleRunArtifacts)
    assert out.test_summary["sharpe"] == 1.0
