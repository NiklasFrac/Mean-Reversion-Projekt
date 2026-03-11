import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest import runner_backtest as rb
from backtest.run.bo_runner import BORunResult
from backtest.utils.run.bo import _bo_key_payload, _sanitize_bo_cfg_for_key
from backtest.utils.run.data import _pair_prefilter_inputs
from backtest.utils.run.io import _deep_merge, _file_fingerprint
from backtest.run.runtime import RuntimeContext


def _make_price_panel(idx: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    data: dict[tuple[str, str], np.ndarray] = {}
    base = np.linspace(100.0, 110.0, len(idx))
    for i, sym in enumerate(symbols):
        drift = base + (i * 2.0)
        data[(sym, "open")] = drift + 0.1
        data[(sym, "high")] = drift + 0.5
        data[(sym, "low")] = drift - 0.4
        data[(sym, "close")] = drift + 0.2
        data[(sym, "volume")] = np.full(len(idx), 1000 + i * 10, dtype=float)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


def _write_pairs_csv(path: Path, t1: str, t2: str) -> None:
    pd.DataFrame([{"t1": t1, "t2": t2}]).to_csv(path, index=False)


def _base_cfg(
    prices_path: Path, pairs_path: Path, adv_path: Path | None = None
) -> dict:
    cfg = {
        "seed": 7,
        "data": {
            "prices_path": str(prices_path),
            "pairs_path": str(pairs_path),
            "prefer_col": "close",
        },
        "pair_prefilter": {"prefilter_active": False},
        "backtest": {
            "initial_capital": 100000.0,
            "splits": {
                "train": {"start": "2024-01-02", "end": "2024-02-15"},
                "test": {"start": "2024-02-16", "end": "2024-03-15"},
            },
        },
        "execution": {
            "mode": "lob",
            "lob": {
                "tick": 0.01,
                "levels": 3,
                "size_per_level": 1000,
                "min_spread_ticks": 1,
                "lam": 1.0,
                "max_add": 10,
                "bias_top": 0.7,
                "cancel_prob": 0.1,
                "max_cancel": 10,
                "steps_per_day": 1,
                "post_costs": {
                    "per_trade": 0.0,
                    "maker_bps": -0.1,
                    "taker_bps": 0.2,
                    "min_fee": 0.0,
                },
            },
        },
        "borrow": {"enabled": False},
        "reporting": {"mode": "core", "test_tearsheet": {"enabled": False}},
    }
    if adv_path is not None:
        cfg["data"]["adv_map_path"] = str(adv_path)
    return cfg


def _empty_bo(cfg_eff: dict | None = None) -> BORunResult:
    return BORunResult(
        cfg_eff=cfg_eff or {},
        pairs_data=None,
        bo_res=None,
        bo_id=None,
        bo_out=None,
        bo_key_payload=None,
        selected_cv_scores=pd.DataFrame(
            columns=["fold_id", "score", "selection_metric", "component"]
        ),
        selection_metric=None,
    )


def _fake_artifact(
    *, cfg_eff: dict | None = None, n_pairs: int = 1, n_trades: int = 1
) -> rb.SingleRunArtifacts:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    eq = pd.Series([100.0, 101.0], index=idx, name="equity")
    stats = pd.DataFrame({"date": idx, "equity": eq.values})
    trades = pd.DataFrame(
        {
            "pair": ["AAA-BBB"] * n_trades,
            "entry_date": ["2024-01-02"] * n_trades,
            "exit_date": ["2024-01-03"] * n_trades,
            "gross_pnl": [1.0] * n_trades,
            "net_pnl": [1.0] * n_trades,
        }
    )
    return rb.SingleRunArtifacts(
        cfg_eff=cfg_eff or {"execution": {"mode": "lob"}},
        bo_run=_empty_bo(cfg_eff),
        pairs_data={f"PAIR-{i}": {} for i in range(n_pairs)},
        borrow_ctx=None,
        stats=stats,
        trades=trades,
        raw_trades=trades.copy(),
        orders=pd.DataFrame(),
        test_equity=eq,
        test_summary={
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
            "n_days": 2,
            "total_return": 0.01,
            "cagr": 0.1,
            "ann_return": 0.1,
            "ann_vol": 0.1,
            "sharpe": 1.0,
            "sortino": 1.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "hit_rate": 1.0,
            "num_trades": n_trades,
        },
        train_refit=None,
    )


def test_deep_merge_overrides_nested() -> None:
    a = {"a": {"b": 1, "c": 2}, "x": 1}
    b = {"a": {"b": 3}, "y": 2}
    out = _deep_merge(a, b)
    assert out == {"a": {"b": 3, "c": 2}, "x": 1, "y": 2}


def test_file_fingerprint_with_missing_and_existing(tmp_path: Path) -> None:
    missing = _file_fingerprint(tmp_path / "nope.txt")
    assert missing == {"path": str(tmp_path / "nope.txt")}

    p = tmp_path / "f.txt"
    p.write_text("abc", encoding="utf-8")
    fp = _file_fingerprint(p)
    assert fp["path"] == str(p)
    assert fp["size"] == 3
    assert isinstance(fp["mtime"], int)


def test_sanitize_bo_cfg_for_key_strips_runtime_fields() -> None:
    cfg = {
        "out_dir": "runs/bo",
        "n_iter": 5,
        "init_points": 2,
        "entry_z_range": [1.0, 2.0],
        "keep": 1,
    }
    out = _sanitize_bo_cfg_for_key(cfg)
    assert "out_dir" not in out
    assert "n_iter" not in out
    assert "init_points" not in out
    assert out["entry_z_range"] == [1.0, 2.0]
    assert out["keep"] == 1


def test_sanitize_bo_cfg_for_key_rejects_legacy_stage_keys() -> None:
    with pytest.raises(ValueError, match="Legacy BO config keys"):
        _sanitize_bo_cfg_for_key({"stage1": {"n_iter": 5}})


def test_bo_key_payload_includes_paths(tmp_path: Path) -> None:
    prices = tmp_path / "prices.pkl"
    pairs = tmp_path / "pairs.json"
    prices.write_text("x", encoding="utf-8")
    pairs.write_text("{}", encoding="utf-8")
    cfg = {
        "seed": 1,
        "data": {"prices_path": str(prices), "pairs_path": str(pairs)},
        "backtest": {"splits": {"train": {"start": "2024-01-01", "end": "2024-01-10"}}},
    }
    payload = _bo_key_payload(cfg)
    assert payload["data"]["prices"]["path"] == str(prices)
    assert payload["data"]["pairs"]["path"] == str(pairs)


def test_bo_key_payload_includes_markov_filter_and_new_bo_keys(tmp_path: Path) -> None:
    prices = tmp_path / "prices.pkl"
    pairs = tmp_path / "pairs.json"
    prices.write_text("x", encoding="utf-8")
    pairs.write_text("{}", encoding="utf-8")
    cfg = {
        "seed": 1,
        "data": {"prices_path": str(prices), "pairs_path": str(pairs)},
        "backtest": {"splits": {"train": {"start": "2024-01-01", "end": "2024-01-10"}}},
        "markov_filter": {"enabled": True, "min_revert_prob": 0.6, "horizon_days": 9},
        "bo": {
            "min_revert_prob_range": [0.5, 0.7],
            "horizon_days_range": [5, 10],
            "markov_init_points": 4,
            "markov_n_iter": 6,
            "markov_patience": 2,
        },
    }

    payload = _bo_key_payload(cfg)

    assert payload["version"] == 9
    assert payload["markov_filter"]["min_revert_prob"] == 0.6
    assert payload["bo"]["min_revert_prob_range"] == [0.5, 0.7]
    assert payload["bo"]["horizon_days_range"] == [5, 10]
    assert payload["bo"]["markov_init_points"] == 4
    assert payload["bo"]["markov_n_iter"] == 6
    assert payload["bo"]["markov_patience"] == 2


def test_bo_key_payload_normalizes_pair_prefilter_legacy_cfg() -> None:
    payload = _bo_key_payload({"pairs_prep": {"disable_prefilter": False}})
    assert "pairs_prep" not in payload
    assert payload["pair_prefilter"]["prefilter_active"] is True


def test_pair_prefilter_inputs_supports_new_and_legacy_keys() -> None:
    disable_prefilter, _ = _pair_prefilter_inputs(
        {"pair_prefilter": {"prefilter_active": True}}
    )
    assert disable_prefilter is False

    disable_prefilter, _ = _pair_prefilter_inputs(
        {"pair_prefilter": {"prefilter_active": False}}
    )
    assert disable_prefilter is True

    disable_prefilter, _ = _pair_prefilter_inputs(
        {"pairs_prep": {"disable_prefilter": False}}
    )
    assert disable_prefilter is False


def test_validate_cfg_strict_errors(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        rb._validate_cfg_strict({"data": {}})
    with pytest.raises(KeyError):
        rb._validate_cfg_strict({"data": {"prices_path": "x"}})

    prices = tmp_path / "prices.pkl"
    pairs = tmp_path / "pairs.json"
    prices.write_text("x", encoding="utf-8")
    pairs.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        rb._validate_cfg_strict(
            {
                "data": {"prices_path": str(prices), "pairs_path": str(pairs)},
                "execution": {"mode": "vectorized"},
            }
        )
    with pytest.raises(ValueError):
        rb._validate_cfg_strict(
            {
                "data": {"prices_path": str(prices), "pairs_path": str(pairs)},
                "execution": {"mode": "lob"},
                "backtest": {"window_end_policy": "force_exit"},
            }
        )
    with pytest.raises(ValueError):
        rb._validate_cfg_strict(
            {
                "data": {"prices_path": str(prices), "pairs_path": str(pairs)},
                "execution": {"mode": "lob"},
                "backtest": {"walkforward": {"carry_open_trades": True}},
            }
        )
    with pytest.raises(ValueError):
        rb._validate_cfg_strict(
            {
                "data": {"prices_path": str(prices), "pairs_path": str(pairs)},
                "execution": {"mode": "lob"},
                "signal": {"execution_lag_bars": 1},
            }
        )
    with pytest.raises(ValueError):
        rb._validate_cfg_strict(
            {
                "data": {"prices_path": str(prices), "pairs_path": str(pairs)},
                "execution": {"mode": "lob", "exec_lob": {"tick": 0.01}},
            }
        )
    with pytest.raises(ValueError):
        rb._validate_cfg_strict(
            {
                "data": {"prices_path": str(prices), "pairs_path": str(pairs)},
                "execution": {"mode": "lob", "lob": {"policy": {"post_only": True}}},
            }
        )
    with pytest.raises(ValueError):
        rb._validate_cfg_strict(
            {
                "data": {"prices_path": str(prices), "pairs_path": str(pairs)},
                "execution": {
                    "mode": "lob",
                    "lob": {"post_costs": {"borrow_bps": 0.0}},
                },
            }
        )


def test_build_strategy_rejects_params() -> None:
    cfg = {"strategy": {"name": "baseline", "params": {"nope": 1}}}
    with pytest.raises(TypeError):
        rb._build_strategy(cfg, borrow_ctx=None)


def test_build_strategy_unknown_name() -> None:
    cfg = {"strategy": {"name": "unknown_strat"}}
    with pytest.raises(KeyError):
        rb._build_strategy(cfg, borrow_ctx=None)


def test_run_backtest_smoke_baseline(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    idx = pd.bdate_range("2024-01-02", periods=80)
    panel = _make_price_panel(idx, ["AAA", "BBB"])
    prices_path = tmp_path / "panel.pkl"
    panel.to_pickle(prices_path)

    pairs_path = tmp_path / "pairs.csv"
    _write_pairs_csv(pairs_path, "AAA", "BBB")

    adv_path = tmp_path / "adv.csv"
    adv_path.write_text(
        "ticker;dollar_adv_hist\nAAA;1000\nBBB;2000\n", encoding="utf-8"
    )

    cfg = _base_cfg(prices_path, pairs_path, adv_path)
    cfg["strategy"] = {"name": "baseline"}
    cfg["signal"] = {"entry_z": 0.5, "exit_z": 0.1, "stop_z": 2.0, "max_hold_days": 5}
    cfg["spread_zscore"] = {"z_window": 5, "z_min_periods": 3}

    out_dir = tmp_path / "run"
    res = rb.run(cfg, out_dir=out_dir, quick=False)
    assert Path(res["out_dir"]).exists()
    assert (out_dir / "config_effective.json").exists()
    assert (out_dir / "report" / "test_summary.json").exists()
    assert (out_dir / "report" / "test_equity.csv").exists()
    assert (out_dir / "report" / "test_trades.csv").exists()

    summary = json.loads(
        (out_dir / "report" / "test_summary.json").read_text(encoding="utf-8")
    )
    assert summary.get("num_trades", 0) >= 0


def test_run_uses_resolved_calendar_and_prefer_col(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=20)
    panel = _make_price_panel(idx, ["AAA", "BBB"])
    prices_path = tmp_path / "panel.pkl"
    panel.to_pickle(prices_path)

    pairs_path = tmp_path / "pairs.csv"
    _write_pairs_csv(pairs_path, "AAA", "BBB")

    cfg = _base_cfg(prices_path, pairs_path)
    seen: dict[str, float | str] = {}

    def _fake_run_once(
        *, prices: pd.DataFrame, out_dir: Path, **_kwargs: object
    ) -> rb.SingleRunArtifacts:
        seen["aaa_first"] = float(prices["AAA"].iloc[0])
        return _fake_artifact(
            cfg_eff={"execution": {"mode": "lob"}}, n_pairs=1, n_trades=0
        )

    def _fake_runtime_loader(
        cfg_in: dict[str, object], *, out_dir: Path
    ) -> RuntimeContext:
        out = dict(cfg_in)
        data = dict(out.get("data", {}))
        data["prefer_col"] = "high"
        out["data"] = data
        seen["out_dir"] = str(out_dir)
        return RuntimeContext(
            cfg=out,
            out_dir=out_dir,
            data_cfg=data,
            prices_path=prices_path,
            pairs_path=pairs_path,
            calendar_name="XNYS",
            prefer_col="high",
            prices_panel=panel,
            prices=panel.xs("high", axis=1, level="field"),
            pairs={"AAA-BBB": {"t1": "AAA", "t2": "BBB"}},
            adv_map=None,
        )

    monkeypatch.setattr(rb, "load_runtime_context", _fake_runtime_loader)
    monkeypatch.setattr(rb, "_run_once", _fake_run_once)

    out_dir = tmp_path / "resolved_run"
    res = rb.run(cfg, out_dir=out_dir, quick=False)

    assert res["n_pairs"] == 1
    assert float(seen["aaa_first"]) == pytest.approx(
        float(panel[("AAA", "high")].iloc[0])
    )


def test_run_walkforward_stubbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=60)
    panel = _make_price_panel(idx, ["AAA", "BBB"])
    prices_path = tmp_path / "panel.pkl"
    panel.to_pickle(prices_path)

    pairs_path = tmp_path / "pairs.csv"
    _write_pairs_csv(pairs_path, "AAA", "BBB")

    cfg = _base_cfg(prices_path, pairs_path)
    cfg["backtest"]["range"] = {"start": "2024-01-02", "end": "2024-03-29"}
    cfg["backtest"]["walkforward"] = {
        "enabled": True,
        "train_mode": "expanding",
        "initial_train_months": 1,
        "test_months": 1,
        "step_months": 1,
    }

    def fake_run_once(*, cfg_eff, out_dir, **_kwargs):
        return _fake_artifact(cfg_eff=cfg_eff, n_pairs=1, n_trades=1)

    monkeypatch.setattr(rb, "_run_once", fake_run_once)
    monkeypatch.setattr(
        rb,
        "_apply_global_positions_ledger",
        lambda df: (df, pd.DataFrame(), {"kept": len(df), "blocked": 0}),
    )
    monkeypatch.setattr(
        rb, "rescale_trades_stateful", lambda df, **_k: (df, {"ok": True})
    )
    monkeypatch.setattr(
        rb,
        "backtest_portfolio_with_yaml_cfg",
        lambda **_k: (
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "equity": [100.0, 101.0],
                }
            ),
            pd.DataFrame({"pair": ["AAA-BBB"], "net_pnl": [1.0]}),
        ),
    )
    out_dir = tmp_path / "wf"
    res = rb.run(cfg, out_dir=out_dir, quick=False)
    assert res["n_windows"] >= 1
    assert (out_dir / "report" / "test_window_summary.csv").exists()


def test_run_walkforward_global_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=60)
    panel = _make_price_panel(idx, ["AAA", "BBB"])
    prices_path = tmp_path / "panel.pkl"
    panel.to_pickle(prices_path)

    pairs_path = tmp_path / "pairs.csv"
    _write_pairs_csv(pairs_path, "AAA", "BBB")

    cfg = _base_cfg(prices_path, pairs_path)
    cfg["reporting"] = {"mode": "core", "test_tearsheet": {"enabled": True, "dpi": 120}}
    cfg["backtest"]["range"] = {"start": "2024-01-02", "end": "2024-03-29"}
    cfg["backtest"]["walkforward"] = {
        "enabled": True,
        "train_mode": "expanding",
        "initial_train_months": 1,
        "test_months": 1,
        "step_months": 1,
    }

    def fake_run_once(*, out_dir, **_kwargs):
        return _fake_artifact(
            cfg_eff={"execution": {"mode": "lob"}}, n_pairs=1, n_trades=1
        )

    monkeypatch.setattr(rb, "_run_once", fake_run_once)
    monkeypatch.setattr(
        rb,
        "_apply_global_positions_ledger",
        lambda df: (df, pd.DataFrame(), {"kept": len(df), "blocked": 0}),
    )
    monkeypatch.setattr(
        rb, "rescale_trades_stateful", lambda df, **_k: (df, {"ok": True})
    )
    monkeypatch.setattr(
        rb,
        "backtest_portfolio_with_yaml_cfg",
        lambda **_k: (
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "equity": [100.0, 101.0],
                }
            ),
            pd.DataFrame({"pair": ["AAA-BBB"], "net_pnl": [1.0]}),
        ),
    )
    out_dir = tmp_path / "wf_global"
    res = rb.run(cfg, out_dir=out_dir, quick=False)
    assert res["n_windows"] >= 1
    assert (out_dir / "report" / "test_summary.json").exists()
    assert (out_dir / "report" / "test_equity.csv").exists()


def test_runner_backtest_main_success_and_missing_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = rb.main(["--cfg", str(tmp_path / "nope.yaml")])
    assert missing == 2

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "data:\n  prices_path: x\n  pairs_path: y\nexecution:\n  mode: lob\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(rb, "run", lambda *_a, **_k: {"out_dir": "x"})
    code = rb.main(["--cfg", str(cfg_path)])
    assert code == 0
