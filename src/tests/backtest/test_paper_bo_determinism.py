from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.optimize import paper_bo


class _DummyBayesianOptimization:
    def __init__(
        self,
        *,
        f: Any,
        pbounds: dict[str, tuple[float, float]],
        random_state: int,
        verbose: int,
        allow_duplicate_points: bool,
    ) -> None:
        _ = random_state
        _ = verbose
        _ = allow_duplicate_points
        self._f = f
        self._keys = sorted(pbounds.keys(), key=str)
        self.res: list[dict[str, Any]] = []
        self.max: dict[str, Any] = {}

    def _refresh_max(self) -> None:
        valid = [r for r in self.res if isinstance(r.get("target"), (int, float))]
        self.max = max(valid, key=lambda r: float(r["target"])) if valid else {}

    def register(self, *, params: dict[str, float], target: float | None) -> None:
        self.res.append(
            {
                "params": dict(params),
                "target": (None if target is None else float(target)),
            }
        )
        self._refresh_max()

    def maximize(self, *, init_points: int, n_iter: int) -> None:
        total = int(init_points) + int(n_iter)
        if not self._keys:
            return
        k0 = self._keys[0]
        for _ in range(total):
            x = float(len(self.res))
            params = {k0: x}
            target = float(self._f(**params))
            self.res.append({"params": params, "target": target})
        self._refresh_max()


def test_bayes_optimize_is_idempotent_after_budget_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paper_bo, "BayesianOptimization", _DummyBayesianOptimization)
    monkeypatch.setattr(paper_bo, "_BAYES_OK", True)

    def objective(x: float) -> float:
        return -abs(float(x) - 2.0)

    best_1, score_1 = paper_bo._bayes_optimize(
        out_dir=tmp_path,
        stage="theta_sig",
        pbounds={"x": (0.0, 10.0)},
        objective=objective,
        seed=7,
        init_points=2,
        n_iter=3,
    )
    trials_1 = paper_bo._load_trials_json(tmp_path, "theta_sig")
    assert len(trials_1) == 5

    best_2, score_2 = paper_bo._bayes_optimize(
        out_dir=tmp_path,
        stage="theta_sig",
        pbounds={"x": (0.0, 10.0)},
        objective=objective,
        seed=7,
        init_points=2,
        n_iter=3,
    )
    trials_2 = paper_bo._load_trials_json(tmp_path, "theta_sig")
    assert len(trials_2) == 5
    assert best_1 == best_2
    assert score_1 == score_2


def test_parse_bo_cv_reads_mode_local_cfg() -> None:
    cfg = {
        "bo": {
            "mode": "fast",
            "fast": {
                "cv": {
                    "enabled": True,
                    "scheme": "blocked",
                    "n_blocks": 5,
                    "k_test_blocks": 3,
                }
            },
            "realistic": {"cv": {"enabled": False, "scheme": "cpcv"}},
        }
    }
    fast_cv = paper_bo._parse_bo_cv(cfg, mode="fast")
    realistic_cv = paper_bo._parse_bo_cv(cfg, mode="realistic")
    assert fast_cv.enabled is True
    assert fast_cv.scheme == "blocked"
    assert realistic_cv.enabled is False
    assert realistic_cv.scheme == "cpcv"


def test_parse_bo_mode_rejects_removed_legacy_values() -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        paper_bo._parse_bo_mode({"bo": {"mode": "lob_rescore"}})
    with pytest.raises(ValueError, match="no longer supported"):
        paper_bo._parse_bo_mode({"bo": {"mode": "lob"}})
    with pytest.raises(ValueError, match="no longer supported"):
        paper_bo._parse_bo_mode({"bo": {"mode": "expensive"}})


def test_parse_bo_cv_rejects_removed_global_keys() -> None:
    with pytest.raises(ValueError, match="bo.cv"):
        paper_bo._parse_bo_cv(
            {"bo": {"mode": "fast", "cv": {"enabled": True}}}, mode="fast"
        )
    with pytest.raises(ValueError, match="bo.rescore"):
        paper_bo._parse_bo_cv(
            {"bo": {"mode": "realistic", "rescore": {"top_k": 5}}}, mode="realistic"
        )


def test_parse_realistic_cfg_reads_metric() -> None:
    cfg = {"bo": {"mode": "realistic", "realistic": {"metric": "calmar"}}}
    realistic_cfg = paper_bo._parse_realistic_cfg(cfg)
    assert realistic_cfg.metric == "calmar"


def test_run_paper_bo_rejects_legacy_stage_cfg(tmp_path: Path) -> None:
    idx = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame({"AAA": range(6), "BBB": range(1, 7)}, index=idx, dtype=float)
    cfg = {
        "backtest": {
            "splits": {
                "train": {"start": str(idx[0]), "end": str(idx[-2])},
                "test": {"start": str(idx[-1]), "end": str(idx[-1])},
            }
        },
        "bo": {"mode": "fast", "stage1": {"z_window_range": [10, 10]}},
    }
    with pytest.raises(ValueError, match="Legacy BO config keys"):
        paper_bo.run_paper_bo_conservative(
            prices=prices,
            prices_panel=None,
            pairs={"AAA-BBB": {"t1": "AAA", "t2": "BBB"}},
            pairs_data=None,
            cfg=cfg,
            out_dir=tmp_path,
        )


def test_run_paper_bo_realistic_requires_prices_panel(tmp_path: Path) -> None:
    idx = pd.bdate_range("2024-01-02", periods=30)
    prices = pd.DataFrame(
        {"AAA": range(30), "BBB": range(1, 31)}, index=idx, dtype=float
    )
    pairs_data = {
        "AAA-BBB": {
            "prices": pd.DataFrame({"y": prices["AAA"], "x": prices["BBB"]}),
            "meta": {"t1": "AAA", "t2": "BBB"},
        }
    }
    cfg = {
        "backtest": {
            "initial_capital": 1000.0,
            "splits": {
                "train": {"start": str(idx[0]), "end": str(idx[-5])},
                "test": {"start": str(idx[-4]), "end": str(idx[-1])},
            },
        },
        "pair_prefilter": {"prefilter_active": False},
        "signal": {
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_z": 3.0,
            "max_hold_days": 10,
            "cooldown_days": 0,
        },
        "spread_zscore": {"z_window": 10},
        "bo": {
            "mode": "realistic",
            "realistic": {"metric": "sharpe", "cv": {"enabled": False}},
            "entry_z_range": [2.0, 2.0],
            "exit_z_range": [0.5, 0.5],
            "stop_z_range": [3.0, 3.0],
        },
    }
    with pytest.raises(ValueError, match="prices_panel"):
        paper_bo.run_paper_bo_conservative(
            prices=prices,
            prices_panel=None,
            pairs={"AAA-BBB": {"t1": "AAA", "t2": "BBB"}},
            pairs_data=pairs_data,
            cfg=cfg,
            out_dir=tmp_path,
        )


def test_run_paper_bo_realistic_omits_legacy_rescore_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backtest.optimize.paper_bo_parts import pipeline as pipeline_mod

    idx = pd.bdate_range("2024-01-02", periods=30)
    prices = pd.DataFrame(
        {"AAA": range(30), "BBB": range(1, 31)}, index=idx, dtype=float
    )
    panel = pd.concat(
        {
            ("AAA", "close"): prices["AAA"],
            ("BBB", "close"): prices["BBB"],
        },
        axis=1,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["symbol", "field"])
    pairs_data = {
        "AAA-BBB": {
            "prices": pd.DataFrame({"y": prices["AAA"], "x": prices["BBB"]}),
            "meta": {"t1": "AAA", "t2": "BBB"},
        }
    }
    cfg = {
        "backtest": {
            "initial_capital": 1000.0,
            "splits": {
                "train": {"start": str(idx[0]), "end": str(idx[-5])},
                "test": {"start": str(idx[-4]), "end": str(idx[-1])},
            },
        },
        "pair_prefilter": {"prefilter_active": False},
        "signal": {
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_z": 3.0,
            "max_hold_days": 10,
            "cooldown_days": 0,
        },
        "spread_zscore": {"z_window": 10},
        "bo": {
            "mode": "realistic",
            "realistic": {"metric": "sharpe", "cv": {"enabled": False}},
            "entry_z_range": [2.0, 2.0],
            "exit_z_range": [0.5, 0.5],
            "stop_z_range": [3.0, 3.0],
        },
    }

    monkeypatch.setattr(pipeline_mod, "_fold_score_realistic", lambda **_kwargs: 1.25)
    res = paper_bo.run_paper_bo_conservative(
        prices=prices,
        prices_panel=panel,
        pairs={"AAA-BBB": {"t1": "AAA", "t2": "BBB"}},
        pairs_data=pairs_data,
        cfg=cfg,
        out_dir=tmp_path,
    )
    assert res["meta"]["mode"] == "realistic"
    assert "rescore" not in res


def test_apply_bo_params_to_cfg_patches_markov_fields() -> None:
    cfg = {
        "signal": {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.0},
        "markov_filter": {"enabled": True, "min_revert_prob": 0.4, "horizon_days": 5},
    }

    out = paper_bo.apply_bo_params_to_cfg(
        cfg,
        {
            "theta_sig_hat": {"entry_z": 1.8, "exit_z": 0.4, "stop_z": 2.8},
            "theta_markov_hat": {"min_revert_prob": 0.7, "horizon_days": 9.4},
        },
    )

    assert out["signal"]["entry_z"] == pytest.approx(1.8)
    assert out["signal"]["exit_z"] == pytest.approx(0.4)
    assert out["signal"]["stop_z"] == pytest.approx(2.8)
    assert out["markov_filter"]["min_revert_prob"] == pytest.approx(0.7)
    assert out["markov_filter"]["horizon_days"] == 9


def test_run_paper_bo_realistic_runs_markov_stage2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backtest.optimize.paper_bo_parts import pipeline as pipeline_mod

    idx = pd.bdate_range("2024-01-02", periods=30)
    prices = pd.DataFrame(
        {"AAA": range(30), "BBB": range(1, 31)}, index=idx, dtype=float
    )
    panel = pd.concat(
        {
            ("AAA", "close"): prices["AAA"],
            ("BBB", "close"): prices["BBB"],
        },
        axis=1,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["symbol", "field"])
    pairs_data = {
        "AAA-BBB": {
            "prices": pd.DataFrame({"y": prices["AAA"], "x": prices["BBB"]}),
            "meta": {"t1": "AAA", "t2": "BBB"},
        }
    }
    cfg = {
        "backtest": {
            "initial_capital": 1000.0,
            "splits": {
                "train": {"start": str(idx[0]), "end": str(idx[-5])},
                "test": {"start": str(idx[-4]), "end": str(idx[-1])},
            },
        },
        "pair_prefilter": {"prefilter_active": False},
        "signal": {
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_z": 3.0,
            "max_hold_days": 10,
            "cooldown_days": 0,
        },
        "markov_filter": {"enabled": True, "min_revert_prob": 0.55, "horizon_days": 10},
        "spread_zscore": {"z_window": 10},
        "bo": {
            "mode": "realistic",
            "realistic": {"metric": "sharpe", "cv": {"enabled": False}},
            "entry_z_range": [2.0, 2.0],
            "exit_z_range": [0.5, 0.5],
            "stop_z_range": [3.0, 3.0],
            "min_revert_prob_range": [0.65, 0.65],
            "horizon_days_range": [7.6, 7.6],
        },
    }
    seen: list[tuple[str, dict[str, Any]]] = []

    def fake_score(**kwargs: Any) -> float:
        seen.append((str(kwargs["component"]), dict(kwargs["theta"])))
        return 1.25 if kwargs["component"] == "theta_sig" else 2.5

    monkeypatch.setattr(pipeline_mod, "_fold_score_realistic", fake_score)

    res = paper_bo.run_paper_bo_conservative(
        prices=prices,
        prices_panel=panel,
        pairs={"AAA-BBB": {"t1": "AAA", "t2": "BBB"}},
        pairs_data=pairs_data,
        cfg=cfg,
        out_dir=tmp_path,
    )

    assert res["theta_sig_hat"] == {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.0}
    assert res["theta_markov_hat"] == {"min_revert_prob": 0.65, "horizon_days": 8}
    assert res["theta_sig_score"] == pytest.approx(1.25)
    assert res["theta_markov_score"] == pytest.approx(2.5)
    assert res["score"] == pytest.approx(2.5)
    assert res["meta"]["selected_component"] == "theta_markov"
    assert seen[0][0] == "theta_sig"
    assert seen[1][0] == "theta_markov"
    assert seen[1][1]["theta_sig_hat"] == res["theta_sig_hat"]
    assert seen[1][1]["theta_markov_hat"] == res["theta_markov_hat"]


def test_run_paper_bo_fast_rejects_markov_stage2(tmp_path: Path) -> None:
    idx = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.DataFrame({"AAA": range(8), "BBB": range(1, 9)}, index=idx, dtype=float)
    cfg = {
        "backtest": {
            "splits": {
                "train": {"start": str(idx[0]), "end": str(idx[-2])},
                "test": {"start": str(idx[-1]), "end": str(idx[-1])},
            }
        },
        "markov_filter": {"enabled": True, "min_revert_prob": 0.55, "horizon_days": 10},
        "bo": {
            "mode": "fast",
            "entry_z_range": [2.0, 2.0],
            "exit_z_range": [0.5, 0.5],
            "stop_z_range": [3.0, 3.0],
        },
    }

    with pytest.raises(ValueError, match="Markov BO requires bo.mode='realistic'"):
        paper_bo.run_paper_bo_conservative(
            prices=prices,
            prices_panel=None,
            pairs={"AAA-BBB": {"t1": "AAA", "t2": "BBB"}},
            pairs_data=None,
            cfg=cfg,
            out_dir=tmp_path,
        )
