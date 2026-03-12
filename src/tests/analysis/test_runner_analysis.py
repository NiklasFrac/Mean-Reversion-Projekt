import argparse
import json
import sys
from pathlib import Path

import analysis.runner_analysis as ra


def test_build_overrides_merges_all(tmp_path: Path):
    ns = argparse.Namespace(
        override_prices=str(tmp_path / "p.pkl"),
        override_out=str(tmp_path / "q.pkl"),
        n_jobs=3,
        n_resamples=77,
        block_size=9,
        disable_bootstrap=False,
        disable_hypothesis_test=False,
        disable_fdr=False,
        train_cutoff_local="2020-01-01",
        train_cutoff_utc=None,
        min_positive_frac=0.9,
        max_nan_frac_cols=0.1,
        drop_policy_rows="any",
    )
    ov = ra.build_overrides(ns)
    assert ov["data"]["prices_path"].endswith("p.pkl")
    assert ov["data"]["pairs_path"].endswith("q.pkl")
    assert ov["data_analysis"]["n_jobs"] == 3
    assert ov["data_analysis"]["bootstrap"]["n_resamples"] == 77
    assert ov["data_analysis"]["bootstrap"]["block_size"] == 9
    assert ov["data_analysis"]["train_cutoff_local"] == "2020-01-01"
    rc = ov["data_analysis"]["returns_cleaning"]
    assert (
        rc["min_positive_frac"] == 0.9
        and rc["max_nan_frac_cols"] == 0.1
        and rc["drop_policy_rows"] == "any"
    )


def test_build_overrides_accepts_legacy_train_cutoff_arg() -> None:
    ns = argparse.Namespace(
        override_prices=None,
        override_out=None,
        n_jobs=None,
        n_resamples=None,
        block_size=None,
        disable_bootstrap=False,
        disable_hypothesis_test=False,
        disable_fdr=False,
        train_cutoff_local=None,
        train_cutoff_utc="2020-01-01T00:00:00Z",
        min_positive_frac=None,
        max_nan_frac_cols=None,
        drop_policy_rows=None,
        max_candidates=None,
        disable_max_candidates=False,
    )
    ov = ra.build_overrides(ns)
    assert ov["data_analysis"]["train_cutoff_local"] == "2020-01-01T00:00:00Z"


def test_build_overrides_accepts_significance_switches() -> None:
    ns = argparse.Namespace(
        override_prices=None,
        override_out=None,
        n_jobs=None,
        n_resamples=None,
        block_size=None,
        disable_bootstrap=True,
        disable_hypothesis_test=True,
        disable_fdr=True,
        train_cutoff_local=None,
        train_cutoff_utc=None,
        min_positive_frac=None,
        max_nan_frac_cols=None,
        drop_policy_rows=None,
        max_candidates=None,
        disable_max_candidates=False,
    )
    ov = ra.build_overrides(ns)
    da = ov["data_analysis"]
    assert da["enable_bootstrap"] is False
    assert da["enable_hypothesis_test"] is False
    assert da["enable_fdr"] is False


def test_resolve_full_yaml_env(monkeypatch, tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps({"data": {"pairs_path": "x"}}), encoding="utf-8")
    monkeypatch.setenv("BACKTEST_CONFIG", str(cfg_path))
    base, src = ra.resolve_full_yaml(None)
    assert "data" in base and "pairs_path" in base["data"]
    assert "BACKTEST_CONFIG" in src


def test_runner_dry_run(monkeypatch, tmp_path: Path, caplog):
    # fake da_main so nothing heavy runs
    called = {"n": 0}

    def fake_main(**kwargs):
        called["n"] += 1

    monkeypatch.setattr(ra, "da_main", fake_main)

    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("BACKTEST_CONFIG", str(cfgp))
    monkeypatch.setattr(sys, "argv", ["runner", "--dry-run"])

    rc = ra.main()
    assert rc == 0
    # dry-run: da_main darf NICHT aufgerufen worden sein
    assert called["n"] == 0


def test_runner_invokes_main_with_cfg_and_overrides(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_main(cfg_path, quick, overrides=None, **_kwargs):
        assert Path(cfg_path).exists()
        captured["quick"] = quick
        captured["overrides"] = overrides or {}

    monkeypatch.setattr(ra, "da_main", fake_main)
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(
        json.dumps({"data": {"pairs_path": str(tmp_path / "pairs.pkl")}}),
        encoding="utf-8",
    )

    # Pass --cfg directly
    monkeypatch.setattr(sys, "argv", ["runner", "--cfg", str(cfgp), "--quick"])
    rc = ra.main()
    assert rc == 0 and captured.get("quick") is True
    assert captured.get("overrides") == {}
