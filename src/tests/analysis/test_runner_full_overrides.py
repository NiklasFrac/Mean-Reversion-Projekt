import types
from pathlib import Path

import analysis.runner_analysis as ra


def test_runner_full_overrides(monkeypatch, tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("BACKTEST_CONFIG", str(base))

    called = {}

    def fake_main(cfg_path, quick, overrides=None, **_kwargs):
        assert Path(cfg_path).exists()
        called["quick"] = quick
        called["overrides"] = overrides or {}

    monkeypatch.setattr(ra, "da_main", fake_main)

    ns = types.SimpleNamespace(
        cfg=None,
        quick=True,
        override_prices=str(tmp_path / "in.pkl"),
        override_out=str(tmp_path / "out.pkl"),
        n_jobs=2,
        n_resamples=80,
        block_size=15,
        disable_bootstrap=False,
        disable_hypothesis_test=False,
        disable_fdr=False,
        train_cutoff_local="2020-02-01",
        train_cutoff_utc=None,
        min_positive_frac=0.9,
        max_nan_frac_cols=0.05,
        drop_policy_rows="any",
        dry_run=False,
        log_level="INFO",
    )
    monkeypatch.setattr(ra, "parse_args", lambda: ns)

    rc = ra.main()
    assert rc == 0
    assert called["quick"] is True
    ov = called["overrides"]
    assert ov["data"]["prices_path"].endswith("in.pkl")
    assert ov["data"]["pairs_path"].endswith("out.pkl")
    assert ov["data_analysis"]["n_jobs"] == 2
    assert ov["data_analysis"]["bootstrap"]["n_resamples"] == 80
    assert ov["data_analysis"]["bootstrap"]["block_size"] == 15
    assert ov["data_analysis"]["train_cutoff_local"] == "2020-02-01"
    rc_clean = ov["data_analysis"]["returns_cleaning"]
    assert rc_clean["min_positive_frac"] == 0.9
    assert rc_clean["max_nan_frac_cols"] == 0.05
    assert rc_clean["drop_policy_rows"] == "any"
