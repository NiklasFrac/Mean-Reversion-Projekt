import argparse
import importlib
import runpy
import sys
from pathlib import Path

import pytest


def _reset_runner_module() -> None:
    """Remove runner module to allow clean re-imports."""
    sys.modules.pop("analysis.runner_analysis", None)


def _args_namespace(**kwargs: object) -> argparse.Namespace:
    defaults = dict(
        cfg=None,
        quick=False,
        override_prices=None,
        override_out=None,
        n_jobs=None,
        n_resamples=None,
        block_size=None,
        train_cutoff_local=None,
        train_cutoff_utc=None,
        min_positive_frac=None,
        max_nan_frac_cols=None,
        drop_policy_rows=None,
        dry_run=False,
        log_level="INFO",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_import_runner_analysis(monkeypatch):
    _reset_runner_module()
    ra = importlib.import_module("analysis.runner_analysis")
    assert callable(ra.da_main)


def test_resolve_full_yaml_prefers_local_config(monkeypatch, tmp_path: Path):
    import analysis.runner_analysis as ra

    monkeypatch.delenv("BACKTEST_CONFIG", raising=False)
    monkeypatch.delenv("STRAT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "runs" / "configs"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text("data:\n  pairs_path: local\n", encoding="utf-8")

    cfg, src = ra.resolve_full_yaml(None)
    assert cfg["data"]["pairs_path"] == "local"
    assert str(cfg_file) in src


def test_resolve_full_yaml_uses_loader(monkeypatch, tmp_path: Path):
    import analysis.runner_analysis as ra

    monkeypatch.delenv("BACKTEST_CONFIG", raising=False)
    monkeypatch.delenv("STRAT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ra, "da_load_config", lambda arg: {"from": "loader"})
    monkeypatch.setattr(ra.Path, "exists", lambda self: False)

    cfg, src = ra.resolve_full_yaml(None)
    assert cfg == {"from": "loader"}
    assert src == "<analysis.load_config()>"


def test_resolve_full_yaml_raises_when_missing(monkeypatch, tmp_path: Path):
    import analysis.runner_analysis as ra

    monkeypatch.delenv("BACKTEST_CONFIG", raising=False)
    monkeypatch.delenv("STRAT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ra.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        ra,
        "da_load_config",
        lambda arg: (_ for _ in ()).throw(FileNotFoundError("no cfg")),
    )
    with pytest.raises(FileNotFoundError):
        ra.resolve_full_yaml(None)


def test_resolve_full_yaml_invalid_cfg_path(tmp_path: Path):
    import analysis.runner_analysis as ra

    with pytest.raises(FileNotFoundError):
        ra.resolve_full_yaml(str(tmp_path / "missing.yaml"))


def test_main_returns_error_when_config_lookup_fails(monkeypatch):
    import analysis.runner_analysis as ra

    args = _args_namespace()
    monkeypatch.setattr(ra, "parse_args", lambda: args)
    monkeypatch.setattr(
        ra,
        "resolve_full_yaml",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("cfg fail")),
    )
    rc = ra.main()
    assert rc == 2


def test_main_handles_da_failure_and_cleanup(monkeypatch):
    import analysis.runner_analysis as ra

    args = _args_namespace(quick=True)
    monkeypatch.setattr(ra, "parse_args", lambda: args)
    monkeypatch.setattr(ra, "resolve_full_yaml", lambda cfg: ({"data": {}}, "dummy"))

    monkeypatch.setattr(
        ra, "da_main", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    rc = ra.main()
    assert rc == 2


def test_runner_analysis_entrypoint(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("BACKTEST_CONFIG", str(cfg))
    monkeypatch.delenv("STRAT_CONFIG", raising=False)

    script = Path(__file__).resolve().parents[2] / "analysis" / "runner_analysis.py"
    old_argv = sys.argv[:]
    sys.argv = ["runner_analysis.py", "--dry-run"]
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(script), run_name="__main__")
        assert exc.value.code == 0
    finally:
        sys.argv = old_argv


def test_main_passes_merged_config_when_cfg_path_is_unresolved(
    monkeypatch, tmp_path: Path
):
    import analysis.runner_analysis as ra

    args = _args_namespace(override_out=str(tmp_path / "pairs_out.pkl"))
    base_cfg = {
        "data": {
            "prices_path": str(tmp_path / "prices.pkl"),
            "pairs_path": str(tmp_path / "pairs_base.pkl"),
        },
        "data_analysis": {"n_jobs": 1},
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(ra, "parse_args", lambda: args)
    monkeypatch.setattr(
        ra, "resolve_full_yaml", lambda cfg: (base_cfg, "<analysis.load_config()>")
    )
    monkeypatch.setattr(
        ra, "_resolve_cfg_path", lambda cfg: (None, "<analysis.load_config()>")
    )

    def fake_main(cfg_path, quick, overrides=None, **_kwargs):
        captured["cfg_path"] = cfg_path
        captured["quick"] = quick
        captured["overrides"] = overrides

    monkeypatch.setattr(ra, "da_main", fake_main)

    rc = ra.main()
    assert rc == 0
    assert captured["cfg_path"] is None
    assert captured["quick"] is False
    merged = captured["overrides"]
    assert isinstance(merged, dict)
    assert merged["data"]["prices_path"] == str(tmp_path / "prices.pkl")
    assert merged["data"]["pairs_path"] == str(tmp_path / "pairs_out.pkl")
