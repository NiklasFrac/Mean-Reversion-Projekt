from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from universe import reset_universe as ru


def _write_cfg(path: Path, *, screener_glob: str = "runs/data/screener*.csv") -> None:
    cfg = {
        "universe": {
            "output_tickers_csv": "runs/data/tickers.csv",
            "output_tickers_ext_csv": "runs/data/tickers_ext.csv",
            "manifest": "runs/data/universe_manifest.json",
            "fundamentals_out": "runs/data/fundamentals.parquet",
            "adv_cache": "runs/data/ticker_adv.pkl",
            "screener_glob": screener_glob,
        },
        "runtime": {
            "checkpoint_path": "runs/data/universe_checkpoint.json",
            "run_scoped_outputs_dir": "runs/data/by_run",
        },
        "data": {
            "adv_path": "runs/data/adv_map_usd.csv",
            "raw_prices_cache": "runs/data/raw_prices.pkl",
            "volume_path": "runs/data/raw_volume.pkl",
            "raw_prices_unadj_warmup_cache": "runs/data/raw_prices_unadj_warmup.pkl",
            "raw_prices_unadj_cache": "runs/data/raw_prices_unadj.pkl",
            "raw_volume_unadj_cache": "runs/data/raw_volume_unadj.pkl",
        },
        "logging": {"file": {"enabled": True, "path": "runs/logs/universe.log"}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def test_gather_targets_includes_hashed_siblings_and_optional_screener(tmp_path: Path):
    root = tmp_path
    (root / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    cfg_path = root / "runs/configs/test_cfg.yaml"
    _write_cfg(cfg_path)

    runs_data = root / "runs/data"
    runs_data.mkdir(parents=True, exist_ok=True)
    (runs_data / "raw_prices.pkl").write_bytes(b"px")
    (runs_data / "raw_prices.abc123.pkl").write_bytes(b"pxh")
    (runs_data / "screener_a.csv").write_text("Symbol\nAAPL\n", encoding="utf-8")

    keep_targets = ru.gather_universe_reset_targets(
        root=root,
        cfg_path=cfg_path,
        keep_screener=True,
        dev_caches=False,
        purge_logs_and_tmp=False,
    )
    purge_targets = ru.gather_universe_reset_targets(
        root=root,
        cfg_path=cfg_path,
        keep_screener=False,
        dev_caches=False,
        purge_logs_and_tmp=False,
    )

    keep_paths = {t.path.resolve() for t in keep_targets}
    purge_paths = {t.path.resolve() for t in purge_targets}

    assert (runs_data / "raw_prices.abc123.pkl").resolve() in keep_paths
    assert (runs_data / "screener_a.csv").resolve() not in keep_paths
    assert (runs_data / "screener_a.csv").resolve() in purge_paths

    # data.adv_path should appear exactly once after de-dup.
    adv_matches = [
        t
        for t in purge_targets
        if t.path.resolve() == (runs_data / "adv_map_usd.csv").resolve()
    ]
    assert len(adv_matches) == 1


def test_delete_target_respects_repo_boundary_and_dry_run(tmp_path: Path):
    root = tmp_path
    inside = root / "runs/data/x.txt"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_text("x", encoding="utf-8")

    did, msg = ru._delete_target(
        root, ru.ResetTarget(inside, "file", "test"), yes=False
    )
    assert not did
    assert "DRY-RUN" in msg
    assert inside.exists()

    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    try:
        did, msg = ru._delete_target(
            root, ru.ResetTarget(outside, "file", "outside"), yes=True
        )
        assert not did
        assert "outside repo" in msg
        assert outside.exists()
    finally:
        outside.unlink(missing_ok=True)


def test_delete_target_removes_file_and_directory(tmp_path: Path):
    root = tmp_path
    f = root / "runs/data/a.txt"
    d = root / "runs/tmp/cache"
    f.parent.mkdir(parents=True, exist_ok=True)
    d.mkdir(parents=True, exist_ok=True)
    f.write_text("x", encoding="utf-8")
    (d / "b.txt").write_text("y", encoding="utf-8")

    did_f, _ = ru._delete_target(root, ru.ResetTarget(f, "file", "file"), yes=True)
    did_d, _ = ru._delete_target(root, ru.ResetTarget(d, "dir", "dir"), yes=True)

    assert did_f and did_d
    assert not f.exists()
    assert not d.exists()


def test_main_default_executes_deletions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root = tmp_path
    (root / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    cfg_path = root / "runs/configs/test_cfg.yaml"
    _write_cfg(cfg_path)
    victim = root / "runs/data/tickers.csv"
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("x", encoding="utf-8")

    rc = ru.main(["--root", str(root), "--cfg", str(cfg_path)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Done. Deleted:" in out
    assert not victim.exists()


def test_main_dry_run_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root = tmp_path
    (root / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    cfg_path = root / "runs/configs/test_cfg.yaml"
    _write_cfg(cfg_path)

    rc = ru.main(["--root", str(root), "--cfg", str(cfg_path), "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Dry run complete." in out


def test_main_raises_when_cfg_missing(tmp_path: Path):
    root = tmp_path
    (root / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    missing = root / "runs/configs/does_not_exist.yaml"
    with pytest.raises(SystemExit):
        ru.main(["--root", str(root), "--cfg", str(missing)])
