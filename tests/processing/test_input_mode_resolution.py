from __future__ import annotations

import json
from pathlib import Path

import pytest

from processing.input_mode import resolve_processing_inputs


def _write_manifest(
    p: Path,
    *,
    run_id: str,
    timestamp: str,
    artifacts: dict[str, str] | None = None,
    data_policy: dict[str, object] | None = None,
) -> None:
    extra: dict[str, object] = {"data_policy": {"allow_incomplete_history": False}}
    if data_policy is not None:
        extra["data_policy"] = data_policy
    if artifacts is not None:
        extra["artifacts"] = artifacts
    p.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "cfg_hash": "deadbeef" * 5,
                "git_commit": "a" * 40,
                "cfg_path": "runs/configs/config_universe.yaml",
                "extra": extra,
            }
        ),
        encoding="utf-8",
    )


def test_resolve_processing_inputs_run_latest_picks_newest_manifest(
    tmp_path: Path,
) -> None:
    base = tmp_path / "runs" / "data"
    by_run = base / "by_run"
    r1 = by_run / "RUN-OLD_aaaa"
    r2 = by_run / "RUN-NEW_bbbb"
    (r1 / "outputs").mkdir(parents=True)
    (r2 / "outputs").mkdir(parents=True)
    _write_manifest(
        r1 / "outputs" / "universe_manifest.json",
        run_id="RUN-OLD",
        timestamp="2025-01-01T00:00:00+00:00",
    )
    _write_manifest(
        r2 / "outputs" / "universe_manifest.json",
        run_id="RUN-NEW",
        timestamp="2025-02-01T00:00:00+00:00",
    )
    (r1 / "outputs" / "raw_prices.pkl").write_text("x", encoding="utf-8")
    (r1 / "outputs" / "raw_volume.pkl").write_text("x", encoding="utf-8")
    (r2 / "outputs" / "raw_prices.pkl").write_text("x", encoding="utf-8")
    (r2 / "outputs" / "raw_volume_unadj.pkl").write_text("x", encoding="utf-8")

    res = resolve_processing_inputs(
        data_cfg={"input_mode": "run_latest", "strict_inputs": True},
        base_data_dir=base,
        price_globs_fallback=[],
        volume_globs_fallback=[],
    )

    assert res.mode == "run_latest"
    assert res.input_dir == (r2 / "outputs")
    assert res.price_globs == [str(r2 / "outputs" / "raw_prices.pkl")]
    assert res.volume_globs == [str(r2 / "outputs" / "raw_volume_unadj.pkl")]
    assert res.universe_meta and res.universe_meta["run_id"] == "RUN-NEW"


def test_resolve_processing_inputs_run_pinned_requires_dir_when_strict(
    tmp_path: Path,
) -> None:
    base = tmp_path / "runs" / "data"
    with pytest.raises(FileNotFoundError):
        resolve_processing_inputs(
            data_cfg={
                "input_mode": "run_pinned",
                "pinned_universe_outputs_dir": str(
                    base / "by_run" / "RUN-X" / "outputs"
                ),
                "strict_inputs": True,
            },
            base_data_dir=base,
            price_globs_fallback=[],
            volume_globs_fallback=[],
        )


def test_resolve_processing_inputs_falls_back_to_legacy_when_allowed(
    tmp_path: Path,
) -> None:
    base = tmp_path / "runs" / "data"
    res = resolve_processing_inputs(
        data_cfg={
            "input_mode": "run_latest",
            "strict_inputs": False,
            "allow_fallback_to_legacy": True,
        },
        base_data_dir=base,
        price_globs_fallback=["runs/data/raw_prices.pkl"],
        volume_globs_fallback=["runs/data/raw_volume.pkl"],
    )
    assert res.mode == "legacy_latest"
    assert res.input_dir == base


def test_resolve_processing_inputs_uses_manifest_artifact_paths(tmp_path: Path) -> None:
    base = tmp_path / "runs" / "data"
    out_dir = base / "by_run" / "RUN-CUSTOM_art" / "outputs"
    out_dir.mkdir(parents=True)

    p_prices = (out_dir / "custom_prices.parquet").resolve()
    p_volume = (out_dir / "custom_volumes.pkl").resolve()
    p_prices.write_text("x", encoding="utf-8")
    p_volume.write_text("x", encoding="utf-8")

    _write_manifest(
        out_dir / "universe_manifest.json",
        run_id="RUN-CUSTOM",
        timestamp="2025-03-01T00:00:00+00:00",
        artifacts={
            "prices_canonical": str(p_prices),
            "volumes_unadjusted": str(p_volume),
        },
    )

    res = resolve_processing_inputs(
        data_cfg={
            "input_mode": "run_pinned",
            "pinned_universe_outputs_dir": str(out_dir),
            "strict_inputs": True,
        },
        base_data_dir=base,
        price_globs_fallback=[],
        volume_globs_fallback=[],
    )

    assert res.price_globs == [str(p_prices)]
    assert res.volume_globs == [str(p_volume)]


def test_resolve_processing_inputs_run_pinned_prefers_local_outputs_over_manifest_relative_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a working-directory mutable artefact that should not win in run_pinned mode.
    monkeypatch.chdir(tmp_path)

    base = tmp_path / "runs" / "data"
    out_dir = base / "by_run" / "RUN-PINNED_local" / "outputs"
    out_dir.mkdir(parents=True)

    mutable_prices = base / "raw_prices.pkl"
    mutable_volume = base / "raw_volume_unadj.pkl"
    mutable_prices.write_text("mutable", encoding="utf-8")
    mutable_volume.write_text("mutable", encoding="utf-8")

    pinned_prices = out_dir / "raw_prices.pkl"
    pinned_volume = out_dir / "raw_volume_unadj.pkl"
    pinned_prices.write_text("pinned", encoding="utf-8")
    pinned_volume.write_text("pinned", encoding="utf-8")

    _write_manifest(
        out_dir / "universe_manifest.json",
        run_id="RUN-PINNED",
        timestamp="2025-03-01T00:00:00+00:00",
        artifacts={
            # Same relative paths as mutable workspace artefacts
            "prices_canonical": "runs/data/raw_prices.pkl",
            "volumes_unadjusted": "runs/data/raw_volume_unadj.pkl",
        },
    )

    res = resolve_processing_inputs(
        data_cfg={
            "input_mode": "run_pinned",
            "pinned_universe_outputs_dir": str(out_dir),
            "strict_inputs": True,
        },
        base_data_dir=base,
        price_globs_fallback=[],
        volume_globs_fallback=[],
    )

    assert res.price_globs == [str(pinned_prices)]
    assert res.volume_globs == [str(pinned_volume)]


def test_resolve_processing_inputs_run_pinned_manifest_relative_basename_fallback(
    tmp_path: Path,
) -> None:
    base = tmp_path / "runs" / "data"
    out_dir = base / "by_run" / "RUN-BASENAME" / "outputs"
    out_dir.mkdir(parents=True)

    p_prices = out_dir / "custom_prices.pkl"
    p_volume = out_dir / "custom_volumes.pkl"
    p_prices.write_text("x", encoding="utf-8")
    p_volume.write_text("x", encoding="utf-8")

    _write_manifest(
        out_dir / "universe_manifest.json",
        run_id="RUN-BASENAME",
        timestamp="2025-03-01T00:00:00+00:00",
        artifacts={
            # Repo-relative style path that does not exist under out_dir/runs/data/...
            "prices_canonical": "runs/data/custom_prices.pkl",
            "volumes_unadjusted": "runs/data/custom_volumes.pkl",
        },
    )

    res = resolve_processing_inputs(
        data_cfg={
            "input_mode": "run_pinned",
            "pinned_universe_outputs_dir": str(out_dir),
            "strict_inputs": True,
        },
        base_data_dir=base,
        price_globs_fallback=[],
        volume_globs_fallback=[],
    )

    assert res.price_globs == [str(p_prices)]
    assert res.volume_globs == [str(p_volume)]
