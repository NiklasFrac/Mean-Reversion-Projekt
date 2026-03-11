from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["ResolvedInputs", "resolve_processing_inputs"]


@dataclass(frozen=True)
class ResolvedInputs:
    mode: str
    input_dir: Path
    price_globs: list[str]
    volume_globs: list[str]
    universe_manifest_path: Path | None
    universe_meta: dict[str, Any] | None


def _read_universe_meta(manifest_path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    keep = {
        k: raw.get(k)
        for k in ["run_id", "timestamp", "cfg_hash", "git_commit", "cfg_path"]
    }
    raw_extra = raw.get("extra")
    extra = raw_extra if isinstance(raw_extra, dict) else {}
    keep["data_policy"] = extra.get("data_policy")
    keep["adv_provenance"] = extra.get("adv_provenance")
    keep["artifacts"] = extra.get("artifacts")
    return keep


def _read_universe_manifest(manifest_path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _manifest_timestamp(manifest_path: Path) -> datetime | None:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        ts = raw.get("timestamp")
        if isinstance(ts, str) and ts:
            # ISO 8601 with timezone offset
            return datetime.fromisoformat(ts)
    except Exception:
        return None
    return None


def _resolve_manifest_path(raw_path: str, *, out_dir: Path) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p
    # In run-scoped modes, prefer output-dir relative interpretation to preserve
    # immutability semantics of pinned/latest run folders.
    candidate = out_dir / p
    if candidate.is_file():
        return candidate
    # Compatibility fallback for manifests that persist repo-relative paths while
    # files are copied into run-scoped outputs with the same basename.
    basename_candidate = out_dir / p.name
    if basename_candidate.is_file():
        return basename_candidate
    return candidate


def _extract_manifest_artifacts(
    manifest_path: Path, *, out_dir: Path
) -> dict[str, Path]:
    raw = _read_universe_manifest(manifest_path)
    if not raw:
        return {}
    extra = raw.get("extra")
    if not isinstance(extra, dict):
        return {}
    artifacts = extra.get("artifacts")
    if not isinstance(artifacts, dict):
        return {}
    out: dict[str, Path] = {}
    for key, value in artifacts.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            continue
        out[key] = _resolve_manifest_path(value, out_dir=out_dir)
    return out


def _pick_first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        try:
            if p.is_file():
                return p
        except Exception:
            continue
    return None


def _resolve_run_scoped_artifacts(
    out_dir: Path, manifest_path: Path
) -> tuple[Path | None, Path | None]:
    artifacts = _extract_manifest_artifacts(manifest_path, out_dir=out_dir)
    price_candidates = [
        out_dir / "raw_prices.pkl",
        artifacts.get("prices_canonical"),
        artifacts.get("prices"),
    ]
    volume_candidates = [
        out_dir / "raw_volume_unadj.pkl",
        out_dir / "raw_volume.pkl",
        artifacts.get("volumes_unadjusted"),
        artifacts.get("volumes_canonical"),
        artifacts.get("volumes"),
    ]
    p_prices = _pick_first_existing([p for p in price_candidates if p is not None])
    p_volume = _pick_first_existing([p for p in volume_candidates if p is not None])
    return p_prices, p_volume


def _find_latest_universe_outputs_dir(by_run_dir: Path) -> Path | None:
    if not by_run_dir.exists() or not by_run_dir.is_dir():
        return None
    candidates: list[tuple[datetime, float, Path]] = []
    for child in by_run_dir.iterdir():
        if not child.is_dir():
            continue
        # Universe runs are written as RUN-... directories; processing uses PRC-...
        if not str(child.name).startswith("RUN-"):
            continue
        out_dir = child / "outputs"
        manifest = out_dir / "universe_manifest.json"
        if not out_dir.is_dir() or not manifest.is_file():
            continue
        ts = _manifest_timestamp(manifest) or datetime.fromtimestamp(
            out_dir.stat().st_mtime
        )
        candidates.append((ts, float(out_dir.stat().st_mtime), out_dir))
    if not candidates:
        return None
    # Prefer manifest timestamp, then mtime for tie-break.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def resolve_processing_inputs(
    *,
    data_cfg: dict[str, Any],
    base_data_dir: Path,
    price_globs_fallback: list[str],
    volume_globs_fallback: list[str],
) -> ResolvedInputs:
    """
    Resolve which raw universe artefacts the processing pipeline should load.

    Modes:
      - run_latest: newest universe run under <base_data_dir>/by_run/RUN-*/outputs
      - run_pinned: exact directory via data.pinned_universe_outputs_dir
      - legacy_latest: use <base_data_dir> and raw_* globs (mutable working dir)

    Guardrails:
      - data.strict_inputs: abort if a run-scoped mode cannot be resolved
      - data.allow_fallback_to_legacy: if enabled, run_latest/run_pinned can fall back to legacy_latest
    """
    mode_raw = str(data_cfg.get("input_mode", "") or "").strip().lower()
    mode = mode_raw or "legacy_latest"
    strict = bool(data_cfg.get("strict_inputs", False))
    allow_fallback = bool(data_cfg.get("allow_fallback_to_legacy", False))

    def _legacy() -> ResolvedInputs:
        return ResolvedInputs(
            mode="legacy_latest",
            input_dir=base_data_dir,
            price_globs=price_globs_fallback,
            volume_globs=volume_globs_fallback,
            universe_manifest_path=None,
            universe_meta=None,
        )

    if mode in {"legacy_latest", "manual", "runs_data"}:
        return _legacy()

    if mode in {"run_latest", "latest", "by_run_latest"}:
        out_dir = _find_latest_universe_outputs_dir(base_data_dir / "by_run")
        if out_dir is None:
            if allow_fallback:
                return _legacy()
            if strict:
                raise FileNotFoundError(
                    f"input_mode={mode} but no universe run outputs found under {base_data_dir / 'by_run'}"
                )
            return _legacy()
        p_manifest = out_dir / "universe_manifest.json"
        p_prices, volume_choice = _resolve_run_scoped_artifacts(
            out_dir=out_dir, manifest_path=p_manifest
        )
        if strict and p_prices is None:
            raise FileNotFoundError(
                f"Missing expected prices file in run outputs {out_dir} (checked manifest artifacts and raw_prices.pkl)."
            )
        if strict and volume_choice is None:
            raise FileNotFoundError(
                f"Missing expected volume file in run outputs {out_dir} (checked manifest artifacts and raw_volume*.pkl)."
            )
        return ResolvedInputs(
            mode="run_latest",
            input_dir=out_dir,
            price_globs=[str(p_prices)]
            if p_prices is not None
            else price_globs_fallback,
            volume_globs=[str(volume_choice)]
            if volume_choice is not None
            else volume_globs_fallback,
            universe_manifest_path=p_manifest if p_manifest.is_file() else None,
            universe_meta=_read_universe_meta(p_manifest)
            if p_manifest.is_file()
            else None,
        )

    if mode in {"run_pinned", "pinned"}:
        pinned = str(data_cfg.get("pinned_universe_outputs_dir", "") or "").strip()
        if not pinned:
            if allow_fallback:
                return _legacy()
            raise ValueError(
                "input_mode=run_pinned requires data.pinned_universe_outputs_dir to be set."
            )
        out_dir = Path(pinned)
        p_manifest = out_dir / "universe_manifest.json"
        p_prices, volume_choice = _resolve_run_scoped_artifacts(
            out_dir=out_dir, manifest_path=p_manifest
        )
        if strict:
            if not out_dir.is_dir():
                raise FileNotFoundError(
                    f"Pinned universe outputs dir not found: {out_dir}"
                )
            if p_prices is None:
                raise FileNotFoundError(
                    f"Missing expected prices file in pinned dir {out_dir} (checked manifest artifacts and raw_prices.pkl)."
                )
            if volume_choice is None:
                raise FileNotFoundError(
                    f"Missing expected volume file in pinned dir {out_dir} (checked manifest artifacts and raw_volume*.pkl)."
                )
            if not p_manifest.is_file():
                raise FileNotFoundError(
                    f"Missing universe_manifest.json in pinned dir: {p_manifest}"
                )
        return ResolvedInputs(
            mode="run_pinned",
            input_dir=out_dir,
            price_globs=[str(p_prices)]
            if p_prices is not None
            else price_globs_fallback,
            volume_globs=[str(volume_choice)]
            if volume_choice is not None
            else volume_globs_fallback,
            universe_manifest_path=p_manifest if p_manifest.is_file() else None,
            universe_meta=_read_universe_meta(p_manifest)
            if p_manifest.is_file()
            else None,
        )

    if allow_fallback:
        return _legacy()
    raise ValueError(f"Unsupported data.input_mode: {mode_raw!r}")
