"""
Universe Reset Script

Deletes Universe Runner caches and outputs to start from a clean slate.

Execute deletions (default):
  python -m universe.reset_universe --cfg runs/configs/config_universe.yaml

Dry-run:
  python -m universe.reset_universe --cfg runs/configs/config_universe.yaml --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from universe.config import load_cfg_or_default, validate_cfg
from universe.storage import hashed_artifact_siblings, resolve_artifact_paths


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start


def _resolve_under_root(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else (root / p)


def _is_within(root: Path, p: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class ResetTarget:
    path: Path
    kind: str  # "file" | "dir"
    reason: str


def _iter_matching_glob(root: Path, pattern: str) -> list[Path]:
    pat = str(pattern).strip()
    if not pat:
        return []
    return list(root.glob(pat))


def gather_universe_reset_targets(
    *,
    root: Path,
    cfg_path: Path,
    keep_screener: bool,
    dev_caches: bool,
    purge_logs_and_tmp: bool,
) -> list[ResetTarget]:
    cfg_raw = load_cfg_or_default(cfg_path)
    cfg = validate_cfg(cfg_raw)

    universe_cfg = cfg.universe
    runtime_cfg = cfg.runtime
    artifact_paths = resolve_artifact_paths(
        universe_cfg=universe_cfg,
        data_cfg=cfg.data,
        runtime_cfg=runtime_cfg,
    )
    logging_cfg = dict(cfg.raw.get("logging", {}) or {})

    targets: list[ResetTarget] = []

    def add_dir(val: Any, *, reason: str) -> None:
        if isinstance(val, Path):
            p_raw = val
        elif isinstance(val, str) and val.strip():
            p_raw = Path(val)
        else:
            return
        p = _resolve_under_root(root, p_raw)
        targets.append(ResetTarget(p, "dir", reason))

    def add_file(val: Any, *, reason: str) -> None:
        if isinstance(val, Path):
            p_raw = val
        elif isinstance(val, str) and val.strip():
            p_raw = Path(val)
        else:
            return
        p = _resolve_under_root(root, p_raw)
        targets.append(ResetTarget(p, "file", reason))

    # Core outputs
    add_file(artifact_paths.output_tickers_csv, reason="universe.output_tickers_csv")
    add_file(
        artifact_paths.output_tickers_ext_csv, reason="universe.output_tickers_ext_csv"
    )
    add_file(artifact_paths.manifest, reason="universe.manifest")
    add_file(artifact_paths.fundamentals_out, reason="universe.fundamentals_out")
    add_file(artifact_paths.adv_cache, reason="universe.adv_cache")

    # Data outputs/caches
    add_file(artifact_paths.adv_csv, reason="data.adv_path")
    add_file(artifact_paths.adv_csv_filtered, reason="data.adv_filtered_path")
    add_file(artifact_paths.raw_prices_cache, reason="data.raw_prices_cache")
    add_file(artifact_paths.volume_path, reason="data.volume_path")
    add_file(
        artifact_paths.raw_prices_unadj_warmup_cache,
        reason="data.raw_prices_unadj_warmup_cache",
    )
    add_file(
        artifact_paths.raw_prices_unadj_cache, reason="data.raw_prices_unadj_cache"
    )
    add_file(
        artifact_paths.raw_volume_unadj_cache, reason="data.raw_volume_unadj_cache"
    )

    # Hashed artifacts siblings (when runtime.use_hashed_artifacts=true).
    for key, canonical in [
        ("raw_prices_cache", artifact_paths.raw_prices_cache),
        ("volume_path", artifact_paths.volume_path),
    ]:
        canonical_under_root = _resolve_under_root(root, canonical)
        for sibling in hashed_artifact_siblings(canonical_under_root):
            targets.append(
                ResetTarget(sibling, "file", f"hashed sibling of data.{key}")
            )

    # Runner checkpoint
    add_file(artifact_paths.checkpoint_path, reason="runtime.checkpoint_path")

    # Immutable run-scoped copies
    add_dir(
        artifact_paths.run_scoped_outputs_dir, reason="runtime.run_scoped_outputs_dir"
    )

    # Logs (if enabled in config)
    file_cfg = logging_cfg.get("file", {}) or {}
    if isinstance(file_cfg, dict) and file_cfg.get("path"):
        add_file(file_cfg.get("path"), reason="logging.file.path")
    json_cfg = logging_cfg.get("json", {}) or {}
    if isinstance(json_cfg, dict) and json_cfg.get("path"):
        add_file(json_cfg.get("path"), reason="logging.json.path")

    # Screener outputs (optional)
    if not keep_screener:
        for p in _iter_matching_glob(
            root, str(universe_cfg.get("screener_glob", "") or "")
        ):
            targets.append(
                ResetTarget(
                    p, "dir" if p.is_dir() else "file", "universe.screener_glob"
                )
            )

    # Dev/tool caches (optional)
    if dev_caches:
        for name in [
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".hypothesis",
            ".artifacts",
        ]:
            targets.append(ResetTarget(root / name, "dir", f"dev cache ({name})"))
        targets.append(
            ResetTarget(root / ".coverage", "file", "coverage file (.coverage)")
        )
        for p in root.rglob("__pycache__"):
            targets.append(ResetTarget(p, "dir", "python cache (__pycache__)"))

    # Runs extras (optional)
    if purge_logs_and_tmp:
        targets.append(
            ResetTarget(root / "runs" / "logs", "dir", "runs logs (runs/logs)")
        )
        targets.append(ResetTarget(root / "runs" / "tmp", "dir", "runs tmp (runs/tmp)"))

    # De-duplicate by resolved path.
    deduped: dict[Path, ResetTarget] = {}
    for t in targets:
        try:
            rp = t.path.resolve()
        except Exception:
            rp = t.path
        if rp not in deduped:
            deduped[rp] = ResetTarget(rp, t.kind, t.reason)
    return list(deduped.values())


def _delete_target(root: Path, t: ResetTarget, *, yes: bool) -> tuple[bool, str]:
    p = t.path
    if not _is_within(root, p):
        return False, f"SKIP (ausserhalb Repo): {p} ({t.reason})"
    if not p.exists():
        return False, f"OK (nicht vorhanden): {p} ({t.reason})"
    if not yes:
        return False, f"DRY-RUN: wuerde loeschen: {p} ({t.reason})"

    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return True, f"DEL: {p} ({t.reason})"
    except Exception as e:
        return False, f"FAIL: {p} ({t.reason}) -> {e}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reset Universe Runner caches/outputs.")
    ap.add_argument(
        "--cfg", type=Path, default=Path("runs/configs/config_universe.yaml")
    )
    ap.add_argument("--root", type=Path, default=None, help="Repo root (optional).")
    ap.add_argument(
        "--yes",
        dest="yes",
        action="store_true",
        default=True,
        help="Execute deletions (default behavior).",
    )
    ap.add_argument(
        "--dry-run",
        dest="yes",
        action="store_false",
        help="Only print what would be deleted, do not delete.",
    )
    ap.add_argument(
        "--keep-screener",
        dest="keep_screener",
        action="store_true",
        default=True,
        help="Keep screener CSVs (universe.screener_glob). Default: enabled.",
    )
    ap.add_argument(
        "--purge-screener",
        dest="keep_screener",
        action="store_false",
        help="Delete screener CSVs (universe.screener_glob).",
    )
    ap.add_argument(
        "--dev-caches",
        action="store_true",
        help="Also delete dev caches like __pycache__, .pytest_cache, .mypy_cache, .ruff_cache, .hypothesis, .coverage.",
    )
    ap.add_argument(
        "--purge-logs-and-tmp",
        action="store_true",
        help="Also delete runs/logs and runs/tmp.",
    )
    args = ap.parse_args(argv)

    root = _find_repo_root(args.root or Path.cwd())
    cfg_path = _resolve_under_root(root, args.cfg)
    if not cfg_path.exists():
        raise SystemExit(f"Config nicht gefunden: {cfg_path}")

    targets = gather_universe_reset_targets(
        root=root,
        cfg_path=cfg_path,
        keep_screener=bool(args.keep_screener),
        dev_caches=bool(args.dev_caches),
        purge_logs_and_tmp=bool(args.purge_logs_and_tmp),
    )
    targets = sorted(targets, key=lambda t: str(t.path).lower())

    deleted = 0
    for t in targets:
        did, msg = _delete_target(root, t, yes=bool(args.yes))
        print(msg)
        if did:
            deleted += 1

    if args.yes:
        print(f"Fertig. Geloescht: {deleted} Eintraege.")
    else:
        print("Dry-run fertig. Mit normalem Start tatsaechlich loeschen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
