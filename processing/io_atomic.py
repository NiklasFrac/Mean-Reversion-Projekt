from __future__ import annotations

import json
import logging
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np
import pandas as pd

CompressionLiteral = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]


def _tmp_path_for(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid4().hex}")


def _fsync_path(path: Path) -> None:
    """
    Best-effort fsync for durability. Safe no-op on platforms/filesystems that
    do not support fsync for a given path.
    """
    try:
        with path.open("rb") as f:
            os.fsync(f.fileno())
    except Exception:
        return


def atomic_write_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(path)
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    _fsync_path(path)


def atomic_write_bytes(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(path)
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_write_parquet(
    df: pd.DataFrame, path: Path, compression: str | None = "zstd"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(path)
    comp = cast(CompressionLiteral | None, compression)
    try:
        if comp is not None:
            df.to_parquet(tmp, compression=comp)
        else:
            df.to_parquet(tmp)
        _fsync_path(tmp)
        tmp.replace(path)
        _fsync_path(path)
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except TypeError:
            if tmp.exists():
                tmp.unlink()
        logging.getLogger("data_processing").warning(
            "Parquet atomic write failed: %s", exc
        )
        raise


def _json_default(o: Any) -> Any:
    """
    Make numpy/pandas types JSON serialisable for diagnostics/manifests.
    """
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if hasattr(pd, "Timestamp") and isinstance(o, pd.Timestamp):
        return o.isoformat()
    if hasattr(pd, "Timedelta") and isinstance(o, pd.Timedelta):
        return o.isoformat()
    if hasattr(pd, "NaT") and o is pd.NaT:
        return None
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def atomic_write_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path_for(path)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    _fsync_path(path)


def file_hash(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    try:
        import hashlib as _h

        h = _h.sha1()
        with path.open("rb") as f:
            for b in iter(lambda: f.read(8192), b""):
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(out.strip())
    except Exception:
        return None


def _pip_freeze(*, project_root: Path | None = None) -> tuple[str, str | None]:
    try:
        root = Path(project_root) if project_root is not None else Path.cwd()
        meta_dir = root / "runs" / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        lock = meta_dir / "requirements.lock"
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        tmp = lock.with_suffix(lock.suffix + ".tmp")
        tmp.write_text(out, encoding="utf-8")
        tmp.replace(lock)
        return str(lock), file_hash(lock)
    except Exception:
        return "", None


def collect_runtime_context(
    *, pip_freeze: bool = True, project_root: Path | None = None
) -> dict[str, Any]:
    pip_freeze_enabled = (
        pip_freeze and os.environ.get("PROCESSING_DISABLE_PIP_FREEZE") is None
    )
    commit = _git_commit()
    dirty = _git_dirty()
    lock_path, lock_sha = ("", None)
    if pip_freeze_enabled:
        try:
            lock_path, lock_sha = _pip_freeze(project_root=project_root)
        except TypeError:
            lock_path, lock_sha = _pip_freeze()
    ctx: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": commit,
        "git_dirty": dirty,
        "pip_lock_path": lock_path or None,
        "pip_lock_sha1": lock_sha,
        "libs": {
            "pandas": getattr(pd, "__version__", None),
            "numpy": getattr(np, "__version__", None),
        },
        "cpu_count": os.cpu_count(),
    }
    return ctx


def make_manifest(
    cfg_path: Path,
    inputs: dict[str, Path | None],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cfg_path": str(cfg_path),
        "cfg_hash": file_hash(cfg_path),
        "inputs": {},
    }
    inputs_map: dict[str, dict[str, str | None]] = {}
    for k, p in inputs.items():
        inputs_map[k] = {
            "path": str(p) if p is not None else None,
            "sha1": file_hash(p) if (p is not None and p.exists()) else None,
        }
    manifest["inputs"] = inputs_map
    if extra is not None:
        manifest["extra"] = extra
    return manifest
