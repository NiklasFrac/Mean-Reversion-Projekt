from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "atomic_write_text",
    "load_json",
    "load_json_dict",
    "load_yaml",
    "load_yaml_dict",
    "safe_write_df",
    "write_json",
]


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml_dict(
    path: Path, *, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    raw = load_yaml(path)
    return raw if isinstance(raw, dict) else (default or {})


def load_json_dict(
    path: Path, *, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    raw = load_json(path)
    return raw if isinstance(raw, dict) else (default or {})


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def write_json(
    path: Path,
    payload: Any,
    *,
    default: Any = str,
    indent: int = 2,
    ensure_ascii: bool = False,
    atomic: bool = False,
) -> None:
    serialized = json.dumps(
        payload,
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=default,
    )
    if atomic:
        atomic_write_text(path, serialized)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def safe_write_df(path: Path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df.to_parquet(path, index=True)
        elif suffix == ".csv":
            df.to_csv(path, index=True)
        else:
            df.to_csv(path.with_suffix(".csv"), index=True)
    except Exception as exc:  # pragma: no cover
        logger.warning("Write DF failed (%s): %s", path, exc)
