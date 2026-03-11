"""Atomic write helpers for artifacts."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from analysis.logging_config import logger


def _fsync_dir(p: Path) -> None:
    try:
        fd = os.open(str(p), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def save_results(
    df_selected: pd.DataFrame, out_path: Path, meta: dict[str, Any]
) -> None:
    pkl_bytes = pickle.dumps(df_selected)
    _atomic_write_bytes(out_path, pkl_bytes)
    try:
        csv_path = out_path.with_suffix(".csv")
        tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
        df_selected.to_csv(tmp, index=False, float_format="%.10g", lineterminator="\n")
        with tmp.open("rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, csv_path)
        _fsync_dir(csv_path.parent)
    except Exception:
        logger.debug("CSV write failed", exc_info=True)
    try:
        meta_path = out_path.with_suffix(".meta.json")
        _atomic_write_text(meta_path, json.dumps(meta, indent=2, default=str))
    except Exception:
        logger.debug("Meta JSON write failed", exc_info=True)


def save_parquet(
    df: pd.DataFrame,
    out_path: Path,
    *,
    compression: str | None = "zstd",
) -> None:
    """Atomically write a parquet file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    Compression = Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None
    comp = cast(Compression, compression)
    df.to_parquet(tmp, index=True, compression=comp)
    try:
        with tmp.open("rb+") as f:
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass
    os.replace(tmp, out_path)
    _fsync_dir(out_path.parent)


def save_json(obj: Any, out_path: Path) -> None:
    """Atomically write JSON text."""
    _atomic_write_text(out_path, json.dumps(obj, indent=2, default=str))


__all__ = [
    "_fsync_dir",
    "_atomic_write_bytes",
    "_atomic_write_text",
    "save_results",
    "save_parquet",
    "save_json",
]
