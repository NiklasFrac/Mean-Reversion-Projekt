"""
Deterministic hashing utilities for tabular artifacts (CSV/Parquet/DataFrames),
plus a production-grade artifact cache keyed by a canonical Config-Hash.

This module now serves two purposes:

1) Stable, deterministic Hashing for DataFrames/CSV/Parquet and JSON
   (unchanged semantics for existing public functions).
2) Tier-1 Artifact Cache with stages raw→features→fills→reports and a
   `load_or_build(...)` API:
      - Canonical config hashing (sorted JSON; schema/code version bits)
      - Locking & atomic writes
      - Meta manifest per artifact (for auditing & parity checks)
      - Pluggable serializers (DataFrame→Parquet/CSV, JSON, …)

Environment variables:
- ARTIFACTS_DIR: Root directory for cached artifacts (default: ".artifacts")
- CODE_VERSION:  Code version string; if unset, tries GIT_SHA; else "dev"
- GIT_SHA:       Alternative way to inject code version (e.g., CI)
- SCHEMA_VERSION: Optional override via code change; baked as constant below.

All public functions are ruff/mypy-clean for Python 3.12.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, Generic, Optional, Protocol, TypeVar, cast

import numpy as np
import pandas as pd
from pandas._typing import WriteBuffer

# -------------------------------- logging -------------------------------------

logger = logging.getLogger(__name__)

# ------------------------------ module constants ---------------------------------

_CANON_FLOAT_DECIMALS: Final[int] = 10
_READ_CHUNK_BYTES: Final[int] = 1 << 20  # 1 MiB
_LINE_TERMINATOR: Final[str] = "\n"  # normalize to LF for stable CSV digests

# Versioning knobs for cache keys / manifests
SCHEMA_VERSION: Final[int] = int(os.getenv("SCHEMA_VERSION", "1"))
CODE_VERSION: Final[str] = os.getenv("CODE_VERSION") or os.getenv("GIT_SHA") or "dev"

__all__ = [
    # Canonicalization/Hashing (unchanged)
    "_canon_df",
    "hash_dataframe_sha256",
    "hash_file_sha256",
    "hash_parquet_sha256",
    "to_json",
    "hash_json_sha256",
    "hash_parquet_canonical_sha256",
    # New: Cache API
    "Stage",
    "Serializer",
    "JSONSerializer",
    "DataFrameParquetSerializer",
    "DataFrameCSVSerializer",
    "ArtifactStore",
    "config_hash",
    "load_or_build",
]

T = TypeVar("T")

# ------------------------------ canonicalization ---------------------------------


def _canon_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a canonicalized view of a DataFrame for deterministic hashing.

    Canonicalization rules (conservative):
      - Index/columns cast to strings.
      - Columns sorted lexicographically (stable column order).
      - Float dtype columns rounded to _CANON_FLOAT_DECIMALS.

    NOTE: Row order is preserved by design.
    """
    d = df.copy(deep=True)

    # Normalize axis labels to strings
    d.index = pd.Index((str(x) for x in d.index), name=None)
    cols_now = [str(c) for c in d.columns]
    if list(d.columns) != cols_now:
        d.columns = pd.Index(cols_now, name=None)

    # Stable column order
    ordered = sorted(cols_now)
    if cols_now != ordered:
        d = pd.concat([d[c] for c in ordered], axis=1)
        d.columns = pd.Index(ordered, name=None)

    # Deterministic rounding only for float columns
    for c in d.columns:
        col = d[c]
        if pd.api.types.is_float_dtype(col.dtype):
            d[c] = (
                pd.to_numeric(col, errors="coerce")
                .astype("float64")
                .round(_CANON_FLOAT_DECIMALS)
            )

    return d


# ------------------------------ hashing primitives ---------------------------------


class _HashLike(Protocol):
    def update(self, data: bytes) -> None: ...
    def hexdigest(self) -> str: ...


class _Sha256TextWriter(io.TextIOBase):
    """
    Text writer that feeds UTF-8 encoded chunks into a sha256 hasher.
    Subclassing TextIOBase satisfies pandas' typing for WriteBuffer[str].
    """

    __slots__ = ("_hasher", "_count")

    def __init__(self, hasher: _HashLike) -> None:
        self._hasher = hasher
        self._count = 0

    def write(self, s: str) -> int:
        b = s.encode("utf-8")
        self._hasher.update(b)
        self._count += len(s)
        return len(s)

    def tell(self) -> int:  # some pandas versions probe this
        return self._count

    def readable(self) -> bool:  # pragma: no cover
        return False

    def writable(self) -> bool:  # pragma: no cover
        return True


def hash_dataframe_sha256(df: pd.DataFrame) -> str:
    """
    Deterministically hash a DataFrame's *content* by serializing a canonical form to CSV.

    - Uses LF newlines only (platform-stable)
    - Includes index
    - Rounds float columns to _CANON_FLOAT_DECIMALS before serialization
    - Streaming writer to keep memory bounded
    """
    d = _canon_df(df)
    hasher = hashlib.sha256()
    writer = _Sha256TextWriter(hasher)
    d.to_csv(
        path_or_buf=cast(WriteBuffer[str], writer),
        index=True,
        lineterminator=_LINE_TERMINATOR,
    )
    return hasher.hexdigest()


def hash_file_sha256(path: Path) -> str:
    """Stream a file in fixed-size chunks and compute its SHA-256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_READ_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_parquet_sha256(path: Path) -> str:
    """
    Hash Parquet file *bytes* as-is.

    WARNING: Parquet encodes metadata/encodings; two semantically identical tables
    may still yield different byte hashes. For content-level parity, use
    `hash_parquet_canonical_sha256`.
    """
    return hash_file_sha256(path)


# ------------------------------ additive helpers (optional) ------------------------


def hash_parquet_canonical_sha256(path: Path) -> str:
    """Content-level hash for Parquet: read -> canonicalize -> CSV digest."""
    df = pd.read_parquet(path)
    return hash_dataframe_sha256(df)


def _to_json_default(obj: Any) -> Any:
    """
    Robust JSON default conversion:
      - numpy scalar -> python scalar
      - pathlib.Path -> str
      - dataclass instance -> dict
      - pandas Timestamp/Timedelta/NA -> string or None
      - numpy arrays -> list
      - fallback -> str(obj)
    """
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if obj is pd.NaT:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def to_json(obj: Any) -> str:
    """Stable JSON dump with sorted keys and UTF-8 output."""
    return json.dumps(
        obj,
        default=_to_json_default,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def hash_json_sha256(obj: Any) -> str:
    """Hash the canonical JSON representation of an object."""
    return hashlib.sha256(to_json(obj).encode("utf-8")).hexdigest()


# ============================== NEW: Cache Layer ===================================


class Stage(str, Enum):
    RAW = "raw"
    FEATURES = "features"
    FILLS = "fills"
    REPORTS = "reports"
    SIGNALS = "signals"
    ORDERS = "orders"


class Serializer(Protocol[T]):
    """Serializer protocol for cached artifacts."""

    extension: str

    def load(self, path: Path) -> T: ...
    def save(self, obj: T, path: Path) -> None: ...


class JSONSerializer(Generic[T]):
    """JSON serializer using our stable JSON (to_json)."""

    extension = ".json"

    def load(self, path: Path) -> T:
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, obj: T, path: Path) -> None:
        path.write_text(to_json(obj), encoding="utf-8")


class DataFrameParquetSerializer:
    """Serialize pandas.DataFrame to Parquet (with index)."""

    extension = ".parquet"

    def load(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def save(self, obj: pd.DataFrame, path: Path) -> None:
        obj.to_parquet(path, index=True)


class DataFrameCSVSerializer:
    """Serialize pandas.DataFrame to CSV with normalized line terminators."""

    extension = ".csv"

    def load(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0)

    def save(self, obj: pd.DataFrame, path: Path) -> None:
        obj.to_csv(path, index=True, lineterminator=_LINE_TERMINATOR)


@dataclass(frozen=True)
class ArtifactPaths:
    data: Path
    meta: Path
    lock: Path


@dataclass
class ArtifactStore:
    """Filesystem-based artifact store with stage/key namespacing."""

    root: Path

    @classmethod
    def default(cls) -> ArtifactStore:
        base = Path(os.getenv("ARTIFACTS_DIR", ".artifacts")).resolve()
        return cls(root=base)

    def paths_for(self, stage: Stage, key: str, ext: str) -> ArtifactPaths:
        sub = self.root / stage.value / key[:2] / key
        data = sub / f"data{ext}"
        meta = sub / "meta.json"
        lock = self.root / ".locks" / stage.value / f"{key}.lock"
        return ArtifactPaths(data=data, meta=meta, lock=lock)

    def exists(self, stage: Stage, key: str, ext: str) -> bool:
        p = self.paths_for(stage, key, ext)
        return p.data.exists() and p.meta.exists()


# ---------- Hashing for cache keys ----------


def _stable_dumps(obj: Any) -> str:
    """Stable, minimal JSON (sorted keys, no spaces)."""
    return json.dumps(
        obj,
        default=_to_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def config_hash(
    config: Mapping[str, Any],
    *,
    stage: Stage,
    extra: Optional[Mapping[str, Any]] = None,
    schema_version: int = SCHEMA_VERSION,
    code_version: str = CODE_VERSION,
) -> str:
    """
    Compute a stable cache key. `extra` may include input fingerprints
    (e.g., dataset version, dependent file hashes).
    """
    payload = {
        "stage": stage.value,
        "schema": schema_version,
        "code": code_version,
        "config": config,
        "extra": extra or {},
    }
    return _sha256(_stable_dumps(payload))


# ---------- Locking & Atomic I/O ----------


class LockTimeout(Exception):
    """Raised when a cache lock cannot be acquired in time."""


def _acquire_lock(lock_path: Path, timeout_s: int = 120) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return
        except FileExistsError:
            if time.monotonic() > deadline:
                raise LockTimeout(f"Could not acquire lock: {lock_path}")
            time.sleep(0.25)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:  # pragma: no cover
        pass


def _atomic_write(write_fn: Callable[[Path], None], final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_name(final_path.name + ".tmp")
    write_fn(tmp)
    os.replace(tmp, final_path)  # atomic on same filesystem


# ---------- Public Cache API ----------


def load_or_build(
    *,
    stage: Stage,
    config: Mapping[str, Any],
    builder: Callable[[], T],
    serializer: Serializer[T],
    store: Optional[ArtifactStore] = None,
    extra: Optional[Mapping[str, Any]] = None,
    force_rebuild: bool = False,
    lock_timeout_s: int = 120,
) -> T:
    """
    Load an artifact from cache (by config-hash) or build and cache it atomically.

    Args:
      stage:        Stage enum (raw/features/fills/reports)
      config:       Mapping that defines the effective configuration
      builder:      Zero-arg callable returning the artifact object
      serializer:   Serializer used to persist/load the artifact
      store:        Optional custom ArtifactStore (default: ArtifactStore.default())
      extra:        Optional extra dict (e.g. input fingerprints) included in the key
      force_rebuild:Bypass cache and rebuild
      lock_timeout_s:Lock acquisition timeout

    Returns:
      The artifact object (loaded or freshly built).
    """
    s = store or ArtifactStore.default()
    key = config_hash(config, stage=stage, extra=extra)
    paths = s.paths_for(stage, key, serializer.extension)

    # Fast path: existing cache (no lock)
    if not force_rebuild and s.exists(stage, key, serializer.extension):
        logger.debug("Cache hit [%s/%s] -> %s", stage.value, key[:8], paths.data)
        return serializer.load(paths.data)

    # Build path with lock/atomics
    _acquire_lock(paths.lock, timeout_s=lock_timeout_s)
    try:
        # Re-check after acquiring the lock
        if not force_rebuild and s.exists(stage, key, serializer.extension):
            logger.debug("Cache hit after lock [%s/%s]", stage.value, key[:8])
            return serializer.load(paths.data)

        logger.info("Cache miss → building [%s/%s]", stage.value, key[:8])
        obj = builder()

        # Persist data atomically
        def _write_data(tmp: Path) -> None:
            serializer.save(obj, tmp)

        _atomic_write(_write_data, paths.data)

        # Write meta manifest atomically
        meta_payload = {
            "stage": stage.value,
            "key": key,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": SCHEMA_VERSION,
            "code_version": CODE_VERSION,
            "serializer": serializer.extension,
            "config": config,
            "extra": extra or {},
        }

        def _write_meta(tmp: Path) -> None:
            tmp.write_text(_stable_dumps(meta_payload), encoding="utf-8")

        _atomic_write(_write_meta, paths.meta)

        return obj
    finally:
        _release_lock(paths.lock)
