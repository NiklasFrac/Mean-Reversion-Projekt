from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from backtest.utils.common.io import load_json_dict as _load_json
from backtest.utils.common.io import load_yaml_dict as _load_yaml
from backtest.utils.common.merge import deep_merge as _deep_merge

__all__ = [
    "_deep_merge",
    "_file_fingerprint",
    "_file_key_payload",
    "_load_json",
    "_load_yaml",
    "_sha256_file",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(str(path))
    try:
        st = p.stat()
    except Exception:
        return {"path": str(p)}
    return {"path": str(p), "size": int(st.st_size), "mtime": int(st.st_mtime)}


def _file_key_payload(
    path: str | Path | None,
    *,
    include_path: bool = True,
    include_hash: bool = False,
) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(str(path))
    try:
        st = p.stat()
    except Exception:
        return {"path": str(p)} if include_path else None
    out: dict[str, Any] = {"size": int(st.st_size)}
    if include_path:
        out["path"] = str(p)
    if include_hash:
        try:
            out["sha256"] = _sha256_file(p)
        except Exception:
            out["sha256"] = None
    return out
