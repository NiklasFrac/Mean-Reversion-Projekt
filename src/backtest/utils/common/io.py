from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml_dict(
    path: Path, *, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else (default or {})


def load_yaml_dict_optional(
    path: Path | None, *, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    if path is None:
        return default or {}
    return load_yaml_dict(path, default=default)


def load_json_dict(
    path: Path, *, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else (default or {})


def load_json_dict_optional(
    path: Path | None, *, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    if path is None:
        return default or {}
    return load_json_dict(path, default=default)


def load_yaml_dict_safe(
    path: Path | None,
    *,
    default: dict[str, Any] | None = None,
    logger: Any | None = None,
    warn_msg: str = "YAML load failed",
) -> dict[str, Any]:
    if path is None:
        return default or {}
    try:
        return load_yaml_dict(path, default=default)
    except Exception as e:
        if logger is not None:
            try:
                logger.warning("%s (%s): %s", warn_msg, path, e)
            except Exception:
                pass
        return default or {}


def load_json_dict_safe(
    path: Path | None,
    *,
    default: dict[str, Any] | None = None,
    logger: Any | None = None,
    warn_msg: str = "JSON load failed",
) -> dict[str, Any]:
    if path is None:
        return default or {}
    p = Path(path)
    if not p.exists():
        return default or {}
    try:
        return load_json_dict(p, default=default)
    except Exception as e:
        if logger is not None:
            try:
                logger.warning("%s (%s): %s", warn_msg, p, e)
            except Exception:
                pass
        return default or {}
