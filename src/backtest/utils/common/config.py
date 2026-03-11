# src/backtest/common/config_utils.py
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, cast

import yaml

from backtest.utils.common.merge import deep_merge as _deep_merge

logger = logging.getLogger("backtest.utils.common.config")
logger.addHandler(logging.NullHandler())

# -------------------------------
# ${ENV[:default]} Expansion
# -------------------------------
_ENV_PAT = re.compile(r"\$\{([^}:\s]+)(?::([^}]+))?\}")


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):

        def repl(m):
            key, default = m.group(1), m.group(2)
            return os.environ.get(key, default if default is not None else "")

        return _ENV_PAT.sub(repl, obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_expand_env(v) for v in obj)
    return obj


# -------------------------------
# !include support
# -------------------------------
class _YamlLoader(yaml.SafeLoader):
    pass


def _construct_include(loader: _YamlLoader, node: yaml.Node):
    root: Path = getattr(_YamlLoader, "_root", Path("."))
    rel = loader.construct_scalar(cast(Any, node))
    inc_path = (root / rel).resolve()
    if not inc_path.exists():
        raise FileNotFoundError(f"!include not found: {inc_path}")
    old_root = getattr(_YamlLoader, "_root", None)
    try:
        _YamlLoader._root = inc_path.parent  # type: ignore[attr-defined]
        with inc_path.open("r", encoding="utf-8") as f:
            return yaml.load(f, Loader=_YamlLoader)
    finally:
        _YamlLoader._root = old_root  # type: ignore[attr-defined]


_YamlLoader.add_constructor("!include", _construct_include)


# -------------------------------
# YAML loader (+ includes + env)
# -------------------------------
def load_yaml_with_includes_and_env(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    _YamlLoader._root = p.parent  # type: ignore[attr-defined]
    with p.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YamlLoader) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping (got {type(data)})")
    return _expand_env(data)


# -------------------------------
# Resolve 'extends' recursively
# -------------------------------
def deep_merge_extends(
    cfg: Dict[str, Any], *, base_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Allows 'extends': <rel/path.yaml> or list thereof. Child overrides parent."""
    base_dir = base_dir or Path(getattr(_YamlLoader, "_root", Path(".")))
    parent = cfg.get("extends")
    if not parent:
        return cfg
    parents = parent if isinstance(parent, list) else [parent]
    merged: Dict[str, Any] = {}
    for rel in parents:
        parent_path = (base_dir / str(rel)).resolve()
        parent_raw = load_yaml_with_includes_and_env(parent_path)
        parent_resolved = deep_merge_extends(parent_raw, base_dir=parent_path.parent)
        merged = _deep_merge(merged, parent_resolved)
    child = {k: v for k, v in cfg.items() if k != "extends"}
    return _deep_merge(merged, child)


# -------------------------------
# Public helpers
# -------------------------------
def load_config_dict(path: str | Path) -> Dict[str, Any]:
    """Single entry: YAML → includes/env/extends → merged dict."""
    raw = load_yaml_with_includes_and_env(path)
    return deep_merge_extends(raw, base_dir=Path(path).parent)


def dump_normalized_config(cfg: Dict[str, Any], out_path: str | Path) -> None:
    """Dump a merged config dict to YAML (repro artifact)."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=True, allow_unicode=True)
