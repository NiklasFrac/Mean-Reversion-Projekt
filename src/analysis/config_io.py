from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from analysis.logging_config import logger


def deep_merge(a: dict, b: dict) -> dict:
    """Deep-merge dict b into a (non-mutating)."""
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def file_sha256(p: Union[str, Path]) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def dict_hash_sha256(d: dict) -> str:
    data = json.dumps(d or {}, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def get_git_sha() -> Optional[str]:
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return sha
    except Exception:
        return os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA")


def load_config(path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """
    Load analysis config (YAML).
    Priority:
      1) explicit `path`
      2) env: ANALYSIS_CONFIG / BACKTEST_CONFIG / STRAT_CONFIG
      3) common project defaults (incl. runs/configs/config_analysis.yaml)
    """
    candidates: list[Path] = []
    if path:
        p = Path(path)
        candidates.extend([p, p.expanduser().resolve()])

    envp = (
        os.environ.get("ANALYSIS_CONFIG")
        or os.environ.get("BACKTEST_CONFIG")
        or os.environ.get("STRAT_CONFIG")
    )
    if envp:
        candidates.insert(0, Path(envp).expanduser().resolve())

    cwd = Path.cwd()
    project_root = cwd.parent if cwd.name == "src" else cwd
    candidates.extend(
        [
            project_root / "runs" / "configs" / "config_analysis.yaml",
            project_root / "runs" / "configs" / "config.yaml",
            cwd / "runs" / "configs" / "config_analysis.yaml",
            cwd / "runs" / "configs" / "config.yaml",
            cwd / "configs" / "config_analysis.yaml",
            project_root / "configs" / "config_analysis.yaml",
            project_root / "config" / "config_analysis.yaml",
            project_root / "config_analysis.yaml",
        ]
    )

    tried: list[str] = []
    for c in candidates:
        tried.append(str(c))
        try:
            if c.exists():
                with c.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if not isinstance(cfg, dict):
                    raise TypeError(
                        f"Config root must be a mapping, got {type(cfg).__name__}"
                    )
                logger.info("Loaded config from %s", c)
                return cfg
        except Exception as e:
            logger.debug("Error reading config %s: %s", c, e)
            continue

    raise FileNotFoundError(
        "Config file not found. Tried:\n  "
        + "\n  ".join(tried)
        + "\nPass --cfg <path> or set ANALYSIS_CONFIG."
    )
