from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast, overload

import yaml

from .logging_utils import logger

__all__ = ["load_config"]


@overload
def load_config(
    path: Path | None = None, *, return_source: Literal[False] = False
) -> dict[str, Any]: ...


@overload
def load_config(
    path: Path | None = None, *, return_source: Literal[True]
) -> tuple[dict[str, Any], Path]: ...


def load_config(
    path: Path | None = None, *, return_source: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], Path]:
    raw_candidates: list[Path] = []
    explicit_candidates: set[str] = set()
    if path:
        explicit = Path(path)
        explicit_resolved = explicit.expanduser().resolve()
        raw_candidates += [explicit, explicit_resolved]
        explicit_candidates = {str(explicit), str(explicit_resolved)}

    envp = (
        os.environ.get("PROCESSING_CONFIG")
        or os.environ.get("BACKTEST_CONFIG")
        or os.environ.get("STRAT_CONFIG")
    )
    if envp:
        # Keep explicit path authoritative when provided. ENV only acts as fallback.
        raw_candidates.append(Path(envp).expanduser().resolve())

    cwd = Path.cwd()
    project_root = cwd.parent if cwd.name == "src" else cwd
    raw_candidates += [
        cwd / "configs" / "config_processing.yaml",
        project_root / "runs" / "configs" / "config_processing.yaml",
        Path(__file__).parent.parent / "configs" / "config_processing.yaml",
    ]

    candidates: list[Path] = []
    seen: set[str] = set()
    for c in raw_candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(c)

    tried: list[str] = []
    for c in candidates:
        if not c:
            continue
        tried.append(str(c))
        try:
            if c.exists():
                with c.open("r", encoding="utf8") as f:
                    cfg_loaded: Any = yaml.safe_load(f)
                cfg: dict[str, Any] = cast(
                    dict[str, Any], cfg_loaded if isinstance(cfg_loaded, dict) else {}
                )
                logger.info("Loaded config from %s", c)
                if return_source:
                    return cfg, c
                return cfg
        except Exception as exc:
            if str(c) in explicit_candidates:
                raise ValueError(f"Failed to parse explicit config path: {c}") from exc
            continue
    raise FileNotFoundError("Config file not found. Tried:\n  " + "\n  ".join(tried))
