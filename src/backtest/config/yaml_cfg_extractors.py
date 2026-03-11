"""
YAML config extractors for the BacktestConfig loader.

This module is intentionally small and dependency-free:
- It normalizes only what the backtest engine needs.
- Only the supported execution modes remain: lob and light.
- The raw YAML remains available via BacktestConfig.raw_yaml.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["_extract_execution_cfg_for_yaml", "_extract_calendar_cfg_for_yaml"]


def _as_dict(x: Any) -> dict[str, Any]:
    return dict(x) if isinstance(x, Mapping) else {}


def _extract_execution_cfg_for_yaml(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize execution config.

    Supported (kept): execution.lob and execution.light.
    Legacy aliases and nested legacy LOB schemas are rejected elsewhere and are
    intentionally not normalized here anymore.
    """
    root = _as_dict(cfg)
    ex = _as_dict(root.get("execution"))

    mode_raw = str(ex.get("mode") or "lob").strip().lower()

    mode = "light" if mode_raw == "light" else "lob"

    lob_raw = _as_dict(ex.get("lob"))

    # Preserve original mode for observability/debugging.
    if mode_raw not in {"lob", "light"}:
        lob_raw.setdefault("_mode_removed", mode_raw)

    light_raw = _as_dict(ex.get("light"))

    return {
        "mode": mode,
        "lob": lob_raw,
        "light": light_raw,
    }


def _extract_calendar_cfg_for_yaml(
    backtest_cfg: Mapping[str, Any], root_cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """
    Normalize calendar-related knobs.

    Sources:
      - backtest.calendar_mapping
      - backtest.settlement_lag_bars
    """
    bt = _as_dict(backtest_cfg)

    out: dict[str, Any] = {}
    out["calendar_mapping"] = bt.get("calendar_mapping") or "prior"
    out["settlement_lag_bars"] = bt.get("settlement_lag_bars", 0)
    return out
