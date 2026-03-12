# src/tests/processing/conftest.py
from __future__ import annotations

import json
import os
import sys
import time as _t
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# -----------------------------------------------------------------------------
# Paths (robust for src layout)
# this file: <repo-root>/src/tests/processing/conftest.py
# -----------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
SRC_DIR = _THIS.parents[2]  # -> <repo-root>/src
REPO_ROOT = _THIS.parents[3]  # -> <repo-root>

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))  # makes "import processing" importable

# -----------------------------------------------------------------------------
# Basic determinism
# -----------------------------------------------------------------------------
os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(1337)
try:
    pd.options.mode.copy_on_write = True  # pandas >= 2.x
except Exception:
    pass


# -----------------------------------------------------------------------------
# Global test environment
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _stable_env(monkeypatch: pytest.MonkeyPatch):
    # fixed TZ and no production config variables
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.delenv("BACKTEST_CONFIG", raising=False)
    monkeypatch.delenv("STRAT_CONFIG", raising=False)
    if hasattr(_t, "tzset"):  # POSIX
        _t.tzset()


@pytest.fixture
def golden_dir() -> Path:
    """
    Goldens under <repo-root>/src/tests/golden/processing.
    """
    gdir = REPO_ROOT / "src" / "tests" / "golden" / "processing"
    gdir.mkdir(parents=True, exist_ok=True)
    return gdir


@pytest.fixture
def update_golden() -> bool:
    return os.environ.get("UPDATE_GOLDEN", "0") == "1"


# -----------------------------------------------------------------------------
# Stubs for optional modules (no repo side effects)
# -----------------------------------------------------------------------------
@pytest.fixture
def stub_processing_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    # processing.liquidity
    liq = types.ModuleType("processing.liquidity")

    def build_adv_map_from_price_and_volume(
        prices: pd.DataFrame, volume: pd.DataFrame, window: int = 21
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for col in prices.columns:
            lp = float(prices[col].iloc[-1]) if len(prices.index) else float("nan")
            out[str(col)] = {"adv": float(window), "last_price": lp}
        return out

    def build_adv_map_with_gates(
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        window: int = 21,
        **kwargs,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        adv = build_adv_map_from_price_and_volume(prices, volume, window=window)
        metrics = {
            str(col): {
                "total_windows": max(0, len(prices.index) - int(window) + 1),
                "valid_windows": max(0, len(prices.index) - int(window) + 1),
                "invalid_windows": 0,
                "valid_window_ratio": 1.0,
                "invalid_window_ratio": 0.0,
                "gate_pass": True,
            }
            for col in prices.columns
        }
        return adv, metrics

    liq.build_adv_map_from_price_and_volume = build_adv_map_from_price_and_volume  # type: ignore[attr-defined]
    liq.build_adv_map_with_gates = build_adv_map_with_gates  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "processing.liquidity", liq)

    # processing.corporate_actions (No-Op)
    ca = types.ModuleType("processing.corporate_actions")

    def apply_corporate_actions(
        df: pd.DataFrame, actions_df: pd.DataFrame
    ) -> pd.DataFrame:
        return df

    ca.apply_corporate_actions = apply_corporate_actions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "processing.corporate_actions", ca)


# -----------------------------------------------------------------------------
# Masking utilities for golden JSONs
# -----------------------------------------------------------------------------
def _round_floats(d: Any, ndigits: int = 6) -> Any:
    if isinstance(d, float):
        return round(d, ndigits)
    if isinstance(d, dict):
        return {k: _round_floats(v, ndigits) for k, v in d.items()}
    if isinstance(d, list):
        return [_round_floats(v, ndigits) for v in d]
    return d


def _mask_diag_payload(payload: dict[str, Any]) -> dict[str, Any]:
    proc = payload.get("processing") or {}
    quality = payload.get("quality") or {}
    pre = (quality.get("pre_exec") or quality.get("pre") or {}).get("checks") or {}
    post = (quality.get("post") or {}).get("checks") or {}
    kept = int(proc.get("kept", 0))
    removed = int(proc.get("removed", 0))

    keep = {
        "pre_quality": {
            "checks": {"rows": int(pre.get("rows", 0)), "cols": int(pre.get("cols", 0))}
        },
        "post_quality": {
            "checks": {
                "rows": int(post.get("rows", 0)),
                "cols": int(post.get("cols", 0)),
            }
        },
        "removed_count": removed,
        "kept_count": kept,
        "processing_agg": {
            "kept": int(proc.get("kept", 0)),
            "removed": int(proc.get("removed", 0)),
            "grid_mode": proc.get("grid_mode"),
            "calendar": proc.get("calendar"),
            "mean_non_na_pct": round(float(proc.get("mean_non_na_pct", 0.0)), 6),
            "max_longest_gap_kept": int(proc.get("max_longest_gap_kept", 0)),
            "sum_outliers_flagged": int(proc.get("sum_outliers_flagged", 0)),
        },
    }
    return _round_floats(keep)


def _mask_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs") or {}
    extra = payload.get("extra") or {}
    proc = (extra.get("processing") or {}) if isinstance(extra, dict) else {}
    metrics = (extra.get("metrics") or {}) if isinstance(extra, dict) else {}

    def _basename_or_none(x: str | None) -> str | None:
        return Path(x).name if x else None

    keep = {
        "cfg_path": Path(payload.get("cfg_path", "config.yaml")).name,
        "inputs": {
            "raw_prices": {
                "path": _basename_or_none((inputs.get("raw_prices") or {}).get("path")),
                "sha1": (inputs.get("raw_prices") or {}).get("sha1"),
            },
            "raw_volume": {
                "path": _basename_or_none((inputs.get("raw_volume") or {}).get("path")),
                "sha1": (inputs.get("raw_volume") or {}).get("sha1"),
            },
        },
        "extra": {
            "metrics": {
                "kept": int(metrics.get("kept", proc.get("kept", 0))),
                "removed": int(metrics.get("removed", proc.get("removed", 0))),
            }
        },
    }
    return _round_floats(keep)


@pytest.fixture
def mask_diag_payload() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return _mask_diag_payload


@pytest.fixture
def mask_manifest_payload() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return _mask_manifest_payload


@pytest.fixture
def write_json() -> Callable[[Path, dict[str, Any]], None]:
    def _write(path: Path, obj: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    return _write
