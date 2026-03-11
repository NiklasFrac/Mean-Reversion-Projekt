from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import logger

__all__ = ["VendorGuardsResult", "apply_vendor_guards"]

_EVENT_COLUMNS = [
    "ts",
    "symbol",
    "stage",
    "field",
    "rule",
    "severity",
    "action",
    "source",
    "value_open",
    "value_high",
    "value_low",
    "value_close",
    "value_volume",
    "metric_value",
    "threshold",
    "note",
]


@dataclass
class VendorGuardsResult:
    df_exec_raw: pd.DataFrame
    panel_fields: dict[str, pd.DataFrame]
    volume_for_processing: pd.DataFrame | None
    anomalies: pd.DataFrame
    vendor_guards_summary: dict[str, Any]
    split_excluded_symbols: list[str]
    zero_volume_stats: dict[str, Any]
    ohlc_sanity_stats: dict[str, Any]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=_EVENT_COLUMNS)


def _shape(df: pd.DataFrame | None) -> dict[str, int]:
    if df is None:
        return {"rows": 0, "cols": 0}
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1])}


def _to_numeric(
    df: pd.DataFrame | None, *, ref_index: pd.DatetimeIndex
) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy().reindex(ref_index)
    out.columns = pd.Index(map(str, out.columns))
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.astype(float)


def _normalize_ts(value: Any, *, ref_index: pd.DatetimeIndex) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    try:
        if ref_index.tz is None:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            else:
                ts = ts.tz_localize(None)
        else:
            if ts.tzinfo is None:
                ts = ts.tz_localize(ref_index.tz)
            else:
                ts = ts.tz_convert(ref_index.tz)
    except Exception:
        return None
    return ts


def _snapshot(
    *,
    ts: pd.Timestamp,
    symbol: str,
    close_df: pd.DataFrame,
    panel_fields: dict[str, pd.DataFrame],
    volume_df: pd.DataFrame | None,
) -> dict[str, float]:
    def _cell(df: pd.DataFrame | None) -> float:
        if df is None or df.empty:
            return float("nan")
        if symbol not in df.columns or ts not in df.index:
            return float("nan")
        try:
            v = float(df.at[ts, symbol])
            return v if np.isfinite(v) else float("nan")
        except Exception:
            return float("nan")

    return {
        "value_open": _cell(panel_fields.get("open")),
        "value_high": _cell(panel_fields.get("high")),
        "value_low": _cell(panel_fields.get("low")),
        "value_close": _cell(close_df),
        "value_volume": _cell(volume_df),
    }


def _target_fields(value: str) -> list[str]:
    v = str(value or "").strip().lower()
    if v in {"*", "all", "ohlc"}:
        return ["open", "high", "low", "close"]
    if not v:
        return []
    return [v]


def apply_vendor_guards(
    *,
    df_exec_raw: pd.DataFrame,
    panel_fields: dict[str, pd.DataFrame],
    volume_for_processing: pd.DataFrame | None,
    config: dict[str, Any] | None = None,
) -> VendorGuardsResult:
    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", True))

    close = _to_numeric(df_exec_raw, ref_index=pd.DatetimeIndex(df_exec_raw.index))
    if close is None:
        close = pd.DataFrame(index=pd.DatetimeIndex(df_exec_raw.index))
    ref_index = pd.DatetimeIndex(close.index)
    panel: dict[str, pd.DataFrame] = {}
    for name, frame in (panel_fields or {}).items():
        key = str(name).strip().lower()
        norm = _to_numeric(frame, ref_index=ref_index)
        if norm is not None:
            panel[key] = norm
    panel.setdefault("close", close.copy())
    volume = _to_numeric(volume_for_processing, ref_index=ref_index)

    before = {
        "prices_exec": _shape(close),
        "volume_for_processing": _shape(volume),
        "panel_fields": {k: _shape(v) for k, v in panel.items()},
    }

    if not enabled:
        summary = {
            "enabled": False,
            "total_events": 0,
            "counts_by_rule": {"manual_bad_row": 0},
            "top_symbols": [],
            "shape_before": before,
            "shape_after": before,
        }
        return VendorGuardsResult(
            df_exec_raw=close,
            panel_fields=panel,
            volume_for_processing=volume,
            anomalies=_empty_events(),
            vendor_guards_summary=summary,
            split_excluded_symbols=[],
            zero_volume_stats={"affected_rows": 0, "affected_symbols": 0, "events": 0},
            ohlc_sanity_stats={
                "order_events": 0,
                "nonpositive_events": 0,
                "hard_row_events": 0,
                "masked_cells": 0,
            },
        )

    bad_rows_cfg = dict(cfg.get("bad_rows", {}) or {})
    bad_rows_enabled = bool(bad_rows_cfg.get("enabled", True))
    bad_rows_path = Path(
        str(bad_rows_cfg.get("path", "runs/data/vendor_bad_rows.parquet"))
    )
    bad_rows_action = str(bad_rows_cfg.get("action", "mask_nan") or "mask_nan")

    rows: list[dict[str, Any]] = []
    if bad_rows_enabled:
        if not bad_rows_path.exists():
            logger.warning(
                "Vendor guards bad_rows enabled but file not found: %s",
                bad_rows_path,
            )
        else:
            try:
                bad_rows_df = (
                    pd.read_parquet(bad_rows_path)
                    if bad_rows_path.suffix.lower() == ".parquet"
                    else pd.read_csv(bad_rows_path)
                )
                bad_rows_df = bad_rows_df.rename(
                    columns={
                        str(c): str(c).strip().lower() for c in bad_rows_df.columns
                    }
                )
                required = {"ts", "symbol", "field"}
                if not required.issubset(set(bad_rows_df.columns)):
                    raise ValueError(
                        f"bad_rows requires columns {sorted(required)}; got {sorted(bad_rows_df.columns)}"
                    )
                for _, row in bad_rows_df.iterrows():
                    ts = _normalize_ts(row.get("ts"), ref_index=ref_index)
                    if ts is None or ts not in ref_index:
                        continue
                    sym = str(row.get("symbol", "")).strip()
                    if not sym:
                        continue
                    reason = str(row.get("reason", "")).strip() or "manual_bad_rows"
                    for field in _target_fields(str(row.get("field", ""))):
                        if field == "close":
                            if sym not in close.columns:
                                continue
                            snap = _snapshot(
                                ts=ts,
                                symbol=sym,
                                close_df=close,
                                panel_fields=panel,
                                volume_df=volume,
                            )
                            close.at[ts, sym] = np.nan
                            panel["close"].at[ts, sym] = np.nan
                        elif field in {"open", "high", "low"}:
                            if field not in panel or sym not in panel[field].columns:
                                continue
                            snap = _snapshot(
                                ts=ts,
                                symbol=sym,
                                close_df=close,
                                panel_fields=panel,
                                volume_df=volume,
                            )
                            panel[field].at[ts, sym] = np.nan
                        elif field == "volume":
                            if volume is None or sym not in volume.columns:
                                continue
                            snap = _snapshot(
                                ts=ts,
                                symbol=sym,
                                close_df=close,
                                panel_fields=panel,
                                volume_df=volume,
                            )
                            volume.at[ts, sym] = np.nan
                            if "volume" in panel and sym in panel["volume"].columns:
                                panel["volume"].at[ts, sym] = np.nan
                        else:
                            continue
                        rows.append(
                            {
                                "ts": ts,
                                "symbol": sym,
                                "stage": "vendor_guards",
                                "field": field,
                                "rule": "manual_bad_row",
                                "severity": "error",
                                "action": bad_rows_action,
                                "source": reason,
                                **snap,
                                "metric_value": np.nan,
                                "threshold": np.nan,
                                "note": None,
                            }
                        )
            except Exception as exc:
                logger.warning(
                    "Vendor guards bad_rows failed for %s: %s",
                    bad_rows_path,
                    exc,
                )

    anomalies = pd.DataFrame(rows, columns=_EVENT_COLUMNS) if rows else _empty_events()
    if not anomalies.empty:
        anomalies = anomalies.sort_values(
            by=["ts", "symbol", "stage", "rule", "field"],
            kind="stable",
        ).reset_index(drop=True)

    counts = {"manual_bad_row": int((anomalies["rule"] == "manual_bad_row").sum())}
    top_symbols: list[dict[str, Any]] = []
    if not anomalies.empty:
        vc = anomalies["symbol"].value_counts().head(10)
        top_symbols = [{"symbol": str(k), "count": int(v)} for k, v in vc.items()]

    after = {
        "prices_exec": _shape(close),
        "volume_for_processing": _shape(volume),
        "panel_fields": {k: _shape(v) for k, v in panel.items()},
    }
    summary = {
        "enabled": True,
        "total_events": int(anomalies.shape[0]),
        "counts_by_rule": counts,
        "top_symbols": top_symbols,
        "shape_before": before,
        "shape_after": after,
    }

    return VendorGuardsResult(
        df_exec_raw=close,
        panel_fields=panel,
        volume_for_processing=volume,
        anomalies=anomalies,
        vendor_guards_summary=summary,
        split_excluded_symbols=[],
        zero_volume_stats={"affected_rows": 0, "affected_symbols": 0, "events": 0},
        ohlc_sanity_stats={
            "order_events": 0,
            "nonpositive_events": 0,
            "hard_row_events": 0,
            "masked_cells": 0,
        },
    )
