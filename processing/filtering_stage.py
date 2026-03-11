from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .outliers import scrub_outliers_causal
from .processing_primitives import cap_extreme_returns, detect_stale
from .quality_helpers import validate_prices_wide
from .timebase import build_tradable_mask, pick_time_grid

EVENT_COLUMNS = [
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
class PreparedInputs:
    close_raw: pd.DataFrame
    panel_fields: dict[str, pd.DataFrame]
    volume: pd.DataFrame | None
    pre_q_raw: dict[str, Any]
    provisional_ref_index: pd.DatetimeIndex
    provisional_tradable_masks: dict[str, pd.Series]


@dataclass
class FinalizedInputs:
    close: pd.DataFrame
    panel_fields: dict[str, pd.DataFrame]
    volume: pd.DataFrame | None
    pre_q_exec: dict[str, Any]
    ref_index: pd.DatetimeIndex
    tradable_masks: dict[str, pd.Series]


@dataclass
class FilteringResult:
    close: pd.DataFrame
    panel_fields: dict[str, pd.DataFrame]
    volume: pd.DataFrame | None
    removed_symbols: list[str]
    drop_reasons: dict[str, str]
    symbol_diag: dict[str, dict[str, Any]]
    events: pd.DataFrame
    stages: dict[str, Any]
    snapshots: dict[str, Any]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _ensure_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_events()
    out = df.copy()
    for col in EVENT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[EVENT_COLUMNS]
    out = out.sort_values(
        by=["ts", "symbol", "stage", "rule", "field"],
        kind="stable",
    )
    return out.reset_index(drop=True)


def _dedupe_symbol_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not df.columns.has_duplicates:
        out = df.copy()
        out.columns = pd.Index(map(str, out.columns))
        return out
    out = df.T.groupby(level=0, sort=False).last().T
    out.columns = pd.Index(map(str, out.columns))
    return out


def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = _dedupe_symbol_columns(df.copy())
    out = out.sort_index()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.astype(float)


def _snapshot_values(
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


def _events_from_mask(
    *,
    mask: pd.DataFrame,
    stage: str,
    field: str,
    rule: str,
    severity: str,
    action: str,
    source: str,
    close_df: pd.DataFrame,
    panel_fields: dict[str, pd.DataFrame],
    volume_df: pd.DataFrame | None,
    metric_value: float | None = None,
    threshold: float | None = None,
    note: str | None = None,
) -> pd.DataFrame:
    if mask.empty:
        return _empty_events()
    mask_bool = mask.fillna(False).astype(bool)
    if not bool(mask_bool.to_numpy().any()):
        return _empty_events()

    hits = mask_bool.stack(future_stack=True)
    hits = hits[hits]
    if hits.empty:
        return _empty_events()

    rows: list[dict[str, Any]] = []
    for ts, symbol in hits.index:
        snapshot = _snapshot_values(
            ts=pd.Timestamp(ts),
            symbol=str(symbol),
            close_df=close_df,
            panel_fields=panel_fields,
            volume_df=volume_df,
        )
        rows.append(
            {
                "ts": pd.Timestamp(ts),
                "symbol": str(symbol),
                "stage": stage,
                "field": field,
                "rule": rule,
                "severity": severity,
                "action": action,
                "source": source,
                **snapshot,
                "metric_value": metric_value,
                "threshold": threshold,
                "note": note,
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _append_symbol_event(
    events: list[dict[str, Any]],
    *,
    ts: pd.Timestamp,
    symbol: str,
    stage: str,
    field: str,
    rule: str,
    severity: str,
    action: str,
    source: str,
    close_df: pd.DataFrame,
    panel_fields: dict[str, pd.DataFrame],
    volume_df: pd.DataFrame | None,
    metric_value: float | None = None,
    threshold: float | None = None,
    note: str | None = None,
) -> None:
    snapshot = _snapshot_values(
        ts=ts,
        symbol=symbol,
        close_df=close_df,
        panel_fields=panel_fields,
        volume_df=volume_df,
    )
    events.append(
        {
            "ts": ts,
            "symbol": str(symbol),
            "stage": stage,
            "field": field,
            "rule": rule,
            "severity": severity,
            "action": action,
            "source": source,
            **snapshot,
            "metric_value": metric_value,
            "threshold": threshold,
            "note": note,
        }
    )


def _build_tradable_masks_by_symbol(
    *,
    ref_index: pd.DatetimeIndex,
    symbols: list[str],
    calendar_code: str,
    rth_only: bool,
    symbol_calendar_map: dict[str, str] | None,
) -> dict[str, pd.Series]:
    per_symbol: dict[str, pd.Series] = {}
    cache: dict[str, pd.Series] = {}
    sc_map = symbol_calendar_map or {}
    for sym in symbols:
        cal = str(sc_map.get(sym, calendar_code) or calendar_code)
        if cal not in cache:
            cache[cal] = build_tradable_mask(
                ref_index,
                calendar_code=cal,
                rth_only=bool(rth_only),
            )
        per_symbol[sym] = cache[cal]
    return per_symbol


def _longest_nan_run(s: pd.Series) -> int:
    arr = s.isna().to_numpy()
    best = cur = 0
    for v in arr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _leading_nas(s: pd.Series) -> int:
    if s.empty:
        return 0
    if s.first_valid_index() is None:
        return int(len(s))
    return int(np.argmax(s.notna().to_numpy()))


def _trailing_nas(s: pd.Series) -> int:
    if s.empty:
        return 0
    if s.last_valid_index() is None:
        return int(len(s))
    return int(np.argmax(s[::-1].notna().to_numpy()))


def _drop_symbols_from_data(
    *,
    symbols: list[str],
    close_df: pd.DataFrame,
    panel_fields: dict[str, pd.DataFrame],
    volume_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame | None]:
    if not symbols:
        return close_df, panel_fields, volume_df
    close = close_df.drop(columns=symbols, errors="ignore")
    panel = {
        k: v.drop(columns=symbols, errors="ignore") for k, v in panel_fields.items()
    }
    volume = (
        volume_df.drop(columns=symbols, errors="ignore")
        if volume_df is not None
        else None
    )
    return close, panel, volume


def prepare_inputs(
    *,
    close_raw: pd.DataFrame,
    panel_fields: dict[str, pd.DataFrame],
    volume: pd.DataFrame | None,
    grid_mode: str,
    calendar_code: str,
    rth_only: bool,
    symbol_calendar_map: dict[str, str] | None,
) -> PreparedInputs:
    close = _to_numeric_df(close_raw)
    panel: dict[str, pd.DataFrame] = {}
    for name, frame in (panel_fields or {}).items():
        key = str(name).strip().lower()
        panel[key] = _to_numeric_df(frame.reindex(close.index))
    if "close" not in panel:
        panel["close"] = close.copy()

    vol: pd.DataFrame | None = None
    if volume is not None and not volume.empty:
        vol = _to_numeric_df(volume.reindex(close.index))

    pre_q_raw = validate_prices_wide(close)
    provisional_ref_index = pick_time_grid(
        close,
        mode=str(grid_mode),
        calendar_code=str(calendar_code),
    )
    provisional_masks = _build_tradable_masks_by_symbol(
        ref_index=provisional_ref_index,
        symbols=list(close.columns),
        calendar_code=str(calendar_code),
        rth_only=bool(rth_only),
        symbol_calendar_map=symbol_calendar_map,
    )

    return PreparedInputs(
        close_raw=close,
        panel_fields=panel,
        volume=vol,
        pre_q_raw=pre_q_raw,
        provisional_ref_index=provisional_ref_index,
        provisional_tradable_masks=provisional_masks,
    )


def finalize_after_vendor(
    *,
    close_after_vendor: pd.DataFrame,
    panel_fields_after_vendor: dict[str, pd.DataFrame],
    volume_after_vendor: pd.DataFrame | None,
    grid_mode: str,
    calendar_code: str,
    rth_only: bool,
    symbol_calendar_map: dict[str, str] | None,
) -> FinalizedInputs:
    close = _to_numeric_df(close_after_vendor)
    ref_index = pick_time_grid(
        close,
        mode=str(grid_mode),
        calendar_code=str(calendar_code),
    )
    close = close.reindex(ref_index)

    panel: dict[str, pd.DataFrame] = {}
    for name, frame in (panel_fields_after_vendor or {}).items():
        key = str(name).strip().lower()
        aligned = _to_numeric_df(frame).reindex(ref_index)
        aligned = aligned.reindex(columns=close.columns)
        panel[key] = aligned
    panel["close"] = close.copy()

    vol: pd.DataFrame | None = None
    if volume_after_vendor is not None and not volume_after_vendor.empty:
        vol = _to_numeric_df(volume_after_vendor).reindex(ref_index)
        vol = vol.reindex(columns=close.columns)

    tradable_masks = _build_tradable_masks_by_symbol(
        ref_index=ref_index,
        symbols=list(close.columns),
        calendar_code=str(calendar_code),
        rth_only=bool(rth_only),
        symbol_calendar_map=symbol_calendar_map,
    )
    pre_q_exec = validate_prices_wide(close)

    return FinalizedInputs(
        close=close,
        panel_fields=panel,
        volume=vol,
        pre_q_exec=pre_q_exec,
        ref_index=ref_index,
        tradable_masks=tradable_masks,
    )


def run_filtering_stages(
    *,
    finalized: FinalizedInputs,
    stage1_cfg: dict[str, Any],
    stage2_cfg: dict[str, Any],
    reverse_split_cfg: dict[str, Any],
    caps_cfg: dict[str, Any],
    outlier_cfg: dict[str, Any],
    staleness_cfg: dict[str, Any],
    close_raw_unadj: pd.DataFrame | None,
    strict_inputs: bool,
) -> FilteringResult:
    close = finalized.close.copy()
    panel = {k: v.copy() for k, v in finalized.panel_fields.items()}
    volume = finalized.volume.copy() if finalized.volume is not None else None
    ref_index = finalized.ref_index
    tradable_masks = finalized.tradable_masks

    drop_reasons: dict[str, str] = {}
    symbol_diag: dict[str, dict[str, Any]] = {}
    event_chunks: list[pd.DataFrame] = []
    event_rows: list[dict[str, Any]] = []

    removed_stage1: list[str] = []
    for sym in list(close.columns):
        mask = (
            tradable_masks.get(sym, pd.Series(True, index=ref_index))
            .reindex(ref_index)
            .fillna(False)
        )
        trad_idx = mask[mask].index
        s_raw = pd.to_numeric(close[sym].reindex(ref_index), errors="coerce").astype(
            float
        )
        if trad_idx.empty:
            removed_stage1.append(sym)
            drop_reasons[sym] = "no_tradable_window"
            _append_symbol_event(
                event_rows,
                ts=ref_index[-1] if len(ref_index) else pd.Timestamp.utcnow(),
                symbol=sym,
                stage="stage1",
                field="close",
                rule="no_tradable_window",
                severity="error",
                action="drop_symbol",
                source="auto",
                close_df=close,
                panel_fields=panel,
                volume_df=volume,
            )
            symbol_diag[sym] = {"reason": "no_tradable_window", "stage": "stage1"}
            continue
        s_trad = s_raw.loc[trad_idx]
        if s_trad.dropna().empty:
            removed_stage1.append(sym)
            drop_reasons[sym] = "empty_after_reindex"
            _append_symbol_event(
                event_rows,
                ts=trad_idx[-1],
                symbol=sym,
                stage="stage1",
                field="close",
                rule="empty_after_reindex",
                severity="error",
                action="drop_symbol",
                source="auto",
                close_df=close,
                panel_fields=panel,
                volume_df=volume,
            )
            symbol_diag[sym] = {"reason": "empty_after_reindex", "stage": "stage1"}
            continue
        nonpos_raw = int((s_trad.dropna() <= 0).sum())
        if nonpos_raw > 0:
            removed_stage1.append(sym)
            drop_reasons[sym] = "nonpositive_raw_values"
            _append_symbol_event(
                event_rows,
                ts=trad_idx[-1],
                symbol=sym,
                stage="stage1",
                field="close",
                rule="nonpositive_raw_values",
                severity="error",
                action="drop_symbol",
                source="auto",
                close_df=close,
                panel_fields=panel,
                volume_df=volume,
                metric_value=float(nonpos_raw),
                threshold=0.0,
            )
            symbol_diag[sym] = {
                "reason": "nonpositive_raw_values",
                "stage": "stage1",
                "nonpositive_raw_values": nonpos_raw,
            }

    close, panel, volume = _drop_symbols_from_data(
        symbols=sorted(set(removed_stage1)),
        close_df=close,
        panel_fields=panel,
        volume_df=volume,
    )

    ohlc_cfg = dict((stage1_cfg.get("ohlc_mask") or {}))
    ohlc_enabled = bool(ohlc_cfg.get("enabled", True))
    eps_abs = float(ohlc_cfg.get("eps_abs", 1.0e-12))
    eps_rel = float(ohlc_cfg.get("eps_rel", 1.0e-8))
    mask_nonpositive = bool(ohlc_cfg.get("mask_nonpositive", True))
    ohlc_stats = {
        "hard_row_events": 0,
        "order_events": 0,
        "nonpositive_events": 0,
        "masked_cells": 0,
    }

    if (
        ohlc_enabled
        and all(f in panel for f in ("open", "high", "low"))
        and not close.empty
    ):
        syms = list(close.columns)
        tradable_df = pd.DataFrame(
            {
                sym: (
                    tradable_masks.get(sym, pd.Series(True, index=ref_index))
                    .reindex(ref_index)
                    .fillna(False)
                )
                for sym in syms
            },
            index=ref_index,
        ).astype(bool)

        open_df = panel["open"].reindex(ref_index).reindex(columns=syms)
        high_df = panel["high"].reindex(ref_index).reindex(columns=syms)
        low_df = panel["low"].reindex(ref_index).reindex(columns=syms)
        close_df = close.reindex(ref_index).reindex(columns=syms)

        arr_open = open_df.to_numpy(dtype=float)
        arr_high = high_df.to_numpy(dtype=float)
        arr_low = low_df.to_numpy(dtype=float)
        arr_close = close_df.to_numpy(dtype=float)
        finite_all = (
            np.isfinite(arr_open)
            & np.isfinite(arr_high)
            & np.isfinite(arr_low)
            & np.isfinite(arr_close)
        )
        stacked = np.abs(np.stack([arr_open, arr_high, arr_low, arr_close], axis=2))
        # Avoid RuntimeWarning("All-NaN slice encountered") by treating non-finite
        # values as zero scale for epsilon construction.
        stacked = np.where(np.isfinite(stacked), stacked, 0.0)
        scale = np.max(stacked, axis=2)
        eps = eps_abs + eps_rel * scale

        hard_cross = finite_all & (arr_high < (arr_low - eps))
        valid_order = finite_all & ~hard_cross
        max_ocl = np.maximum(np.maximum(arr_open, arr_close), arr_low)
        min_och = np.minimum(np.minimum(arr_open, arr_close), arr_high)
        high_order = valid_order & (arr_high < (max_ocl - eps))
        low_order = valid_order & (arr_low > (min_och + eps))
        nonpos_open = np.isfinite(arr_open) & (arr_open <= eps)
        nonpos_high = np.isfinite(arr_high) & (arr_high <= eps)
        nonpos_low = np.isfinite(arr_low) & (arr_low <= eps)
        nonpos_close = np.isfinite(arr_close) & (arr_close <= eps)

        hard_mask = (
            pd.DataFrame(hard_cross, index=ref_index, columns=syms) & tradable_df
        )
        high_mask = (
            pd.DataFrame(high_order, index=ref_index, columns=syms) & tradable_df
        )
        low_mask = pd.DataFrame(low_order, index=ref_index, columns=syms) & tradable_df
        nonpos_masks = {
            "open": pd.DataFrame(nonpos_open, index=ref_index, columns=syms)
            & tradable_df,
            "high": pd.DataFrame(nonpos_high, index=ref_index, columns=syms)
            & tradable_df,
            "low": pd.DataFrame(nonpos_low, index=ref_index, columns=syms)
            & tradable_df,
            "close": pd.DataFrame(nonpos_close, index=ref_index, columns=syms)
            & tradable_df,
        }

        event_chunks.append(
            _events_from_mask(
                mask=hard_mask,
                stage="stage1",
                field="ohlc",
                rule="ohlc_hard_cross",
                severity="error",
                action="mask_row",
                source="auto",
                close_df=close,
                panel_fields=panel,
                volume_df=volume,
                threshold=eps_abs,
            )
        )
        event_chunks.append(
            _events_from_mask(
                mask=high_mask,
                stage="stage1",
                field="high",
                rule="ohlc_high_order_violation",
                severity="warning",
                action="mask_field",
                source="auto",
                close_df=close,
                panel_fields=panel,
                volume_df=volume,
                threshold=eps_abs,
            )
        )
        event_chunks.append(
            _events_from_mask(
                mask=low_mask,
                stage="stage1",
                field="low",
                rule="ohlc_low_order_violation",
                severity="warning",
                action="mask_field",
                source="auto",
                close_df=close,
                panel_fields=panel,
                volume_df=volume,
                threshold=eps_abs,
            )
        )

        for field_name in ("open", "high", "low", "close"):
            frame = panel[field_name] if field_name in panel else close
            frame = frame.reindex(ref_index).reindex(columns=syms)
            frame = frame.mask(hard_mask)
            if field_name == "high":
                frame = frame.mask(high_mask)
            if field_name == "low":
                frame = frame.mask(low_mask)
            if mask_nonpositive:
                frame = frame.mask(nonpos_masks[field_name])
                event_chunks.append(
                    _events_from_mask(
                        mask=nonpos_masks[field_name],
                        stage="stage1",
                        field=field_name,
                        rule="ohlc_nonpositive",
                        severity="error",
                        action="mask_field",
                        source="auto",
                        close_df=close,
                        panel_fields=panel,
                        volume_df=volume,
                        threshold=eps_abs,
                    )
                )
            if field_name == "close":
                close = frame
            else:
                panel[field_name] = frame
        panel["close"] = close.copy()

        ohlc_stats = {
            "hard_row_events": int(hard_mask.to_numpy().sum()),
            "order_events": int(
                hard_mask.to_numpy().sum()
                + high_mask.to_numpy().sum()
                + low_mask.to_numpy().sum()
            ),
            "nonpositive_events": int(
                sum(nonpos_masks[f].to_numpy().sum() for f in nonpos_masks)
            ),
            "masked_cells": int(
                (hard_mask.to_numpy().sum() * 4)
                + high_mask.to_numpy().sum()
                + low_mask.to_numpy().sum()
                + sum(nonpos_masks[f].to_numpy().sum() for f in nonpos_masks)
            ),
        }

    for sym in close.columns:
        s = pd.to_numeric(close[sym], errors="coerce").astype(float)
        if bool(caps_cfg.get("enabled", True)):
            s = cap_extreme_returns(
                s,
                lower=float(caps_cfg.get("lower", -0.9)),
                upper=float(caps_cfg.get("upper", 3.0)),
                exclude_dates=None,
                max_gap_bars=int(stage2_cfg.get("max_gap_bars", 3)),
                mode=str(caps_cfg.get("mode", "nan") or "nan"),
            )
        if bool(outlier_cfg.get("enabled", True)):
            s, n_out = scrub_outliers_causal(
                s,
                zscore=float(outlier_cfg.get("zscore", 8.0)),
                window=int(outlier_cfg.get("window", 21)),
                use_log_returns=bool(outlier_cfg.get("use_log_returns", True)),
                exclude_dates=None,
                max_gap_bars=int(stage2_cfg.get("max_gap_bars", 3)),
            )
            symbol_diag.setdefault(sym, {})["outliers_flagged"] = int(n_out)
        close[sym] = s
    panel["close"] = close.copy()

    zero_cfg = dict((stage1_cfg.get("zero_volume_with_price") or {}))
    zero_enabled = bool(zero_cfg.get("enabled", True))
    min_price_for_zero_rule = float(zero_cfg.get("min_price_for_zero_volume_rule", 0.0))
    zero_stats = {"affected_rows": 0, "affected_symbols": 0, "events": 0}
    if zero_enabled and volume is not None and not volume.empty and not close.empty:
        vol_view = volume.reindex(index=close.index, columns=close.columns)
        mask = vol_view.eq(0.0) & close.notna()
        if min_price_for_zero_rule > 0:
            mask &= close >= min_price_for_zero_rule
        tradable_df = pd.DataFrame(
            {
                sym: (
                    tradable_masks.get(sym, pd.Series(True, index=ref_index))
                    .reindex(ref_index)
                    .fillna(False)
                )
                for sym in close.columns
            },
            index=ref_index,
        ).astype(bool)
        mask &= tradable_df
        if bool(mask.to_numpy().any()):
            event_chunks.append(
                _events_from_mask(
                    mask=mask,
                    stage="stage1",
                    field="volume",
                    rule="zero_volume_with_price",
                    severity="warning",
                    action="mask_field",
                    source="auto",
                    close_df=close,
                    panel_fields=panel,
                    volume_df=volume,
                    threshold=min_price_for_zero_rule,
                )
            )
            volume = volume.mask(mask)
            zero_stats = {
                "affected_rows": int(mask.any(axis=1).sum()),
                "affected_symbols": int(mask.any(axis=0).sum()),
                "events": int(mask.to_numpy().sum()),
            }

    removed_stage2: list[str] = []
    keep_pct_threshold = float(stage2_cfg.get("keep_pct_threshold", 0.8))
    max_start_na = int(stage2_cfg.get("max_start_na", 3))
    max_end_na = int(stage2_cfg.get("max_end_na", 3))
    max_gap = int(stage2_cfg.get("max_gap_bars", 3))
    hard_drop = bool(stage2_cfg.get("hard_drop", True))
    min_trad_rows_for_ohl_gate = int(
        stage2_cfg.get("min_tradable_rows_for_ohl_gate", 40)
    )
    ohl_missing_pct_max = float(stage2_cfg.get("ohl_missing_pct_max", 0.25))

    for sym in list(close.columns):
        mask = (
            tradable_masks.get(sym, pd.Series(True, index=ref_index))
            .reindex(ref_index)
            .fillna(False)
        )
        trad_idx = mask[mask].index
        s = pd.to_numeric(close[sym].reindex(ref_index), errors="coerce").astype(float)
        s_trad = s.loc[trad_idx]
        ref_len = int(len(trad_idx))
        non_na_pct = (
            float(s_trad.notna().sum()) / float(ref_len) if ref_len > 0 else 0.0
        )
        start_na = _leading_nas(s_trad)
        end_na = _trailing_nas(s_trad)
        longest_gap = _longest_nan_run(s_trad)

        ohl_any_missing_pct = 0.0
        if all(f in panel for f in ("open", "high", "low")) and ref_len > 0:
            ohl_any_missing = (
                panel["open"].reindex(ref_index)[sym].loc[trad_idx].isna()
                | panel["high"].reindex(ref_index)[sym].loc[trad_idx].isna()
                | panel["low"].reindex(ref_index)[sym].loc[trad_idx].isna()
            )
            ohl_any_missing_pct = (
                float(ohl_any_missing.mean()) if len(ohl_any_missing) else 0.0
            )

        diag = {
            "non_na_pct": non_na_pct,
            "start_na": int(start_na),
            "end_na": int(end_na),
            "longest_gap": int(longest_gap),
            "ohl_any_missing_pct": float(ohl_any_missing_pct),
            "tradable_rows_count": int(ref_len),
        }
        symbol_diag.setdefault(sym, {}).update(diag)

        reason: str | None = None
        if non_na_pct < keep_pct_threshold:
            reason = "insufficient_coverage"
        elif start_na > max_start_na:
            reason = "excess_leading_na"
        elif end_na > max_end_na:
            reason = "excess_trailing_na"
        elif longest_gap > max_gap:
            if hard_drop:
                reason = "large_gap"
            else:
                _append_symbol_event(
                    event_rows,
                    ts=(
                        trad_idx[-1]
                        if len(trad_idx)
                        else (
                            ref_index[-1] if len(ref_index) else pd.Timestamp.utcnow()
                        )
                    ),
                    symbol=sym,
                    stage="stage2",
                    field="close",
                    rule="large_gap",
                    severity="warning",
                    action="keep_with_gap",
                    source="auto",
                    close_df=close,
                    panel_fields=panel,
                    volume_df=volume,
                    metric_value=float(longest_gap),
                    threshold=float(max_gap),
                )
        elif (
            ref_len >= min_trad_rows_for_ohl_gate
            and ohl_any_missing_pct > ohl_missing_pct_max
        ):
            reason = "ohl_missing_pct_exceeded"
        else:
            if bool(staleness_cfg.get("enabled", False)) and ref_len > 0:
                vser = None
                if (
                    bool(staleness_cfg.get("use_volume", False))
                    and volume is not None
                    and sym in volume.columns
                ):
                    vser = pd.to_numeric(volume[sym].reindex(trad_idx), errors="coerce")
                if detect_stale(
                    s_trad,
                    k=int(staleness_cfg.get("k", 5)),
                    vol=vser,
                    vol_thresh=float(staleness_cfg.get("volume_threshold", 0.0)),
                    eps=float(staleness_cfg.get("eps", 1.0e-12)),
                    min_run=int(staleness_cfg.get("min_run", 2)),
                ):
                    reason = "stale_segment"

        if reason is not None:
            removed_stage2.append(sym)
            drop_reasons[sym] = reason
            symbol_diag[sym]["reason"] = reason
            _append_symbol_event(
                event_rows,
                ts=(
                    trad_idx[-1]
                    if len(trad_idx)
                    else (ref_index[-1] if len(ref_index) else pd.Timestamp.utcnow())
                ),
                symbol=sym,
                stage="stage2",
                field="close",
                rule=reason,
                severity="error",
                action="drop_symbol",
                source="auto",
                close_df=close,
                panel_fields=panel,
                volume_df=volume,
            )

    close, panel, volume = _drop_symbols_from_data(
        symbols=sorted(set(removed_stage2)),
        close_df=close,
        panel_fields=panel,
        volume_df=volume,
    )

    snapshot_pre_reverse = {
        "rows": int(len(ref_index)),
        "kept_count": int(close.shape[1]),
        "removed_count": int(len(removed_stage1) + len(removed_stage2)),
        "symbols_after_stage2": list(map(str, close.columns)),
        "drop_reasons": dict(drop_reasons),
    }

    reverse_enabled = bool(reverse_split_cfg.get("enabled", True))
    reverse_max_gap = int(reverse_split_cfg.get("max_split_jump_gap", max_gap))
    jump_threshold = float(reverse_split_cfg.get("jump_threshold", 1.10))
    level_window = int(reverse_split_cfg.get("level_window", 63))
    level_mad_k = float(reverse_split_cfg.get("level_mad_k", 6.0))
    post_persist_bars = int(reverse_split_cfg.get("post_persist_bars", 5))
    factor_tolerance = float(reverse_split_cfg.get("factor_tolerance", 0.20))
    weights = dict(reverse_split_cfg.get("weights") or {})
    w_jump = float(weights.get("jump", 0.35))
    w_level = float(weights.get("level", 0.25))
    w_persistence = float(weights.get("persistence", 0.25))
    w_factor = float(weights.get("factor", 0.15))
    score_flag_threshold = float(reverse_split_cfg.get("score_flag_threshold", 0.60))
    score_drop_threshold = float(reverse_split_cfg.get("score_drop_threshold", 0.80))
    drop_on_score_only = bool(reverse_split_cfg.get("drop_on_score_only", False))
    drop_on_flag_threshold = bool(
        reverse_split_cfg.get("drop_on_flag_threshold", False)
    )
    eps = 1.0e-12

    reverse_dropped: list[str] = []
    reverse_flagged: list[str] = []
    if reverse_enabled and not close.empty:
        if close_raw_unadj is None or close_raw_unadj.empty:
            if strict_inputs:
                raise FileNotFoundError(
                    "reverse_split_v2 requires close_raw_unadj but no unadjusted close input was found."
                )
            _append_symbol_event(
                event_rows,
                ts=ref_index[-1] if len(ref_index) else pd.Timestamp.utcnow(),
                symbol="*",
                stage="reverse_split",
                field="close",
                rule="reverse_split_unadjusted_missing",
                severity="warning",
                action="skip",
                source="auto",
                close_df=close,
                panel_fields=panel,
                volume_df=volume,
                note="strict_inputs=false",
            )
        else:
            unadj = (
                _to_numeric_df(close_raw_unadj)
                .reindex(ref_index)
                .reindex(columns=close.columns)
            )
            plausible_factors = np.array(
                [2.0, 3.0, 5.0, 10.0, 0.5, 1.0 / 3.0, 0.2, 0.1],
                dtype=float,
            )
            for sym in list(close.columns):
                adj_s = pd.to_numeric(close[sym], errors="coerce").astype(float)
                raw_s = pd.to_numeric(unadj[sym], errors="coerce").astype(float)
                valid = adj_s.gt(eps) & raw_s.gt(eps) & adj_s.notna() & raw_s.notna()
                if int(valid.sum()) < 2:
                    continue
                r = (raw_s / adj_s).where(valid)
                valid_pos = np.where(r.notna().to_numpy())[0]
                if valid_pos.size < 2:
                    continue

                best_abs_d = 0.0
                best_d = 0.0
                best_idx = None
                best_dist = None
                for j in range(1, len(valid_pos)):
                    cur = int(valid_pos[j])
                    prev = int(valid_pos[j - 1])
                    dist = cur - prev
                    if dist > reverse_max_gap:
                        continue
                    rv = float(r.iloc[cur])
                    pv = float(r.iloc[prev])
                    if (
                        not np.isfinite(rv)
                        or not np.isfinite(pv)
                        or rv <= eps
                        or pv <= eps
                    ):
                        continue
                    d = float(np.log(rv) - np.log(pv))
                    if abs(d) > best_abs_d:
                        best_abs_d = abs(d)
                        best_d = d
                        best_idx = cur
                        best_dist = dist

                if best_idx is None:
                    continue
                candidate_jump = bool(best_abs_d >= jump_threshold)
                start = max(0, int(best_idx) - level_window + 1)
                level_slice = r.iloc[start : int(best_idx) + 1].dropna()
                candidate_level = False
                if not level_slice.empty:
                    med = float(level_slice.median())
                    mad = float((level_slice - med).abs().median())
                    if np.isfinite(mad) and mad > eps:
                        z = abs(float(r.iloc[best_idx]) - med) / (1.4826 * mad)
                        candidate_level = bool(z >= level_mad_k)

                post = r.iloc[
                    int(best_idx) + 1 : int(best_idx) + 1 + post_persist_bars
                ].dropna()
                candidate_persistent = bool(
                    len(post) >= post_persist_bars
                    and np.all(
                        np.abs(
                            (post.to_numpy(dtype=float) / float(r.iloc[best_idx])) - 1.0
                        )
                        <= factor_tolerance
                    )
                )

                jump_factor = float(np.exp(best_d))
                candidate_factor = bool(
                    np.any(
                        np.abs((jump_factor - plausible_factors) / plausible_factors)
                        <= factor_tolerance
                    )
                )

                score = (
                    (w_jump if candidate_jump else 0.0)
                    + (w_level if candidate_level else 0.0)
                    + (w_persistence if candidate_persistent else 0.0)
                    + (w_factor if candidate_factor else 0.0)
                )

                d_sym = symbol_diag.get(sym, {})
                quality_problem = bool(
                    float(d_sym.get("non_na_pct", 1.0)) < 0.9
                    or int(d_sym.get("longest_gap", 0)) > max(1, max_gap // 2)
                    or float(d_sym.get("ohl_any_missing_pct", 0.0))
                    > (ohl_missing_pct_max * 0.5)
                )

                ts = ref_index[int(best_idx)]
                drop_by_flag = bool(score >= score_flag_threshold)
                drop_by_score = bool(score >= score_drop_threshold)
                if drop_on_flag_threshold:
                    should_drop = drop_by_flag
                    drop_threshold = score_flag_threshold
                else:
                    should_drop = bool(
                        drop_by_score and (drop_on_score_only or quality_problem)
                    )
                    drop_threshold = score_drop_threshold

                if should_drop:
                    reverse_dropped.append(sym)
                    drop_reasons[sym] = "reverse_split_corporate_action_artifact"
                    symbol_diag.setdefault(sym, {})["reason"] = (
                        "reverse_split_corporate_action_artifact"
                    )
                    _append_symbol_event(
                        event_rows,
                        ts=ts,
                        symbol=sym,
                        stage="reverse_split",
                        field="close",
                        rule="reverse_split_corporate_action_artifact",
                        severity="error",
                        action="drop_symbol",
                        source="auto",
                        close_df=close,
                        panel_fields=panel,
                        volume_df=volume,
                        metric_value=float(score),
                        threshold=float(drop_threshold),
                        note=(
                            f"d={best_d:.6f},dist={best_dist},"
                            f"qprob={int(quality_problem)},score_only={int(drop_on_score_only)},"
                            f"flag_drop={int(drop_on_flag_threshold)}"
                        ),
                    )
                elif drop_by_flag:
                    reverse_flagged.append(sym)
                    _append_symbol_event(
                        event_rows,
                        ts=ts,
                        symbol=sym,
                        stage="reverse_split",
                        field="close",
                        rule="corporate_action_like_keep",
                        severity="warning",
                        action="keep_symbol",
                        source="auto",
                        close_df=close,
                        panel_fields=panel,
                        volume_df=volume,
                        metric_value=float(score),
                        threshold=float(score_flag_threshold),
                        note=(
                            f"d={best_d:.6f},dist={best_dist},"
                            f"qprob={int(quality_problem)},score_only={int(drop_on_score_only)},"
                            f"flag_drop={int(drop_on_flag_threshold)}"
                        ),
                    )

    close, panel, volume = _drop_symbols_from_data(
        symbols=sorted(set(reverse_dropped)),
        close_df=close,
        panel_fields=panel,
        volume_df=volume,
    )

    event_frames = [df for df in event_chunks if df is not None and not df.empty]
    if event_rows:
        event_frames.append(pd.DataFrame(event_rows, columns=EVENT_COLUMNS))
    if event_frames:
        event_records: list[dict[str, Any]] = []
        for frame in event_frames:
            event_records.extend(frame.to_dict(orient="records"))
        events_raw = pd.DataFrame(event_records, columns=EVENT_COLUMNS)
    else:
        events_raw = _empty_events()
    events_df = _ensure_event_columns(events_raw)

    removed_symbols = sorted(set(removed_stage1 + removed_stage2 + reverse_dropped))
    stages_summary = {
        "stage1": {
            "removed": sorted(set(removed_stage1)),
            "kept": int(close.shape[1] + len(set(removed_stage2 + reverse_dropped))),
            "ohlc": ohlc_stats,
            "zero_volume_with_price": zero_stats,
        },
        "stage2": {
            "removed": sorted(set(removed_stage2)),
            "thresholds": {
                "keep_pct_threshold": keep_pct_threshold,
                "max_start_na": max_start_na,
                "max_end_na": max_end_na,
                "max_gap_bars": max_gap,
                "hard_drop": hard_drop,
                "min_tradable_rows_for_ohl_gate": min_trad_rows_for_ohl_gate,
                "ohl_missing_pct_max": ohl_missing_pct_max,
            },
        },
        "reverse_split": {
            "enabled": reverse_enabled,
            "dropped": sorted(set(reverse_dropped)),
            "flagged": sorted(set(reverse_flagged)),
            "n_dropped": int(len(set(reverse_dropped))),
            "n_flagged": int(len(set(reverse_flagged))),
            "thresholds": {
                "score_flag_threshold": score_flag_threshold,
                "score_drop_threshold": score_drop_threshold,
                "drop_on_score_only": drop_on_score_only,
                "drop_on_flag_threshold": drop_on_flag_threshold,
            },
        },
    }

    snapshots = {
        "pre_reverse_split": snapshot_pre_reverse,
    }

    return FilteringResult(
        close=close,
        panel_fields=panel,
        volume=volume,
        removed_symbols=removed_symbols,
        drop_reasons=drop_reasons,
        symbol_diag=symbol_diag,
        events=events_df,
        stages=stages_summary,
        snapshots=snapshots,
    )
