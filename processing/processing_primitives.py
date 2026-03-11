from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .fill import fill_gaps_safely
from .timebase import build_tradable_mask, pick_time_grid

__all__ = [
    "SymbolResult",
    "detect_stale",
    "cap_extreme_returns",
    "_process_symbol",
    "process_and_fill_prices",
]


def _dedupe_symbol_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not df.columns.has_duplicates:
        return df
    # Keep the declared column order (sort=False) and collapse duplicate symbols
    # by taking the last non-null value per timestamp.
    out = df.T.groupby(level=0, sort=False).last().T
    out.columns = pd.Index(map(str, out.columns))
    return out


@dataclass
class SymbolResult:
    symbol: str
    kept: bool
    diagnostics: dict[str, Any]
    series: pd.Series | None


def detect_stale(
    series: pd.Series,
    *,
    k: int = 5,
    vol: pd.Series | None = None,
    vol_thresh: float = 0.0,
    eps: float = 1e-12,
    min_run: int = 2,
) -> bool:
    """
    "Stale" nur, wenn k aufeinanderfolgende flache Schritte (|dP|<eps)
    mindestens 'min_run' mal hintereinander auftreten; optional mit Volumenkriterium.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.dropna().size < (k + 1):
        return False

    flat_step = s.diff().abs() < float(eps)
    flat_k = flat_step.rolling(k, min_periods=k).sum() == k
    flat_runs = (
        flat_k.astype(int).rolling(min_run, min_periods=min_run).sum() == min_run
    )
    stale = bool(flat_runs.any())

    if stale and vol is not None:
        v = pd.to_numeric(vol.reindex(series.index), errors="coerce").fillna(0.0)
        vol_flat = v.rolling(k, min_periods=k).sum() <= float(vol_thresh)
        stale = bool((flat_runs & vol_flat).any())

    return stale


def cap_extreme_returns(
    series: pd.Series,
    lower: float = -0.9,
    upper: float = 3.0,
    exclude_dates: set[pd.Timestamp] | None = None,
    max_gap_bars: int | None = None,
    *,
    mode: str = "nan",
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s[s <= 0] = np.nan
    # Compute simple returns against the last valid observation (causal) so missing bars don't
    # suppress return-based caps (e.g. vendor gaps).
    prev = s.ffill().shift(1).replace(0.0, np.nan)
    r = (s / prev) - 1.0
    if max_gap_bars is not None:
        max_gap_bars = int(max_gap_bars)
        bar_pos = pd.Series(np.arange(len(s), dtype="float64"), index=s.index)
        prev_valid_pos = bar_pos.where(s.notna()).ffill().shift(1)
        dist = bar_pos - prev_valid_pos
        r = r.where(dist <= float(max_gap_bars + 1))
    if exclude_dates:
        ex = pd.DatetimeIndex(list(exclude_dates))
        idx = cast(pd.DatetimeIndex, s.index)
        r.loc[idx.normalize().isin(ex.normalize())] = np.nan
    bad = (r < lower) | (r > upper)
    if not bool(bad.any()):
        return s

    mode_n = str(mode or "nan").strip().lower()
    if mode_n in {"clip", "winsor", "winsorize"}:
        # Clip the *return* instead of dropping the point, applied sequentially so capped
        # values become the reference for subsequent returns.
        s_out = s.copy()
        ex_dates: set[pd.Timestamp] | None = None
        if exclude_dates:
            ex = pd.DatetimeIndex(list(exclude_dates))
            ex_dates = set(ex.normalize())

        prev_val: float | None = None
        prev_valid_idx: int | None = None
        max_gap_n = int(max_gap_bars) if max_gap_bars is not None else None

        for i, (ts, val) in enumerate(s.items()):
            if not np.isfinite(val):
                continue
            if prev_val is None:
                prev_val = float(val)
                prev_valid_idx = i
                continue

            gap_ok = True
            if max_gap_n is not None and prev_valid_idx is not None:
                if (i - prev_valid_idx) > (max_gap_n + 1):
                    gap_ok = False
            if ex_dates is not None:
                try:
                    ts_norm = pd.Timestamp(cast(Any, ts)).normalize()
                    if ts_norm in ex_dates:
                        gap_ok = False
                except (TypeError, ValueError):
                    pass

            if gap_ok:
                r_val = (float(val) / float(prev_val)) - 1.0
                if (r_val < lower) or (r_val > upper):
                    r_val = float(np.clip(r_val, lower, upper))
                    val = float(prev_val) * (1.0 + r_val)

            s_out.at[ts] = val
            prev_val = float(val)
            prev_valid_idx = i

        s_out[s_out <= 0] = np.nan
        return s_out

    # Legacy default: drop extreme-return points and let the downstream filler decide.
    s.loc[bad[bad].index] = np.nan
    return s


def _process_symbol(
    symbol: str,
    series: pd.Series,
    ref_index: pd.DatetimeIndex,
    *,
    max_gap: int,
    keep_pct_threshold: float,
    max_start_na: int,
    max_end_na: int,
    outlier_cfg: dict[str, Any],
    tradable_mask: pd.Series | None = None,
    caps_cfg: dict[str, Any] | None = None,
    staleness_cfg: dict[str, Any] | None = None,
    vol_series: pd.Series | None = None,
    causal_only: bool = False,
    hard_drop: bool = True,
) -> SymbolResult:
    tradable_mask = (
        tradable_mask.reindex(ref_index).fillna(False).astype(bool)
        if tradable_mask is not None
        else pd.Series(True, index=ref_index, dtype=bool)
    )
    s = pd.to_numeric(series.reindex(ref_index), errors="coerce").astype(float)
    tradable_index = tradable_mask[tradable_mask].index
    if tradable_index.empty:
        return SymbolResult(
            symbol=symbol,
            kept=False,
            diagnostics={"reason": "no_tradable_window", "phase": "fill_backstop"},
            series=None,
        )

    s_tradable = s.loc[tradable_index]
    if s_tradable.dropna().empty:
        return SymbolResult(
            symbol=symbol,
            kept=False,
            diagnostics={"reason": "empty_after_reindex", "phase": "fill_backstop"},
            series=None,
        )

    ref_len = len(tradable_index)
    pre_non_na_pct = (
        float(s_tradable.notna().sum()) / float(ref_len) if ref_len > 0 else 0.0
    )
    pre_longest_gap = 0
    cur_pre = 0
    for v in s_tradable.isna().to_numpy():
        if v:
            cur_pre += 1
            pre_longest_gap = max(pre_longest_gap, cur_pre)
        else:
            cur_pre = 0

    filled, removed_flag, fill_diag = fill_gaps_safely(
        s,
        max_gap=int(max_gap),
        tradable_mask=tradable_mask,
        causal_only=bool(causal_only),
        hard_drop=bool(hard_drop),
    )

    arr2 = filled.loc[tradable_index].isna().to_numpy()
    best2 = 0
    cur2 = 0
    for v in arr2:
        if v:
            cur2 += 1
            best2 = max(best2, cur2)
        else:
            cur2 = 0

    diag_post = {
        "non_na_pct": pre_non_na_pct,
        "longest_gap": int(pre_longest_gap),
        "post_non_na_pct": (
            float(filled.loc[tradable_index].notna().sum()) / float(ref_len)
            if ref_len > 0
            else 0.0
        ),
        "post_longest_gap": int(best2),
        "filling": fill_diag,
    }

    if removed_flag:
        d = {
            **diag_post,
            "reason": "large_gap_or_nontradable",
            "phase": "fill_backstop",
        }
        return SymbolResult(symbol=symbol, kept=False, diagnostics=d, series=None)

    return SymbolResult(
        symbol=symbol,
        kept=True,
        diagnostics=diag_post,
        series=filled,
    )


def process_and_fill_prices(
    prices: pd.DataFrame,
    *,
    max_gap: int = 3,
    keep_pct_threshold: float = 0.8,
    n_jobs: int = 1,
    grid_mode: str = "leader",
    calendar_code: str = "XNYS",
    max_start_na: int = 5,
    max_end_na: int = 3,
    outlier_cfg: dict[str, Any] | None = None,
    rth_only: bool = True,
    symbol_calendar_map: dict[str, str] | None = None,
    caps_cfg: dict[str, Any] | None = None,
    staleness_cfg: dict[str, Any] | None = None,
    volume_df: pd.DataFrame | None = None,
    causal_only: bool = False,
    hard_drop: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, Any]]]:
    if prices.empty:
        return pd.DataFrame(), [], {}
    prices = _dedupe_symbol_columns(prices)

    symbols = list(prices.columns)
    ref_index = pick_time_grid(prices, mode=grid_mode, calendar_code=calendar_code)

    # Reuse tradable masks across symbols (and panel fields) to avoid repeatedly
    # rebuilding market calendars, which is expensive for large universes.
    calendar_for_symbol = symbol_calendar_map or {}
    unique_calendars = {calendar_for_symbol.get(sym, calendar_code) for sym in symbols}
    tradable_masks: dict[str, pd.Series] = {}
    for cal in unique_calendars:
        cal_str = str(cal) if cal is not None else calendar_code
        tradable_masks[cal_str] = build_tradable_mask(
            ref_index, calendar_code=cal_str, rth_only=rth_only
        )

    volume_aligned: pd.DataFrame | None = None
    if volume_df is not None:
        try:
            volume_aligned = _dedupe_symbol_columns(volume_df).reindex(ref_index)
        except Exception:
            volume_aligned = _dedupe_symbol_columns(volume_df)

    def _mask_for(sym: str) -> pd.Series:
        cal = (symbol_calendar_map or {}).get(sym, calendar_code)
        cal_str = str(cal) if cal is not None else calendar_code
        mask = tradable_masks.get(cal_str)
        if mask is None:
            mask = build_tradable_mask(
                ref_index, calendar_code=cal_str, rth_only=rth_only
            )
            tradable_masks[cal_str] = mask
        return mask

    def _vol_for(sym: str) -> pd.Series | None:
        if volume_aligned is None:
            return None
        try:
            if sym in volume_aligned.columns:
                return pd.to_numeric(volume_aligned[sym], errors="coerce").astype(float)
        except Exception:
            pass
        return None

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_symbol)(
            sym,
            prices[sym],
            ref_index,
            max_gap=max_gap,
            keep_pct_threshold=keep_pct_threshold,
            max_start_na=max_start_na,
            max_end_na=max_end_na,
            outlier_cfg=outlier_cfg or {},
            tradable_mask=_mask_for(sym),
            caps_cfg=caps_cfg or {},
            staleness_cfg=staleness_cfg or {},
            vol_series=_vol_for(sym),
            causal_only=bool(causal_only),
            hard_drop=bool(hard_drop),
        )
        for sym in symbols
    )

    kept: dict[str, pd.Series] = {}
    removed: list[str] = []
    diagnostics: dict[str, dict[str, Any]] = {}

    for r in results:
        diagnostics[r.symbol] = r.diagnostics
        if r.kept and r.series is not None:
            kept[r.symbol] = r.series
        else:
            removed.append(r.symbol)

    if not kept:
        return pd.DataFrame(index=ref_index), removed, diagnostics

    df = pd.DataFrame(kept, index=ref_index)
    return df, removed, diagnostics
