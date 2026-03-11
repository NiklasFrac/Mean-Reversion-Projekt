from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

logger = logging.getLogger("liquidity")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def compute_adv_series(
    vol: pd.DataFrame,
    window: int = 20,
    min_periods: int | None = None,
    *,
    stat: str = "mean",
) -> pd.DataFrame:
    """
    Rollierender ADV je Spalte als Zeitreihe (DataFrame).
    """
    if vol.empty:
        return vol.copy()
    if min_periods is None:
        min_periods = window
    stat_n = str(stat or "mean").strip().lower()
    roll = vol.rolling(window=window, min_periods=min_periods)
    if stat_n in {"mean", "avg", "average"}:
        return roll.mean()
    if stat_n in {"median", "med"}:
        return roll.median()
    raise ValueError(f"Unsupported adv stat: {stat!r} (use 'mean' or 'median')")


def compute_adv(
    vol: pd.DataFrame,
    window: int = 20,
    min_periods: int | None = None,
    *,
    stat: str = "mean",
) -> pd.Series:
    """
    Aktueller ADV je Spalte als Skalar (Serie mit einem Wert pro Spalte).
    = letzte Zeile von compute_adv_series(...).
    """
    r = compute_adv_series(vol, window=window, min_periods=min_periods, stat=stat)
    return r.iloc[-1].astype(float) if not r.empty else pd.Series(dtype=float)


def build_adv_map_from_price_and_volume(
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    window: int = 21,
    *,
    adv_mode: str = "shares",
    stat: str = "mean",
    fill_limit: int | None = None,
) -> dict[str, dict[str, Any]]:
    if price_df.empty or volume_df.empty:
        return {}

    # Keep index alignment explicit; volume is already chosen as split-neutral input (unadjusted).
    # price_df is the execution price series (typically adjusted close after cleaning).
    vol_aligned = volume_df.reindex(price_df.index)
    if fill_limit is not None:
        vol_aligned = vol_aligned.ffill(limit=int(fill_limit))

    mode_n = str(adv_mode or "shares").strip().lower()
    if mode_n in {"share", "shares", "volume"}:
        base = vol_aligned
    elif mode_n in {"dollar", "usd", "notional"}:
        px = price_df.reindex(vol_aligned.index).reindex(columns=vol_aligned.columns)
        base = px.astype(float) * vol_aligned.astype(float)
    else:
        raise ValueError(
            f"Unsupported adv_mode: {adv_mode!r} (use 'shares' or 'dollar')"
        )

    adv_series = compute_adv(base, window=window, stat=stat)
    last_price = price_df.iloc[-1]

    result: dict[str, dict[str, Any]] = {}
    for col in price_df.columns:
        adv_val = float(adv_series.get(col, 0.0))
        lp_val_raw = last_price.get(col)
        lp_val = float(lp_val_raw) if pd.notna(lp_val_raw) else float("nan")
        result[col] = {"adv": adv_val, "last_price": lp_val}
    return result


def build_adv_map_with_gates(
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    window: int = 21,
    *,
    adv_mode: str = "shares",
    stat: str = "mean",
    min_valid_ratio: float = 0.80,
    min_total_windows_for_adv_gate: int = 20,
    max_invalid_window_ratio: float = 0.35,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if price_df.empty or volume_df.empty:
        return {}, {}

    vol_aligned = volume_df.reindex(price_df.index).reindex(columns=price_df.columns)
    px_aligned = price_df.reindex(vol_aligned.index).reindex(
        columns=vol_aligned.columns
    )
    mode_n = str(adv_mode or "shares").strip().lower()
    if mode_n in {"share", "shares", "volume"}:
        base = vol_aligned.astype(float)
    elif mode_n in {"dollar", "usd", "notional"}:
        base = px_aligned.astype(float) * vol_aligned.astype(float)
    else:
        raise ValueError(
            f"Unsupported adv_mode: {adv_mode!r} (use 'shares' or 'dollar')"
        )

    window_n = max(1, int(window))
    min_valid = int(math.ceil(float(min_valid_ratio) * float(window_n)))
    min_valid = max(1, min(window_n, min_valid))
    valid_obs = base.notna()
    valid_count = valid_obs.rolling(window=window_n, min_periods=window_n).sum()
    valid_window = valid_count >= float(min_valid)
    total_windows = valid_count.notna().sum()
    valid_windows = valid_window.sum()
    invalid_windows = total_windows - valid_windows
    invalid_ratio = (
        (invalid_windows / total_windows).replace([pd.NA], 1.0).fillna(1.0)
        if not total_windows.empty
        else pd.Series(dtype=float)
    )
    valid_ratio = (
        (valid_windows / total_windows).replace([pd.NA], 0.0).fillna(0.0)
        if not total_windows.empty
        else pd.Series(dtype=float)
    )

    adv_series = compute_adv_series(
        base, window=window_n, min_periods=min_valid, stat=stat
    )
    last_window_adv = (
        adv_series.where(valid_window).iloc[-1]
        if not adv_series.empty
        else pd.Series(dtype=float)
    )
    last_price = price_df.iloc[-1] if not price_df.empty else pd.Series(dtype=float)

    metrics: dict[str, dict[str, Any]] = {}
    out: dict[str, dict[str, Any]] = {}
    for col in base.columns:
        t_w = int(total_windows.get(col, 0))
        v_w = int(valid_windows.get(col, 0))
        i_w = int(invalid_windows.get(col, t_w - v_w))
        v_ratio = float(valid_ratio.get(col, 0.0))
        i_ratio = float(invalid_ratio.get(col, 1.0))
        adv_val_raw = last_window_adv.get(col)
        adv_val = float(adv_val_raw) if pd.notna(adv_val_raw) else float("nan")
        has_current_adv = bool(pd.notna(adv_val_raw))
        gate_pass = bool(
            t_w >= int(min_total_windows_for_adv_gate)
            and i_ratio <= float(max_invalid_window_ratio)
            and v_w > 0
            and has_current_adv
        )
        lp_raw = last_price.get(col)
        lp_val = float(lp_raw) if pd.notna(lp_raw) else float("nan")
        metrics[str(col)] = {
            "total_windows": t_w,
            "valid_windows": v_w,
            "invalid_windows": i_w,
            "valid_window_ratio": v_ratio,
            "invalid_window_ratio": i_ratio,
            "gate_pass": gate_pass,
            "has_current_adv": has_current_adv,
        }
        if gate_pass:
            out[str(col)] = {"adv": adv_val, "last_price": lp_val}
    return out, metrics
