from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd


def build_trading_index(
    raw: pd.DataFrame,
    mode: str = "leader",
    calendar: str = "XNYS",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """
    Bestimmt das Referenz-Grid:
      - leader: nimmt die Index-Achse der Spalte mit den meisten Nicht-NaNs
      - calendar: Boersenkalender (XNYS) via pandas_market_calendars; Fallback auf leader
      - intersection/union: konservativ/expansiv
    """
    mode = (mode or "leader").lower()
    if raw.empty:
        return pd.DatetimeIndex([], name=raw.index.name)

    if mode == "leader":
        non_na = raw.notna().sum(axis=0)
        ref = raw[non_na.idxmax()].dropna().index
        return pd.DatetimeIndex(ref)

    if mode in ("intersection", "union"):
        cols = [c for c in raw.columns if raw[c].dropna().size]
        if not cols:
            return pd.DatetimeIndex([])
        idxs = [pd.DatetimeIndex(raw[c].dropna().index) for c in cols]
        ref = idxs[0]
        for idx in idxs[1:]:
            ref = ref.intersection(idx) if mode == "intersection" else ref.union(idx)
        return pd.DatetimeIndex(sorted(ref.unique()))

    if mode == "calendar":
        try:
            import pandas_market_calendars as mcal

            if start is None:
                start = pd.to_datetime(raw.index.min())
            if end is None:
                end = pd.to_datetime(raw.index.max())
            cal = mcal.get_calendar(calendar)
            sched = cal.schedule(start_date=start.date(), end_date=end.date())
            ref = pd.DatetimeIndex(sched.index.tz_localize(None))
            return ref
        except Exception:
            if start is None:
                start = pd.to_datetime(raw.index.min())
            if end is None:
                end = pd.to_datetime(raw.index.max())
            return pd.bdate_range(start=start, end=end)

    non_na = raw.notna().sum(axis=0)
    return pd.DatetimeIndex(raw[non_na.idxmax()].dropna().index)


def _safe_log_series(x: pd.Series) -> pd.Series:
    """Numerisch robuste Log-Transformation fuer Preise (<=0/inf -> NaN)."""
    s = pd.to_numeric(x, errors="coerce").astype("float64")
    mask = (s > 0) & np.isfinite(s.to_numpy())
    return cast(pd.Series, np.log(s.where(mask)))


def robust_outlier_mask(
    series: pd.Series,
    zscore: float = 8.0,
    window: int = 21,
    use_log_returns: bool = True,
) -> pd.Series:
    """
    Markiert Return-Ausreisser (True = Ausreisser) auf Basis eines robusten Z-Scores (Median/MAD).
    Numerisch robust (Log-Returns nur fuer positive Preise).
    """
    s = pd.to_numeric(series, errors="coerce").astype("float64").copy()
    if s.dropna().size < 5:
        return pd.Series(False, index=s.index, dtype=bool)

    r = _safe_log_series(s).diff() if use_log_returns else s.pct_change()
    r = r.replace([np.inf, -np.inf], np.nan)

    minp = max(5, window // 2)
    med = r.rolling(window, center=True, min_periods=minp).median()
    mad = (r - med).abs().rolling(
        window, center=True, min_periods=minp
    ).median() * 1.4826

    z = (r - med) / mad.replace(0, np.nan)
    mask = z.abs() > float(zscore)
    return mask.fillna(False)


def scrub_outliers(
    series: pd.Series,
    zscore: float = 8.0,
    window: int = 21,
    use_log_returns: bool = True,
) -> tuple[pd.Series, int]:
    """
    Setzt die *Zielwerte* der als Ausreisser erkannten Returns
    auf NaN (am rechten Rand der Differenz), damit sie spaeter
    regulaer gefuellt werden.
    """
    mask = robust_outlier_mask(
        series, zscore=zscore, window=window, use_log_returns=use_log_returns
    )
    if not bool(mask.any()):
        return series, 0
    s = series.copy()
    to_nan = mask[mask].index
    s.loc[to_nan] = np.nan
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    s[s <= 0] = np.nan
    return s, int(mask.sum())


def longest_nan_run(series: pd.Series) -> int:
    arr = series.isna().to_numpy()
    if arr.size == 0:
        return 0
    best = 0
    cur = 0
    for v in arr:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def quality_gates(
    series: pd.Series,
    ref_len: int,
    *,
    non_na_min_pct: float = 0.7,
    max_gap: int = 12,
    max_start_na: int = 5,
    max_end_na: int = 3,
    forbid_nonpositive: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """
    Prueft Datenqualitaet gegen Grenzwerte. Gibt (ok, diagnostics) zurueck.
    """
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    non_na_pct = float(s.notna().sum()) / float(ref_len) if ref_len > 0 else 0.0

    if s.first_valid_index() is None:
        start_na = ref_len
    else:
        start_na = int(np.argmax(s.notna().to_numpy()))
    if s.last_valid_index() is None:
        end_na = ref_len
    else:
        end_na = int(np.argmax(s[::-1].notna().to_numpy()))

    lgap = longest_nan_run(s)
    nonpos = int((s.dropna() <= 0).sum()) if forbid_nonpositive else 0

    diag: dict[str, Any] = {
        "non_na_pct": non_na_pct,
        "start_na": start_na,
        "end_na": end_na,
        "longest_gap": lgap,
        "nonpositive_count": nonpos,
    }
    if non_na_pct < non_na_min_pct:
        diag["reason"] = "insufficient_coverage"
        return False, diag
    if lgap > max_gap:
        diag["reason"] = "large_gap"
        return False, diag
    if start_na > max_start_na:
        diag["reason"] = "excess_leading_na"
        return False, diag
    if end_na > max_end_na:
        diag["reason"] = "excess_trailing_na"
        return False, diag
    if forbid_nonpositive and nonpos > 0:
        diag["reason"] = "nonpositive_prices"
        return False, diag
    return True, diag
