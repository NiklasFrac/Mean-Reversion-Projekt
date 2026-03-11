from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def find_nan_gaps(series: pd.Series) -> list[tuple[int, int]]:
    arr = series.isna().values
    gaps: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(arr):
        if v and start is None:
            start = i
        if not v and start is not None:
            gaps.append((start, i))
            start = None
    if start is not None:
        gaps.append((start, len(arr)))
    return gaps


def kalman_smoother_level(
    y: np.ndarray, q: float = 1e-4, r: float | None = None
) -> np.ndarray:
    n = len(y)
    if n == 0:
        return y
    mask = np.isfinite(y)
    if mask.sum() == 0:
        return np.full(n, np.nan)
    if r is None:
        var = float(np.nanvar(y[mask]))
        if not np.isfinite(var) or var <= 0:
            var = 1e-6
        r = var + 1e-8
    Q = float(q)
    R = float(r)
    x_filt = np.zeros(n)
    P_filt = np.zeros(n)
    x_pred = np.zeros(n)
    P_pred = np.zeros(n)
    first_idx = int(np.argmax(mask))
    x_filt[first_idx] = y[first_idx]
    P_filt[first_idx] = 1.0
    for t in range(first_idx + 1, n):
        x_pred[t] = x_filt[t - 1]
        P_pred[t] = P_filt[t - 1] + Q
        if mask[t]:
            K = P_pred[t] / (P_pred[t] + R)
            x_filt[t] = x_pred[t] + K * (y[t] - x_pred[t])
            P_filt[t] = (1 - K) * P_pred[t]
        else:
            x_filt[t] = x_pred[t]
            P_filt[t] = P_pred[t]
    x_smooth = x_filt.copy()
    for t in range(n - 2, -1, -1):
        if P_filt[t] == 0:
            continue
        C = P_filt[t] / (P_filt[t] + Q)
        x_smooth[t] = x_filt[t] + C * (x_smooth[t + 1] - x_pred[t + 1])
    if first_idx > 0:
        x_smooth[:first_idx] = x_smooth[first_idx]
    return x_smooth


def fill_gap_segment(
    series: pd.Series,
    start: int,
    end: int,
    *,
    max_gap: int,
    method_policy: tuple[bool, bool, bool] = (True, True, True),
    causal_only: bool = False,
) -> tuple[np.ndarray, str] | None:
    """
    Bounded policy (mit Methode als Label):
    - len==1 -> single-step: avg(ffill,bfill) / ffill / bfill
    - len<=2 -> linear (lokal)
    - len<=max_gap -> kurzer Kalman (nur wenn beidseitig Nachbarn)
    - else -> None

    Bei causal_only=True:
      - keine Zukunft verwenden: reines ffill (falls linker Wert existiert), sonst None
    """
    length = end - start
    if length <= 0:
        return np.array([], dtype=float), "filled_noop"
    if length > max_gap:
        return None

    left = series.iloc[start - 1] if start - 1 >= 0 else np.nan
    right = series.iloc[end] if end < len(series) else np.nan

    if causal_only:
        if np.isfinite(left):
            return np.full(length, float(left), dtype=float), "filled_ffill"
        return None

    # Kontextfenster um die Luecke herum
    window_start = max(0, start - 5)
    window_end = min(len(series), end + 5)
    window = series.iloc[window_start:window_end].copy()

    if length == 1 and method_policy[0]:
        if np.isfinite(left) and np.isfinite(right):
            return np.array([(left + right) / 2.0], dtype=float), "filled_single_avg"
        if np.isfinite(left):
            return np.array([left], dtype=float), "filled_ffill"
        if np.isfinite(right):
            return np.array([right], dtype=float), "filled_bfill"
        return None

    if length <= 2 and method_policy[1]:
        x0 = start - 1
        x1 = end
        if x0 >= 0 and x1 < len(series):
            y0 = series.iloc[x0]
            y1 = series.iloc[x1]
            if np.isfinite(y0) and np.isfinite(y1):
                vals = np.linspace(float(y0), float(y1), num=length + 2)[1:-1].astype(
                    float
                )
                return vals, "filled_linear"
        return None

    if length <= max_gap and method_policy[2]:
        if np.isfinite(left) and np.isfinite(right):
            arr = window.to_numpy(dtype=float)
            sm = kalman_smoother_level(arr, q=1e-4, r=float(np.nanvar(arr)))

            # Gap-Position relativ zum Fenster
            rel_start = start - window_start
            rel_end = end - window_start

            vals = sm[rel_start:rel_end].astype(float)
            return vals, "filled_kalman"
        return None

    return None


def fill_gaps_safely(
    series: pd.Series,
    *,
    max_gap: int,
    tradable_mask: pd.Series | None = None,
    causal_only: bool = False,
    hard_drop: bool = True,
) -> tuple[pd.Series, bool, dict[str, Any]]:
    """
    Fuellt *nur* innerhalb erlaubter Tradable-Bereiche.
    Kein globales ffill/bfill. Rueckgabe: (filled_series, removed_flag, diagnostics)

    Bei causal_only=True: ausschliesslich Vergangenheitsinformation (ffill-artig).
    Non-tradable Gaps werden uebersprungen (kein Drop), um False-Drops zu vermeiden.
    Bei hard_drop=False: unzulaessige Gaps werden NICHT gefuellt, aber auch NICHT zum Symbol-Drop.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float).copy()
    gaps = find_nan_gaps(s)
    diagnostics: dict[str, Any] = {
        "gaps": [],
        "methods": [],
        "causal_only": bool(causal_only),
    }

    if not gaps:
        return s, False, diagnostics

    tradable = (
        tradable_mask if tradable_mask is not None else pd.Series(True, index=s.index)
    )

    for start, end in gaps:
        idx = s.index[start:end]
        tradable_slice = tradable.loc[idx].fillna(False)

        # If the gap has no tradable timestamps, leave it untouched.
        if not bool(tradable_slice.any()):
            diagnostics["methods"].append(
                {"gap": (start, end), "method": "skip_nontradable"}
            )
            continue

        # Process contiguous tradable sub-segments inside the gap so that max_gap rules
        # are enforced on actual trading bars even when non-tradable rows exist.
        true_positions = np.where(tradable_slice.to_numpy())[0]
        runs: list[tuple[int, int]] = []
        if true_positions.size:
            run_start = true_positions[0]
            prev = true_positions[0]
            for pos in true_positions[1:]:
                if pos == prev + 1:
                    prev = pos
                    continue
                runs.append((run_start, prev + 1))
                run_start = pos
                prev = pos
            runs.append((run_start, prev + 1))

        for rstart, rend in runs:
            abs_start = start + rstart
            abs_end = start + rend
            seg_idx = s.index[abs_start:abs_end]

            ret = fill_gap_segment(
                s,
                abs_start,
                abs_end,
                max_gap=max_gap,
                causal_only=bool(causal_only),
            )

            seg_len = abs_end - abs_start
            if ret is None:
                is_edge_gap = abs_start == 0 or abs_end == len(s)
                method = "skip_edge_unfilled" if is_edge_gap else "drop"
                diagnostics["methods"].append(
                    {"gap": (abs_start, abs_end), "method": method}
                )
                diagnostics["gaps"].append(
                    {"start": int(abs_start), "end": int(abs_end), "len": int(seg_len)}
                )
                if is_edge_gap or not hard_drop:
                    continue
                return s, True, diagnostics

            vals, mlabel = ret

            s.loc[seg_idx] = pd.Series(
                np.asarray(vals, dtype=float), index=seg_idx, dtype=float
            )
            diagnostics["gaps"].append(
                {"start": int(abs_start), "end": int(abs_end), "len": int(seg_len)}
            )
            diagnostics["methods"].append(
                {"gap": (abs_start, abs_end), "method": mlabel, "len": seg_len}
            )

    return s, False, diagnostics


def fill_gap_with_context(
    series: pd.Series,
    start: int,
    end: int,
    *,
    max_gap: int,
    causal_only: bool = False,
) -> np.ndarray | None:
    """
    Legacy helper retained for tests/backwards-compatibility. Returns the inferred values for
    a single gap using the same policy as fill_gaps_safely without mutating the input series.
    """
    res = fill_gap_segment(
        series,
        start,
        end,
        max_gap=max_gap,
        causal_only=bool(causal_only),
    )
    if res is None:
        return None
    vals, _ = res
    return np.asarray(vals, dtype=float)
