from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

REQUIRED_SPLIT_COLUMNS = {"symbol", "effective_date", "ratio"}


def _as_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    idx = pd.to_datetime(out.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx
    out = out.loc[out.index.notna()].sort_index()
    return out


def _as_norm_ts(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(cast(Any, value))
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def audit_corporate_actions(
    raw_prices: pd.DataFrame,
    adjusted_prices: pd.DataFrame,
    *,
    splits: pd.DataFrame | None = None,
    tolerance_return: float = 0.4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Simple corporate-actions audit focused on split-day continuity in adjusted prices.

    A split event should not produce a large discontinuity in adjusted close returns.
    If abs(adjusted_return_on_event_day) exceeds tolerance_return, we emit a
    `continuity_fail` issue.
    """
    raw = _as_dt_index(raw_prices)
    adj = _as_dt_index(adjusted_prices)
    common_idx = raw.index.intersection(adj.index)
    if common_idx.empty:
        issues = pd.DataFrame(columns=["type", "symbol", "effective_date", "message"])
        return issues, {"n_issues": 0, "types": {}}

    if splits is None or splits.empty:
        issues = pd.DataFrame(columns=["type", "symbol", "effective_date", "message"])
        return issues, {"n_issues": 0, "types": {}}

    split_df = splits.copy()
    split_df.columns = [str(c).strip().lower() for c in split_df.columns]
    if not REQUIRED_SPLIT_COLUMNS.issubset(set(split_df.columns)):
        missing = sorted(REQUIRED_SPLIT_COLUMNS - set(split_df.columns))
        raise ValueError(f"splits is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for _, split in split_df.iterrows():
        symbol = str(split.get("symbol", "")).strip().upper()
        if not symbol or symbol not in raw.columns or symbol not in adj.columns:
            continue

        event_day = _as_norm_ts(split.get("effective_date"))
        if event_day is None:
            continue

        # Evaluate continuity around the event on both boundaries:
        # 1) pre-event -> event day
        # 2) event day -> next trading day
        on_or_after = common_idx[common_idx >= event_day]
        if on_or_after.empty:
            continue
        event_anchor = pd.Timestamp(on_or_after[0])

        candidates: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
        prev_candidates = common_idx[common_idx < event_anchor]
        if not prev_candidates.empty:
            candidates.append(
                (pd.Timestamp(prev_candidates[-1]), event_anchor, "pre_to_event")
            )
        next_candidates = common_idx[common_idx > event_anchor]
        if not next_candidates.empty:
            candidates.append(
                (event_anchor, pd.Timestamp(next_candidates[0]), "event_to_next")
            )

        for prev_ts, event_ts, segment in candidates:
            raw_prev = pd.to_numeric(
                pd.Series([raw.at[prev_ts, symbol]]), errors="coerce"
            ).iloc[0]
            raw_curr = pd.to_numeric(
                pd.Series([raw.at[event_ts, symbol]]), errors="coerce"
            ).iloc[0]
            adj_prev = pd.to_numeric(
                pd.Series([adj.at[prev_ts, symbol]]), errors="coerce"
            ).iloc[0]
            adj_curr = pd.to_numeric(
                pd.Series([adj.at[event_ts, symbol]]), errors="coerce"
            ).iloc[0]

            if any(pd.isna(v) for v in (raw_prev, raw_curr, adj_prev, adj_curr)):
                continue
            if float(adj_prev) == 0.0:
                continue

            adj_ret = float(adj_curr / adj_prev - 1.0)
            if abs(adj_ret) > float(tolerance_return):
                rows.append(
                    {
                        "type": "continuity_fail",
                        "symbol": symbol,
                        "effective_date": str(event_day.date()),
                        "segment": segment,
                        "raw_return": float(raw_curr / raw_prev - 1.0)
                        if raw_prev != 0
                        else np.nan,
                        "adjusted_return": adj_ret,
                        "tolerance_return": float(tolerance_return),
                        "message": (
                            "Adjusted return "
                            f"{adj_ret:.3f} exceeds tolerance {float(tolerance_return):.3f}"
                        ),
                    }
                )

    issues = pd.DataFrame(rows)
    if issues.empty:
        issues = pd.DataFrame(columns=["type", "symbol", "effective_date", "message"])
        summary = {"n_issues": 0, "types": {}}
    else:
        counts = issues["type"].value_counts().to_dict()
        summary = {
            "n_issues": int(len(issues)),
            "types": {str(k): int(v) for k, v in counts.items()},
        }
    return issues, summary
