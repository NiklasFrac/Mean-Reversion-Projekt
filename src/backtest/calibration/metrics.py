from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

__all__ = [
    "CalibrationMetrics",
    "derive_exec_metrics",
    "bucket_by_adv_decile",
    "bucket_by_participation",
    "summarize_metrics",
    "report_by_adv_decile",
    "report_by_participation",
]


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den2 = _to_float(den).replace(0.0, np.nan)
    return _to_float(num) / den2


def _harmonic_min(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = _to_float(a)
    bb = _to_float(b)
    ok = np.isfinite(aa) & np.isfinite(bb) & (aa > 0) & (bb > 0)
    out = pd.Series(np.nan, index=aa.index, dtype=float)
    # harmonic mean is conservative for "pair liquidity"; use min-like behavior
    out.loc[ok] = 2.0 / (1.0 / aa.loc[ok] + 1.0 / bb.loc[ok])
    return out


@dataclass(frozen=True)
class CalibrationMetrics:
    n_trades: int
    total_exec_bps_median: float
    total_exec_bps_p90: float
    outlier_frac: float
    top_decile_spread_ticks_median: float | None


def derive_exec_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns for calibration scoring.

    Requires:
      - gross_notional
      - fees, slippage_cost, impact_cost
      - lob_adv_usd_y/x, lob_spread_ticks_y/x (preferred; created by LOB annotator)
    """
    if trades is None or trades.empty:
        return pd.DataFrame()

    df = trades.copy()

    gross_notional = _to_float(cast(pd.Series, df.get("gross_notional", np.nan)))
    fees = _to_float(cast(pd.Series, df.get("fees", 0.0)))
    slip = _to_float(cast(pd.Series, df.get("slippage_cost", 0.0)))
    imp = _to_float(cast(pd.Series, df.get("impact_cost", 0.0)))
    buyin = _to_float(cast(pd.Series, df.get("buyin_penalty_cost", 0.0)))
    emergency = _to_float(cast(pd.Series, df.get("exec_emergency_penalty_cost", 0.0)))
    diag_only = (
        pd.Series(df.get("exec_diag_costs_only", False), index=df.index)
        .fillna(False)
        .astype(bool)
    )

    realized_costs_cash = fees + buyin + emergency
    realized_costs_cash = realized_costs_cash + slip.where(~diag_only, 0.0)
    realized_costs_cash = realized_costs_cash + imp.where(~diag_only, 0.0)
    diagnostic_costs_cash = slip.where(diag_only, 0.0) + imp.where(diag_only, 0.0)
    total_costs_cash = realized_costs_cash + diagnostic_costs_cash

    df["exec_realized_costs"] = realized_costs_cash
    df["exec_diagnostic_costs"] = diagnostic_costs_cash
    df["exec_total_costs"] = total_costs_cash
    df["exec_realized_bps"] = -_safe_div(realized_costs_cash, gross_notional) * 1e4
    df["exec_diagnostic_bps"] = -_safe_div(diagnostic_costs_cash, gross_notional) * 1e4
    df["exec_total_bps"] = -_safe_div(total_costs_cash, gross_notional) * 1e4
    df["exec_slippage_bps"] = -_safe_div(slip, gross_notional) * 1e4
    df["exec_impact_bps"] = -_safe_div(imp, gross_notional) * 1e4
    df["exec_fees_bps"] = -_safe_div(fees, gross_notional) * 1e4

    adv_y = _to_float(cast(pd.Series, df.get("lob_adv_usd_y", np.nan)))
    adv_x = _to_float(cast(pd.Series, df.get("lob_adv_usd_x", np.nan)))
    adv_pair = _harmonic_min(adv_y, adv_x)
    df["lob_adv_usd_pair"] = adv_pair

    df["participation_usd"] = _safe_div(
        _to_float(cast(pd.Series, df.get("gross_notional", np.nan))), adv_pair
    ).clip(lower=0.0)

    # spread ticks (entry-state; per leg)
    st_y = _to_float(cast(pd.Series, df.get("lob_spread_ticks_y", np.nan)))
    st_x = _to_float(cast(pd.Series, df.get("lob_spread_ticks_x", np.nan)))
    df["lob_spread_ticks_pair"] = pd.concat([st_y, st_x], axis=1).max(
        axis=1, skipna=True
    )

    return df


def bucket_by_adv_decile(
    df: pd.DataFrame, *, col_adv: str = "lob_adv_usd_pair"
) -> pd.Series:
    a = _to_float(cast(pd.Series, df.get(col_adv, np.nan)))
    # deciles: 1..10 (1=least liquid)
    try:
        q = pd.qcut(a.rank(method="first"), 10, labels=False, duplicates="drop")
    except Exception:
        q = pd.Series(np.nan, index=df.index, dtype=float)
    if q.isna().all():
        return pd.Series(np.nan, index=df.index, dtype=float)
    return (q.astype(float) + 1.0).rename("adv_decile")


def bucket_by_participation(
    df: pd.DataFrame, *, col_part: str = "participation_usd"
) -> pd.Series:
    p = _to_float(cast(pd.Series, df.get(col_part, np.nan)))
    # bins in % of ADV (USD) — broad
    bins = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, np.inf]
    labels = [
        "0-0.1%",
        "0.1-0.25%",
        "0.25-0.5%",
        "0.5-1%",
        "1-2%",
        "2-5%",
        "5-10%",
        "10%+",
    ]
    return pd.cut(p, bins=bins, labels=labels, include_lowest=True).rename(
        "participation_bin"
    )


def summarize_metrics(
    df: pd.DataFrame,
    *,
    outlier_bps_threshold: float,
    top_decile_spread_ticks: bool = True,
) -> CalibrationMetrics:
    if df is None or df.empty:
        return CalibrationMetrics(
            n_trades=0,
            total_exec_bps_median=float("nan"),
            total_exec_bps_p90=float("nan"),
            outlier_frac=float("nan"),
            top_decile_spread_ticks_median=None,
        )

    bps = _to_float(cast(pd.Series, df.get("exec_total_bps", np.nan)))
    bps = bps[np.isfinite(bps)]
    n = int(len(bps))
    med = float(np.nanmedian(bps)) if n else float("nan")
    p90 = float(np.nanpercentile(bps, 90.0)) if n else float("nan")
    out = float((bps > float(outlier_bps_threshold)).mean()) if n else float("nan")

    spread_med: float | None = None
    if top_decile_spread_ticks:
        adv_dec = bucket_by_adv_decile(df)
        st = _to_float(cast(pd.Series, df.get("lob_spread_ticks_pair", np.nan)))
        mask = (adv_dec == 10) & np.isfinite(st)
        if bool(mask.any()):
            spread_med = float(np.nanmedian(st.loc[mask]))

    return CalibrationMetrics(
        n_trades=n,
        total_exec_bps_median=med,
        total_exec_bps_p90=p90,
        outlier_frac=out,
        top_decile_spread_ticks_median=spread_med,
    )


def report_by_adv_decile(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["adv_decile"] = bucket_by_adv_decile(d)
    g = d.dropna(subset=["adv_decile"]).groupby("adv_decile", as_index=False)
    out = g.agg(
        n=("exec_total_bps", "count"),
        exec_total_bps_median=("exec_total_bps", "median"),
        exec_total_bps_p90=(
            "exec_total_bps",
            lambda x: float(np.nanpercentile(x, 90.0)) if len(x) else np.nan,
        ),
        spread_ticks_median=("lob_spread_ticks_pair", "median"),
        participation_median=("participation_usd", "median"),
    )
    return out.sort_values("adv_decile").reset_index(drop=True)


def report_by_participation(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["participation_bin"] = bucket_by_participation(d)
    g = d.dropna(subset=["participation_bin"]).groupby(
        "participation_bin", as_index=False, observed=True
    )
    out = g.agg(
        n=("exec_total_bps", "count"),
        exec_total_bps_median=("exec_total_bps", "median"),
        exec_impact_bps_median=("exec_impact_bps", "median"),
        exec_slippage_bps_median=("exec_slippage_bps", "median"),
    )
    return out
