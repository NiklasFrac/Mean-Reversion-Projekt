from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from backtest.calibration.metrics import bucket_by_participation, report_by_adv_decile
from backtest.calibration.targets import DEFAULT_TARGETS, CostTargets

__all__ = ["ScoreBreakdown", "score_trades", "DEFAULT_TARGETS"]


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    penalties: dict[str, float]
    notes: dict[str, Any]


def _nan0(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _monotone_non_decreasing(vals: list[float]) -> bool:
    v = [x for x in vals if np.isfinite(x)]
    if len(v) < 3:
        return True
    return all(v[i] <= v[i + 1] + 1e-12 for i in range(len(v) - 1))


def score_trades(
    df_raw: pd.DataFrame, *, targets: CostTargets = DEFAULT_TARGETS
) -> ScoreBreakdown:
    """
    Compute a plausibility score for execution costs.

    Lower is better. Score is a sum of penalties; each penalty is scaled so that
    “obviously bad” settings dominate.
    """
    if df_raw is None or df_raw.empty:
        return ScoreBreakdown(score=1e9, penalties={"empty": 1e9}, notes={})

    df = df_raw.copy()
    bps = pd.to_numeric(
        cast(pd.Series, df.get("exec_total_bps", np.nan)), errors="coerce"
    ).astype(float)
    bps = bps.replace([np.inf, -np.inf], np.nan)

    penalties: dict[str, float] = {}
    notes: dict[str, Any] = {}

    # --- Outliers (too many huge costs) --------------------------------------
    ok = np.isfinite(bps)
    if not bool(ok.any()):
        return ScoreBreakdown(score=1e9, penalties={"no_numeric_bps": 1e9}, notes={})

    outlier_frac = float((bps.loc[ok] > float(targets.outlier_bps_threshold)).mean())
    notes["outlier_frac"] = outlier_frac
    penalties["outlier_frac"] = (
        max(0.0, (outlier_frac - float(targets.outlier_frac_hi))) * 10_000.0
    )

    # --- ADV-decile medians inside broad ranges ------------------------------
    rep_dec = report_by_adv_decile(df)
    if rep_dec is None or rep_dec.empty:
        penalties["adv_deciles_missing"] = 5_000.0
    else:
        med_by_dec: dict[int, float] = {}
        for _, r in rep_dec.iterrows():
            try:
                d = int(cast(Any, r.get("adv_decile")))
            except Exception:
                continue
            med = float(cast(Any, r.get("exec_total_bps_median")))
            med_by_dec[d] = med
        notes["median_bps_by_adv_decile"] = med_by_dec

        p = 0.0
        for d in range(1, 11):
            if d not in med_by_dec or not np.isfinite(med_by_dec[d]):
                p += 250.0
                continue
            lo = float(targets.total_bps_median_lo[d - 1])
            hi = float(targets.total_bps_median_hi[d - 1])
            v = float(med_by_dec[d])
            if v < lo:
                p += (lo - v) * 50.0
            elif v > hi:
                p += (v - hi) * 25.0
        penalties["median_bps_by_adv_decile"] = float(p)

        # Spread ticks sanity in top decile
        try:
            st_med = float(
                rep_dec.loc[rep_dec["adv_decile"] == 10, "spread_ticks_median"].iloc[0]
            )
        except Exception:
            st_med = float("nan")
        notes["top_decile_spread_ticks_median"] = st_med
        penalties["top_decile_spread_ticks"] = (
            max(0.0, _nan0(st_med) - float(targets.top_decile_spread_ticks_hi)) * 250.0
        )

        # Monotonic: median cost should decrease with liquidity (decile)
        seq = [med_by_dec.get(d, float("nan")) for d in range(1, 11)]
        # cost decreasing => -cost increasing
        seq2 = [-float(x) if np.isfinite(x) else float("nan") for x in seq]
        notes["median_seq"] = seq
        if not _monotone_non_decreasing(seq2):
            penalties["non_monotone_adv"] = 1_000.0

    # --- Participation monotonicity (impact) ---------------------------------
    if bool(targets.require_participation_monotone):
        df["participation_bin"] = bucket_by_participation(df)
        imp = pd.to_numeric(
            cast(pd.Series, df.get("exec_impact_bps", np.nan)), errors="coerce"
        ).astype(float)
        gb = (
            df.assign(exec_impact_bps=imp)
            .dropna(subset=["participation_bin"])
            .groupby("participation_bin", observed=True)["exec_impact_bps"]
        )
        med_series = gb.median()
        seq = [float(med_series.get(k, np.nan)) for k in med_series.index.tolist()]
        notes["impact_median_by_participation_bin"] = {
            str(k): float(med_series.get(k, np.nan)) for k in med_series.index.tolist()
        }
        if not _monotone_non_decreasing(seq):
            penalties["non_monotone_participation"] = 750.0

    score = float(sum(penalties.values()))
    return ScoreBreakdown(score=score, penalties=penalties, notes=notes)
