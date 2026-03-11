from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.overfit import analyze_bo_trials, pbo_cpcv, summarize_overfit_from_equity


def test_pbo_cpcv_rank_based() -> None:
    is_scores = np.array(
        [
            [3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    oos_scores = np.array(
        [
            [0.1, 0.2, 0.3],  # IS-best is worst OOS -> lambda > 0
            [0.3, 0.2, 0.1],  # IS-best is best OOS -> lambda < 0
        ],
        dtype=float,
    )
    pbo = pbo_cpcv(is_scores, oos_scores)
    assert abs(pbo - 0.5) < 1e-9


def test_deflated_sharpe_decreases_with_trials() -> None:
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(7)
    eq = pd.Series(
        100.0
        * (
            1.0 + pd.Series(rng.normal(0.0005, 0.01, size=len(idx)), index=idx)
        ).cumprod()
    )
    s1 = summarize_overfit_from_equity(eq, candidate_sharpes=[0.1])
    s2 = summarize_overfit_from_equity(eq, candidate_sharpes=list(range(30)))
    assert s2.dsr <= s1.dsr + 1e-9


def test_analyze_bo_trials_filters_component_metric(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "component": ["other", "theta_sig", "theta_sig", "other"],
            "metric": ["sharpe", "sharpe", "sharpe", "cagr"],
            "model_id": ["A", "B", "A", "B"],
            "score": [0.5, 0.6, 0.55, 0.2],
            "fold": [0, 0, 1, 1],
            "is_score": [0.7, 0.6, 0.8, 0.4],
            "oos_score": [0.1, 0.2, -0.1, 0.05],
        }
    )
    trials_path = tmp_path / "bo_trials.csv"
    df.to_csv(trials_path, index=False)
    eq = pd.Series(
        np.linspace(100.0, 120.0, 50),
        index=pd.date_range("2020-01-01", periods=50, freq="B"),
    )

    out_path = analyze_bo_trials(
        trials_path,
        equity_curve=eq,
        out_path=tmp_path / "overfit_summary.json",
        component_filter="theta_sig",
        metric_filter="sharpe",
    )
    data = json.loads(Path(out_path).read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    assert "theta_sig" in str(meta.get("component", ""))
