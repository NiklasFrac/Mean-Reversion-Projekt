from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.utils.run.io import _load_json


def _write_walkforward_overfit_summary(
    out_dir: Path, *, windows: list[Any], out_relpath: str
) -> None:
    rows: list[dict[str, Any]] = []
    for w in windows:
        wf_dir = out_dir / f"WF-{int(w.i):03d}"
        sum_path = wf_dir / out_relpath
        if not sum_path.exists():
            continue
        data = _load_json(sum_path)
        if not isinstance(data, dict) or not data:
            continue
        row: dict[str, Any] = {"wf_i": int(w.i), "wf_dir": str(wf_dir)}
        row.update(data)
        meta_path = wf_dir / "overfit.json"
        if meta_path.exists():
            meta = _load_json(meta_path)
            if isinstance(meta, dict):
                if "trials_paths" in meta:
                    row["trials_paths"] = meta.get("trials_paths")
                if "error" in meta:
                    row["overfit_error"] = meta.get("error")
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "walkforward_overfit_summary.csv", index=False)

    def _numeric_series(name: str) -> pd.Series:
        raw = df[name] if name in df.columns else pd.Series(index=df.index, dtype=float)
        s = pd.to_numeric(raw, errors="coerce")
        return s

    def _finite_series(name: str) -> pd.Series:
        s = _numeric_series(name)
        s = s.replace([float("inf"), float("-inf")], float("nan")).dropna()
        return s

    dsr = _finite_series("dsr")
    memmel = _finite_series("memmel_p_one_sided")
    n_obs = _finite_series("n_test_obs")
    n_cand = _finite_series("n_candidates")
    n_folds = _numeric_series("n_folds").fillna(0.0)
    pbo = _numeric_series("pbo")
    pbo_sel = pbo[(n_folds > 0) & pbo.notna()]

    agg = {
        "windows": int(len(rows)),
        "out_relpath": str(out_relpath),
        "dsr_mean": float(dsr.mean()) if not dsr.empty else None,
        "dsr_median": float(dsr.median()) if not dsr.empty else None,
        "memmel_p_one_sided_mean": float(memmel.mean()) if not memmel.empty else None,
        "memmel_p_one_sided_median": float(memmel.median())
        if not memmel.empty
        else None,
        "total_n_test_obs": int(n_obs.sum()) if not n_obs.empty else 0,
        "n_candidates_median": float(n_cand.median()) if not n_cand.empty else None,
        "pbo_mean": float(pbo_sel.mean()) if not pbo_sel.empty else None,
        "pbo_windows": int((n_folds > 0).sum()),
    }
    (out_dir / "walkforward_overfit_summary.json").write_text(
        json.dumps(
            {"aggregate": agg, "windows": rows},
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
