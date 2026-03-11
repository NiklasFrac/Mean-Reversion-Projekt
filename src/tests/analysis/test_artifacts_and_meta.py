# src/tests/analysis/test_artifacts_and_meta.py
import json
from pathlib import Path

import pandas as pd

from analysis.atomic_io import save_results


def test_save_results_writes_all(tmp_path: Path):
    out = tmp_path / "pairs.pkl"
    df = pd.DataFrame([{"pair": "A-B", "mean_corr": 0.9, "pval": 0.01}])
    # timmings explizit mitgeben, weil der Test darauf prüft
    meta = {"analysis_schema_version": 1, "thresholds": {"corr": 0.8}, "timings": {}}
    save_results(df, out, meta)

    assert out.exists()
    assert out.with_suffix(".csv").exists()

    m = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    # "golden-ish": Schema-Keys anstatt exakter Werte
    assert "thresholds" in m and "timings" in m
