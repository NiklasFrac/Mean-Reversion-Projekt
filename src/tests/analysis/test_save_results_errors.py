import pandas as pd

import analysis.data_analysis as da


def test_save_results_survives_csv_error(tmp_path, monkeypatch):
    out = tmp_path / "pairs.pkl"
    df = pd.DataFrame([{"pair": "A-B", "mean_corr": 0.9}])
    meta = {"analysis_schema_version": 1, "thresholds": {"corr": 0.8}, "timings": {}}

    # Make DataFrame.to_csv raise – save_results should still write pkl + meta
    monkeypatch.setattr(
        pd.DataFrame, "to_csv", lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
    )

    da.save_results(df, out, meta)
    assert out.exists()
    assert out.with_suffix(".meta.json").exists()
