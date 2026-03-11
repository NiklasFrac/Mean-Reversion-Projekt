import json
from pathlib import Path

import numpy as np
import pandas as pd

# robuster Import, funktioniert für beide Layouts
from analysis.pipeline import main


def test_index_duplicates_handled(make_prices_file, tmp_path: Path):
    # --- Synthetic Preise mit doppelten Zeitstempeln ---
    idx = pd.bdate_range("2022-01-03", periods=30, tz=None)
    # Dupliziere die letzten 3 Timestamps:
    idx_dup = idx.append(idx[-3:])
    # Konstruiere zwei hoch-korrellierte Serien (B = A * 2)
    rng = np.random.default_rng(123)
    ret = 0.0005 + 0.01 * rng.standard_normal(len(idx_dup))
    px_a = 100.0 * np.exp(np.cumsum(ret))
    px_b = 2.0 * px_a
    df = pd.DataFrame({"A": px_a, "B": px_b}, index=idx_dup)

    prices_path = make_prices_file(df, "prices_with_dups.pkl")
    out_path = tmp_path / "pairs_out.pkl"

    cfg = {
        "data": {
            "prices_path": str(prices_path),
            "pairs_path": str(out_path),
        },
        "data_analysis": {
            "rolling_window": 20,
            "rolling_step": 5,
            "pair_corr_threshold": 0.40,
            "pct_thr1": 0.30,
            "pct_thr2": 0.35,
            "pair_pct_threshold": 0.8,  # als Anteil (0..1)
            "pair_mean_corr_min": 0.30,
            "bootstrap": {"n_resamples": 50, "block_size": 5},
            "fdr_alpha": 0.10,
            "n_jobs": 1,
            "rng_seed": 42,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
            # kein train_cutoff_utc, wir testen nur Duplikat-Handling
        },
        "perf": {"blas_threads": 1, "numexpr_threads": 1},
    }

    df_selected = main(cfg_path=None, quick=False, overrides=cfg)

    # Integrationserwartungen:
    # 1) Lauf erfolgreich, Output existiert
    assert isinstance(df_selected, pd.DataFrame)
    assert out_path.exists()
    # 2) Meta enthält Timings/Env (Observability) – indirekter Beleg, dass Pipeline durchlief
    meta = json.loads(out_path.with_suffix(".meta.json").read_text())
    assert "timings" in meta and isinstance(meta["timings"], dict)
    assert "env" in meta and isinstance(meta["env"], dict)
    # 3) Paar A-B sollte wegen identischer Struktur erscheinen
    assert "pair" in df_selected.columns
    assert any(x in ("A-B", "B-A") for x in df_selected["pair"].astype(str).tolist())
