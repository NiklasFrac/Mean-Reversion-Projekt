from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processing.processing_primitives import process_and_fill_prices
from processing.quality_helpers import validate_prices_wide


def _mini_wide_prices() -> pd.DataFrame:
    # extrem klein & deterministisch
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    a = pd.Series(np.linspace(100, 109, len(idx)), index=idx)
    b = pd.Series(np.linspace(200, 209, len(idx)), index=idx)
    b.iloc[3:5] = np.nan  # LÃ¼cke (len=2)
    a.iloc[7] = 500.0  # Outlier-Spike -> sollte geflaggt werden
    return pd.DataFrame({"A": a, "B": b})


def test_processing_golden_smoke(
    tmp_path: Path,
    golden_dir: Path,
    update_golden: bool,
    mask_diag_payload,
    mask_manifest_payload,
    monkeypatch: pytest.MonkeyPatch,
):
    # Keep calendar behavior deterministic across environments with/without
    # pandas_market_calendars installed.
    fake = types.ModuleType("pandas_market_calendars")
    fake.get_calendar = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", fake)

    # 1) Schweres & Volatiles vollstÃ¤ndig neutralisieren
    # No runtime-context collection is executed here; this test runs fully in-memory.

    # 2) Minimaler Wide-Input (keine Datei-I/O)
    prices_wide = _mini_wide_prices()

    # 3) Parameter wie in "echtem" Lauf â€“ aber ohne I/O
    max_gap = 5
    keep_pct = 0.7
    grid_mode = "leader"
    calendar = "XNYS"
    max_start_na = 5
    max_end_na = 3
    outlier_cfg = {"enabled": True, "zscore": 6.0, "window": 7, "use_log_returns": True}

    # 4) Verarbeitung (rein in-memory)
    filled_df, removed, diagnostics = process_and_fill_prices(
        prices=prices_wide,
        max_gap=max_gap,
        keep_pct_threshold=keep_pct,
        n_jobs=1,
        grid_mode=grid_mode,
        calendar_code=calendar,
        max_start_na=max_start_na,
        max_end_na=max_end_na,
        outlier_cfg=outlier_cfg,
        hard_drop=False,
    )

    # 5) Diagnosestruktur nachbilden (wie main, aber in-memory)
    pre_q = validate_prices_wide(prices_wide)
    post_q = validate_prices_wide(filled_df) if not filled_df.empty else {"checks": {}}

    removed_set = set(removed)
    pre_non_na = [
        d.get("non_na_pct", 0.0) for _, d in diagnostics.items() if "non_na_pct" in d
    ]
    pre_longest_kept = [
        d.get("longest_gap", 0)
        for sym, d in diagnostics.items()
        if sym not in removed_set
    ]
    agg = {
        "kept": int(filled_df.shape[1] if not filled_df.empty else 0),
        "removed": int(len(removed)),
        "mean_non_na_pct": float(np.mean(pre_non_na or [0.0])),
        "max_longest_gap_kept": int(max(pre_longest_kept or [0])),
        "sum_outliers_flagged": int(
            sum(int(d.get("outliers_flagged", 0)) for d in diagnostics.values())
        ),
        "grid_mode": grid_mode,
        "calendar": calendar,
    }
    diag_payload = {
        "schema_version": 3,
        "quality": {"pre_raw": pre_q, "pre_exec": pre_q, "post": post_q},
        "processing": agg,
    }

    # Manifest-Ã¤hnliche Mini-Struktur (fÃ¼r Konsistenz der Goldens)
    manifest_like = {
        "cfg_path": "config.yaml",
        "inputs": {
            "raw_prices": {"path": "in-memory", "sha1": None},
            "raw_volume": {"path": None, "sha1": None},
        },
        "extra": {"processing": {"kept": agg["kept"], "removed": agg["removed"]}},
    }

    masked_diag = mask_diag_payload(diag_payload)
    masked_man = mask_manifest_payload(manifest_like)

    # 6) Schreiben/Vergleichen der Goldens (winzige JSONs)
    g_diag = golden_dir / "processing_smoke.diag.json"
    g_man = golden_dir / "processing_smoke.manifest.json"

    if update_golden:
        g_diag.parent.mkdir(parents=True, exist_ok=True)
        g_diag.write_text(
            json.dumps(masked_diag, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        g_man.write_text(
            json.dumps(masked_man, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pytest.skip("Goldens aktualisiert (UPDATE_GOLDEN=1).")
    else:
        exp_diag = json.loads(g_diag.read_text(encoding="utf-8"))
        exp_man = json.loads(g_man.read_text(encoding="utf-8"))
        assert masked_diag == exp_diag
        assert masked_man == exp_man
