from __future__ import annotations

import pandas as pd


from processing.timebase import ensure_utc_index


def test_ensure_utc_index_normalizes_and_deduplicates():
    # tz-aware + Duplikate
    idx = pd.to_datetime(
        ["2020-01-01 10:00Z", "2020-01-01 10:00Z", "2020-01-01 11:00Z"]
    )
    df = pd.DataFrame(
        {"A": [1.0, 3.0, 5.0], "B": [2.0, 4.0, 6.0]},
        index=idx,
        dtype="float64",
    )
    out = ensure_utc_index(df)
    # tz-naiv (UTC konventionell), sortiert, Duplikat zusammengeführt via Median
    assert out.index.tz is None
    assert out.shape == (2, 2)
    # Median von 1.0 und 3.0 = 2.0  (für Spalte A bei dupliziertem Timestamp)
    assert float(out.loc[pd.Timestamp("2020-01-01 10:00"), "A"]) == 2.0
