from __future__ import annotations

import pandas as pd

from processing.vendor_guards import apply_vendor_guards


def test_vendor_guards_ignores_ohlc_sanity_rules():
    idx = pd.date_range("2024-01-01", periods=4, tz="America/New_York")
    close = pd.DataFrame({"AAA": [9.5, 10.5, 11.0, 9.8]}, index=idx)
    panel_fields = {
        "open": pd.DataFrame({"AAA": [10.0, 10.0, 11.0, 9.6]}, index=idx),
        "high": pd.DataFrame({"AAA": [9.0, 12.0, 12.0, 9.0]}, index=idx),
        "low": pd.DataFrame({"AAA": [8.0, 11.0, 0.0, 10.0]}, index=idx),
        "close": close.copy(),
    }
    volume = pd.DataFrame({"AAA": [100.0, 100.0, 100.0, 100.0]}, index=idx)

    out = apply_vendor_guards(
        df_exec_raw=close,
        panel_fields=panel_fields,
        volume_for_processing=volume,
        config={
            "enabled": True,
            "ohlc_sanity": {"enabled": True},
            "bad_rows": {"enabled": False},
        },
    )

    pd.testing.assert_frame_equal(out.df_exec_raw, close.astype(float))
    pd.testing.assert_frame_equal(
        out.panel_fields["high"], panel_fields["high"].astype(float)
    )
    assert out.vendor_guards_summary["counts_by_rule"]["manual_bad_row"] == 0
    assert out.anomalies.empty
