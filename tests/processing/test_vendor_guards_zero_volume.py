from __future__ import annotations

import pandas as pd

from processing.vendor_guards import apply_vendor_guards


def test_vendor_guards_ignores_zero_volume_rule():
    idx = pd.date_range("2024-04-01", periods=3, tz="America/New_York")
    close = pd.DataFrame({"AAA": [10.0, 11.0, float("nan")]}, index=idx)
    volume = pd.DataFrame({"AAA": [100.0, 0.0, 0.0]}, index=idx)
    panel_fields = {"close": close.copy(), "volume": volume.copy()}

    cfg = {
        "enabled": True,
        "bad_rows": {"enabled": False},
        "zero_volume_with_price": {"enabled": True},
    }

    out = apply_vendor_guards(
        df_exec_raw=close,
        panel_fields=panel_fields,
        volume_for_processing=volume,
        config=cfg,
    )

    pd.testing.assert_frame_equal(out.df_exec_raw, close.astype(float))
    pd.testing.assert_frame_equal(out.volume_for_processing, volume.astype(float))
    assert out.vendor_guards_summary["counts_by_rule"]["manual_bad_row"] == 0
