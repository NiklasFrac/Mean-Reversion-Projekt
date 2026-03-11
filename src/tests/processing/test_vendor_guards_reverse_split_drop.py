from __future__ import annotations

import pandas as pd

from processing.vendor_guards import apply_vendor_guards


def test_vendor_guards_reverse_split_drops_symbol():
    idx = pd.date_range("2024-03-01", periods=12, tz="America/New_York")
    split_like = [
        220.0,
        210.0,
        205.0,
        200.0,
        198.0,
        195.0,
        6.0,
        5.0,
        4.0,
        3.5,
        3.0,
        2.5,
    ]
    stable = [50.0] * len(idx)
    close = pd.DataFrame({"SPLT": split_like, "OKAY": stable}, index=idx)
    volume = pd.DataFrame(
        {
            "SPLT": [1.0] * len(idx),
            "OKAY": [2000.0] * len(idx),
        },
        index=idx,
    )
    panel_fields = {"close": close.copy()}

    cfg = {
        "enabled": True,
        "bad_rows": {"enabled": False},
        "reverse_split": {"enabled": True},
    }

    out = apply_vendor_guards(
        df_exec_raw=close,
        panel_fields=panel_fields,
        volume_for_processing=volume,
        config=cfg,
    )

    assert out.split_excluded_symbols == []
    assert list(out.df_exec_raw.columns) == ["SPLT", "OKAY"]
    assert list(out.panel_fields["close"].columns) == ["SPLT", "OKAY"]
    assert list(out.volume_for_processing.columns) == ["SPLT", "OKAY"]
    assert out.vendor_guards_summary["counts_by_rule"]["manual_bad_row"] == 0
