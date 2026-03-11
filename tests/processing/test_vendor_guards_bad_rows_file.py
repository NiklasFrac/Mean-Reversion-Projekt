from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from processing.vendor_guards import apply_vendor_guards


def test_vendor_guards_bad_rows_file_masks_manual_rows(tmp_path: Path):
    idx = pd.date_range("2024-02-01", periods=3, tz="America/New_York")
    close = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=idx)
    panel_fields = {
        "open": pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=idx),
        "high": pd.DataFrame({"AAA": [10.5, 11.5, 12.5]}, index=idx),
        "low": pd.DataFrame({"AAA": [9.5, 10.5, 11.5]}, index=idx),
        "close": close.copy(),
        "volume": pd.DataFrame({"AAA": [100.0, 101.0, 102.0]}, index=idx),
    }
    volume = panel_fields["volume"].copy()

    bad_rows_path = tmp_path / "bad_rows.parquet"
    pd.DataFrame(
        {
            "ts": [idx[0], idx[1]],
            "symbol": ["AAA", "AAA"],
            "field": ["close", "volume"],
            "reason": ["known_bad_close", "known_bad_volume"],
        }
    ).to_parquet(bad_rows_path, index=False)

    cfg = {
        "enabled": True,
        "ohlc_sanity": {"enabled": False},
        "bad_rows": {
            "enabled": True,
            "path": str(bad_rows_path),
            "action": "mask_nan",
        },
        "reverse_split": {"enabled": False},
        "zero_volume_with_price": {"enabled": False},
    }

    out = apply_vendor_guards(
        df_exec_raw=close,
        panel_fields=panel_fields,
        volume_for_processing=volume,
        config=cfg,
    )

    assert pd.isna(out.df_exec_raw.at[idx[0], "AAA"])
    assert pd.isna(out.panel_fields["close"].at[idx[0], "AAA"])
    assert pd.isna(out.volume_for_processing.at[idx[1], "AAA"])

    counts = out.vendor_guards_summary["counts_by_rule"]
    assert counts["manual_bad_row"] == 2
    assert set(out.anomalies["source"]) == {"known_bad_close", "known_bad_volume"}


def test_vendor_guards_bad_rows_disabled_does_not_require_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    idx = pd.date_range("2024-02-01", periods=3, tz="America/New_York")
    close = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=idx)
    panel_fields = {
        "close": close.copy(),
        "volume": pd.DataFrame({"AAA": [100.0, 101.0, 102.0]}, index=idx),
    }
    volume = panel_fields["volume"].copy()

    cfg = {
        "enabled": True,
        "ohlc_sanity": {"enabled": False},
        "bad_rows": {
            "enabled": False,
            "path": str(tmp_path / "missing_bad_rows.parquet"),
            "action": "mask_nan",
        },
        "reverse_split": {"enabled": False},
        "zero_volume_with_price": {"enabled": False},
    }

    with caplog.at_level("WARNING"):
        out = apply_vendor_guards(
            df_exec_raw=close,
            panel_fields=panel_fields,
            volume_for_processing=volume,
            config=cfg,
        )

    assert out.df_exec_raw.equals(close)
    assert out.panel_fields["close"].equals(close)
    assert out.volume_for_processing.equals(volume)
    assert out.vendor_guards_summary["counts_by_rule"]["manual_bad_row"] == 0
    assert not any(
        "bad_rows enabled but file not found" in rec.message for rec in caplog.records
    )
