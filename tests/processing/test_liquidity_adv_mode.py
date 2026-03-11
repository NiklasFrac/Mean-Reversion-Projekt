from __future__ import annotations

import numpy as np
import pandas as pd

from processing.liquidity import (
    build_adv_map_from_price_and_volume,
    build_adv_map_with_gates,
)


def test_adv_mode_shares_uses_volume_mean() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    px = pd.DataFrame({"AAA": [10.0, 10.0, 10.0]}, index=idx)
    vol = pd.DataFrame({"AAA": [100.0, 200.0, 300.0]}, index=idx)

    out = build_adv_map_from_price_and_volume(
        px, vol, window=3, adv_mode="shares", stat="mean"
    )

    assert "AAA" in out
    assert out["AAA"]["adv"] == (100.0 + 200.0 + 300.0) / 3.0
    assert out["AAA"]["last_price"] == 10.0


def test_adv_mode_dollar_uses_price_times_volume_mean() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    px = pd.DataFrame({"AAA": [10.0, 20.0, 30.0]}, index=idx)
    vol = pd.DataFrame({"AAA": [100.0, 100.0, 100.0]}, index=idx)

    out = build_adv_map_from_price_and_volume(
        px, vol, window=3, adv_mode="dollar", stat="mean"
    )

    assert out["AAA"]["adv"] == (10.0 * 100.0 + 20.0 * 100.0 + 30.0 * 100.0) / 3.0
    assert out["AAA"]["last_price"] == 30.0


def test_adv_stat_median_supported() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    px = pd.DataFrame({"AAA": [10.0, 10.0, 10.0]}, index=idx)
    vol = pd.DataFrame({"AAA": [1.0, 100.0, 2.0]}, index=idx)

    out = build_adv_map_from_price_and_volume(
        px, vol, window=3, adv_mode="shares", stat="median"
    )

    assert out["AAA"]["adv"] == 2.0


def test_adv_gates_use_min_valid_threshold_for_window_mean() -> None:
    idx = pd.date_range("2024-01-01", periods=41, freq="D")
    px = pd.DataFrame({"AAA": [1.0] * 41}, index=idx)
    vol = pd.DataFrame({"AAA": [float(i) for i in range(1, 42)]}, index=idx)
    # Last window has 20/21 valid observations; this is valid for min_valid_ratio=0.8.
    vol.iloc[-1, 0] = np.nan

    out, metrics = build_adv_map_with_gates(
        px,
        vol,
        window=21,
        adv_mode="dollar",
        stat="mean",
        min_valid_ratio=0.8,
        min_total_windows_for_adv_gate=20,
        max_invalid_window_ratio=0.35,
    )

    assert metrics["AAA"]["total_windows"] == 21
    assert metrics["AAA"]["valid_windows"] == 21
    assert metrics["AAA"]["invalid_windows"] == 0
    assert metrics["AAA"]["gate_pass"] is True
    assert metrics["AAA"]["has_current_adv"] is True
    assert "AAA" in out
    assert out["AAA"]["adv"] == 30.5


def test_adv_gates_do_not_ffill_stale_adv_when_latest_window_invalid() -> None:
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    px = pd.DataFrame({"AAA": [1.0] * 60}, index=idx)
    vol = pd.DataFrame({"AAA": [100.0] * 60}, index=idx)
    # Last window has only 16/21 valid observations -> invalid at ratio 0.8.
    vol.iloc[-5:, 0] = np.nan

    out, metrics = build_adv_map_with_gates(
        px,
        vol,
        window=21,
        adv_mode="dollar",
        stat="mean",
        min_valid_ratio=0.8,
        min_total_windows_for_adv_gate=20,
        max_invalid_window_ratio=0.35,
    )

    assert metrics["AAA"]["total_windows"] == 40
    assert metrics["AAA"]["valid_windows"] == 39
    assert metrics["AAA"]["invalid_windows"] == 1
    assert metrics["AAA"]["gate_pass"] is False
    assert metrics["AAA"]["has_current_adv"] is False
    assert "AAA" not in out
