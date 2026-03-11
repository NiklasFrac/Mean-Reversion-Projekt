import pandas as pd
import pytest

from backtest.windowing import walkforward


def test_walkforward_rejects_overlapping_test_windows():
    cal = pd.bdate_range("2020-01-01", "2022-12-31", freq="B")
    cfg = {
        "backtest": {
            "range": {"start": "2020-01-02", "end": "2022-12-30"},
            "walkforward": {
                "enabled": True,
                "train_mode": "expanding",
                "initial_train_months": 12,
                "test_months": 6,
                "step_months": 3,  # overlap
            },
        }
    }

    with pytest.raises(ValueError, match="overlapping test windows are not allowed"):
        walkforward.generate_walkforward_windows_from_cfg(calendar=cal, cfg=cfg)


def test_walkforward_expanding_generates_windows():
    cal = pd.bdate_range("2020-01-01", "2022-12-31", freq="B")
    cfg = {
        "backtest": {
            "range": {"start": "2020-01-02", "end": "2022-12-30"},
            "walkforward": {
                "enabled": True,
                "train_mode": "expanding",
                "initial_train_months": 12,
                "test_months": 6,
                "step_months": 6,
            },
        }
    }

    windows, meta = walkforward.generate_walkforward_windows_from_cfg(
        calendar=cal, cfg=cfg
    )
    assert meta["enabled"] is True
    assert int(meta["n_windows"]) >= 1

    w0 = windows[0]
    assert w0.train_start == pd.Timestamp("2020-01-02")
    assert w0.train_end >= w0.train_start
    assert w0.test_start > w0.train_end
    assert w0.test_end >= w0.test_start

    if len(windows) >= 2:
        assert windows[1].train_end > windows[0].train_end
        assert windows[1].test_start > windows[0].test_start
        assert windows[1].test_start > windows[0].test_end


def test_walkforward_truncates_last_window():
    cal = pd.bdate_range("2020-01-01", "2021-12-31", freq="B")
    cfg = {
        "backtest": {
            "range": {"start": "2020-01-02", "end": "2021-03-15"},
            "walkforward": {
                "enabled": True,
                "train_mode": "expanding",
                "initial_train_months": 12,
                "test_months": 6,
                "step_months": 6,
            },
        }
    }

    windows, meta = walkforward.generate_walkforward_windows_from_cfg(
        calendar=cal, cfg=cfg
    )
    assert int(meta["n_windows"]) >= 1
    assert windows[0].truncated is True
    assert windows[0].test_end == pd.Timestamp("2021-03-15")


def test_walkforward_requires_month_params():
    cal = pd.bdate_range("2020-01-01", "2020-12-31", freq="B")
    cfg = {
        "backtest": {"range": {"start": "2020-01-02"}, "walkforward": {"enabled": True}}
    }
    with pytest.raises(ValueError):
        _ = walkforward.generate_walkforward_windows_from_cfg(calendar=cal, cfg=cfg)


def test_walkforward_disabled_and_non_expanding() -> None:
    cal = pd.bdate_range("2020-01-01", "2020-06-30", freq="B")
    cfg_disabled = {"backtest": {"walkforward": {"enabled": False}}}
    windows, meta = walkforward.generate_walkforward_windows_from_cfg(
        calendar=cal, cfg=cfg_disabled
    )
    assert windows == []
    assert meta["enabled"] is False

    cfg_bad = {
        "backtest": {
            "walkforward": {
                "enabled": True,
                "train_mode": "bad",
                "initial_train_months": 3,
                "test_months": 1,
                "step_months": 1,
            }
        }
    }
    with pytest.raises(ValueError):
        walkforward.generate_walkforward_windows_from_cfg(calendar=cal, cfg=cfg_bad)


def test_walkforward_rolling_train_windows():
    cal = pd.bdate_range("2020-01-01", "2021-12-31", freq="B")
    cfg = {
        "backtest": {
            "range": {"start": "2020-01-02", "end": "2021-12-30"},
            "walkforward": {
                "enabled": True,
                "train_mode": "rolling",
                "initial_train_months": 6,
                "test_months": 3,
                "step_months": 3,
            },
        }
    }
    windows, meta = walkforward.generate_walkforward_windows_from_cfg(
        calendar=cal, cfg=cfg
    )
    assert meta["enabled"] is True
    assert int(meta["n_windows"]) >= 1
    if len(windows) >= 2:
        assert windows[1].train_start > windows[0].train_start
        assert windows[1].train_end > windows[0].train_end
        assert windows[1].test_start > windows[0].test_end


def test_walkforward_rolling_uses_nominal_start_anchor_without_global_shift() -> None:
    cal = pd.bdate_range("2020-01-01", "2020-06-30", freq="B")
    cfg = {
        "backtest": {
            "range": {"start": "2020-01-04", "end": "2020-06-30"},
            "walkforward": {
                "enabled": True,
                "train_mode": "rolling",
                "initial_train_months": 1,
                "test_months": 1,
                "step_months": 1,
            },
        }
    }

    windows, meta = walkforward.generate_walkforward_windows_from_cfg(
        calendar=cal, cfg=cfg
    )
    assert meta["warnings"] == []
    assert len(windows) >= 2
    assert windows[0].train_start == pd.Timestamp("2020-01-06")
    assert windows[0].train_end == pd.Timestamp("2020-02-04")
    assert windows[1].train_start == pd.Timestamp("2020-02-04")
    assert windows[1].train_end == pd.Timestamp("2020-03-04")


def test_walkforward_rolling_uses_nominal_test_boundaries_without_overlap_repairs() -> (
    None
):
    cal = pd.bdate_range("2022-03-01", "2025-08-29", freq="B")
    cal = cal.drop(
        pd.DatetimeIndex([pd.Timestamp("2023-09-04"), pd.Timestamp("2024-09-02")])
    )
    cfg = {
        "backtest": {
            "range": {"start": "2022-03-01", "end": "2025-08-29"},
            "walkforward": {
                "enabled": True,
                "train_mode": "rolling",
                "initial_train_months": 12,
                "test_months": 6,
                "step_months": 6,
            },
        }
    }

    windows, meta = walkforward.generate_walkforward_windows_from_cfg(
        calendar=cal, cfg=cfg
    )
    assert meta["warnings"] == []
    assert len(windows) == 5
    assert windows[1].test_start == pd.Timestamp("2023-09-05")
    assert windows[1].test_end == pd.Timestamp("2024-03-01")
    assert windows[2].test_start == pd.Timestamp("2024-03-04")
    assert windows[2].test_end == pd.Timestamp("2024-08-30")
    assert windows[3].test_start == pd.Timestamp("2024-09-03")
    assert windows[1].test_end < windows[2].test_start
    assert windows[2].test_end < windows[3].test_start


def test_infer_backtest_start_from_analysis_cfg(tmp_path) -> None:
    cal = pd.bdate_range("2020-01-01", "2020-01-10", freq="B")
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(
        "data_analysis:\n  train_cutoff_local: 2020-01-02\n", encoding="utf-8"
    )
    start = walkforward.infer_backtest_start_from_analysis_cfg(
        calendar=cal, analysis_cfg_path=cfg_path
    )
    assert start == pd.Timestamp("2020-01-02")


def test_generate_walkforward_uses_analysis_cfg_start(tmp_path) -> None:
    cal = pd.bdate_range("2020-01-01", "2020-06-30", freq="B")
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(
        "data_analysis:\n  train_cutoff_utc: 2020-01-03\n", encoding="utf-8"
    )
    cfg = {
        "backtest": {
            "range": {"analysis_cfg_path": str(cfg_path)},
            "walkforward": {
                "enabled": True,
                "train_mode": "expanding",
                "initial_train_months": 1,
                "test_months": 1,
                "step_months": 1,
            },
        }
    }
    windows, meta = walkforward.generate_walkforward_windows_from_cfg(
        calendar=cal, cfg=cfg
    )
    assert meta["range_sources"]["start"].startswith("analysis_cfg:")
    assert windows


def test_next_session_and_coerce_range_boundary_errors() -> None:
    cal = pd.bdate_range("2020-01-01", "2020-01-06", freq="B")
    assert walkforward._next_session(cal, pd.Timestamp("2020-01-03")) == pd.Timestamp(
        "2020-01-06"
    )
    assert walkforward._next_session(cal, pd.Timestamp("2020-01-06")) is None

    with pytest.raises(ValueError):
        walkforward._coerce_range_boundary(
            calendar=pd.DatetimeIndex([]), ts="2020-01-01", policy="next", name="start"
        )


def test_walkforward_parse_months_and_missing_analysis_cfg(tmp_path) -> None:
    with pytest.raises(ValueError):
        walkforward._parse_months("bad", name="months")

    empty = pd.DatetimeIndex([])
    assert walkforward._next_session(empty, pd.Timestamp("2020-01-01")) is None

    missing_path = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        walkforward.infer_backtest_start_from_analysis_cfg(
            calendar=pd.bdate_range("2020-01-01", "2020-01-10"),
            analysis_cfg_path=missing_path,
        )
