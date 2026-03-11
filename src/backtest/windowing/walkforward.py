from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import pandas as pd
import yaml

from backtest.calendars import map_to_calendar
from backtest.utils.tz import align_ts_to_index, to_naive_local

logger = logging.getLogger("backtest.windowing.walkforward")


@dataclass(frozen=True)
class WalkForwardWindow:
    i: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    truncated: bool = False

    def as_splits(self) -> dict[str, dict[str, str]]:
        return {
            "train": {
                "start": str(self.train_start.date()),
                "end": str(self.train_end.date()),
            },
            "test": {
                "start": str(self.test_start.date()),
                "end": str(self.test_end.date()),
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "i": int(self.i),
            "train": {"start": str(self.train_start), "end": str(self.train_end)},
            "test": {"start": str(self.test_start), "end": str(self.test_end)},
            "truncated": bool(self.truncated),
        }


def _next_session(calendar: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    if calendar.empty:
        return None
    t = pd.Timestamp(ts)
    # Ensure tz compatibility for comparisons/lookup.
    try:
        t = align_ts_to_index(t, calendar)
    except Exception:
        t = pd.Timestamp(to_naive_local(t))

    if t not in calendar:
        t_m = map_to_calendar(t, calendar, policy="prior")
        if t_m is None:
            return None
        t = t_m

    pos = int(calendar.get_indexer([t])[0])
    if pos < 0:
        return None
    j = pos + 1
    if j >= len(calendar):
        return None
    return pd.Timestamp(calendar[j])


def _resolve_analysis_cfg_boundary(
    da: Mapping[str, Any],
    *,
    preferred_key: str,
    legacy_key: str,
) -> Any:
    preferred = da.get(preferred_key)
    legacy = da.get(legacy_key)

    if preferred not in (None, ""):
        if legacy not in (None, "") and legacy != preferred:
            logger.warning(
                "Both data_analysis.%s and deprecated data_analysis.%s are set; using %s.",
                preferred_key,
                legacy_key,
                preferred_key,
            )
        return preferred

    if legacy not in (None, ""):
        logger.warning(
            "data_analysis.%s is deprecated; use data_analysis.%s.",
            legacy_key,
            preferred_key,
        )
        return legacy

    return None


def _parse_months(x: Any, *, name: str) -> int:
    try:
        v = int(x)
    except Exception as e:
        raise ValueError(f"{name} must be an int (got {x!r})") from e
    if v <= 0:
        raise ValueError(f"{name} must be > 0 (got {v})")
    return v


def infer_backtest_start_from_analysis_cfg(
    *, calendar: pd.DatetimeIndex, analysis_cfg_path: Path
) -> pd.Timestamp:
    if not analysis_cfg_path.exists():
        raise FileNotFoundError(f"analysis_cfg_path not found: {analysis_cfg_path}")
    raw = yaml.safe_load(analysis_cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("analysis cfg must be a YAML mapping")

    da_raw = raw.get("data_analysis")
    da = cast(Mapping[str, Any], da_raw) if isinstance(da_raw, Mapping) else {}
    cutoff = _resolve_analysis_cfg_boundary(
        da,
        preferred_key="train_cutoff_local",
        legacy_key="train_cutoff_utc",
    )
    if not cutoff:
        raise KeyError(
            "analysis cfg missing data_analysis.train_cutoff_local "
            "(legacy fallback: data_analysis.train_cutoff_utc)"
        )

    cutoff_raw = pd.to_datetime(cutoff, errors="coerce")
    if pd.isna(cutoff_raw):
        raise ValueError(
            "Invalid data_analysis.train_cutoff_local "
            f"(legacy fallback: train_cutoff_utc): {cutoff!r}"
        )
    cutoff_ts = align_ts_to_index(
        pd.Timestamp(cutoff_raw), calendar, naive_is_utc=False
    )

    start = map_to_calendar(pd.Timestamp(cutoff_ts), calendar, policy="next")
    if start is None:
        raise ValueError("Could not map analysis cutoff to calendar (start=None)")
    return start


def _coerce_range_boundary(
    *,
    calendar: pd.DatetimeIndex,
    ts: Any,
    policy: str,
    name: str,
) -> pd.Timestamp:
    if calendar.empty:
        raise ValueError("calendar is empty")
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        raise ValueError(f"Invalid {name}: {ts!r}")
    mapped = map_to_calendar(pd.Timestamp(t), calendar, policy=policy)
    if mapped is None:
        raise ValueError(
            f"Could not map {name} to calendar with policy={policy!r}: {ts!r}"
        )
    return mapped


def _parse_nominal_boundary(*, ts: Any, name: str) -> pd.Timestamp:
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        raise ValueError(f"Invalid {name}: {ts!r}")
    return pd.Timestamp(t)


def generate_walkforward_windows_from_cfg(
    *,
    calendar: pd.DatetimeIndex,
    cfg: Mapping[str, Any],
    default_analysis_cfg_path: Path = Path("runs/configs/config_analysis.yaml"),
) -> tuple[list[WalkForwardWindow], dict[str, Any]]:
    """
    Expanding or rolling walk-forward generator.

    Expects:
      backtest:
        range:
          start: optional (default: analysis_end + 1 session, inferred via analysis cfg cutoff)
          end: optional (default: data end)
          analysis_cfg_path: optional path override
        walkforward:
          enabled: true
          train_mode: expanding | rolling
          initial_train_months: int
          test_months: int
          step_months: int
    """
    if calendar.empty:
        raise ValueError("calendar is empty")

    bt = cfg.get("backtest", {}) if isinstance(cfg.get("backtest"), Mapping) else {}
    wf = bt.get("walkforward", {}) if isinstance(bt.get("walkforward"), Mapping) else {}
    if not bool(wf.get("enabled", False)):
        return [], {"enabled": False}

    train_mode = str(wf.get("train_mode", "expanding")).strip().lower()
    if train_mode not in {"expanding", "rolling"}:
        raise ValueError(
            "Only backtest.walkforward.train_mode in {'expanding','rolling'} is supported"
        )

    initial_train_months = _parse_months(
        wf.get("initial_train_months", None),
        name="backtest.walkforward.initial_train_months",
    )
    test_months = _parse_months(
        wf.get("test_months", None), name="backtest.walkforward.test_months"
    )
    step_months = _parse_months(
        wf.get("step_months", None), name="backtest.walkforward.step_months"
    )

    warnings: list[str] = []
    if step_months < test_months:
        raise ValueError(
            "backtest.walkforward.step_months must be >= backtest.walkforward.test_months "
            "(overlapping test windows are not allowed)"
        )

    r = bt.get("range", {}) if isinstance(bt.get("range"), Mapping) else {}
    start_raw = r.get("start", None)
    end_raw = r.get("end", None)
    analysis_cfg_raw = r.get("analysis_cfg_path", None)
    analysis_cfg_path = (
        Path(str(analysis_cfg_raw)) if analysis_cfg_raw else default_analysis_cfg_path
    )

    if start_raw:
        nominal_range_start = _parse_nominal_boundary(
            ts=start_raw, name="backtest.range.start"
        )
        range_start = _coerce_range_boundary(
            calendar=calendar, ts=start_raw, policy="next", name="backtest.range.start"
        )
        start_source = "explicit"
    else:
        range_start = infer_backtest_start_from_analysis_cfg(
            calendar=calendar, analysis_cfg_path=analysis_cfg_path
        )
        nominal_range_start = pd.Timestamp(range_start)
        start_source = f"analysis_cfg:{analysis_cfg_path}"

    if end_raw:
        range_end = _coerce_range_boundary(
            calendar=calendar, ts=end_raw, policy="prior", name="backtest.range.end"
        )
        end_source = "explicit"
    else:
        range_end = pd.Timestamp(calendar[-1])
        end_source = "data_end"

    # Range sanity
    if range_start not in calendar:
        range_start = _coerce_range_boundary(
            calendar=calendar,
            ts=range_start,
            policy="next",
            name="backtest.range.start",
        )
    if range_end not in calendar:
        range_end = _coerce_range_boundary(
            calendar=calendar, ts=range_end, policy="prior", name="backtest.range.end"
        )
    if range_start > range_end:
        raise ValueError(
            f"Invalid backtest.range: start={range_start} > end={range_end}"
        )

    meta = {
        "enabled": True,
        "train_mode": train_mode,
        "initial_train_months": int(initial_train_months),
        "test_months": int(test_months),
        "step_months": int(step_months),
        "range": {"start": str(range_start), "end": str(range_end)},
        "range_sources": {"start": start_source, "end": end_source},
        "warnings": warnings,
    }

    windows: list[WalkForwardWindow] = []
    i = 0
    while True:
        if train_mode == "expanding":
            nominal_train_start = pd.Timestamp(nominal_range_start)
            nominal_train_end = nominal_range_start + pd.DateOffset(
                months=int(initial_train_months + i * step_months)
            )
        else:
            nominal_train_start = nominal_range_start + pd.DateOffset(
                months=int(i * step_months)
            )
            nominal_train_end = nominal_train_start + pd.DateOffset(
                months=int(initial_train_months)
            )

        train_start = map_to_calendar(
            pd.Timestamp(nominal_train_start), calendar, policy="next"
        )
        train_end = map_to_calendar(
            pd.Timestamp(nominal_train_end), calendar, policy="prior"
        )
        if train_start is None or train_end is None:
            break
        train_start = pd.Timestamp(train_start)
        train_end = pd.Timestamp(train_end)

        if train_start > range_end:
            break
        if train_end < train_start:
            break

        test_start = _next_session(calendar, train_end)
        if test_start is None or test_start > range_end:
            break

        test_end_target = pd.Timestamp(nominal_train_end) + pd.DateOffset(
            months=int(test_months)
        )
        test_end_mapped = map_to_calendar(
            pd.Timestamp(test_end_target), calendar, policy="prior"
        )
        if test_end_mapped is None:
            break

        test_end = pd.Timestamp(test_end_mapped)
        truncated = False
        if test_end > range_end:
            truncated = True
            test_end = range_end

        if test_end < test_start:
            break

        if truncated:
            warnings.append(
                f"Window {i}: truncated test_end to range.end ({test_end})."
            )
            logger.warning(
                "Walkforward window %d: truncating test_end to range.end (%s).",
                i,
                str(test_end),
            )

        windows.append(
            WalkForwardWindow(
                i=int(i),
                train_start=pd.Timestamp(train_start),
                train_end=pd.Timestamp(train_end),
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),
                truncated=bool(truncated),
            )
        )
        i += 1

    prev_test_end: pd.Timestamp | None = None
    for w in windows:
        if prev_test_end is not None and w.test_start <= prev_test_end:
            raise ValueError(
                "Walkforward produced overlapping test windows after calendar mapping "
                f"(window {int(w.i)} starts at {w.test_start} while previous test_end is {prev_test_end})."
            )
        prev_test_end = pd.Timestamp(w.test_end)

    meta["n_windows"] = int(len(windows))
    return windows, meta
