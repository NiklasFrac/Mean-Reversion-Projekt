from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

from backtest.utils.tz import align_ts_to_index


def require_backtest_splits(
    cfg: Mapping[str, Any],
    *,
    keys: Iterable[str] = ("train", "test"),
    err_cls: type[Exception] = KeyError,
    err_msg: str = "backtest.splits.{train,test} missing",
) -> Mapping[str, Any]:
    bt = cfg.get("backtest") if isinstance(cfg.get("backtest"), Mapping) else {}
    splits = bt.get("splits") if isinstance(bt, Mapping) else None
    if not isinstance(splits, Mapping):
        raise err_cls(err_msg)
    for key in keys:
        if key not in splits:
            raise err_cls(err_msg)
    return splits


def require_split_start_end(
    splits: Mapping[str, Any],
    key: str,
    *,
    err_cls: type[Exception] = ValueError,
    err_msg: str = "split window must include start/end",
) -> Mapping[str, Any]:
    win = splits.get(key)
    if not isinstance(win, Mapping) or ("start" not in win) or ("end" not in win):
        raise err_cls(err_msg)
    return win


def coerce_ts_to_index_tz(value: Any, idx: pd.DatetimeIndex) -> pd.Timestamp:
    return align_ts_to_index(value, idx)
