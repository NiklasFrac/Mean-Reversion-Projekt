from __future__ import annotations

import pandas as pd
import pytest

from backtest.utils import tz as tz_utils


def test_get_ex_tz_prefers_env_and_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WF_EXCHANGE_TZ", "US/Eastern")
    assert tz_utils.get_ex_tz({}) == tz_utils.NY_TZ
    monkeypatch.delenv("WF_EXCHANGE_TZ", raising=False)

    cfg = {"backtest": {"timezone": "UTC"}}
    assert tz_utils.get_ex_tz(cfg) == "UTC"


def test_align_ts_to_series_and_to_naive_local() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    s = pd.Series([1.0, 2.0], index=idx)
    t = tz_utils.align_ts_to_series(pd.Timestamp("2024-01-01"), s)
    assert t.tz is not None

    df = pd.DataFrame({"x": [1, 2]}, index=idx)
    out = tz_utils.to_naive_local(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None


def test_same_tz_or_raise_handles_naive_and_mismatch() -> None:
    idx_naive = pd.date_range("2024-01-01", periods=2, freq="D")
    idx_utc = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")

    tz_utils.same_tz_or_raise(idx_naive, idx_naive, allow_naive_pair=True)

    with pytest.raises(ValueError):
        tz_utils.same_tz_or_raise(idx_naive, idx_utc, allow_naive_pair=False)


def test_ensure_index_tz_inplace_and_to_naive_series() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    s = pd.Series(idx)
    out = tz_utils.ensure_index_tz(s, "UTC", inplace=True)
    assert out is s

    naive = tz_utils.to_naive_local(s)
    assert hasattr(naive, "dt")
    assert naive.dt.tz is None


def test_align_ts_to_series_drops_tz_for_naive_series() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    s = pd.Series([1.0, 2.0], index=idx)
    ts = tz_utils.align_ts_to_series(pd.Timestamp("2024-01-01", tz="UTC"), s)
    assert ts.tz is None


def test_get_ex_tz_from_prices_and_alias_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {"backtest": {"calendar": {"tz": "America/NewYork"}}}
    assert tz_utils.get_ex_tz(cfg) == tz_utils.NY_TZ

    monkeypatch.delenv("WF_EXCHANGE_TZ", raising=False)
    prices = pd.Series(
        [1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, tz="UTC")
    )
    assert tz_utils.get_ex_tz({}, prices) == "UTC"


def test_ensure_index_tz_non_datetime_index_copy() -> None:
    s = pd.Series([1.0, 2.0], index=[0, 1])
    out = tz_utils.ensure_index_tz(s, "UTC")
    assert out.index.equals(s.index)
    assert out is not s


def test_align_ts_to_series_uses_series_tz() -> None:
    s = pd.Series(pd.date_range("2024-01-01", periods=2, tz="UTC"))
    t = tz_utils.align_ts_to_series(pd.Timestamp("2024-01-01"), s)
    assert str(t.tz) == "UTC"


def test_to_naive_local_timestamp_and_index() -> None:
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    out_ts = tz_utils.to_naive_local(ts)
    assert out_ts.tz is None

    idx = pd.date_range("2024-01-01", periods=2, tz="UTC")
    out_idx = tz_utils.to_naive_local(idx)
    assert out_idx.tz is None


def test_same_tz_or_raise_on_both_naive() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    with pytest.raises(ValueError):
        tz_utils.same_tz_or_raise(idx, idx, allow_naive_pair=False)


def test_tz_internal_helpers_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tz_utils._normalize_tz_name("") is None
    assert tz_utils._tz_to_str("UTC") == "UTC"
    assert tz_utils._extract_tz_from_index_like(object()) is None

    monkeypatch.delenv("WF_EXCHANGE_TZ", raising=False)
    assert tz_utils.get_ex_tz({}, None, default="UTC") == "UTC"


def test_ensure_index_tz_inplace_with_naive_index() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame({"x": [1, 2]}, index=idx)
    out = tz_utils.ensure_index_tz(df, "UTC", inplace=True)
    assert out is df
    assert str(df.index.tz) == "UTC"


def test_align_ts_to_series_convert_and_to_naive_local_series() -> None:
    idx = pd.date_range("2024-01-01", periods=2, tz="UTC")
    s = pd.Series([1.0, 2.0], index=idx)
    ts = tz_utils.align_ts_to_series(
        pd.Timestamp("2024-01-01", tz="America/New_York"), s
    )
    assert str(ts.tz) == "UTC"

    s_num = pd.Series([1.0, 2.0])
    out = tz_utils.to_naive_local(s_num)
    assert out is s_num


def test_align_ts_to_series_same_tz() -> None:
    idx = pd.date_range("2024-01-01", periods=2, tz="UTC")
    s = pd.Series([1.0, 2.0], index=idx)
    ts = tz_utils.align_ts_to_series(pd.Timestamp("2024-01-01", tz="UTC"), s)
    assert str(ts.tz) == "UTC"


def test_extract_tz_and_align_ts_naive_series(monkeypatch: pytest.MonkeyPatch) -> None:
    s_dt = pd.Series(pd.date_range("2024-01-01", periods=2, tz="UTC"))
    assert tz_utils._extract_tz_from_index_like(s_dt) == "UTC"

    s_num = pd.Series([1.0, 2.0])
    ts = tz_utils.align_ts_to_series(pd.Timestamp("2024-01-01"), s_num)
    assert ts.tz is None

    orig = tz_utils._tz_to_str

    def boom_on_utc(tz):
        if tz is not None and str(tz) == "UTC":
            raise ValueError("boom")
        return orig(tz)

    monkeypatch.setattr(tz_utils, "_tz_to_str", boom_on_utc)
    out = tz_utils.to_naive_local(
        pd.Series(pd.date_range("2024-01-01", periods=2, tz="UTC"))
    )
    assert isinstance(out, pd.Series)
    ts2 = tz_utils.align_ts_to_series(pd.Timestamp("2024-01-01"), s_dt)
    assert ts2.tz is None
    assert tz_utils.to_naive_local(123) == 123
