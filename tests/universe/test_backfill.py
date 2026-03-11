from __future__ import annotations

import threading

import pandas as pd
import pytest

from universe import backfill


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_token_bucket_respects_rate_and_reset() -> None:
    clock = _FakeClock()
    bucket = backfill.TokenBucket(
        rate_per_sec=2.0, burst=2, now_fn=clock.time, sleep_fn=clock.sleep
    )

    bucket.take()
    bucket.take()
    bucket.take()  # must wait ~0.5s to refill

    assert clock.slept and pytest.approx(clock.slept[0], rel=1e-3) == 0.5
    bucket.reset()
    assert bucket.tokens == bucket.capacity


def test_postfill_fetch_chunk_handles_multiindex_and_junk(monkeypatch) -> None:
    idx = pd.date_range("2024-01-01", periods=3)
    cols = pd.MultiIndex.from_tuples(
        [
            ("Adj Close", "AAA"),
            ("Adj Close", "BBB"),
            ("Volume", "AAA"),
            ("Volume", "BBB"),
        ]
    )
    data = [
        [10.0, 20.0, 100.0, 200.0],
        [11.0, 21.0, 110.0, 210.0],
        [12.0, 22.0, 120.0, 220.0],
    ]
    df = pd.DataFrame(data, index=idx, columns=cols)
    monkeypatch.setattr(backfill.yf, "download", lambda *a, **k: df)

    res = backfill._postfill_fetch_chunk(
        ["AAA", "BBB", "JUNK"], junk_filter=lambda s: s == "JUNK"
    )

    assert res["AAA"] == (12.0, 110.0)
    assert res["BBB"] == (22.0, 210.0)
    assert "JUNK" not in res


def test_postfill_missing_price_volume_fills_and_tracks(monkeypatch) -> None:
    df = pd.DataFrame(
        {"price": [None, 5.0], "volume": [10.0, None]},
        index=pd.Index(["AAA", "BBB"], name="ticker"),
    )

    def fake_fetch(chunk_syms: list[str], *, junk_filter=None, **_unused):
        return {
            sym: (
                1.0 if pd.isna(df.at[sym, "price"]) else df.at[sym, "price"],
                2.0 if pd.isna(df.at[sym, "volume"]) else df.at[sym, "volume"],
            )
            for sym in chunk_syms
        }

    monkeypatch.setattr(backfill, "_postfill_fetch_chunk", fake_fetch)
    monitoring: dict[str, object] = {}

    backfill.postfill_missing_price_volume(
        df,
        workers=2,
        rate_limit_per_sec=100.0,
        stop_event=None,
        ensure_not_cancelled=lambda ev: None,
        monitoring=monitoring,
        incomplete_sample_limit=5,
        junk_filter=None,
        chunk_size=1,
    )

    assert df.at["AAA", "price"] == 1.0
    assert df.at["BBB", "volume"] == 2.0
    assert monitoring["postfill_unresolved_total"] == 0
    assert monitoring["postfill_total"] == 2
    assert monitoring["postfill_completed_chunks"] == monitoring["postfill_chunks"]


def test_postfill_missing_price_volume_respects_cancellation(monkeypatch) -> None:
    df = pd.DataFrame(
        {"price": [None], "volume": [None]}, index=pd.Index(["AAA"], name="ticker")
    )
    stop_event = threading.Event()
    stop_event.set()

    monkeypatch.setattr(
        backfill, "_postfill_fetch_chunk", lambda *a, **k: {"AAA": (99.0, 99.0)}
    )
    monitoring: dict[str, object] = {}

    with pytest.raises(RuntimeError):
        backfill.postfill_missing_price_volume(
            df,
            workers=1,
            rate_limit_per_sec=100.0,
            stop_event=stop_event,
            ensure_not_cancelled=lambda ev: None,
            monitoring=monitoring,
            incomplete_sample_limit=3,
            junk_filter=None,
            chunk_size=1,
        )

    assert monitoring.get("postfill_cancelled") is True


def test_token_bucket_clamps_extreme_burst(monkeypatch) -> None:
    clock = _FakeClock()
    bucket = backfill.TokenBucket(
        rate_per_sec=1.0,
        burst=5_000_000,  # should clamp internally
        now_fn=clock.time,
        sleep_fn=clock.sleep,
    )

    assert bucket.capacity == 1_000_000
    bucket.take()
    assert bucket.tokens <= bucket.capacity


def test_token_bucket_take_cancels_during_wait() -> None:
    clock = _FakeClock()
    stop_event = threading.Event()
    sleep_calls = {"count": 0}

    def fake_sleep(seconds: float) -> None:
        sleep_calls["count"] += 1
        clock.now += seconds
        if sleep_calls["count"] == 1:
            stop_event.set()

    bucket = backfill.TokenBucket(
        rate_per_sec=0.2, burst=1, now_fn=clock.time, sleep_fn=fake_sleep
    )
    bucket.take()  # consume the initial token

    with pytest.raises(RuntimeError, match="cancelled"):
        bucket.take(stop_event=stop_event)

    assert sleep_calls["count"] == 1


def test_fetch_price_volume_respects_stop_event(monkeypatch):
    calls = {"sleep": 0}
    stop_event = threading.Event()
    stop_event.set()

    monkeypatch.setattr(
        backfill.yf,
        "download",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        backfill.time, "sleep", lambda s: calls.__setitem__("sleep", calls["sleep"] + 1)
    )

    res = backfill._fetch_price_volume_for_symbols(
        ["AAA"],
        retries=2,
        backoff=0.1,
        stop_event=stop_event,
    )

    assert res == {}
    assert calls["sleep"] == 0  # cooperative sleep short-circuits on cancellation


def test_postfill_missing_price_volume_validates_vol_lookback(monkeypatch):
    df = pd.DataFrame(
        {"price": [None], "volume": [None]}, index=pd.Index(["AAA"], name="ticker")
    )
    monkeypatch.setattr(
        backfill, "_postfill_fetch_chunk", lambda *a, **k: {"AAA": (1.0, 2.0)}
    )
    monitoring: dict[str, object] = {}

    backfill.postfill_missing_price_volume(
        df,
        workers=1,
        rate_limit_per_sec=10.0,
        stop_event=None,
        ensure_not_cancelled=lambda _: None,
        monitoring=monitoring,
        incomplete_sample_limit=3,
        junk_filter=None,
        chunk_size=1,
        vol_lookback_priorities=(5,),  # malformed; should fall back safely
    )

    assert df.at["AAA", "price"] == 1.0
    assert df.at["AAA", "volume"] == 2.0


def test_postfill_raises_on_chunk_error(monkeypatch):
    df = pd.DataFrame(
        {"price": [None], "volume": [None]}, index=pd.Index(["AAA"], name="ticker")
    )
    monitoring: dict[str, object] = {}

    def boom(*a, **k):
        raise RuntimeError("fail-chunk")

    monkeypatch.setattr(backfill, "_postfill_fetch_chunk", boom)

    with pytest.raises(RuntimeError):
        backfill.postfill_missing_price_volume(
            df,
            workers=1,
            rate_limit_per_sec=10.0,
            stop_event=None,
            ensure_not_cancelled=lambda _: None,
            monitoring=monitoring,
            incomplete_sample_limit=3,
            junk_filter=None,
            chunk_size=1,
        )

    assert monitoring.get("postfill_failed_chunks", 0) >= 1
    assert monitoring.get("postfill_errors")
