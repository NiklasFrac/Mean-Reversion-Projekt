from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

import universe.fundamentals as funda
from universe.checkpoint import Checkpointer
from universe.filters import apply_filters_with_reasons


class DummyVendor:
    def fetch_quote_bundle(self, *args, **kwargs):
        raise RuntimeError("Dummy vendor should not be invoked in this test.")

    def download_history(self, *args, **kwargs):
        raise RuntimeError("Dummy vendor should not be invoked in this test.")

    def ticker_history(self, *args, **kwargs):
        raise RuntimeError("Dummy vendor should not be invoked in this test.")


def test_apply_filters_with_reasons_respects_all_guards():
    df = pd.DataFrame(
        {
            "price": [2.0, 12.0, 10.0, 10.0, 22.0],
            "market_cap": [4e8, 6e8, 6e8, 6e8, 6e8],
            "volume": [1_200_000, 1_300_000, 0.0, 1_100_000, 1_100_000],
            "float_pct": [0.30, 0.40, 0.55, 0.10, 0.20],
            "dividend": [False, True, True, True, True],
            "shares_out": [1e8] * 5,
        },
        index=["AAA", "BBB", "ZERO", "FLOAT", "EQX"],
    )
    df.index.name = "ticker"

    filters = {
        "drop_na": True,
        "drop_zero": True,
        "min_price": 5.0,
        "min_market_cap": 5e8,
        "min_avg_volume": 500_000,
        "min_float_pct": 0.30,
        "require_dividend": True,
    }

    filtered, reasons = apply_filters_with_reasons(df, filters)

    assert list(filtered.index) == ["BBB"]
    assert reasons["REASON_LOW_PRICE"] == 1  # AAA
    assert reasons["REASON_ZERO_CORE_FIELD"] == 1  # ZERO
    assert reasons["REASON_FLOAT_PCT"] == 2  # FLOAT, EQX


def test_fetch_fundamentals_parallel_produces_expected_frame(monkeypatch):
    def fake_fetch(
        sym: str,
        *,
        vendor=None,
        request_timeout=None,
        max_attempts=3,
        backoff_factor=1.8,
    ) -> funda.Funda:
        _ = (request_timeout, max_attempts, backoff_factor)
        base = float(len(sym))
        return funda.Funda(
            ticker=sym,
            price=10.0 + base,
            market_cap=5e8 + base * 1e6,
            volume=1_000_000 + base * 10_000,
            float_pct=0.4,
            dividend=True,
            is_etf=False,
            sector="Tech",
            industry="Software",
            country="US",
            shares_out=100_000_000 + base * 1_000,
        )

    monkeypatch.setattr(funda, "_fetch_fundamentals_one", fake_fetch)

    dummy_vendor = DummyVendor()
    df, monitoring = funda.fetch_fundamentals_parallel(
        tickers=["AAA", "BBB", "$OTC"],  # junk symbol should be skipped
        workers=2,
        show_progress=False,
        rate_limit_per_sec=100.0,
        breaker=None,
        checkpoint=None,
        checkpoint_filter=None,
        checkpoint_cfg_hash=None,
        checkpoint_ttl=None,
        vendor=dummy_vendor,
    )

    assert list(df.index) == ["AAA", "BBB"]
    assert monitoring.get("failed") == []
    assert (
        pytest.approx(df.loc["AAA", "dollar_adv"])
        == df.loc["AAA", "price"] * df.loc["AAA", "volume"]
    )
    assert "updated_at" in df.columns


def test_fetch_fundamentals_parallel_honors_postfill_mode(monkeypatch):
    def fake_fetch(
        sym: str,
        *,
        vendor=None,
        request_timeout=None,
        max_attempts=3,
        backoff_factor=1.8,
    ) -> funda.Funda:
        _ = (request_timeout, max_attempts, backoff_factor)
        return funda.Funda(
            ticker=sym,
            price=None,
            market_cap=5e8,
            volume=None,
            float_pct=0.4,
            dividend=False,
            is_etf=False,
        )

    observed: dict[str, object] = {}

    def fake_postfill(df: pd.DataFrame, **kwargs) -> None:
        observed["drop_instead_of_fill"] = kwargs.get("drop_instead_of_fill")
        df.loc[:, "price"] = 11.0
        df.loc[:, "volume"] = 12345.0

    monkeypatch.setattr(funda, "_fetch_fundamentals_one", fake_fetch)
    monkeypatch.setattr(funda, "postfill_missing_price_volume", fake_postfill)

    df, monitoring = funda.fetch_fundamentals_parallel(
        tickers=["AAA"],
        workers=1,
        show_progress=False,
        rate_limit_per_sec=100.0,
        postfill_mode="fill",
    )

    assert observed["drop_instead_of_fill"] is False
    assert monitoring.get("postfill_mode") == "fill"
    assert float(df.loc["AAA", "price"]) == pytest.approx(11.0)
    assert float(df.loc["AAA", "volume"]) == pytest.approx(12345.0)


def test_fetch_fundamentals_parallel_passes_request_controls(monkeypatch):
    seen: dict[str, float | int | None] = {}

    def fake_fetch(
        sym: str,
        *,
        vendor=None,
        request_timeout=None,
        max_attempts=3,
        backoff_factor=1.8,
    ) -> funda.Funda:
        seen["request_timeout"] = request_timeout
        seen["max_attempts"] = max_attempts
        seen["backoff_factor"] = backoff_factor
        return funda.Funda(
            ticker=sym,
            price=10.0,
            market_cap=1e9,
            volume=1_000_000,
            float_pct=0.5,
            dividend=False,
            is_etf=False,
        )

    monkeypatch.setattr(funda, "_fetch_fundamentals_one", fake_fetch)

    df, _ = funda.fetch_fundamentals_parallel(
        tickers=["AAA"],
        workers=1,
        show_progress=False,
        rate_limit_per_sec=100.0,
        request_timeout=9.5,
        request_retries=4,
        request_backoff=2.2,
    )

    assert list(df.index) == ["AAA"]
    assert seen["request_timeout"] == pytest.approx(9.5)
    assert seen["max_attempts"] == 5
    assert seen["backoff_factor"] == pytest.approx(2.2)


def test_fetch_fundamentals_parallel_marks_timeouts(monkeypatch):
    def slow_fetch(
        sym: str,
        *,
        vendor=None,
        request_timeout=None,
        max_attempts=3,
        backoff_factor=1.8,
    ) -> funda.Funda:
        _ = (vendor, request_timeout, max_attempts, backoff_factor)
        time.sleep(0.35)
        return funda.Funda(
            ticker=sym,
            price=10.0,
            market_cap=1e9,
            volume=1_000_000,
            float_pct=0.5,
            dividend=False,
            is_etf=False,
        )

    monkeypatch.setattr(funda, "_fetch_fundamentals_one", slow_fetch)

    started = time.perf_counter()
    df, monitoring = funda.fetch_fundamentals_parallel(
        tickers=["AAA", "BBB"],
        workers=2,
        show_progress=False,
        rate_limit_per_sec=100.0,
        request_timeout=0.05,
        request_retries=0,
    )
    elapsed = time.perf_counter() - started

    assert df.empty
    assert sorted(monitoring.get("failed", [])) == ["AAA", "BBB"]
    assert monitoring.get("timeouts") == 2
    assert elapsed < 1.0


def test_fetch_fundamentals_parallel_keeps_submitting_after_failures(monkeypatch):
    seen: list[str] = []

    def fail_fetch(
        sym: str,
        *,
        vendor=None,
        request_timeout=None,
        max_attempts=3,
        backoff_factor=1.8,
    ) -> funda.Funda:
        _ = (vendor, request_timeout, max_attempts, backoff_factor)
        seen.append(sym)
        raise RuntimeError("boom")

    monkeypatch.setattr(funda, "_fetch_fundamentals_one", fail_fetch)

    df, monitoring = funda.fetch_fundamentals_parallel(
        tickers=["AAA", "BBB", "CCC", "DDD"],
        workers=2,
        show_progress=False,
        rate_limit_per_sec=100.0,
        request_retries=0,
        use_token_bucket=False,
    )

    assert df.empty
    assert sorted(monitoring.get("failed", [])) == ["AAA", "BBB", "CCC", "DDD"]
    assert sorted(seen) == ["AAA", "BBB", "CCC", "DDD"]


def test_fetch_fundamentals_parallel_does_not_abort_on_worker_keyboard_interrupt(
    monkeypatch,
):
    def interrupted_fetch(
        sym: str,
        *,
        vendor=None,
        request_timeout=None,
        max_attempts=3,
        backoff_factor=1.8,
    ) -> funda.Funda:
        _ = (vendor, request_timeout, max_attempts, backoff_factor)
        raise KeyboardInterrupt("simulated worker interrupt")

    monkeypatch.setattr(funda, "_fetch_fundamentals_one", interrupted_fetch)

    df, monitoring = funda.fetch_fundamentals_parallel(
        tickers=["AAA", "BBB", "CCC"],
        workers=2,
        show_progress=False,
        rate_limit_per_sec=100.0,
        request_retries=0,
        use_token_bucket=False,
    )

    assert df.empty
    assert sorted(monitoring.get("failed", [])) == ["AAA", "BBB", "CCC"]


def test_fetch_fundamentals_parallel_trips_breaker(monkeypatch):
    import universe.fundamentals as funda

    monkeypatch.setattr(
        funda,
        "_fetch_fundamentals_one",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    breaker = funda.CircuitBreaker(max_consec_fail=1)

    with pytest.raises(RuntimeError):
        funda.fetch_fundamentals_parallel(
            tickers=["AAA", "BBB"],
            workers=1,
            show_progress=False,
            rate_limit_per_sec=100.0,
            breaker=breaker,
            checkpoint=None,
            checkpoint_filter=None,
            checkpoint_cfg_hash=None,
            checkpoint_ttl=None,
        )


def test_fetch_fundamentals_parallel_reports_cancelled_count(monkeypatch):
    monkeypatch.setattr(
        funda,
        "_fetch_fundamentals_one",
        lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("should not run when cancelled")
        ),
    )

    stop_event = threading.Event()
    stop_event.set()

    df, monitoring = funda.fetch_fundamentals_parallel(
        tickers=["AAA", "BBB", "CCC"],
        workers=2,
        show_progress=False,
        rate_limit_per_sec=100.0,
        breaker=None,
        checkpoint=None,
        checkpoint_filter=None,
        checkpoint_cfg_hash=None,
        checkpoint_ttl=None,
        stop_event=stop_event,
    )

    assert df.empty
    assert monitoring.get("cancelled") == 3


def test_fetch_fundamentals_parallel_cancels_while_rate_limited(monkeypatch):
    monkeypatch.setattr(
        funda,
        "_fetch_fundamentals_one",
        lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("fetch should not run while waiting in token bucket")
        ),
    )

    seen: dict[str, bool] = {"stop_event_forwarded": False}

    def fake_take(self, *, stop_event=None):
        seen["stop_event_forwarded"] = stop_event is not None
        deadline = time.perf_counter() + 2.0
        while True:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("cancelled")
            if time.perf_counter() >= deadline:
                raise RuntimeError("cancel_timeout")
            time.sleep(0.01)

    monkeypatch.setattr(funda.TokenBucket, "take", fake_take)

    stop_event = threading.Event()

    def _trigger_cancel() -> None:
        time.sleep(0.05)
        stop_event.set()

    threading.Thread(target=_trigger_cancel, daemon=True).start()
    started = time.perf_counter()
    df, monitoring = funda.fetch_fundamentals_parallel(
        tickers=["AAA", "BBB", "CCC"],
        workers=1,
        show_progress=False,
        rate_limit_per_sec=0.2,
        stop_event=stop_event,
        request_retries=0,
        heartbeat_logging=False,
    )
    elapsed = time.perf_counter() - started

    assert df.empty
    assert monitoring.get("cancelled") == 3
    assert seen["stop_event_forwarded"] is True
    assert elapsed < 1.0


def test_checkpointer_persistence_and_symbol_seed(tmp_path):
    ckpt_path = tmp_path / "ckpt.json"
    cp = Checkpointer(ckpt_path)
    cp.load()

    now = time.time()
    cp.mark_done("AAA", cfg_hash="cfg", timestamp=now)
    cp.mark_done("STALE", cfg_hash="cfg", timestamp=now - 10)
    assert cp.is_done("AAA", cfg_hash="cfg")

    dropped = cp.purge_invalid(cfg_hash="cfg", max_age=1)
    assert dropped == 1
    assert "STALE" not in cp.entries()

    cp.store_symbol_seed(["AAA", "BBB"], cfg_hash="cfg")
    seed = cp.symbol_seed(cfg_hash="cfg")
    assert seed == ["AAA", "BBB"]

    cp2 = Checkpointer(ckpt_path)
    cp2.load()
    assert cp2.is_done("AAA", cfg_hash="cfg")
    assert cp2.symbol_seed(cfg_hash="cfg") == ["AAA", "BBB"]

    removed = cp2.retain_only(["AAA"])
    assert removed == 0
    cp2.drop_many(["AAA"])
    assert not cp2.is_done("AAA", cfg_hash="cfg")
