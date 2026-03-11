from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import pandas as pd
import yfinance as yf

from universe.backfill import TokenBucket, postfill_missing_price_volume
from universe.checkpoint import Checkpointer, norm_symbol
from universe.coercion import clamp01, coerce_float, coerce_int, safe_float
from universe.numeric_utils import replace_inf_with_nan
from universe.symbol_filter_defaults import DEFAULT_DROP_CONTAINS, DEFAULT_DROP_SUFFIXES
from universe.timeout_utils import run_with_timeout

tqdm: Any | None
try:  # pragma: no cover - optional progress bar
    from tqdm import tqdm as _tqdm

    tqdm = _tqdm
except Exception:  # pragma: no cover
    tqdm = None

logger = logging.getLogger("runner_universe")

INCOMPLETE_SAMPLE_LIMIT = 200

_BASE_JUNK_SUFFIXES = frozenset(DEFAULT_DROP_SUFFIXES)
_BASE_JUNK_CONTAINS = frozenset(DEFAULT_DROP_CONTAINS)
_JUNK_OVERRIDES: dict[str, set[str] | None] = {"suffixes": None, "contains": None}

__all__ = [
    "CircuitBreaker",
    "INCOMPLETE_SAMPLE_LIMIT",
    "fetch_fundamentals_parallel",
    "is_junk",
    "set_junk_overrides",
]

_FUNDAMENTALS_COLUMNS_FULL = [
    "price",
    "market_cap",
    "volume",
    "float_pct",
    "dividend",
    "is_etf",
    "shares_out",
    "sector",
    "industry",
    "country",
    "long_name",
    "currency",
    "float_quality",
    "is_etn",
    "is_baby_bond",
    "is_adr",
    "is_trust",
    "exchange_code",
    "market",
    "quote_type",
]

_FUNDAMENTALS_COLUMNS_MINIMAL = [
    "price",
    "market_cap",
    "volume",
    "float_pct",
    "dividend",
    "is_etf",
    "shares_out",
]


def _empty_fundamentals_frame(*, full: bool) -> pd.DataFrame:
    columns = _FUNDAMENTALS_COLUMNS_FULL if full else _FUNDAMENTALS_COLUMNS_MINIMAL
    out = pd.DataFrame(columns=columns)
    out.index.name = "ticker"
    return out


def set_junk_overrides(
    *, suffixes: Iterable[str] | None = None, contains: Iterable[str] | None = None
) -> None:
    if suffixes is None:
        _JUNK_OVERRIDES["suffixes"] = None
    else:
        suf = {str(s).upper() for s in suffixes if str(s).strip()}
        _JUNK_OVERRIDES["suffixes"] = suf

    if contains is None:
        _JUNK_OVERRIDES["contains"] = None
    else:
        cont = {str(s).upper() for s in contains if str(s).strip()}
        _JUNK_OVERRIDES["contains"] = cont


def _effective_junk_suffixes() -> set[str]:
    overrides = _JUNK_OVERRIDES["suffixes"]
    if overrides is None:
        return set(_BASE_JUNK_SUFFIXES)
    return set(overrides)


def _effective_junk_contains() -> set[str]:
    overrides = _JUNK_OVERRIDES["contains"]
    if overrides is None:
        return set(_BASE_JUNK_CONTAINS)
    return set(overrides)


def is_junk(sym: str) -> bool:
    s = str(sym).upper().strip()
    suffixes = _effective_junk_suffixes()
    contains = _effective_junk_contains()
    return (
        s.startswith("$")
        or any(s.endswith(x) for x in suffixes)
        or any(c in s for c in contains)
    )


@dataclass
class Funda:
    ticker: str
    price: float | None
    market_cap: float | None
    volume: float | None
    float_pct: float | None
    dividend: bool
    is_etf: bool
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    shares_out: float | None = None
    long_name: str | None = None
    currency: str | None = None
    float_quality: str | None = None
    is_etn: bool = False
    is_baby_bond: bool = False
    is_adr: bool = False
    is_trust: bool = False
    exchange_code: str | None = None
    market: str | None = None
    quote_type: str | None = None


class CircuitBreaker:
    def __init__(self, max_consec_fail: int = 50):
        self.max = int(max_consec_fail)
        self._fails = 0
        self._lock = threading.Lock()

    def ok(self) -> None:
        with self._lock:
            self._fails = 0

    def err(self) -> None:
        with self._lock:
            self._fails += 1

    def open(self) -> bool:
        with self._lock:
            return self._fails >= self.max

    def reset(self) -> None:
        with self._lock:
            self._fails = 0


def _safe_float(x: Any) -> float | None:
    return safe_float(x)


def _sleep_backoff(
    attempt: int, base: float = 0.3, factor: float = 1.8, jitter: float = 0.5
) -> float:
    t = base * (factor ** max(0, attempt - 1))
    return t + random.random() * jitter


def _run_with_timeout(func: Callable[[], Any], timeout: float | None) -> Any:
    return run_with_timeout(func, timeout, err_prefix="Fundamentals call")


def _clip01(x: float) -> float:
    return clamp01(x, default=0.0)


def _extract_major_holders_frame(
    raw: Any,
) -> tuple[float | None, float | None, float | None]:
    insiders = inst = inst_of_float = None
    try:
        import pandas as _pd

        if raw is None:
            return None, None, None
        df = raw
        if isinstance(df, _pd.Series):
            df = df.to_frame()
        try:
            for i in range(min(5, len(df))):
                vals = list(df.iloc[i].values)
                val, label = (
                    (vals[0], str(vals[1]).lower()) if len(vals) == 2 else (vals[0], "")
                )
                try:
                    fv = float(val)
                    if fv > 1.0:
                        fv /= 100.0
                except Exception:
                    continue
                if "insider" in label:
                    insiders = fv
                elif "institut" in label and "float" in label:
                    inst_of_float = fv
                elif "institut" in label:
                    inst = fv
        except Exception:
            pass
        try:
            idx = [str(x).lower() for x in getattr(df, "index", [])]
            for i, lab in enumerate(idx[:5]):
                try:
                    raw_val = _pd.to_numeric(df.iloc[i, 0], errors="coerce")
                    if _pd.isna(raw_val):
                        continue
                    val = float(raw_val)
                    if val > 1.0:
                        val /= 100.0
                except Exception:
                    continue
                if "insider" in lab:
                    insiders = val
                elif "float" in lab and "institut" in lab:
                    inst_of_float = val
                elif "institut" in lab:
                    inst = val
        except Exception:
            pass
    except Exception:
        return None, None, None
    return insiders, inst, inst_of_float


def _extract_major_holders(t: Any) -> tuple[float | None, float | None, float | None]:
    try:
        df = (
            t.get_major_holders()
            if hasattr(t, "get_major_holders")
            else getattr(t, "major_holders", None)
        )
        return _extract_major_holders_frame(df)
    except Exception:
        return None, None, None


def _best_float_pct(
    info: dict[str, Any],
    shares_out: float | None,
    mh: tuple[float | None, float | None, float | None] | None,
    held_inst: float | None,
) -> tuple[float | None, str]:
    fs = _safe_float(info.get("floatShares"))
    if fs and shares_out:
        val = fs / float(shares_out)
        if 0.0 < val <= 1.2:
            return _clip01(val), "native"
    spof = _safe_float(info.get("shortPercentOfFloat"))
    shares_short = _safe_float(info.get("sharesShort"))
    if shares_out and spof and shares_short and spof > 0:
        est_fs = shares_short / spof
        if 0.0 < est_fs <= shares_out * 1.5:
            return _clip01(est_fs / float(shares_out)), "derived"
    candidates: list[float] = []
    hpi = _safe_float(info.get("heldPercentInsiders"))
    if hpi is not None:
        candidates.append(_clip01(1.0 - hpi))
    if mh is not None:
        insiders, inst, inst_of_float = mh
        if insiders is not None:
            candidates.append(_clip01(1.0 - float(insiders)))
        if inst_of_float is not None:
            candidates.append(_clip01(float(inst_of_float)))
        if inst is not None:
            candidates.append(_clip01(float(inst)))
    if held_inst is not None:
        candidates.append(_clip01(float(held_inst)))
    if candidates:
        # Use a conservative estimate to avoid overstating float.
        return min(candidates), "conservative"
    return None, "low_confidence"


def _fetch_fundamentals_one(
    ticker: str,
    *,
    vendor: Any | None = None,
    request_timeout: float | None = None,
    max_attempts: int = 3,
    backoff_factor: float = 1.8,
) -> Funda:
    if is_junk(ticker):
        raise RuntimeError("junk_skip")

    yf_symbol = norm_symbol(ticker)

    last_exc: Exception | None = None
    timeout_sec_raw = safe_float(request_timeout)
    timeout_sec = (
        timeout_sec_raw if timeout_sec_raw is not None and timeout_sec_raw > 0 else None
    )
    attempts = coerce_int(max_attempts, 3, min_value=1, field_name="max_attempts")
    sleep_factor = coerce_float(
        backoff_factor,
        1.8,
        strictly_positive=True,
        field_name="request_backoff",
    )
    if sleep_factor <= 1.0:
        sleep_factor = 1.8

    def _fallback_price_volume(sym: str) -> tuple[float | None, float | None]:
        try:
            import pandas as _pd

            window_kwargs: dict[str, Any] = {
                "interval": "1d",
                "auto_adjust": True,
                "progress": False,
                "threads": False,
            }
            window_kwargs["period"] = "3mo"

            df = pd.DataFrame()
            if vendor is not None and hasattr(vendor, "download_history"):
                try:
                    try:
                        df = vendor.download_history(
                            [sym],
                            period="3mo",
                            interval="1d",
                            auto_adjust=True,
                            progress=False,
                            threads=False,
                            request_timeout=timeout_sec,
                        )
                    except TypeError:
                        df = vendor.download_history(
                            [sym],
                            period="3mo",
                            interval="1d",
                            auto_adjust=True,
                            progress=False,
                            threads=False,
                        )
                except Exception:
                    df = pd.DataFrame()
            if df is None or df.empty:
                if timeout_sec is not None:
                    window_kwargs["timeout"] = timeout_sec
                df = yf.download(sym, **window_kwargs)
            if df is None or df.empty:
                return None, None

            if isinstance(df.columns, _pd.MultiIndex):
                sub = None
                for level in range(df.columns.nlevels):
                    try:
                        sub = df.xs(sym, axis=1, level=level)
                        break
                    except Exception:
                        continue
                if isinstance(sub, _pd.DataFrame):
                    df = sub

            close_series = None
            for c in ("Adj Close", "Close"):
                if c in df.columns:
                    close_series = _pd.to_numeric(df[c], errors="coerce").dropna()
                    if not close_series.empty:
                        break
            px = (
                float(close_series.iloc[-1])
                if close_series is not None and not close_series.empty
                else None
            )

            vol = None
            if "Volume" in df.columns:
                vs = _pd.to_numeric(df["Volume"], errors="coerce").dropna()
                if not vs.empty:
                    tail_n = 21 if len(vs) >= 21 else (10 if len(vs) >= 10 else len(vs))
                    vol = float(vs.tail(tail_n).mean()) if tail_n > 0 else None
            return px, vol
        except Exception:
            return None, None

    for attempt in range(1, attempts + 1):
        try:
            t = None
            price: float | None = None
            mcap: float | None = None
            vol: float | None = None
            float_pct: float | None = None
            div = False
            is_etf = False
            sector = industry = country = None

            fi = None
            info: dict[str, Any] = {}
            mh_raw = None
            if vendor is not None and hasattr(vendor, "fetch_quote_bundle"):
                try:
                    try:
                        payload = vendor.fetch_quote_bundle(
                            yf_symbol,
                            include_major_holders=True,
                            request_timeout=timeout_sec,
                        )
                    except TypeError:
                        payload = vendor.fetch_quote_bundle(
                            yf_symbol, include_major_holders=True
                        )
                    if isinstance(payload, dict):
                        fi = payload.get("fast_info")
                        info_raw = payload.get("info")
                        if isinstance(info_raw, dict):
                            info = info_raw
                        mh_raw = payload.get("major_holders")
                except Exception:
                    fi = None
                    info = {}
                    mh_raw = None

            # Fallback to direct yfinance only when no vendor wrapper is provided.
            if (fi is None or not info) and vendor is None:
                t = yf.Ticker(yf_symbol)
                if fi is None:
                    fi = getattr(t, "fast_info", None)
                if not info:
                    try:
                        info = (
                            t.get_info() if hasattr(t, "get_info") else (t.info or {})
                        )
                    except Exception:
                        info = {}

            def _figet(key: str, _fi: Any = fi) -> Any:
                if _fi is None:
                    return None
                try:
                    return (
                        _fi.get(key) if hasattr(_fi, "get") else getattr(_fi, key, None)
                    )
                except Exception:
                    return None

            price = _safe_float(_figet("lastPrice")) or _safe_float(
                _figet("regularMarketPrice")
            )
            mcap = _safe_float(_figet("marketCap"))
            vol = (
                _safe_float(_figet("tenDayAverageVolume"))
                or _safe_float(_figet("threeMonthAverageVolume"))
                or _safe_float(_figet("lastVolume"))
            )

            if price is None:
                price = _safe_float(info.get("regularMarketPrice")) or _safe_float(
                    info.get("previousClose")
                )
            if mcap is None:
                mcap = _safe_float(info.get("marketCap"))
            if vol is None:
                vol = (
                    _safe_float(info.get("averageDailyVolume10Day"))
                    or _safe_float(info.get("averageVolume"))
                    or _safe_float(info.get("volume"))
                )

            shares_out = (
                _safe_float(info.get("sharesOutstanding"))
                or _safe_float(_figet("sharesOutstanding"))
                or _safe_float(info.get("impliedSharesOutstanding"))
            )

            if mcap is None and shares_out and price:
                mcap = float(shares_out) * float(price)

            if price is None or vol is None:
                px2, vol2 = _fallback_price_volume(yf_symbol)
                if price is None:
                    price = px2
                if vol is None:
                    vol = vol2

            if mh_raw is not None:
                mh = _extract_major_holders_frame(mh_raw)
            elif t is not None:
                mh = _extract_major_holders(t)
            else:
                mh = (None, None, None)
            held_inst = _safe_float(info.get("heldPercentInstitutions"))
            float_pct, float_quality = _best_float_pct(info, shares_out, mh, held_inst)
            if not float_quality:
                float_quality = "low_confidence"

            dy = _safe_float(info.get("dividendYield"))
            div = bool(dy is not None and dy > 0)
            qt_raw = str(info.get("quoteType") or _figet("quoteType") or "").upper()
            is_etf = qt_raw == "ETF"
            is_etn = qt_raw == "ETN"
            is_baby_bond = (
                "baby bond"
                in (
                    str(info.get("longName") or "")
                    + " "
                    + str(info.get("shortName") or "")
                ).lower()
                or qt_raw == "BOND"
            )
            adr_hint = " ".join(
                str(x or "")
                for x in (
                    info.get("shortName"),
                    info.get("longName"),
                    info.get("quoteSourceName"),
                )
            ).lower()
            is_adr = qt_raw == "ADR" or " adr" in adr_hint
            trust_hint = " ".join(
                str(x or "") for x in (info.get("shortName"), info.get("longName"))
            ).lower()
            is_trust = "trust" in trust_hint or qt_raw == "TRUST"

            sector = str(info.get("sector") or None) or None
            industry = str(info.get("industry") or None) or None
            country = str(info.get("country") or None) or None
            currency = (
                str(
                    info.get("currency")
                    or info.get("financialCurrency")
                    or _figet("currency")
                    or ""
                ).upper()
                or None
            )
            exchange_code = (
                str(
                    info.get("exchange")
                    or info.get("fullExchangeName")
                    or _figet("exchange")
                    or ""
                )
                or None
            )
            market = str(info.get("market") or _figet("market") or "").lower() or None
            long_name = (
                str(info.get("longName") or info.get("shortName") or "").strip() or None
            )

            return Funda(
                ticker=ticker,
                price=price,
                market_cap=mcap,
                volume=vol,
                float_pct=float_pct,
                dividend=div,
                is_etf=is_etf,
                sector=sector or None,
                industry=industry or None,
                country=country or None,
                shares_out=shares_out,
                long_name=long_name,
                currency=currency,
                float_quality=float_quality,
                is_etn=is_etn,
                is_baby_bond=is_baby_bond,
                is_adr=is_adr,
                is_trust=is_trust,
                exchange_code=exchange_code,
                market=market,
                quote_type=qt_raw or None,
            )
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                time.sleep(_sleep_backoff(attempt, factor=sleep_factor))
    raise last_exc if last_exc else RuntimeError("Unknown fetch error")


def _missing_core_fields(payload: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in ("price", "market_cap", "volume", "float_pct"):
        val = payload.get(field)
        if val is None:
            missing.append(field)
            continue
        try:
            fv = float(val)
        except Exception:
            continue
        if math.isnan(fv):
            missing.append(field)
    return missing


def fetch_fundamentals_parallel(
    tickers: list[str],
    workers: int,
    show_progress: bool,
    rate_limit_per_sec: float,
    breaker: CircuitBreaker | None = None,
    checkpoint: Checkpointer | None = None,
    checkpoint_filter: set[str] | None = None,
    checkpoint_cfg_hash: str | None = None,
    checkpoint_ttl: float | None = None,
    *,
    max_inflight: int | None = None,
    stop_event: threading.Event | None = None,
    ensure_not_cancelled: Callable[[threading.Event | None], None] | None = None,
    incomplete_sample_limit: int = INCOMPLETE_SAMPLE_LIMIT,
    junk_filter: Callable[[str], bool] | None = None,
    vendor: Any | None = None,
    vol_lookback_priorities: tuple[int, int] | None = None,
    postfill_mode: str = "drop",
    use_token_bucket: bool = True,
    request_timeout: float | None = None,
    request_retries: int = 3,
    request_backoff: float = 1.8,
    heartbeat_logging: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from concurrent.futures import (
        CancelledError,
        ThreadPoolExecutor,
        TimeoutError as FuturesTimeoutError,
        as_completed,
    )

    monitoring: dict[str, Any] = {"failed": [], "timeouts": 0}
    records: list[dict[str, Any]] = []
    bucket = (
        TokenBucket(rate_per_sec=rate_limit_per_sec, burst=max(2, workers * 2))
        if use_token_bucket
        else None
    )
    if bucket is not None:
        bucket.reset()
    if breaker is not None:
        breaker.reset()

    def _safe_cancel_check() -> None:
        if ensure_not_cancelled is not None:
            ensure_not_cancelled(stop_event)

    if checkpoint is not None:
        if checkpoint_filter:
            skip = {norm_symbol(s) for s in checkpoint_filter}
            tickers = [t for t in tickers if norm_symbol(t) not in skip]
        else:
            tickers = [
                t
                for t in tickers
                if not checkpoint.is_done(
                    norm_symbol(t),
                    cfg_hash=checkpoint_cfg_hash,
                    max_age=checkpoint_ttl,
                )
            ]
    junk_pred = junk_filter or is_junk
    tickers = [t for t in tickers if not junk_pred(t)]
    total = len(tickers)
    if total == 0:
        return _empty_fundamentals_frame(full=True), monitoring

    max_workers = coerce_int(workers, 1, min_value=1, field_name="workers")
    inflight_cap = max_inflight if max_inflight is not None else max_workers
    inflight_cap = max(1, min(inflight_cap, total))
    retries_eff = coerce_int(
        request_retries, 3, min_value=0, field_name="request_retries"
    )
    backoff_eff = coerce_float(
        request_backoff,
        1.8,
        strictly_positive=True,
        field_name="request_backoff",
    )
    request_timeout_eff_raw = safe_float(request_timeout)
    request_timeout_eff = (
        request_timeout_eff_raw
        if request_timeout_eff_raw is not None and request_timeout_eff_raw > 0
        else None
    )
    per_symbol_timeout: float | None = None
    if request_timeout_eff is not None:
        retry_span = float(max(1, retries_eff + 1))
        # Keep timeout bounded so the progress loop is responsive even when
        # upstream calls stall for a long time.
        per_symbol_timeout = request_timeout_eff * retry_span * 1.5
        if request_timeout_eff >= 1.0:
            per_symbol_timeout = min(per_symbol_timeout, 120.0)
    # Poll periodically so the main loop can emit heartbeats and observe cancellation
    # even when all workers are waiting on slow vendor calls.
    poll_timeout_sec = 5.0
    if per_symbol_timeout is not None:
        # Keep polling at most every 5 seconds so users see regular progress
        # heartbeats even when all in-flight calls are still pending.
        poll_timeout_sec = max(1.0, min(5.0, per_symbol_timeout / 6.0))

    def one(sym: str) -> tuple[str, Funda | None, str | None]:
        if breaker is not None and breaker.open():
            return sym, None, "circuit_open"
        if stop_event is not None and stop_event.is_set():
            return sym, None, "cancelled"
        if junk_pred(sym):
            return sym, None, "junk_skip"
        try:
            if bucket is not None:
                bucket.take(stop_event=stop_event)

            def fetch_call() -> Funda:
                return _fetch_fundamentals_one(
                    sym,
                    vendor=vendor,
                    request_timeout=request_timeout_eff,
                    max_attempts=max(1, retries_eff + 1),
                    backoff_factor=backoff_eff,
                )

            if per_symbol_timeout is not None:
                try:
                    f = _run_with_timeout(fetch_call, per_symbol_timeout)
                except TimeoutError:
                    return sym, None, "timeout"
            else:
                f = fetch_call()
            return sym, f, None
        except KeyboardInterrupt as e:
            if stop_event is not None and stop_event.is_set():
                return sym, None, "cancelled"
            detail = str(e).strip() or "KeyboardInterrupt"
            return sym, None, f"worker_keyboard_interrupt:{detail}"
        except TimeoutError:
            return sym, None, "timeout"
        except Exception as e:
            return sym, None, str(e)

    cancelled = 0
    submitted = 0
    breaker_tripped = False
    cancel_requested = False
    pbar = None
    stalled_polls = 0

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            active: set[Any] = set()
            future_symbol: dict[Any, str] = {}

            def _submit_next() -> bool:
                nonlocal submitted, breaker_tripped, cancel_requested
                if submitted >= total:
                    return False
                if stop_event is not None and stop_event.is_set():
                    cancel_requested = True
                    return False
                if breaker is not None and breaker.open():
                    breaker_tripped = True
                    return False
                sym = norm_symbol(tickers[submitted])
                submitted += 1
                fut = pool.submit(one, sym)
                active.add(fut)
                future_symbol[fut] = sym
                return True

            while len(active) < inflight_cap and _submit_next():
                pass

            if show_progress and tqdm is not None:
                pbar = tqdm(total=total, desc="Fundamentals", unit="tick")
                pbar.set_postfix(active=len(active), submitted=submitted, done=0)
                pbar.refresh()

            logger.info(
                "Fundamentals fetch started: total=%d workers=%d inflight=%d "
                "timeout_per_symbol=%s rate_limit_per_sec=%.3f.",
                total,
                max_workers,
                inflight_cap,
                (
                    f"{per_symbol_timeout:.1f}s"
                    if per_symbol_timeout is not None
                    else "none"
                ),
                float(rate_limit_per_sec),
            )

            while active:
                if stop_event is not None and stop_event.is_set():
                    cancel_requested = True
                    for pending in list(active):
                        pending.cancel()
                    # Count submitted-but-unprocessed tasks that were cancelled.
                    cancelled += len(active)
                    future_symbol.clear()
                    active.clear()
                    break
                try:
                    fut = next(as_completed(active, timeout=poll_timeout_sec))
                    stalled_polls = 0
                except FuturesTimeoutError:
                    stalled_polls += 1
                    if pbar is not None:
                        done_count = (
                            len(records)
                            + len(monitoring.get("failed", []))
                            + int(monitoring.get("timeouts", 0))
                            + int(cancelled)
                        )
                        pbar.set_postfix(
                            active=len(active),
                            submitted=submitted,
                            done=done_count,
                        )
                        pbar.refresh()
                    if heartbeat_logging and (
                        stalled_polls == 1 or stalled_polls % 6 == 0
                    ):
                        logger.info(
                            "Fundamentals warten auf Antworten: active=%d submitted=%d/%d "
                            "ok=%d failed=%d timeouts=%d.",
                            len(active),
                            submitted,
                            total,
                            len(records),
                            len(monitoring.get("failed", [])),
                            int(monitoring.get("timeouts", 0)),
                        )
                    continue
                active.remove(fut)
                submitted_sym = future_symbol.pop(fut, "")
                try:
                    sym, f, err = fut.result()
                except CancelledError:
                    sym, f, err = submitted_sym, None, "cancelled"
                except KeyboardInterrupt as e:
                    if stop_event is not None and stop_event.is_set():
                        sym, f, err = submitted_sym, None, "cancelled"
                    else:
                        detail = str(e).strip() or "KeyboardInterrupt"
                        sym, f, err = (
                            submitted_sym,
                            None,
                            f"worker_keyboard_interrupt:{detail}",
                        )
                        monitoring["worker_interrupts"] = (
                            int(monitoring.get("worker_interrupts", 0)) + 1
                        )
                        logger.warning(
                            "Unexpected worker KeyboardInterrupt treated as symbol failure: %s",
                            sym,
                        )
                except Exception as e:
                    if stop_event is not None and stop_event.is_set():
                        sym, f, err = submitted_sym, None, "cancelled"
                    else:
                        sym, f, err = submitted_sym, None, f"worker_exception:{e}"
                        logger.warning(
                            "Worker exception treated as symbol failure: %s (%s)",
                            sym,
                            e,
                        )
                if pbar is not None:
                    pbar.update(1)

                allow_submit = True
                if err == "cancelled":
                    cancel_requested = True
                    cancelled += 1
                    allow_submit = False
                elif err == "circuit_open":
                    breaker_tripped = True
                    cancel_requested = True
                    cancelled += 1
                    allow_submit = False
                elif err == "junk_skip":
                    # Treat as processed and continue with the next symbol.
                    pass
                else:
                    if err == "timeout":
                        monitoring["timeouts"] = int(monitoring.get("timeouts", 0)) + 1
                    if f is None:
                        monitoring["failed"].append(sym)
                        if breaker is not None:
                            breaker.err()
                    else:
                        payload = asdict(f)
                        records.append(payload)
                        missing = _missing_core_fields(payload)
                        if missing:
                            monitoring["incomplete_core_total"] = (
                                int(monitoring.get("incomplete_core_total", 0)) + 1
                            )
                            sample = monitoring.setdefault("incomplete_core", [])
                            if len(sample) < incomplete_sample_limit:
                                sample.append({"ticker": sym, "fields": missing})
                            counts = monitoring.setdefault("missing_field_counts", {})
                            for field in missing:
                                counts[field] = int(counts.get(field, 0)) + 1
                        if breaker is not None:
                            breaker.ok()

                if allow_submit and len(active) < inflight_cap:
                    _submit_next()
                if pbar is not None:
                    done_count = (
                        len(records)
                        + len(monitoring.get("failed", []))
                        + int(monitoring.get("timeouts", 0))
                        + int(cancelled)
                    )
                    pbar.set_postfix(
                        active=len(active),
                        submitted=submitted,
                        done=done_count,
                    )

    finally:
        if pbar is not None:
            pbar.close()

    if cancel_requested or breaker_tripped:
        remaining = max(0, total - submitted)
        cancelled += remaining
    if cancelled:
        monitoring["cancelled"] = int(cancelled)

    if cancel_requested:
        _safe_cancel_check()

    if breaker is not None and breaker.open():
        raise RuntimeError("Circuit breaker opened during fundamentals fetch.")

    if not records:
        return _empty_fundamentals_frame(full=False), monitoring
    df = pd.DataFrame.from_records(records).set_index("ticker").sort_index()
    df.index = df.index.map(norm_symbol)
    df["updated_at"] = float(time.time())

    need_price_mask = (
        df["price"].isna()
        if "price" in df.columns
        else pd.Series(False, index=df.index)
    )
    need_vol_mask = (
        df["volume"].isna()
        if "volume" in df.columns
        else pd.Series(False, index=df.index)
    )
    to_fix = df.index[need_price_mask | need_vol_mask].tolist()
    if to_fix:
        mode = str(postfill_mode or "drop").strip().lower()
        if mode not in {"drop", "fill"}:
            logger.warning(
                "Invalid postfill_mode=%r; falling back to 'drop'.", postfill_mode
            )
            mode = "drop"
        monitoring["postfill_mode"] = mode
        postfill_missing_price_volume(
            df,
            workers=workers,
            rate_limit_per_sec=rate_limit_per_sec,
            stop_event=stop_event,
            ensure_not_cancelled=ensure_not_cancelled or (lambda _: None),
            monitoring=monitoring,
            incomplete_sample_limit=incomplete_sample_limit,
            junk_filter=junk_pred,
            chunk_size=50,
            vol_lookback_priorities=vol_lookback_priorities,
            drop_instead_of_fill=(mode != "fill"),
        )

    if "market_cap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
        mc_missing = df["market_cap"].isna()
        if mc_missing.any() and "shares_out" in df.columns and "price" in df.columns:
            so = pd.to_numeric(df["shares_out"], errors="coerce")
            px = pd.to_numeric(df["price"], errors="coerce")
            mc_series = replace_inf_with_nan(so * px)
            df.loc[mc_missing, "market_cap"] = mc_series.loc[mc_missing]

    if {"price", "volume"}.issubset(df.columns):
        dadv = pd.to_numeric(df["price"], errors="coerce") * pd.to_numeric(
            df["volume"], errors="coerce"
        )
        df["dollar_adv"] = replace_inf_with_nan(dadv)
    else:
        df["dollar_adv"] = float("nan")

    if df.index.name is None:
        df.index.name = "ticker"
    return df, monitoring
