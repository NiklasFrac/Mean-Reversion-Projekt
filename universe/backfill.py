from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Sequence

import pandas as pd

try:  # pragma: no cover - optional dependency is required at runtime
    import yfinance as yf
except Exception as e:  # pragma: no cover
    raise ImportError(
        "universe.backfill requires 'yfinance' (pip install yfinance)."
    ) from e

from universe.monitoring import logger

__all__ = ["TokenBucket", "postfill_missing_price_volume"]


class TokenBucket:
    """
    Thread-safe token bucket rate limiter.

    Parameters
    ----------
    rate_per_sec :
        Tokens replenished per second (minimum 1e-4).
    burst :
        Maximum number of tokens that can accumulate (initial tokens = burst).
    now_fn :
        Optional time source; defaults to time.monotonic.
    sleep_fn :
        Optional sleep function; defaults to time.sleep.

    Behaviour
    ---------
    - `take()` blocks until at least one token is available.
    - Designed to be safe under multi-threaded access.
    """

    def __init__(
        self,
        rate_per_sec: float = 8.0,
        burst: int = 32,
        now_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.rate = max(0.0001, float(rate_per_sec))

        capacity = max(1.0, float(burst))
        if capacity > 1_000_000:
            # Extremely large bursts are almost certainly a configuration bug.
            logger.warning(
                "TokenBucket burst=%s is very large; clamping to 1_000_000.", capacity
            )
            capacity = 1_000_000
        self.capacity = capacity
        self.tokens = self.capacity

        # Use a monotonic clock by default to avoid issues with system clock jumps.
        self._now = now_fn or time.monotonic
        self._sleep = sleep_fn or time.sleep
        self.last = self._now()
        self.lock = threading.Lock()

    def reset(self) -> None:
        """
        Reset the bucket to a full state (capacity tokens, current time baseline).
        """
        with self.lock:
            self.tokens = self.capacity
            self.last = self._now()

    def take(self, *, stop_event: threading.Event | None = None) -> None:
        """
        Block until a token is available, then consume one token.
        """
        while True:
            with self.lock:
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("cancelled")
                now = self._now()
                elapsed = now - self.last
                if elapsed < 0:
                    # Should not happen with a monotonic clock, but guard anyway.
                    elapsed = 0.0
                refill = elapsed * self.rate
                self.tokens = min(self.capacity, self.tokens + refill)
                self.last = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return

                needed = 1.0 - self.tokens
                sleep = needed / self.rate

            # Sleep outside the lock to avoid blocking other waiters.
            sleep_sec = max(0.0, sleep)
            if stop_event is None:
                self._sleep(sleep_sec)
                continue
            while sleep_sec > 0:
                if stop_event.is_set():
                    raise RuntimeError("cancelled")
                chunk = min(0.1, sleep_sec)
                self._sleep(max(0.0, chunk))
                sleep_sec -= chunk
            if stop_event.is_set():
                raise RuntimeError("cancelled")


def _cooperative_sleep(seconds: float, stop_event: threading.Event | None) -> None:
    """
    Sleep in small increments so cancellation can be observed promptly.
    """
    if seconds <= 0:
        return
    if stop_event is None:
        time.sleep(seconds)
        return
    deadline = time.time() + seconds
    while time.time() < deadline:
        if stop_event.is_set():
            return
        chunk = min(0.1, deadline - time.time())
        if chunk > 0:
            time.sleep(chunk)


def _determine_ticker_level(raw: pd.DataFrame, symbols: Sequence[str]) -> int | None:
    """
    Attempt to determine which MultiIndex level contains ticker symbols.

    Returns the level index if a plausible level is found, else None.
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        return None

    for lvl in range(raw.columns.nlevels):
        try:
            level_values = set(map(str, raw.columns.get_level_values(lvl)))
        except Exception:
            continue
        if any(sym in level_values for sym in symbols):
            return lvl
    return None


def _fetch_price_volume_for_symbols(
    symbols: Sequence[str],
    *,
    junk_filter: Callable[[str], bool] | None = None,
    vol_lookback_priorities: tuple[int, int] = (21, 10),
    retries: int = 2,
    backoff: float = 0.6,
    monitoring: dict[str, Any] | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, tuple[float | None, float | None]]:
    """
    Fetch last close and average volume for a batch of symbols using yfinance.

    Parameters
    ----------
    symbols :
        Iterable of ticker symbols.
    junk_filter :
        Optional predicate to skip unwanted/junk symbols.
    vol_lookback_priorities :
        Tuple `(primary, secondary)` for the volume averaging window in trading
        days, applied as:
            - if len(volume) >= primary: use `primary`
            - elif len(volume) >= secondary: use `secondary`
            - else: use `len(volume)`

    Returns
    -------
    dict
        Mapping symbol -> (last_close, avg_volume). Values are `None` when not
        available. Symbols filtered by `junk_filter` are omitted.
    """
    clean_symbols = [
        t for t in symbols if t and not (junk_filter(t) if junk_filter else False)
    ]
    if not clean_symbols:
        return {}

    download_kwargs: dict[str, Any] = {
        "interval": "1d",
        "auto_adjust": True,
        "progress": False,
        "threads": False,
    }
    download_kwargs["period"] = "3mo"

    raw = None
    errors: list[str] = []
    try:
        retries_int = max(0, int(retries))
    except Exception:
        retries_int = 0
    try:
        backoff_val = float(backoff)
        if backoff_val <= 0:
            backoff_val = 0.1
    except Exception:
        backoff_val = 0.6

    for attempt in range(retries_int + 1):
        if stop_event is not None and stop_event.is_set():
            return {}
        try:
            raw = yf.download(clean_symbols, **download_kwargs)
        except Exception as e:
            errors.append(str(e))
            if stop_event is not None and stop_event.is_set():
                return {}
            if attempt < retries_int:
                _cooperative_sleep(backoff_val * (2**attempt), stop_event)
                continue
            logger.warning(
                "Post-fill chunk download failed (%d tickers) after %d attempts: %s",
                len(clean_symbols),
                retries_int + 1,
                "; ".join(errors),
            )
            if monitoring is not None:
                monitoring.setdefault("postfill_errors", []).append(
                    f"download_failed:{errors[-1]}"
                )
                monitoring["postfill_failed_chunks"] = (
                    int(monitoring.get("postfill_failed_chunks", 0)) + 1
                )
            return {}
        if raw is None or raw.empty:
            if stop_event is not None and stop_event.is_set():
                return {}
            if attempt < retries_int:
                _cooperative_sleep(backoff_val * (2**attempt), stop_event)
                continue
            logger.warning(
                "Post-fill chunk download empty for %d tickers after %d attempts.",
                len(clean_symbols),
                retries_int + 1,
            )
            if monitoring is not None:
                monitoring.setdefault("postfill_errors", []).append("download_empty")
                monitoring["postfill_failed_chunks"] = (
                    int(monitoring.get("postfill_failed_chunks", 0)) + 1
                )
            return {}
        break

    assert raw is not None
    assert isinstance(raw, pd.DataFrame)
    ticker_level = _determine_ticker_level(raw, clean_symbols)

    def _slice_symbol(sym: str) -> pd.DataFrame | None:
        data = raw
        if isinstance(data.columns, pd.MultiIndex):
            sub = None

            # Preferred: use the detected ticker level.
            if ticker_level is not None:
                try:
                    sub = data.xs(sym, axis=1, level=ticker_level)
                except Exception:
                    sub = None

            # Fallback: heuristic over all levels (legacy behaviour).
            if sub is None:
                for level in range(data.columns.nlevels):
                    try:
                        sub = data.xs(sym, axis=1, level=level)
                        break
                    except Exception:
                        continue

            if sub is None:
                return None
            if isinstance(sub, pd.Series):
                sub = sub.to_frame().T
            return sub if isinstance(sub, pd.DataFrame) else None

        # Single-symbol DataFrame: treat the full frame as that symbol.
        return data if isinstance(data, pd.DataFrame) else None

    def _last_close(sub: pd.DataFrame) -> float | None:
        for col in ("Adj Close", "Close"):
            if col in sub.columns:
                ser = pd.to_numeric(sub[col], errors="coerce").dropna()
                if not ser.empty:
                    return float(ser.iloc[-1])
        if sub.columns.size:
            first = str(sub.columns[0])
            ser = pd.to_numeric(sub[first], errors="coerce").dropna()
            if not ser.empty:
                return float(ser.iloc[-1])
        return None

    def _avg_vol(sub: pd.DataFrame) -> float | None:
        if "Volume" not in sub.columns:
            return None
        vs = pd.to_numeric(sub["Volume"], errors="coerce").dropna()
        if vs.empty:
            return None

        primary, secondary = vol_lookback_priorities
        primary = max(1, int(primary))
        secondary = max(1, int(secondary))

        if len(vs) >= primary:
            n = primary
        elif len(vs) >= secondary:
            n = secondary
        else:
            n = len(vs)

        return float(vs.tail(n).mean()) if n > 0 else None

    out: dict[str, tuple[float | None, float | None]] = {}
    for sym in clean_symbols:
        sub = _slice_symbol(sym)
        if sub is None:
            continue
        px = _last_close(sub)
        vol = _avg_vol(sub)
        if px is None and vol is None:
            continue
        out[sym] = (px, vol)
    return out


def _postfill_fetch_chunk(
    chunk: list[str],
    *,
    junk_filter: Callable[[str], bool] | None = None,
    vol_lookback_priorities: tuple[int, int] = (21, 10),
    monitoring: dict[str, Any] | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, tuple[float | None, float | None]]:
    """
    Backwards-compatible wrapper for fetching last close and average volume.

    This keeps the original function name used by `postfill_missing_price_volume`,
    while delegating to the more general `_fetch_price_volume_for_symbols`.
    """
    return _fetch_price_volume_for_symbols(
        chunk,
        junk_filter=junk_filter,
        vol_lookback_priorities=vol_lookback_priorities,
        monitoring=monitoring,
        stop_event=stop_event,
    )


def postfill_missing_price_volume(
    df: pd.DataFrame,
    *,
    workers: int,
    rate_limit_per_sec: float,
    stop_event: threading.Event | None,
    ensure_not_cancelled: Callable[[threading.Event | None], None],
    monitoring: dict[str, Any],
    incomplete_sample_limit: int,
    junk_filter: Callable[[str], bool] | None = None,
    chunk_size: int = 50,
    strict_columns: bool = False,
    vol_lookback_priorities: tuple[int, int] | None = None,
    drop_instead_of_fill: bool = False,
) -> None:
    """
    Fill missing `price` and `volume` fields in-place via yfinance.

    Parameters
    ----------
    df :
        DataFrame indexed by ticker symbol. Must have `price` and `volume` columns
        unless `strict_columns=False`, in which case missing columns are treated
        as "no missing values".
    workers :
        Maximum number of concurrent worker threads.
    rate_limit_per_sec :
        Rate limit in requests per second (enforced via TokenBucket) for the
        backfill phase.
    stop_event :
        Optional threading.Event used for cooperative cancellation.
    ensure_not_cancelled :
        Callback that is periodically invoked as `ensure_not_cancelled(stop_event)`.
        It may raise to abort the operation (e.g. if a higher-level cancellation
        is requested).
    monitoring :
        Dict to which monitoring / diagnostic stats are written.
    incomplete_sample_limit :
        Max number of unresolved symbols to sample into `monitoring["postfill_unresolved"]`.
    junk_filter :
        Optional predicate to skip "junk" symbols (e.g. warrants, rights).
    chunk_size :
        Number of symbols per yfinance chunk request.
    strict_columns :
        If True, require `price` and `volume` columns to exist, else raise.
        If False (default), silently treat missing columns as having no missing values
        (legacy behaviour).
    vol_lookback_priorities :
        Optional override of `(primary, secondary)` volume lookback windows.
        If None, defaults to `(21, 10)`.

    Returns
    -------
    None
        Mutates `df` in-place and updates `monitoring`.
    """
    workers = max(1, int(workers))
    chunk_size = max(1, int(chunk_size))

    required_cols = ("price", "volume")
    if strict_columns:
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"postfill_missing_price_volume requires columns {required_cols}, "
                f"but {missing_cols} are missing."
            )

    need_price_mask: pd.Series[Any]
    if "price" in df.columns:
        need_price_mask = df["price"].isna()
    else:
        need_price_mask = pd.Series(False, index=df.index, dtype=bool)

    need_vol_mask: pd.Series[Any]
    if "volume" in df.columns:
        need_vol_mask = df["volume"].isna()
    else:
        need_vol_mask = pd.Series(False, index=df.index, dtype=bool)

    to_fix = df.index[need_price_mask | need_vol_mask].tolist()
    if not to_fix:
        return

    if drop_instead_of_fill:
        monitoring["postfill_total"] = len(to_fix)
        monitoring["postfill_chunks"] = 0
        monitoring["postfill_completed_chunks"] = 0
        monitoring["postfill_duration_sec"] = 0.0
        monitoring["postfill_unresolved_total"] = len(to_fix)
        monitoring["postfill_unresolved"] = to_fix[:incomplete_sample_limit]
        monitoring["postfill_unresolved_all"] = list(to_fix)
        if to_fix:
            sample = ", ".join(to_fix[:5])
            logger.warning(
                "Post-fill disabled (drop-only); dropping %d unresolved symbols: %s",
                len(to_fix),
                sample,
            )
            try:
                df.drop(index=to_fix, errors="ignore", inplace=True)
            except Exception as exc:
                logger.debug(
                    "Failed to drop unresolved symbols during drop-only postfill: %s",
                    exc,
                )
        return

    chunks = [to_fix[i : i + chunk_size] for i in range(0, len(to_fix), chunk_size)]
    postfill_workers = min(max(1, workers), len(chunks))

    logger.info(
        "Post-fill missing price/volume via yf.download for %d tickers (chunk=%d, workers=%d).",
        len(to_fix),
        chunk_size,
        postfill_workers,
    )

    monitoring["postfill_total"] = len(to_fix)
    monitoring["postfill_chunks"] = len(chunks)
    monitoring.setdefault("postfill_errors", [])
    monitoring.setdefault("postfill_failed_chunks", 0)

    bucket_pf = TokenBucket(rate_per_sec=rate_limit_per_sec, burst=max(2, workers * 2))
    bucket_pf.reset()

    start_postfill = time.perf_counter()
    completed_chunks = 0
    progress_step = max(1, len(chunks) // 10)

    missing_price_symbols = set(need_price_mask[need_price_mask].index)
    missing_vol_symbols = set(need_vol_mask[need_vol_mask].index)

    effective_vol_lookback = vol_lookback_priorities or (21, 10)
    if (
        not isinstance(effective_vol_lookback, (list, tuple))
        or len(effective_vol_lookback) != 2
    ):
        effective_vol_lookback = (21, 10)
    try:
        effective_vol_lookback = tuple(max(1, int(v)) for v in effective_vol_lookback)  # type: ignore
    except Exception:
        effective_vol_lookback = (21, 10)

    errors: list[str] = []
    cancelled = False

    def _postfill_worker(
        chunk_syms: list[str],
    ) -> dict[str, tuple[float | None, float | None]]:
        if stop_event is not None and stop_event.is_set():
            return {}
        try:
            bucket_pf.take(stop_event=stop_event)
        except RuntimeError:
            return {}
        return _postfill_fetch_chunk(
            chunk_syms,
            junk_filter=junk_filter,
            vol_lookback_priorities=effective_vol_lookback,
            monitoring=monitoring,
            stop_event=stop_event,
        )

    with ThreadPoolExecutor(max_workers=postfill_workers) as pool:
        futures = {pool.submit(_postfill_worker, chunk): chunk for chunk in chunks}
        for fut in as_completed(futures):
            # Allow higher-level code to abort if needed.
            ensure_not_cancelled(stop_event)
            if stop_event is not None and stop_event.is_set():
                cancelled = True
            completed_chunks += 1
            try:
                result = fut.result()
            except Exception as e:
                msg = str(e)
                if stop_event is not None and stop_event.is_set():
                    cancelled = True
                errors.append(msg)
                monitoring.setdefault("postfill_errors", []).append(msg)
                monitoring["postfill_failed_chunks"] = (
                    int(monitoring.get("postfill_failed_chunks", 0)) + 1
                )
                logger.warning(
                    "Post-fill worker failed (%d tickers): %s", len(futures[fut]), msg
                )
                continue

            for sym, (pxv, vv) in result.items():
                if sym in missing_price_symbols and pxv is not None:
                    df.at[sym, "price"] = pxv
                    missing_price_symbols.discard(sym)
                if sym in missing_vol_symbols and vv is not None:
                    df.at[sym, "volume"] = vv
                    missing_vol_symbols.discard(sym)

            if completed_chunks % progress_step == 0 or completed_chunks == len(chunks):
                pct = (completed_chunks / len(chunks)) * 100.0
                logger.info(
                    "Post-fill progress: %d/%d chunks (%.1f%%)",
                    completed_chunks,
                    len(chunks),
                    pct,
                )

    duration_postfill = time.perf_counter() - start_postfill
    monitoring["postfill_completed_chunks"] = completed_chunks
    monitoring["postfill_duration_sec"] = float(duration_postfill)

    if stop_event is not None and stop_event.is_set():
        cancelled = True
    if cancelled:
        monitoring["postfill_cancelled"] = True
        raise RuntimeError("postfill cancelled")
    if errors:
        raise RuntimeError(f"postfill failed for {len(errors)} chunks")

    unresolved = sorted(missing_price_symbols.union(missing_vol_symbols))
    if unresolved:
        monitoring["postfill_unresolved_total"] = len(unresolved)
        monitoring["postfill_unresolved"] = unresolved[:incomplete_sample_limit]
        monitoring["postfill_unresolved_all"] = list(unresolved)
        sample = ", ".join(unresolved[:5])
        logger.warning(
            "Post-fill unresolved symbols after retries (dropping %d): %s",
            len(unresolved),
            sample,
        )
        try:
            df.drop(index=unresolved, errors="ignore", inplace=True)
        except Exception as exc:  # defensive
            logger.debug("Failed to drop unresolved symbols from frame: %s", exc)
        missing_price_symbols.clear()
        missing_vol_symbols.clear()
    else:
        monitoring["postfill_unresolved_total"] = 0
        monitoring["postfill_unresolved"] = []
        monitoring["postfill_unresolved_all"] = []
