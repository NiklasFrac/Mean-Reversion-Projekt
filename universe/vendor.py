from __future__ import annotations

import datetime as dt
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

import pandas as pd

from universe.coercion import cfg_bool, cfg_float, cfg_int
from universe.datetime_utils import coerce_utc_naive_timestamp
from universe.panel_utils import (
    coerce_utc_naive_index,
    merge_duplicate_columns_prefer_non_null,
)
from universe.ticker_sets import price_tickers, volume_tickers
from universe.timeout_utils import run_with_timeout

try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    raise ImportError(
        "universe.vendor requires 'yfinance' (pip install yfinance)."
    ) from e

tqdm: Any | None
try:  # pragma: no cover - optional progress bar
    from tqdm import tqdm as _tqdm

    tqdm = _tqdm
except Exception:  # pragma: no cover
    tqdm = None

logger = logging.getLogger("runner_universe")

__all__ = [
    "VendorConfig",
    "UniverseVendor",
    "YFinanceVendor",
    "_clean_panel",
    "_close_ticker_set",
    "_volume_ticker_set",
    "_download_chunk",
    "fetch_price_volume_data",
    "_retry_missing_history",
    "period_start_from_end",
]


def _period_to_offset(
    period: str | None, fallback_months: int = 3
) -> pd.DateOffset | pd.Timedelta:
    """
    Convert a yfinance-style period string into a pandas DateOffset/Timedelta.
    Falls back to `fallback_months` if parsing fails.
    """
    if not period:
        return pd.DateOffset(months=fallback_months)
    p = str(period).strip().lower()
    m = re.match(
        r"^(?P<num>\d+)\s*(?P<unit>d|day|days|w|wk|week|weeks|mo|m|month|months|y|yr|yrs|year|years)$",
        p,
    )
    if m:
        num = int(m.group("num"))
        unit = m.group("unit")
        if unit in {"d", "day", "days"}:
            return pd.Timedelta(days=num)
        if unit in {"w", "wk", "week", "weeks"}:
            return pd.Timedelta(weeks=num)
        if unit in {"mo", "m", "month", "months"}:
            return pd.DateOffset(months=num)
        return pd.DateOffset(years=num)
    if p == "ytd":
        # Use `period_start_from_end` for calendar-accurate YTD handling.
        return pd.DateOffset(years=1)
    if p == "max":
        return pd.DateOffset(years=50)
    return pd.DateOffset(months=fallback_months)


def period_start_from_end(
    *,
    end_ts: pd.Timestamp,
    period: str | None,
    fallback_months: int = 3,
) -> pd.Timestamp:
    """
    Derive a normalized start timestamp from a period and end timestamp.

    Notes:
    - `ytd` maps to Jan 1 of `end_ts.year` (calendar-year semantics).
    - Other periods reuse `_period_to_offset`.
    """
    p = str(period or "").strip().lower()
    end_norm = end_ts.normalize()
    if p == "ytd":
        return pd.Timestamp(year=end_norm.year, month=1, day=1)
    return (
        end_norm - _period_to_offset(period, fallback_months=fallback_months)
    ).normalize()


def _clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = coerce_utc_naive_index(df.index, normalize=False)
    df.index = idx
    df = df.sort_index()
    return merge_duplicate_columns_prefer_non_null(df)


def _close_ticker_set(
    prices_df: pd.DataFrame, *, require_data: bool = False
) -> set[str]:
    return price_tickers(
        prices_df,
        require_data=bool(require_data),
        include_bare_columns=False,
    )


def _volume_ticker_set(
    volumes_df: pd.DataFrame, *, require_data: bool = False
) -> set[str]:
    return volume_tickers(volumes_df, require_data=bool(require_data))


def _coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric values with NaN fallback instead of relying on
    deprecated errors=\"ignore\" behaviour.
    """
    if df.empty:
        return df
    try:
        return df.apply(pd.to_numeric, errors="coerce")
    except Exception:
        converted: dict[str, pd.Series] = {}
        for col in df.columns:
            converted[col] = pd.to_numeric(df[col], errors="coerce")
        return pd.DataFrame(converted, index=df.index)


def _with_optional_timeout(
    kwargs: dict[str, Any], request_timeout: float | None
) -> dict[str, Any]:
    out = dict(kwargs)
    if request_timeout is None:
        return out
    try:
        timeout = float(request_timeout)
    except Exception:
        return out
    if timeout <= 0:
        return out
    out["timeout"] = timeout
    return out


def _download_chunk(
    chunk: list[str],
    interval: str,
    pause: float,
    retries: int,
    backoff: float,
    *,
    auto_adjust: bool,
    start_date: str | dt.date | pd.Timestamp | None,
    end_date: str | dt.date | pd.Timestamp | None,
    request_timeout: float | None = None,
    junk_filter: Callable[[str], bool] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    import pandas as pd

    if junk_filter is not None:
        chunk = [t for t in chunk if not junk_filter(t)]
    if not chunk:
        return pd.DataFrame(), pd.DataFrame()

    FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    attempt = 0
    if start_date is None and end_date is None:
        raise ValueError("start_date/end_date cannot both be None for vendor download.")

    def _to_ts(value: str | dt.date | pd.Timestamp | None) -> pd.Timestamp | None:
        if value is None:
            return None
        if isinstance(value, str):
            raw = value.strip()
            lowered = raw.lower()
            if lowered in {"today", "now"}:
                return pd.Timestamp(dt.datetime.now(dt.UTC).date())
            ts = coerce_utc_naive_timestamp(raw, normalize=True, errors="raise")
            return ts
        ts = coerce_utc_naive_timestamp(value, normalize=True, errors="raise")
        return ts

    start_ts = _to_ts(start_date)
    end_ts = _to_ts(end_date)
    if start_ts is None:
        raise ValueError("start_date must be provided.")
    if end_ts is None:
        raise ValueError("end_date must be provided.")
    if end_ts < start_ts:
        raise ValueError(
            f"end_date {end_ts.date().isoformat()} is before start_date {start_ts.date().isoformat()}."
        )

    start = start_ts.date().isoformat()
    # yfinance uses end as an exclusive bound
    end_excl = (end_ts + pd.Timedelta(days=1)).date().isoformat()

    while attempt <= retries:
        try:
            dl_kwargs: dict[str, Any] = {
                "interval": interval,
                "auto_adjust": auto_adjust,
                "progress": False,
                "threads": False,
            }
            hist_kwargs: dict[str, Any] = {
                "interval": interval,
                "auto_adjust": auto_adjust,
            }
            dl_kwargs["start"] = start
            dl_kwargs["end"] = end_excl
            hist_kwargs["start"] = start
            hist_kwargs["end"] = end_excl

            dl_kwargs_timeout = _with_optional_timeout(dl_kwargs, request_timeout)
            hist_kwargs_timeout = _with_optional_timeout(hist_kwargs, request_timeout)

            if len(chunk) > 1:
                try:
                    df = yf.download(chunk, **dl_kwargs_timeout)
                except TypeError:
                    df = yf.download(chunk, **dl_kwargs)
            else:
                tkr = chunk[0]
                try:
                    hist = yf.Ticker(tkr).history(**hist_kwargs_timeout)
                except TypeError:
                    hist = yf.Ticker(tkr).history(**hist_kwargs)
                if hist is None or hist.empty:
                    try:
                        df = yf.download(tkr, **dl_kwargs_timeout)
                    except TypeError:
                        df = yf.download(tkr, **dl_kwargs)
                else:
                    df = hist

            if df is None or df.empty:
                return pd.DataFrame(), pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                lvl0 = list(map(str, df.columns.get_level_values(0)))
                lvl1 = list(map(str, df.columns.get_level_values(1)))

                def detect_levels() -> tuple[int, int]:
                    if any(f in lvl0 for f in FIELDS) and not any(
                        f in lvl1 for f in FIELDS
                    ):
                        return 0, 1
                    if any(f in lvl1 for f in FIELDS) and not any(
                        f in lvl0 for f in FIELDS
                    ):
                        return 1, 0
                    c0 = sum(1 for f in FIELDS if f in lvl0)
                    c1 = sum(1 for f in FIELDS if f in lvl1)
                    return (0, 1) if c0 >= c1 else (1, 0)

                field_level, ticker_level = detect_levels()

                tickers_in_df = sorted({str(col[ticker_level]) for col in df.columns})

                close_cols: dict[str, pd.Series] = {}
                vol_cols: dict[str, pd.Series] = {}
                ohlc_extra: dict[str, pd.Series] = {}

                for tkr in tickers_in_df:
                    try:
                        sub = df.xs(tkr, axis=1, level=ticker_level)
                    except Exception:
                        continue
                    if isinstance(sub, pd.Series):
                        sub = sub.to_frame().T

                    sub = _coerce_numeric_frame(sub)

                    s_close = None
                    if auto_adjust:
                        if "Adj Close" in sub.columns:
                            s_close = pd.to_numeric(sub["Adj Close"], errors="coerce")
                        elif "Close" in sub.columns:
                            s_close = pd.to_numeric(sub["Close"], errors="coerce")
                    else:
                        if "Close" in sub.columns:
                            s_close = pd.to_numeric(sub["Close"], errors="coerce")
                        elif "Adj Close" in sub.columns:
                            s_close = pd.to_numeric(sub["Adj Close"], errors="coerce")

                    if s_close is not None and not s_close.isna().all():
                        close_cols[f"{tkr}_close"] = s_close

                    if "Volume" in sub.columns:
                        s_vol = pd.to_numeric(sub["Volume"], errors="coerce")
                        if not s_vol.isna().all():
                            vol_cols[tkr] = s_vol

                    for fld, suf in (
                        ("Open", "_open"),
                        ("High", "_high"),
                        ("Low", "_low"),
                    ):
                        if fld in sub.columns:
                            s = pd.to_numeric(sub[fld], errors="coerce")
                            if not s.isna().all():
                                ohlc_extra[f"{tkr}{suf}"] = s

                close_df = pd.DataFrame(close_cols)
                vol_df = pd.DataFrame(vol_cols)

                if not close_df.empty:
                    want = []
                    for t in chunk:
                        name = f"{t}_close"
                        if name in close_df.columns:
                            want.append(name)
                    rest = [c for c in close_df.columns if c not in want]
                    close_df = close_df.loc[:, want + rest]
                if not vol_df.empty:
                    ordered = [t for t in chunk if t in vol_df.columns]
                    vol_df = vol_df.loc[
                        :, ordered + [c for c in vol_df.columns if c not in ordered]
                    ]

                for name, s in ohlc_extra.items():
                    close_df[name] = s

                return close_df, vol_df

            if len(chunk) == 1:
                tkr = chunk[0]
                src = df

                src = _coerce_numeric_frame(src)

                name_close = f"{tkr}_close"
                if auto_adjust:
                    if "Adj Close" in src.columns:
                        close_df = src[["Adj Close"]].rename(
                            columns={"Adj Close": name_close}
                        )
                    elif "Close" in src.columns:
                        close_df = src[["Close"]].rename(columns={"Close": name_close})
                    else:
                        first = str(src.columns[0])
                        close_df = src[[first]].rename(columns={first: name_close})
                else:
                    if "Close" in src.columns:
                        close_df = src[["Close"]].rename(columns={"Close": name_close})
                    elif "Adj Close" in src.columns:
                        close_df = src[["Adj Close"]].rename(
                            columns={"Adj Close": name_close}
                        )
                    else:
                        first = str(src.columns[0])
                        close_df = src[[first]].rename(columns={first: name_close})

                if "Volume" in src.columns:
                    vol_df = src[["Volume"]].rename(columns={"Volume": tkr})
                else:
                    vol_df = pd.DataFrame()

                for fld, suf in (("Open", "_open"), ("High", "_high"), ("Low", "_low")):
                    if fld in src.columns:
                        close_df[f"{tkr}{suf}"] = pd.to_numeric(
                            src[fld], errors="coerce"
                        )

                return close_df, vol_df

            return pd.DataFrame(), pd.DataFrame()

        except Exception as e:
            if any(code in str(e) for code in ("401", "403")):
                logger.warning(
                    "Yahoo auth error (chunk, %d tickers): %s - returning empty chunk",
                    len(chunk),
                    e,
                )
                return pd.DataFrame(), pd.DataFrame()

            attempt += 1
            if attempt <= retries:
                sleep_time = pause * (backoff ** (attempt - 1))
                logger.warning(
                    "Download failed for chunk size %d (attempt %d/%d), retry in %.1fs: %s",
                    len(chunk),
                    attempt,
                    retries,
                    sleep_time,
                    e,
                )
                time.sleep(sleep_time)
            else:
                logger.error(
                    "Chunk permanently failed after %d retries. Last error: %s",
                    retries,
                    e,
                )

    return pd.DataFrame(), pd.DataFrame()


def _pause_between_chunks(pause: float, stop_event: threading.Event | None) -> None:
    try:
        pause_sec = float(pause)
    except Exception:
        pause_sec = 0.0
    if pause_sec <= 0:
        return
    if stop_event is None:
        time.sleep(pause_sec)
        return
    deadline = time.monotonic() + pause_sec
    while True:
        if stop_event.is_set():
            raise KeyboardInterrupt("Universe runner cancelled via signal.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def fetch_price_volume_data(
    tickers: list[str],
    start_date: str | dt.date | pd.Timestamp | None,
    end_date: str | dt.date | pd.Timestamp | None,
    interval: str,
    batch_size: int,
    pause: float,
    retries: int,
    backoff: float,
    use_threads: bool,
    *,
    max_workers: int | None = None,
    stop_event: threading.Event | None = None,
    auto_adjust: bool = True,
    show_progress: bool = False,
    request_timeout: float | None = None,
    junk_filter: Callable[[str], bool] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices_list: list[pd.DataFrame] = []
    volumes_list: list[pd.DataFrame] = []
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    before = len(tickers)
    if junk_filter is not None:
        tickers = [t for t in tickers if not junk_filter(t)]
    dropped = before - len(tickers)
    if dropped > 0:
        logger.info("Download-Guard entfernte Junk-Ticker: %d", dropped)

    if stop_event is not None and stop_event.is_set():
        raise KeyboardInterrupt("Universe runner cancelled via signal.")

    total = len(tickers)
    if total == 0:
        return pd.DataFrame(), pd.DataFrame()
    logger.info(
        "Starting download for %d tickers (start=%s, end=%s, interval=%s)",
        total,
        start_date,
        end_date,
        interval,
    )

    progress_bar = None
    if show_progress and tqdm is not None:
        progress_bar = tqdm(total=total, desc="Prices", unit="tick")

    def _update_progress(n: int) -> None:
        if progress_bar is not None:
            progress_bar.update(n)

    if not use_threads or total <= 1:
        for i in range(0, total, batch_size):
            if stop_event is not None and stop_event.is_set():
                raise KeyboardInterrupt("Universe runner cancelled via signal.")
            chunk = tickers[i : i + batch_size]
            res = _download_chunk(
                chunk,
                interval,
                pause,
                retries,
                backoff,
                auto_adjust=auto_adjust,
                start_date=start_date,
                end_date=end_date,
                request_timeout=request_timeout,
                junk_filter=junk_filter,
            )
            if res:
                df_close, df_vol = res
                if df_close is not None and not df_close.empty:
                    prices_list.append(_clean_panel(df_close))
                if df_vol is not None and not df_vol.empty:
                    volumes_list.append(_clean_panel(df_vol))
            _update_progress(len(chunk))
            if i + batch_size < total:
                _pause_between_chunks(pause, stop_event)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunk_indices = list(range(0, total, batch_size))
        auto_workers = min(len(chunk_indices), (os.cpu_count() or 4))
        worker_count = (
            auto_workers
            if max_workers is None
            else max(1, min(max_workers, len(chunk_indices)))
        )
        logger.info("Using %d threads for downloading in parallel.", worker_count)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_chunk: dict[Any, int] = {}
            for pos, i in enumerate(chunk_indices):
                if stop_event is not None and stop_event.is_set():
                    raise KeyboardInterrupt("Universe runner cancelled via signal.")
                fut = executor.submit(
                    _download_chunk,
                    tickers[i : i + batch_size],
                    interval,
                    pause,
                    retries,
                    backoff,
                    auto_adjust=auto_adjust,
                    start_date=start_date,
                    end_date=end_date,
                    request_timeout=request_timeout,
                    junk_filter=junk_filter,
                )
                future_to_chunk[fut] = i
                if pos < len(chunk_indices) - 1:
                    _pause_between_chunks(pause, stop_event)
            ordered_results: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
            for future in as_completed(future_to_chunk):
                if stop_event is not None and stop_event.is_set():
                    raise KeyboardInterrupt("Universe runner cancelled via signal.")
                i = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        ordered_results[i] = result
                except Exception as e:
                    logger.error(
                        "Exception in parallel download for chunk starting at %d: %s",
                        i,
                        e,
                    )
                finally:
                    _update_progress(min(batch_size, total - i))
            for i in sorted(ordered_results):
                df_close, df_vol = ordered_results[i]
                if df_close is not None and not df_close.empty:
                    prices_list.append(_clean_panel(df_close))
                if df_vol is not None and not df_vol.empty:
                    volumes_list.append(_clean_panel(df_vol))

    prices_df = (
        _clean_panel(pd.concat(prices_list, axis=1)) if prices_list else pd.DataFrame()
    )
    volumes_df = (
        _clean_panel(pd.concat(volumes_list, axis=1))
        if volumes_list
        else pd.DataFrame()
    )
    if progress_bar is not None:
        progress_bar.close()
    return prices_df, volumes_df


def _retry_missing_history(
    tickers: list[str],
    prices_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    *,
    start_date: str | dt.date | pd.Timestamp | None,
    end_date: str | dt.date | pd.Timestamp | None,
    interval: str,
    pause: float,
    retries: int,
    backoff: float,
    stop_event: threading.Event | None = None,
    auto_adjust: bool = True,
    show_progress: bool = False,
    request_timeout: float | None = None,
    junk_filter: Callable[[str], bool] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    want = {str(t).strip() for t in tickers if str(t).strip()}
    missing_prices = want - _close_ticker_set(prices_df, require_data=True)
    missing_volumes = want - _volume_ticker_set(volumes_df, require_data=True)
    missing = sorted(missing_prices | missing_volumes)
    if not missing:
        return prices_df, volumes_df, []

    logger.warning(
        "Price/volume download missing %d tickers; retrying sequentially (examples: %s)",
        len(missing),
        ", ".join(missing[:5]),
    )

    if stop_event is not None and stop_event.is_set():
        raise KeyboardInterrupt("Universe runner cancelled via signal.")

    retry_prices, retry_vols = fetch_price_volume_data(
        missing,
        start_date,
        end_date,
        interval,
        batch_size=1,
        pause=pause,
        retries=retries,
        backoff=backoff,
        use_threads=False,
        stop_event=stop_event,
        auto_adjust=auto_adjust,
        show_progress=show_progress,
        request_timeout=request_timeout,
        junk_filter=junk_filter,
    )

    if not retry_prices.empty:
        prices_df = _clean_panel(pd.concat([prices_df, retry_prices], axis=1))
    if not retry_vols.empty:
        volumes_df = _clean_panel(pd.concat([volumes_df, retry_vols], axis=1))

    remaining = sorted(
        (want - _close_ticker_set(prices_df, require_data=True))
        | (want - _volume_ticker_set(volumes_df, require_data=True))
    )
    return prices_df, volumes_df, remaining


@dataclass
class VendorConfig:
    rate_limit_per_sec: float = 0.5
    max_retries: int = 2
    base_backoff: float = 0.75
    backoff_factor: float = 2.0
    cooldown_after_rate_limit: float = 30.0
    use_internal_rate_limiter: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "VendorConfig":
        data = data or {}
        return cls(
            rate_limit_per_sec=cfg_float(
                data,
                "rate_limit_per_sec",
                0.5,
                strictly_positive=True,
                logger=logger,
                section_name="vendor",
            ),
            max_retries=cfg_int(
                data,
                "max_retries",
                2,
                min_value=0,
                logger=logger,
                section_name="vendor",
            ),
            base_backoff=cfg_float(
                data,
                "base_backoff",
                0.75,
                min_value=0.0,
                logger=logger,
                section_name="vendor",
            ),
            backoff_factor=cfg_float(
                data,
                "backoff_factor",
                2.0,
                strictly_positive=True,
                logger=logger,
                section_name="vendor",
            ),
            cooldown_after_rate_limit=cfg_float(
                data,
                "cooldown_after_rate_limit",
                30.0,
                min_value=0.0,
                logger=logger,
                section_name="vendor",
            ),
            use_internal_rate_limiter=cfg_bool(data, "use_internal_rate_limiter", True),
        )


class UniverseVendor(Protocol):
    def fetch_quote_bundle(
        self,
        symbol: str,
        *,
        include_major_holders: bool,
        request_timeout: float | None = None,
    ) -> dict[str, Any]: ...

    def download_history(
        self,
        symbols: Iterable[str] | str,
        *,
        period: str,
        interval: str,
        auto_adjust: bool,
        progress: bool = False,
        threads: bool = False,
        request_timeout: float | None = None,
    ) -> pd.DataFrame: ...

    def ticker_history(
        self,
        symbol: str,
        *,
        period: str,
        interval: str,
        auto_adjust: bool,
        request_timeout: float | None = None,
    ) -> pd.DataFrame: ...


class _RateLimiter:
    def __init__(self, rate_per_sec: float) -> None:
        self.rate = max(0.01, float(rate_per_sec))
        # Capacity must allow at least one full token, otherwise rates < 1.0
        # can never satisfy `tokens >= 1.0` and callers block forever.
        self.capacity = max(1.0, self.rate)
        self.tokens = self.capacity
        self.last = time.time()
        self._lock = threading.Lock()

    def take(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self.last
                if elapsed < 0:
                    elapsed = 0.0
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                to_sleep = (1.0 - self.tokens) / self.rate
            time.sleep(max(0.0, to_sleep))


class YFinanceVendor(UniverseVendor):
    def __init__(self, cfg: VendorConfig):
        self.cfg = cfg
        self._limiter = (
            _RateLimiter(cfg.rate_limit_per_sec)
            if cfg.use_internal_rate_limiter
            else None
        )

    @staticmethod
    def _run_with_timeout(func: Callable[[], Any], timeout: float | None) -> Any:
        return run_with_timeout(func, timeout, err_prefix="Vendor call")

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            ("429" in text) or ("too many requests" in text) or ("rate limit" in text)
        )

    def _with_retries(
        self,
        func,
        *args,
        max_retries: int | None = None,
        backoff_factor: float | None = None,
        base_backoff: float | None = None,
        **kwargs,
    ):
        last_exc: Exception | None = None
        retries = (
            self.cfg.max_retries if max_retries is None else max(0, int(max_retries))
        )
        factor = (
            self.cfg.backoff_factor if backoff_factor is None else float(backoff_factor)
        )
        base = self.cfg.base_backoff if base_backoff is None else float(base_backoff)
        for attempt in range(retries + 1):
            if self._limiter is not None:
                self._limiter.take()
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    break
                sleep = base * (factor**attempt)
                if self._is_rate_limited(exc):
                    sleep = max(sleep, float(self.cfg.cooldown_after_rate_limit))
                time.sleep(sleep)
        if last_exc:
            raise last_exc
        raise RuntimeError("Vendor call failed without exception.")

    def fetch_quote_bundle(
        self,
        symbol: str,
        *,
        include_major_holders: bool,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        # Constructing the Ticker object itself is local and should not consume
        # rate-limited request budget.
        ticker = yf.Ticker(symbol)
        try:
            fast_info_raw = self._with_retries(
                lambda: self._run_with_timeout(
                    lambda: getattr(ticker, "fast_info", None), request_timeout
                )
            )
        except Exception:
            fast_info_raw = None
        payload: dict[str, Any] = {
            "fast_info": fast_info_raw,
            "info": {},
            "major_holders": None,
        }

        def _load_info() -> dict[str, Any]:
            if hasattr(ticker, "get_info"):
                info_raw = self._run_with_timeout(
                    lambda: ticker.get_info(), request_timeout
                )
            else:
                info_raw = self._run_with_timeout(
                    lambda: ticker.info or {}, request_timeout
                )
            return info_raw if isinstance(info_raw, dict) else {}

        try:
            payload["info"] = self._with_retries(_load_info)
        except Exception:
            payload["info"] = {}
        if include_major_holders:

            def _load_holders() -> Any:
                if hasattr(ticker, "get_major_holders"):
                    return self._run_with_timeout(
                        lambda: ticker.get_major_holders(), request_timeout
                    )
                return self._run_with_timeout(
                    lambda: getattr(ticker, "major_holders", None), request_timeout
                )

            try:
                payload["major_holders"] = self._with_retries(_load_holders)
            except Exception:
                payload["major_holders"] = None
        return payload

    def download_history(
        self,
        symbols: Iterable[str] | str,
        *,
        period: str,
        interval: str,
        auto_adjust: bool,
        progress: bool = False,
        threads: bool = False,
        request_timeout: float | None = None,
    ) -> pd.DataFrame:
        symbols_arg = list(symbols) if not isinstance(symbols, str) else symbols
        kwargs: dict[str, Any] = {
            "period": period,
            "interval": interval,
            "auto_adjust": auto_adjust,
            "progress": progress,
            "threads": threads,
        }
        kwargs = _with_optional_timeout(kwargs, request_timeout)
        return self._with_retries(
            yf.download,
            symbols_arg,
            **kwargs,
        )

    def ticker_history(
        self,
        symbol: str,
        *,
        period: str,
        interval: str,
        auto_adjust: bool,
        request_timeout: float | None = None,
    ) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        kwargs: dict[str, Any] = {
            "period": period,
            "interval": interval,
            "auto_adjust": auto_adjust,
        }
        kwargs = _with_optional_timeout(kwargs, request_timeout)
        return self._with_retries(
            ticker.history,
            **kwargs,
        )
