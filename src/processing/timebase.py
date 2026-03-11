from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise index to UTC (localise naive indices), sort, dedupe, and return a
    frame with tz-naive UTC timestamps for compatibility helpers/tests.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    else:
        idx = df.index
        out = df.copy()
        if idx.tz is None:
            out.index = idx.tz_localize("UTC")
        else:
            out.index = idx.tz_convert("UTC")

    out = out.sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).median()

    idx_any = out.index
    if isinstance(idx_any, pd.DatetimeIndex):
        if idx_any.tz is None:
            out.index = idx_any.tz_localize("UTC").tz_localize(None)
        else:
            out.index = idx_any.tz_convert("UTC").tz_localize(None)
    return out


def _to_naive_normalized(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    out = idx
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _is_intraday_like(index: pd.DatetimeIndex) -> bool:
    freqstr = getattr(index, "freqstr", None)
    is_daily_freq = bool(freqstr and str(freqstr).upper().startswith(("B", "D")))
    has_time_component = ((index - index.normalize()) != pd.Timedelta(0)).any()
    is_intraday = bool(has_time_component and not is_daily_freq)
    if not is_intraday:
        return False

    try:
        # One observation per day is typically daily data whose timestamp was shifted by timezone conversion.
        if index.normalize().nunique() == len(index):
            return False
    except Exception:
        pass

    try:
        unique_times = (index - index.normalize()).unique()
        offsets: list[pd.Timedelta] = [td for td in unique_times if not pd.isna(td)]
        if len(offsets) == 1:
            td = offsets[0]
            # Treat shifted EOD-style daily bars (e.g. 19:00/20:00 after TZ conversion) as daily.
            if td < pd.Timedelta(hours=6) or td > pd.Timedelta(hours=18):
                return False
        elif offsets:
            outside_window = all(
                td < pd.Timedelta(hours=6) or td > pd.Timedelta(hours=18)
                for td in offsets
            )
            if outside_window:
                return False
    except Exception:
        pass
    return True


def ensure_ny_index(df: pd.DataFrame, *, vendor_tz: str | None = "UTC") -> pd.DataFrame:
    """
    Canonicalisiert Zeitachse auf America/New_York (tz-aware), sortiert und dedupliziert.
    - Naiver Index wird als vendor_tz (Default UTC) interpretiert.
    - Doppelte Walltimes (DST Fall-Back) -> keep='last' (Korrektur bevorzugt).
    """
    if df.empty:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index, errors="coerce")

    if getattr(idx, "tz", None) is None:
        tz = vendor_tz or "UTC"
        idx_dt = pd.DatetimeIndex(idx)
        if not _is_intraday_like(idx_dt):
            # Daily-like bars are treated as date labels. Do not shift dates through
            # UTC->NY conversion (e.g. 00:00 UTC -> previous day 19:00 NY).
            logger.warning(
                "Naive daily-like index detected; preserving calendar dates in America/New_York (vendor_tz=%s).",
                tz,
            )
            idx = idx_dt.tz_localize(
                "America/New_York",
                nonexistent="shift_forward",
                ambiguous="NaT",
            )
        else:
            logger.warning(
                "Naive timestamp index detected; interpreting as %s then converting to America/New_York.",
                tz,
            )
            idx = idx_dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
            idx = idx.tz_convert("America/New_York")
    else:
        idx = idx.tz_convert("America/New_York")

    out.index = idx
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    return out


def pick_time_grid(
    raw: pd.DataFrame,
    *,
    mode: str = "leader",
    calendar_code: str = "XNYS",
) -> pd.DatetimeIndex:
    """
    Referenz-Grid fuer Wide-Frame. Fuer Intraday bleibt die Grid-Wahl unkritisch,
    weil wir *keine* erfundenen Fills zulassen (Fuellen nur auf tradable Mask).
    """
    mode = str(mode or "leader").lower()
    if raw.empty:
        return pd.DatetimeIndex([], name=raw.index.name)

    if mode == "leader":
        non_na = raw.notna().sum(axis=0)
        ref = raw[non_na.idxmax()].dropna().index
        return pd.DatetimeIndex(ref)

    if mode in ("intersection", "union"):
        cols = [c for c in raw.columns if raw[c].dropna().size]
        if not cols:
            return pd.DatetimeIndex([])
        idxs = [pd.DatetimeIndex(raw[c].dropna().index) for c in cols]
        ref = idxs[0]
        for idx in idxs[1:]:
            ref = ref.intersection(idx) if mode == "intersection" else ref.union(idx)
        return pd.DatetimeIndex(sorted(ref.unique()))

    if mode == "calendar":
        raw_idx = pd.DatetimeIndex(raw.index)
        if _is_intraday_like(raw_idx):
            raise ValueError(
                "grid_mode=calendar is not supported for intraday data; use leader/union/intersection."
            )
        # Preserve a stable daily timestamp offset (e.g. 16:00 close) so
        # downstream reindexing aligns with the raw daily bars.
        offset = pd.Timedelta(0)
        try:
            offs = pd.Series(raw_idx - raw_idx.normalize()).dropna()
            if not offs.empty:
                mode_vals = offs.mode()
                if not mode_vals.empty:
                    offset = pd.Timedelta(mode_vals.iloc[0])
        except Exception:
            offset = pd.Timedelta(0)
        start = pd.to_datetime(raw.index.min())
        end = pd.to_datetime(raw.index.max())
        try:
            import pandas_market_calendars as mcal

            cal = mcal.get_calendar(calendar_code)
            sched = cal.schedule(start_date=start.date(), end_date=end.date())
            trade_days = pd.DatetimeIndex(sched.index)
            if trade_days.tz is None:
                trade_days = trade_days.tz_localize("America/New_York")
            else:
                trade_days = trade_days.tz_convert("America/New_York")
            return trade_days + offset
        except Exception as exc:
            # Keep the observed vendor session axis when calendar resolution fails.
            # Using bdate_range here would inject market holidays as synthetic rows.
            logger.warning(
                "Calendar grid resolution failed for %s; using observed sessions from raw index (%s).",
                calendar_code,
                exc,
            )
            observed_days = pd.DatetimeIndex(raw_idx.normalize().unique()).sort_values()
            return observed_days + offset

    non_na = raw.notna().sum(axis=0)
    return pd.DatetimeIndex(raw[non_na.idxmax()].dropna().index)


def build_tradable_mask(
    index: pd.DatetimeIndex,
    *,
    calendar_code: str = "XNYS",
    rth_only: bool = True,
) -> pd.Series:
    """
    Liefert pro Timestamp (NY tz) ein Bool (True = im zulaessigen Handelsfenster).
    Fuer Daily: True wenn Tag ein Handelstag ist.
    Fuer Intraday: True falls innerhalb (open, close) des Kalenders (RTH wenn rth_only).
    Fallback: Business-Day fuer Daily; fuer Intraday -> True (keine Erfindung durch Fuell-Policy).
    """
    if len(index) == 0:
        return pd.Series([], dtype=bool, index=index)

    is_intraday = _is_intraday_like(index)
    if not is_intraday:
        # Daily data: use explicit calendar when available so weekend/holiday rows
        # do not count toward coverage/gap statistics.
        try:
            import pandas_market_calendars as mcal

            cal = mcal.get_calendar(calendar_code)
            sched = cal.schedule(
                start_date=index.min().date(), end_date=index.max().date()
            )
            trade_days = pd.DatetimeIndex(sched.index.tz_localize("America/New_York"))
            trade_days_naive = trade_days.tz_localize(None)
            idx_norm = index.normalize()
            if idx_norm.tz is not None:
                idx_norm = idx_norm.tz_localize(None)
            return pd.Series(idx_norm.isin(trade_days_naive), index=index)
        except Exception:
            idx_norm_naive = _to_naive_normalized(index)
            bdays = pd.bdate_range(start=idx_norm_naive.min(), end=idx_norm_naive.max())
            return pd.Series(idx_norm_naive.isin(bdays), index=index)

    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar(calendar_code)
        start = index.min().date()
        end = index.max().date()
        sched = cal.schedule(start_date=start, end_date=end)
        opens = sched["market_open"].dt.tz_convert("America/New_York")
        closes = sched["market_close"].dt.tz_convert("America/New_York")
        open_norm = opens.copy()
        close_norm = closes.copy()
        if open_norm.index.tz is None:
            open_norm.index = open_norm.index.tz_localize("America/New_York")
        if close_norm.index.tz is None:
            close_norm.index = close_norm.index.tz_localize("America/New_York")
        open_norm.index = open_norm.index.tz_localize(None).normalize()
        close_norm.index = close_norm.index.tz_localize(None).normalize()

        mask_vals: list[bool] = []
        for ts in index:
            date_key = (
                ts.tz_localize(None).normalize()
                if ts.tz is not None
                else ts.normalize()
            )
            if date_key not in open_norm.index:
                mask_vals.append(False)
                continue
            o = open_norm.loc[date_key]
            c = close_norm.loc[date_key]
            if pd.isna(o) or pd.isna(c):
                mask_vals.append(False)
                continue
            mask_vals.append(bool((ts >= o) and (ts <= c)))
        mask = pd.Series(mask_vals, index=index)
        if rth_only:
            return mask
        trade_days = pd.DatetimeIndex(sched.index.tz_localize("America/New_York"))
        return pd.Series(index.normalize().isin(trade_days), index=index)

    except Exception:
        if not is_intraday:
            idx_norm_naive = _to_naive_normalized(index)
            bdays = pd.bdate_range(start=idx_norm_naive.min(), end=idx_norm_naive.max())
            return pd.Series(idx_norm_naive.isin(bdays), index=index)
        return pd.Series(True, index=index)
