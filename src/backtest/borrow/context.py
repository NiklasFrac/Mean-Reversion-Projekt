from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from backtest.config.types import BorrowAvailabilityPoint, BorrowConfig, BorrowRatePoint
from backtest.utils.tz import to_naive_day

logger = logging.getLogger("backtest.borrow")


def _coerce_date(x: Any) -> pd.Timestamp | None:
    try:
        t = pd.Timestamp(x)
    except Exception:
        return None
    if pd.isna(t):
        return None
    return to_naive_day(t)


def _parse_per_asset_rates_any(obj: Any) -> dict[str, float]:
    """
    Parse canonical per-asset borrow rates from config data.

    Supported:
      - Mapping: {SYMBOL: rate_annual, ...}
      - List[Mapping]: [{"symbol": "...", "rate_annual": ...}, ...]
    """
    if obj is None:
        return {}

    if isinstance(obj, Mapping):
        out: dict[str, float] = {}
        for k, v in obj.items():
            sym = str(k or "").strip().upper()
            if not sym:
                continue
            try:
                val = float(pd.to_numeric(v, errors="coerce"))
            except Exception:
                continue
            if np.isfinite(val) and val >= 0:
                out[sym] = float(val)
        return out

    if isinstance(obj, list):
        out_list: dict[str, float] = {}
        for row in obj:
            if not isinstance(row, Mapping):
                continue
            sym = str(row.get("symbol") or "").strip().upper()
            if not sym:
                continue
            try:
                val = float(pd.to_numeric(row.get("rate_annual"), errors="coerce"))
            except Exception:
                continue
            if np.isfinite(val) and val >= 0:
                out_list[sym] = float(val)
        return out_list

    return {}


def _parse_rate_series_by_symbol_any(obj: Any) -> dict[str, pd.Series]:
    """
    Parse canonical per-symbol borrow rate series from config data.

    Supported:
      - Mapping: {SYMBOL: {date: rate, ...}, ...}
      - Mapping: {SYMBOL: [{"date": "...", "rate_annual": ...}, ...], ...}
      - List[Mapping] (long): [{"date": "...", "symbol": "...", "rate_annual": ...}, ...]
    """
    if obj is None:
        return {}

    def _norm_series(dates: list[Any], vals: list[Any]) -> pd.Series:
        idx = to_naive_day(pd.to_datetime(pd.Series(dates), errors="coerce"))
        v = pd.to_numeric(pd.Series(vals), errors="coerce")
        mask = (~idx.isna()) & (~v.isna())
        s = pd.Series(
            v[mask].to_numpy(dtype=float), index=pd.DatetimeIndex(idx[mask])
        ).sort_index()
        return s.groupby(level=0).last()

    out: dict[str, pd.Series] = {}

    # long list: [{date, symbol, rate_annual}, ...]
    if isinstance(obj, list):
        rows = [r for r in obj if isinstance(r, Mapping)]
        if not rows:
            return {}
        df = pd.DataFrame(rows)
        if df.empty:
            return {}
        if not {"date", "symbol", "rate_annual"}.issubset(df.columns):
            return {}
        tmp = df[["date", "symbol", "rate_annual"]].copy()
        tmp["date"] = to_naive_day(pd.to_datetime(tmp["date"], errors="coerce"))
        tmp["symbol"] = tmp["symbol"].astype(str).str.strip().str.upper()
        tmp["rate_annual"] = pd.to_numeric(tmp["rate_annual"], errors="coerce")
        tmp = tmp.dropna(subset=["date", "symbol", "rate_annual"])
        for sym, sub in tmp.groupby("symbol", sort=False):
            s = _norm_series(sub["date"].tolist(), sub["rate_annual"].tolist())
            if not s.empty:
                out[str(sym)] = s
        return out

    # mapping: {SYMBOL -> ...}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            sym = str(k or "").strip().upper()
            if not sym:
                continue

            if isinstance(v, Mapping):
                # {date -> rate}
                s = _norm_series(list(v.keys()), list(v.values()))
                if not s.empty:
                    out[sym] = s
                continue

            if isinstance(v, list):
                dates: list[Any] = []
                vals: list[Any] = []
                for row in v:
                    if isinstance(row, Mapping):
                        dates.append(row.get("date"))
                        vals.append(row.get("rate_annual"))
                s = _norm_series(dates, vals)
                if not s.empty:
                    out[sym] = s
                continue

        return out

    return {}


def _parse_availability_long_any(obj: Any) -> pd.DataFrame | None:
    """
    Parse canonical availability data from config data.

    Expected: List[{"date": "...", "symbol": "...", "available": 0/1}, ...]
    """
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, list):
        rows = [r for r in obj if isinstance(r, Mapping)]
        if not rows:
            return None
        df = pd.DataFrame(rows)
    else:
        return None
    if df.empty:
        return None
    if not {"date", "symbol", "available"}.issubset(df.columns):
        return None
    out = df[["date", "symbol", "available"]].copy()
    out["date"] = to_naive_day(pd.to_datetime(out["date"], errors="coerce"))
    out = out.dropna(subset=["date", "symbol"])
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    av_vals = pd.to_numeric(out["available"], errors="coerce")
    if av_vals.isna().all():
        av_vals = (
            out["available"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": 1,
                    "t": 1,
                    "yes": 1,
                    "y": 1,
                    "1": 1,
                    "false": 0,
                    "f": 0,
                    "no": 0,
                    "n": 0,
                    "0": 0,
                }
            )
            .fillna(0)
        )
    out["available"] = av_vals.fillna(0.0).astype(float)
    return out[["date", "symbol", "available"]]


@dataclass(slots=True)
class BorrowContext:
    enabled: bool
    day_basis: int = 252
    # Accrual semantics (paper-friendly defaults can be set in YAML)
    # - accrual_mode:
    #     * "entry_notional": borrow on entry notional × n_days (legacy/default)
    #     * "mtm_daily": sum daily borrow on mark-to-market notional using price_data (if available)
    # - day_count:
    #     * "busdays": numpy busdays (legacy/default)
    #     * "calendar_days": every calendar day (weekends included; price asof last close)
    #     * "sessions": exchange sessions from engine calendar
    # - include_exit_day: if true, accrues borrow for the exit day as well (conservative for daily bars)
    accrual_mode: str = "entry_notional"
    day_count: str = "busdays"
    include_exit_day: bool = False
    min_days: int = 1
    default_rate_annual: float = 0.0
    per_asset_rate_annual: dict[str, float] = field(default_factory=dict)
    rate_series_by_symbol: dict[str, pd.Series] = field(default_factory=dict)
    availability_long: pd.DataFrame | None = None

    def resolve_borrow_rate(self, symbol: str, day: Any) -> float:
        if not self.enabled:
            return 0.0
        sym = str(symbol or "").strip().upper()
        d = _coerce_date(day)
        if not sym or d is None:
            return float(self.default_rate_annual)

        explicit = False
        s = self.rate_series_by_symbol.get(sym)
        if isinstance(s, pd.Series) and not s.empty:
            try:
                # asof: last observation on/before day
                idx = pd.DatetimeIndex(pd.to_datetime(s.index, errors="coerce"))
                idx = to_naive_day(idx)
                s2 = (
                    pd.Series(pd.to_numeric(s.to_numpy(), errors="coerce"), index=idx)
                    .dropna()
                    .sort_index()
                )
                if not s2.empty:
                    pos = s2.index.searchsorted(d, side="right") - 1
                    if pos >= 0:
                        val = float(s2.iloc[pos])
                        if np.isfinite(val) and val >= 0:
                            explicit = True
                            base_rate = val
                        else:
                            base_rate = float(self.default_rate_annual)
                    else:
                        base_rate = float(self.default_rate_annual)
                else:
                    base_rate = float(self.default_rate_annual)
            except Exception:
                base_rate = float(self.default_rate_annual)
        else:
            base_rate = float(self.default_rate_annual)

        if (not explicit) and sym in self.per_asset_rate_annual:
            try:
                val = float(self.per_asset_rate_annual[sym])
                if np.isfinite(val) and val >= 0:
                    explicit = True
                    base_rate = val
            except Exception:
                pass

        return float(base_rate)

    def events_for_range(
        self, symbols: list[str] | tuple[str, ...], start: Any, end: Any
    ) -> pd.DataFrame:
        from backtest.borrow.events import generate_borrow_events

        if not self.enabled:
            return pd.DataFrame(
                columns=[
                    "date",
                    "symbol",
                    "type",
                    "rate_annual",
                    "locate_fee_bps",
                    "lead_days",
                    "notes",
                ]
            )
        d0 = _coerce_date(start)
        d1 = _coerce_date(end)
        if d0 is None or d1 is None:
            return pd.DataFrame(
                columns=[
                    "date",
                    "symbol",
                    "type",
                    "rate_annual",
                    "locate_fee_bps",
                    "lead_days",
                    "notes",
                ]
            )
        syms = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
        frames: list[pd.DataFrame] = []
        for d in pd.date_range(d0, d1, freq="D"):
            frames.append(
                generate_borrow_events(
                    universe=syms,
                    day=d,
                    borrow_cfg=None,
                    lead_days=None,
                    locate_fee_bps=None,
                    availability_df=self.availability_long,
                    borrow_ctx=self,
                )
            )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    get_borrow_events = events_for_range


def _borrow_rate_rows(rows: tuple[BorrowRatePoint, ...]) -> list[dict[str, Any]]:
    return [
        {"date": row.date, "symbol": row.symbol, "rate_annual": row.rate_annual}
        for row in rows
    ]


def _borrow_availability_rows(
    rows: tuple[BorrowAvailabilityPoint, ...],
) -> list[dict[str, Any]]:
    return [
        {"date": row.date, "symbol": row.symbol, "available": row.available}
        for row in rows
    ]


def build_borrow_context(cfg: BorrowConfig) -> BorrowContext | None:
    if not isinstance(cfg, BorrowConfig):
        raise TypeError("cfg must be a BorrowConfig")
    if not cfg.enabled:
        return None

    per_asset_rates = _parse_per_asset_rates_any(cfg.per_asset_rate_annual)
    rate_series_by_symbol = _parse_rate_series_by_symbol_any(_borrow_rate_rows(cfg.rates))
    availability_long = _parse_availability_long_any(
        _borrow_availability_rows(cfg.availability)
    )

    return BorrowContext(
        enabled=True,
        day_basis=int(cfg.day_basis),
        accrual_mode=str(cfg.accrual_mode),
        day_count=str(cfg.day_count),
        include_exit_day=bool(cfg.include_exit_day),
        min_days=max(0, int(cfg.min_days)),
        default_rate_annual=float(cfg.default_rate_annual),
        per_asset_rate_annual=per_asset_rates,
        rate_series_by_symbol=rate_series_by_symbol,
        availability_long=availability_long,
    )
