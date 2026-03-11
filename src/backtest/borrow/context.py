from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

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


def _find_col(lc: dict[str, str], *cands: str) -> str | None:
    for c in cands:
        key = str(c).lower()
        if key in lc:
            return lc[key]
    return None


def _parse_per_asset_rates_any(obj: Any) -> dict[str, float]:
    """
    Parse per-asset borrow rates from inline YAML data.

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
            sym = (
                str(row.get("symbol") or row.get("ticker") or row.get("asset") or "")
                .strip()
                .upper()
            )
            if not sym:
                continue
            try:
                val = float(
                    pd.to_numeric(
                        row.get("rate_annual", row.get("rate")), errors="coerce"
                    )
                )
            except Exception:
                continue
            if np.isfinite(val) and val >= 0:
                out_list[sym] = float(val)
        return out_list

    return {}


def _parse_rate_series_by_symbol_any(obj: Any) -> dict[str, pd.Series]:
    """
    Parse per-symbol time series from inline YAML data.

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
        df.columns = [str(c) for c in df.columns]
        lc = {c.lower(): c for c in df.columns}
        date_col = _find_col(lc, "date", "day", "ts", "timestamp", "datetime")
        sym_col = _find_col(lc, "symbol", "ticker", "asset", "secid")
        rate_col = _find_col(
            lc,
            "rate_annual",
            "rate",
            "borrow_rate",
            "annual_rate",
            "borrow_rate_annual",
        )
        if not (date_col and sym_col and rate_col):
            return {}
        tmp = df[[date_col, sym_col, rate_col]].copy()
        tmp[date_col] = to_naive_day(pd.to_datetime(tmp[date_col], errors="coerce"))
        tmp[sym_col] = tmp[sym_col].astype(str).str.strip().str.upper()
        tmp[rate_col] = pd.to_numeric(tmp[rate_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col, sym_col, rate_col])
        for sym, sub in tmp.groupby(sym_col, sort=False):
            s = _norm_series(sub[date_col].tolist(), sub[rate_col].tolist())
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
                        dates.append(row.get("date", row.get("day", row.get("ts"))))
                        vals.append(row.get("rate_annual", row.get("rate")))
                    elif isinstance(row, (list, tuple)) and len(row) >= 2:
                        dates.append(row[0])
                        vals.append(row[1])
                s = _norm_series(dates, vals)
                if not s.empty:
                    out[sym] = s
                continue

        return out

    return {}


def _parse_availability_long_any(obj: Any) -> pd.DataFrame | None:
    """
    Parse availability data from inline YAML.

    Expected (preferred): List[{"date": "...", "symbol": "...", "available": 0/1}, ...]
    Also supports a DataFrame-like list of dicts with flexible column names.
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
    df.columns = [str(c) for c in df.columns]
    lc = {c.lower(): c for c in df.columns}
    date_col = _find_col(lc, "date", "day", "dt", "ts", "timestamp", "datetime")
    sym_col = _find_col(lc, "symbol", "ticker", "asset", "secid")
    av_col = _find_col(
        lc,
        "available",
        "avail",
        "is_available",
        "borrowable",
        "availability",
        "locates",
        "locates_available",
        "shares_available",
        "borrow_avail",
    )
    if not (date_col and sym_col and av_col):
        return None
    out = df[[date_col, sym_col, av_col]].copy()
    out = out.rename(columns={date_col: "date", sym_col: "symbol", av_col: "available"})
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
                    cfg_path=None,
                    lead_days=None,
                    locate_fee_bps=None,
                    availability_df=self.availability_long,
                    borrow_ctx=self,
                )
            )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    get_borrow_events = events_for_range


def build_borrow_context(cfg: Mapping[str, Any]) -> BorrowContext | None:
    """Public wrapper to construct a BorrowContext from the YAML config dict."""
    return _build_borrow_ctx_from_cfg(cfg)


def _build_borrow_ctx_from_cfg(cfg: Mapping[str, Any]) -> BorrowContext | None:
    bcfg = (cfg.get("borrow", {}) or {}) if isinstance(cfg, Mapping) else {}
    enabled = bool(bcfg.get("enabled", False))
    if not enabled:
        return None

    try:
        day_basis = int(bcfg.get("day_basis", 252))
    except Exception:
        day_basis = 252
    try:
        default_rate = float(bcfg.get("default_rate_annual", 0.0))
    except Exception:
        default_rate = 0.0

    accrual_mode = (
        str(bcfg.get("accrual_mode", "entry_notional") or "entry_notional")
        .strip()
        .lower()
    )
    day_count = str(bcfg.get("day_count", "busdays") or "busdays").strip().lower()
    include_exit_day = bool(bcfg.get("include_exit_day", False))
    try:
        min_days = int(bcfg.get("min_days", 1))
    except Exception:
        min_days = 1

    # Inline-only inputs (YAML / dicts / lists)
    per_asset_rates = _parse_per_asset_rates_any(
        bcfg.get("per_asset_rate_annual", bcfg.get("per_asset_rates"))
    )

    rate_series_by_symbol: dict[str, pd.Series] = {}
    # Preferred: long rows under borrow.rates
    rs_long = _parse_rate_series_by_symbol_any(bcfg.get("rates"))
    if rs_long:
        rate_series_by_symbol.update(rs_long)
    # Alternative: pre-grouped mapping under borrow.rate_series_by_symbol
    rs_map = _parse_rate_series_by_symbol_any(bcfg.get("rate_series_by_symbol"))
    if rs_map:
        rate_series_by_symbol.update(rs_map)

    availability_long = _parse_availability_long_any(
        bcfg.get("availability", bcfg.get("availability_long"))
    )

    return BorrowContext(
        enabled=True,
        day_basis=day_basis,
        accrual_mode=accrual_mode,
        day_count=day_count,
        include_exit_day=include_exit_day,
        min_days=max(0, min_days),
        default_rate_annual=default_rate,
        per_asset_rate_annual=per_asset_rates,
        rate_series_by_symbol=rate_series_by_symbol,
        availability_long=availability_long,
    )
