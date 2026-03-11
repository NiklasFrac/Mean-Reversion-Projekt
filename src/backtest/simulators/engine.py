from __future__ import annotations

import heapq
import importlib
import logging
from collections.abc import Mapping
from typing import Any, Callable, cast
from typing import Mapping as TMapping

import numpy as np
import pandas as pd

from backtest.reporting.perf_timer import measure_runtime

from ..calendars import apply_settlement_lag
from ..config.cfg import BacktestConfig, make_config_from_yaml
from ..config.types import BorrowCtx
from ..utils.tz import NY_TZ, coerce_ts_to_tz, to_naive_local
from . import engine_state as _state
from .engine_mtm import _compute_equity_and_stats, _map_trades_to_daily_pnl
from .engine_pipeline import (
    _build_calendar_and_window,
    _calendar_name_from_cfg,
    _collect_and_normalize_trades,
    _ensure_exit_after_entry,
    _flat_equity_stats,
    _select_eval_trades,
)
from .engine_trades import (
    _asof_price_for_ts,
    _finalize_costs_and_net,
    _finalize_trade_columns,
    _get_first_present,
    _infer_leg_payload,
    _recompute_holding_days_inplace,
    _restore_essential_columns,
)
from .engine_tz import _to_ex_tz_series, _to_ex_tz_timestamp

__all__ = [
    "backtest_portfolio",
    "backtest_portfolio_with_yaml_cfg",
    "_to_ex_tz_series",
    "_to_ex_tz_timestamp",
]

# ---------- Optional deps (risk, execution overlays)
try:  # pragma: no cover
    _rm_mod = importlib.import_module("backtest.strat.risk_manager")
    RiskManager: Any | None = getattr(_rm_mod, "RiskManager", None)
    _RM_OK = RiskManager is not None
except Exception:  # pragma: no cover
    RiskManager = None
    _RM_OK = False

try:  # pragma: no cover
    _lob_mod = importlib.import_module("backtest.simulators.lob")
    annotate_with_lob: Any | None = getattr(_lob_mod, "annotate_with_lob", None)
    _LOB_OK = annotate_with_lob is not None
except Exception:  # pragma: no cover
    annotate_with_lob = None
    _LOB_OK = False

try:  # pragma: no cover
    _light_mod = importlib.import_module("backtest.simulators.light")
    annotate_with_light: Any | None = getattr(_light_mod, "annotate_with_light", None)
    _LIGHT_OK = annotate_with_light is not None
except Exception:  # pragma: no cover
    annotate_with_light = None
    _LIGHT_OK = False

logger = logging.getLogger("backtest")
logger.propagate = True

# --- Timezone policy (single source of truth) ----------------------------------
# Backtest runs on processing-normalized prices. Exchange TZ is taken from the
# loaded price series when available, otherwise NY_TZ is used as the fallback.
_EX_TZ: str = _state._EX_TZ
_NAIVE_IS_UTC: bool = _state._NAIVE_IS_UTC


# ============================== Perf helpers ===================================
def _perf_run(name: str, fn: Callable[[], Any]) -> Any:
    res = measure_runtime(fn)
    logger.info("PERF %-22s | %6.3fs", name, float(res.runtime_sec))
    return res.value


def _apply_risk_gating(
    trades_df: pd.DataFrame,
    e0: pd.Timestamp,
    e1: pd.Timestamp,
    initial_capital: float,
    risk_cfg: dict[str, Any],
    price_data: TMapping[str, pd.Series] | None = None,
    market_data_panel: pd.DataFrame | None = None,
    adv_map: Mapping[str, float] | None = None,
    settlement_lag_bars: int = 0,
    calendar: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if trades_df.empty:
        return trades_df, {"accepted": 0, "blocked": 0}

    df = trades_df.copy()
    if "entry_date" not in df.columns or "exit_date" not in df.columns:
        logger.warning("Risk gating: missing time cols - skip")
        return trades_df, {"accepted": len(trades_df), "blocked": 0}

    if not _RM_OK or RiskManager is None:
        logger.warning("RiskManager not available -> risk gating skipped.")
        return trades_df, {"accepted": len(trades_df), "blocked": 0}

    rm_cls = cast(Any, RiskManager)
    rm = rm_cls(initial_capital, cfg=risk_cfg or {})

    short_policy = getattr(rm, "short_availability_heuristic", None)
    short_enabled = bool(getattr(short_policy, "enabled", False))
    block_on_missing = bool(getattr(short_policy, "block_on_missing", True))

    short_inputs_available = bool(adv_map)
    if not short_inputs_available and price_data:
        try:
            short_inputs_available = len(price_data) > 0
        except Exception:
            short_inputs_available = False
    if not short_inputs_available:
        price_cols = (
            "exec_entry_vwap_y",
            "exec_entry_vwap_x",
            "entry_price_y",
            "entry_price_x",
            "price_y",
            "price_x",
            "entry_price",
        )
        for col in price_cols:
            if col in df.columns and df[col].notna().any():
                short_inputs_available = True
                break
    if short_enabled and not short_inputs_available:
        short_enabled = False

    df["risk_blocked"] = False
    df["risk_reason"] = ""
    accepted_mask = pd.Series(False, index=df.index)

    def _coerce_ts(ts: pd.Timestamp) -> pd.Timestamp:
        tz0 = getattr(e0, "tz", None)
        try:
            if tz0 is None:
                return to_naive_local(ts)
            tzname = str(tz0)
            return coerce_ts_to_tz(ts, tzname)
        except Exception:
            return pd.Timestamp(to_naive_local(ts))

    def _coerce_sym(x: Any) -> str | None:
        if x is None:
            return None
        s = str(x).strip().upper()
        return s or None

    def _infer_units_and_entry_prices(
        row: pd.Series, *, entry_ts: pd.Timestamp
    ) -> tuple[
        tuple[str | None, str | None],
        tuple[float, float],
        tuple[float | None, float | None],
    ]:
        y_sym = _coerce_sym(
            row.get("y_symbol") or row.get("t1_symbol") or row.get("leg1_symbol")
        )
        x_sym = _coerce_sym(
            row.get("x_symbol") or row.get("t2_symbol") or row.get("leg2_symbol")
        )

        y_units = _get_first_present(row, ("y_units", "units_y"))
        x_units = _get_first_present(row, ("x_units", "units_x"))

        size = _get_first_present(row, ("size", "qty", "quantity", "units"))
        sig = _get_first_present(row, ("signal",))
        sig = 1.0 if sig is None else (1.0 if float(sig) >= 0 else -1.0)
        beta = _get_first_present(row, ("beta_entry", "beta"))
        beta = 1.0 if beta is None else float(beta)

        if (
            (y_units is None or x_units is None)
            and size is not None
            and np.isfinite(float(size))
            and float(size) != 0.0
        ):
            y_units = abs(float(size)) * (1.0 if sig >= 0 else -1.0)
            x_units = abs(float(size)) * (-1.0 if sig >= 0 else 1.0) * float(beta)

        py0 = _get_first_present(
            row, ("exec_entry_vwap_y", "entry_price_y", "price_y", "entry_price")
        )
        px0 = _get_first_present(
            row, ("exec_entry_vwap_x", "entry_price_x", "price_x", "entry_price")
        )

        if y_units is None and py0 is not None and np.isfinite(py0) and py0 != 0.0:
            ny = _get_first_present(row, ("notional_y",))
            if ny is not None and np.isfinite(ny):
                y_units = float(ny) / float(py0)
        if x_units is None and px0 is not None and np.isfinite(px0) and px0 != 0.0:
            nx = _get_first_present(row, ("notional_x",))
            if nx is not None and np.isfinite(nx):
                x_units = float(nx) / float(px0)

        if py0 is None and y_sym and price_data is not None:
            py0 = _asof_price_for_ts(price_data.get(y_sym), entry_ts)
        if px0 is None and x_sym and price_data is not None:
            px0 = _asof_price_for_ts(price_data.get(x_sym), entry_ts)

        y_units_f = (
            float(y_units)
            if y_units is not None and np.isfinite(float(y_units))
            else 0.0
        )
        x_units_f = (
            float(x_units)
            if x_units is not None and np.isfinite(float(x_units))
            else 0.0
        )
        py0_f = float(py0) if py0 is not None and np.isfinite(float(py0)) else None
        px0_f = float(px0) if px0 is not None and np.isfinite(float(px0)) else None
        return (y_sym, x_sym), (y_units_f, x_units_f), (py0_f, px0_f)

    # open positions inherited at e0
    pre = df[(df["entry_date"] < e0) & (df["exit_date"] >= e0)].copy()
    pre_to_close: list[
        tuple[
            pd.Timestamp,
            str,
            tuple[str | None, str | None],
            tuple[float, float],
            tuple[float, float],
            tuple[float | None, float | None],
            float,
        ]
    ] = []
    for _, r in pre.iterrows():
        (y_sym, x_sym), (ny, nx), gross = _infer_leg_payload(r)
        y_sym = _coerce_sym(y_sym)
        x_sym = _coerce_sym(x_sym)
        if gross <= 0:
            continue
        pair = str(r.get("pair", ""))
        try:
            dt_entry = _coerce_ts(pd.Timestamp(r["entry_date"]))
            rm.register_open_pair(pair, (y_sym, x_sym), (ny, nx))
        except Exception:
            rm.register_open(pair, signed_notional=0.0)
        try:
            dt_exit = _coerce_ts(pd.Timestamp(r["exit_date"]))
        except Exception:
            continue
        pnl = float(r.get("net_pnl", r.get("gross_pnl", 0.0)) or 0.0)
        _, units, entry_px = _infer_units_and_entry_prices(r, entry_ts=dt_entry)
        pre_to_close.append(
            (dt_exit, pair, (y_sym, x_sym), (ny, nx), units, entry_px, pnl)
        )

    # candidates ordered by entry
    cand = df[(df["entry_date"] >= e0) & (df["entry_date"] <= e1)].copy()
    cand = cand.sort_values(["entry_date", "exit_date"]).reset_index(drop=False)
    idx_map = cand["index"].to_numpy()

    heap_close: list[
        tuple[
            pd.Timestamp,
            int,
            str,
            tuple[str | None, str | None],
            tuple[float, float],
            tuple[float, float],
            tuple[float | None, float | None],
        ]
    ] = []
    heap_settle: list[tuple[pd.Timestamp, int, float]] = []
    heapq.heapify(heap_close)
    heapq.heapify(heap_settle)
    heap_id = 0
    realized_pnl = 0.0
    settle_lag = int(max(0, settlement_lag_bars))

    def _settle_ts(ts: pd.Timestamp) -> pd.Timestamp:
        if settle_lag <= 0 or calendar is None:
            return ts
        try:
            return apply_settlement_lag(ts, calendar, lag_bars=settle_lag)
        except Exception:
            return ts

    for dt_exit, pair, syms, notionals, units, entry_px, pnl in pre_to_close:
        heapq.heappush(
            heap_close, (dt_exit, heap_id, pair, syms, notionals, units, entry_px)
        )
        heapq.heappush(heap_settle, (_settle_ts(dt_exit), heap_id, float(pnl)))
        heap_id += 1

    def _flush_until(ts: pd.Timestamp) -> None:
        nonlocal realized_pnl
        while heap_close and heap_close[0][0] <= ts:
            dt_exit, _, p, syms, notionals, _units, _entry_px = heapq.heappop(
                heap_close
            )
            try:
                rm.register_close_pair(p, syms, notionals)
            except Exception:
                pass
        while heap_settle and heap_settle[0][0] <= ts:
            _dt_settle, _, pnl = heapq.heappop(heap_settle)
            realized_pnl += float(pnl)

    def _flush_all() -> None:
        nonlocal realized_pnl
        while heap_close:
            dt_exit, _, p, syms, notionals, _units, _entry_px = heapq.heappop(
                heap_close
            )
            try:
                rm.register_close_pair(p, syms, notionals)
            except Exception:
                pass
        while heap_settle:
            _dt_settle, _, pnl = heapq.heappop(heap_settle)
            realized_pnl += float(pnl)

    def _mtm_unrealized(ts: pd.Timestamp) -> float:
        if price_data is None:
            return 0.0
        total = 0.0
        for _dt_exit, _hid, _pair, syms, _notionals, units, entry_px in heap_close:
            y_sym, x_sym = syms
            y_units, x_units = units
            py0, px0 = entry_px

            if y_units != 0.0 and py0 is not None:
                py1 = _asof_price_for_ts(price_data.get(y_sym), ts) if y_sym else None
                if py1 is None:
                    py1 = py0
                total += float(y_units) * (float(py1) - float(py0))

            if x_units != 0.0 and px0 is not None:
                px1 = _asof_price_for_ts(price_data.get(x_sym), ts) if x_sym else None
                if px1 is None:
                    px1 = px0
                total += float(x_units) * (float(px1) - float(px0))

        return float(total)

    for _, r in cand.iterrows():
        dt_entry = _coerce_ts(pd.Timestamp(r["entry_date"]))
        dt_exit = _coerce_ts(pd.Timestamp(r["exit_date"]))
        _flush_until(dt_entry)
        rm.update_capital(initial_capital + realized_pnl + _mtm_unrealized(dt_entry))

        (y_sym, x_sym), (ny, nx), gross = _infer_leg_payload(r)
        y_sym = _coerce_sym(y_sym)
        x_sym = _coerce_sym(x_sym)
        pair = str(r.get("pair", ""))
        _, units, entry_px = _infer_units_and_entry_prices(r, entry_ts=dt_entry)

        allow = True
        reason = ""
        try:
            if short_enabled and allow:

                def _adv_asof(sym: str | None) -> float | None:
                    if sym is None or not sym:
                        return None
                    if adv_map is None:
                        return None
                    try:
                        out = float(adv_map.get(sym, float("nan")))
                        return out if np.isfinite(out) else None
                    except Exception:
                        return None

                short_reason = rm.short_availability_pair_reason(
                    leg_symbols=(y_sym, x_sym),
                    leg_notionals=(float(ny), float(nx)),
                    leg_units=(float(units[0]), float(units[1])),
                    leg_entry_prices=(entry_px[0], entry_px[1]),
                    leg_adv_usd=(_adv_asof(y_sym), _adv_asof(x_sym)),
                    block_on_missing=block_on_missing,
                )
                if short_reason:
                    allow = False
                    reason = short_reason

            if allow:
                if gross <= 0.0 and ny == 0.0 and nx == 0.0:
                    allow = True
                else:
                    if (y_sym or x_sym) or (ny != 0.0 or nx != 0.0):
                        allow = rm.can_open_pair(pair, (y_sym, x_sym), (ny, nx))
                        if allow:
                            rm.register_open_pair(pair, (y_sym, x_sym), (ny, nx))
                    else:
                        allow = rm.can_open(pair, notional=gross)
                        if allow:
                            rm.register_open(pair, signed_notional=0.0)
        except Exception as e:
            allow = True
            reason = f"rm_error:{e}"

        row_pos = cast(int, r.name)
        row_key = idx_map[row_pos]
        if allow:
            accepted_mask.at[row_key] = True
            pnl = float(r.get("net_pnl", r.get("gross_pnl", 0.0)) or 0.0)
            heapq.heappush(
                heap_close,
                (dt_exit, heap_id, pair, (y_sym, x_sym), (ny, nx), units, entry_px),
            )
            heapq.heappush(heap_settle, (_settle_ts(dt_exit), heap_id, float(pnl)))
            heap_id += 1
        else:
            df.loc[row_key, "risk_blocked"] = True
            df.loc[row_key, "risk_reason"] = reason or "cap_violation"

    _flush_all()

    df.loc[accepted_mask, "risk_blocked"] = False
    keep = (df["entry_date"] < e0) | (df["entry_date"] > e1) | accepted_mask
    out = df.loc[keep].copy()
    report = {
        "accepted": int(accepted_mask.sum()),
        "blocked": int(len(cand) - int(accepted_mask.sum())),
    }
    return out, report


# ============================== Borrow enforcement =============================
def _apply_borrow_event_enforcement(
    trades_df: pd.DataFrame,
    *,
    raw_yaml: dict[str, Any] | None,
    borrow_ctx: Any | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Apply borrow events to trades:
      recall_notice -> optional forced exit after 'recall_grace_days'
      buy_in_effective -> penalty costs (bps on gross) + optional exit clip
    Availability <= threshold synthesizes buy_in events.
    """
    raw = raw_yaml or {}
    enf = (raw.get("borrow") or {}).get("enforcement") or {}

    enabled = bool(enf.get("enabled", False))
    mode_raw = str(enf.get("mode", "penalty_only"))
    enf_mode = mode_raw.strip().lower()  # "penalty_only" | "clip_exit"
    grace = int(enf.get("recall_grace_days", 2))
    buyin_penalty_bps = float(enf.get("buyin_penalty_bps", 0.0))

    try:
        ftd_threshold = float(
            ((raw.get("borrow") or {}).get("ftd_block_threshold") or 0.0)
        )
        if not np.isfinite(ftd_threshold):
            ftd_threshold = 0.0
    except Exception:
        ftd_threshold = 0.0

    if not enabled or trades_df is None or trades_df.empty:
        return trades_df, {"changed_exits": 0, "buyin_penalties": 0}

    df = trades_df.copy()

    # ensure entry/exit are present and tz-correct
    def _to_dt(s: Any) -> pd.Series:
        s = _to_ex_tz_series(pd.to_datetime(s, errors="coerce"), _EX_TZ, _NAIVE_IS_UTC)
        return s

    if "entry_date" not in df.columns or df["entry_date"].isna().all():
        for cand in (
            "exec_entry_ts",
            "entry_ts",
            "timestamp",
            "open_dt",
            "start_dt",
            "fill_ts",
        ):
            if cand in df.columns and not df[cand].isna().all():
                df["entry_date"] = _to_dt(df[cand])
                break
    else:
        df["entry_date"] = _to_dt(df["entry_date"])

    if "exit_date" not in df.columns or df["exit_date"].isna().all():
        for cand in ("exec_exit_ts", "exit_ts", "close_dt", "end_dt", "fill_ts"):
            if cand in df.columns and not df[cand].isna().all():
                df["exit_date"] = _to_dt(df[cand])
                break
    else:
        df["exit_date"] = _to_dt(df["exit_date"])

    mask_bad = df["entry_date"].isna() | df["exit_date"].isna()
    if mask_bad.any():
        df = df.loc[~mask_bad].copy()
        if df.empty:
            return df, {"changed_exits": 0, "buyin_penalties": 0}

    # collect symbols for window
    def _symbols_from_row(row: pd.Series) -> set[str]:
        syms_row: set[str] = set()
        for a, b in (
            ("y_symbol", "x_symbol"),
            ("t1_symbol", "t2_symbol"),
            ("leg1_symbol", "leg2_symbol"),
            ("y_ticker", "x_ticker"),
            ("secid1", "secid2"),
            ("sid1", "sid2"),
            ("sym1", "sym2"),
        ):
            if a in row.index and pd.notna(row[a]):
                syms_row.add(str(row[a]).upper().strip())
            if b in row.index and pd.notna(row[b]):
                syms_row.add(str(row[b]).upper().strip())
        for k in ("symbol", "asset", "ticker", "secid", "sid"):
            if k in row.index and pd.notna(row[k]):
                syms_row.add(str(row[k]).upper().strip())
        for pk in ("pair_key", "pair", "pairid", "pair_id"):
            if pk in row.index and pd.notna(row[pk]):
                vv = str(row[pk]).upper()
                for sep in ("/", "|", "-", "_"):
                    if sep in vv:
                        a, *rest = vv.split(sep)
                        if a:
                            syms_row.add(a.strip())
                        if rest and rest[0]:
                            syms_row.add(rest[0].strip())
                        break
        return {s for s in syms_row if s and s != "NAN"}

    syms = set()
    for _, r in df.iterrows():
        syms |= _symbols_from_row(r)
    if not syms:
        return df, {"changed_exits": 0, "buyin_penalties": 0}

    # window
    e0d = pd.Timestamp(df["entry_date"].min()).normalize()
    e1d = pd.Timestamp(df["exit_date"].max()).normalize()

    # provider events
    ev = pd.DataFrame()
    try:
        if borrow_ctx is not None:
            events_fn = getattr(borrow_ctx, "events_for_range", None)
            if callable(events_fn):
                ev = events_fn(sorted(syms), e0d, e1d)
            else:
                events_fn = getattr(borrow_ctx, "get_borrow_events", None)
                if callable(events_fn):
                    ev = events_fn(sorted(syms), e0d, e1d)
            if (ev is None or ev.empty) and hasattr(borrow_ctx, "borrow_events"):
                ev0 = getattr(borrow_ctx, "borrow_events", None)
                if isinstance(ev0, pd.DataFrame) and not ev0.empty:
                    ev = ev0.copy()
    except Exception as e:
        logger.warning("borrow enforcement: events provider failed: %s", e)
        ev = pd.DataFrame()

    # availability (optional) for synthetic buy-ins
    avail = None
    try:
        if borrow_ctx is not None:
            avail = getattr(borrow_ctx, "availability_long", None)
            if avail is None:
                avail = getattr(borrow_ctx, "_availability_long", None)
    except Exception:
        avail = None

    if isinstance(avail, pd.DataFrame) and not avail.empty:
        a = avail.copy()
        lc = {c.lower(): c for c in a.columns}
        date_col = (
            lc.get("date") or lc.get("day") or lc.get("dt") or lc.get("timestamp")
        )
        sym_col = lc.get("symbol") or lc.get("ticker") or lc.get("secid")
        cand_av = [
            "available",
            "avail",
            "shares_available",
            "availability",
            "shares",
            "qty",
            "quantity",
            "locates",
            "locates_available",
            "is_available",
            "borrowable",
            "borrow_avail",
        ]
        av_col = next((lc[c] for c in cand_av if c in lc), None)
        if date_col and sym_col and av_col:
            a = a.rename(
                columns={date_col: "date", sym_col: "symbol", av_col: "available"}
            )
            a["date"] = _to_ex_tz_series(
                pd.to_datetime(a["date"], errors="coerce"), _EX_TZ, _NAIVE_IS_UTC
            ).dt.normalize()
            a = a.dropna(subset=["date", "symbol"])
            a["symbol"] = a["symbol"].astype(str).str.upper()
            a = a[(a["date"] >= e0d) & (a["date"] <= e1d) & (a["symbol"].isin(syms))]
            av_vals = pd.to_numeric(a["available"], errors="coerce")
            if av_vals.isna().all():
                av_vals = (
                    a["available"]
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
            a["available_num"] = av_vals.fillna(0.0)
            avail = a[["date", "symbol", "available_num"]].copy()
        else:
            avail = None
    else:
        avail = None

    def _norm_types(df_in: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df_in, pd.DataFrame) or df_in.empty:
            return pd.DataFrame(columns=["date", "symbol", "type"])
        d = df_in.copy()
        lc = {c.lower(): c for c in d.columns}
        if "date" not in d.columns and "day" in lc:
            d = d.rename(columns={lc["day"]: "date"})
        if "symbol" not in d.columns and "ticker" in lc:
            d = d.rename(columns={lc["ticker"]: "symbol"})
        if "type" not in d.columns and "event" in lc:
            d = d.rename(columns={lc["event"]: "type"})
        if "date" in d.columns:
            d["date"] = _to_ex_tz_series(
                pd.to_datetime(d["date"], errors="coerce"), _EX_TZ, _NAIVE_IS_UTC
            ).dt.normalize()
        d = d.dropna(subset=["date", "symbol"])
        d["symbol"] = d["symbol"].astype(str).str.upper()
        if "type" not in d.columns:
            d["type"] = ""
        d["type"] = d["type"].astype(str).str.lower().str.replace("-", "_").str.strip()

        ALIAS = {
            "buyin": "buy_in_effective",
            "buy_in": "buy_in_effective",
            "buyin_effective": "buy_in_effective",
            "buy_in_eff": "buy_in_effective",
            "no_availability": "buy_in_effective",
            "availability_zero": "buy_in_effective",
            "no_borrow": "buy_in_effective",
            "ftd": "buy_in_effective",
            "recallnotice": "recall_notice",
            "recall_notice": "recall_notice",
            "recall": "recall_notice",
        }
        d["type"] = d["type"].map(lambda t: ALIAS.get(t, t))
        for c in ("available", "available_num"):
            if c in d.columns:
                vv = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
                d.loc[vv <= max(0.0, ftd_threshold), "type"] = d.loc[
                    vv <= max(0.0, ftd_threshold), "type"
                ].replace("", "buy_in_effective")
        return d

    ev = _norm_types(ev)

    if avail is not None and not avail.empty:
        bmask = avail["available_num"] <= max(0.0, ftd_threshold)
        if bool(bmask.any()):
            ev_av = avail.loc[bmask, ["date", "symbol"]].copy()
            ev_av["type"] = "buy_in_effective"
            ev = (
                ev_av
                if (ev is None or ev.empty)
                else pd.concat([ev, ev_av], ignore_index=True).drop_duplicates(
                    subset=["date", "symbol", "type"], keep="first"
                )
            )

    if not isinstance(ev, pd.DataFrame) or ev.empty:
        return df, {"changed_exits": 0, "buyin_penalties": 0}

    ev = ev[
        (ev["date"] >= e0d) & (ev["date"] <= e1d) & (ev["symbol"].isin(syms))
    ].copy()
    if ev.empty:
        return df, {"changed_exits": 0, "buyin_penalties": 0}

    for c, default in (
        ("hard_exit", False),
        ("hard_exit_reason", ""),
        ("buyin_penalty_cost", 0.0),
    ):
        if c not in df.columns:
            df[c] = default
    df["buyin_penalty_cost"] = (
        pd.to_numeric(df["buyin_penalty_cost"], errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    orig_exit = pd.to_datetime(df["exit_date"], errors="coerce").dt.normalize().copy()

    BUYIN_NAMES = {"buy_in_effective"}
    RECALL_NAMES = {"recall_notice"}
    ev_by_sym = {s: g.sort_values("date") for s, g in ev.groupby("symbol")}

    if "_had_buyin" not in df.columns:
        df["_had_buyin"] = False

    def _gross_from_row(row: pd.Series) -> float:
        for k in ("gross_notional", "gross", "notional", "dollar_exposure"):
            try:
                v = float(row.get(k, np.nan))
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass
        for a, b in (
            ("notional_y", "notional_x"),
            ("y_notional", "x_notional"),
            ("t1_notional", "t2_notional"),
            ("leg1_notional", "leg2_notional"),
        ):
            try:
                ny = abs(float(row.get(a, 0.0)))
                nx = abs(float(row.get(b, 0.0)))
                if ny + nx > 0:
                    return ny + nx
            except Exception:
                pass
        sizes = [
            k
            for k in row.index
            if "qty" in k.lower() or "size" in k.lower() or "shares" in k.lower()
        ]
        prices = [k for k in row.index if "price" in k.lower() or "px" in k.lower()]
        tot = 0.0
        for s in sizes:
            try:
                sv = abs(float(row[s]))
                if not np.isfinite(sv) or sv <= 0:
                    continue
            except Exception:
                continue
            best_px = 0.0
            for p in prices:
                try:
                    pv = abs(float(row[p]))
                    if np.isfinite(pv) and pv > 0:
                        best_px = max(best_px, pv)
                except Exception:
                    continue
            if best_px > 0:
                tot += sv * best_px
        return float(tot)

    for idx, row in df.iterrows():
        row_idx = cast(Any, idx)
        try:
            t0 = pd.Timestamp(row["entry_date"]).normalize()
            t1 = pd.Timestamp(row["exit_date"]).normalize()
        except Exception:
            continue

        symbols = _symbols_from_row(row)
        if not symbols:
            continue

        recall_dt: pd.Timestamp | None = None
        buyin_dt: pd.Timestamp | None = None
        for s in symbols:
            g = ev_by_sym.get(s)
            if g is None or g.empty:
                continue
            gs = g[(g["date"] >= t0) & (g["date"] <= t1)]
            if gs.empty:
                continue
            types = gs["type"].astype(str).str.lower()
            recall_rows = gs.loc[types.isin(RECALL_NAMES)]
            buyin_rows = gs.loc[types.isin(BUYIN_NAMES)]
            if not recall_rows.empty:
                dt = pd.Timestamp(recall_rows["date"].iloc[0])
                recall_dt = dt if (recall_dt is None or dt < recall_dt) else recall_dt
            if not buyin_rows.empty:
                dt = pd.Timestamp(buyin_rows["date"].iloc[0])
                buyin_dt = dt if (buyin_dt is None or dt < buyin_dt) else buyin_dt

        recall_due: pd.Timestamp | None = None
        if recall_dt is not None and grace >= 0:
            recall_due = pd.Timestamp(recall_dt) + pd.Timedelta(days=int(grace))

        final_exit = t1
        final_reason = ""

        if enf_mode == "clip_exit":
            candidates: list[tuple[str, pd.Timestamp]] = []
            if buyin_dt is not None:
                candidates.append(("buy_in", buyin_dt))
            if recall_due is not None:
                candidates.append(("recall_grace", recall_due))
            if candidates:
                reason, dt = min(candidates, key=lambda x: x[1])
                if dt < final_exit:
                    final_exit = dt
                    final_reason = reason
            if final_exit < t1:
                df.at[row_idx, "exit_date"] = pd.Timestamp(final_exit)
                df.at[row_idx, "hard_exit"] = True
                df.at[row_idx, "hard_exit_reason"] = final_reason
        else:
            if recall_due is not None and recall_due < t1:
                df.at[row_idx, "hard_exit"] = True
                if not str(df.at[row_idx, "hard_exit_reason"]):
                    df.at[row_idx, "hard_exit_reason"] = "recall_grace"

        if buyin_dt is not None:
            exit_for_penalty = final_exit if enf_mode == "clip_exit" else t1
            if buyin_dt <= exit_for_penalty:
                df.at[row_idx, "_had_buyin"] = True
                if buyin_penalty_bps > 0:
                    gross = _gross_from_row(row)
                    if np.isfinite(gross) and gross > 0:
                        # Cash cost (<=0): penalty bps on gross notional.
                        penalty = -float(gross) * (float(buyin_penalty_bps) * 1e-4)
                        prev = pd.to_numeric(
                            df.at[row_idx, "buyin_penalty_cost"], errors="coerce"
                        )
                        df.at[row_idx, "buyin_penalty_cost"] = float(
                            (0.0 if pd.isna(prev) else prev) + penalty
                        )

    new_exit = pd.to_datetime(df["exit_date"], errors="coerce").dt.normalize()
    changed_exits = int((new_exit != orig_exit).sum())
    try:
        buyin_penalties = int(
            pd.Series(df["_had_buyin"]).fillna(False).astype(bool).sum()
        )
    except Exception:
        buyin_penalties = int(
            bool("_had_buyin" in df.columns) and df["_had_buyin"].sum()
        )

    return df, {
        "changed_exits": int(changed_exits),
        "buyin_penalties": int(buyin_penalties),
    }


# ============================== Engine pipeline helpers ========================
def _resolve_engine_timezone(
    _cfg: BacktestConfig, price_data: TMapping[str, pd.Series]
) -> str:
    ex_tz = NY_TZ
    for v in price_data.values():
        if isinstance(v, pd.Series) and isinstance(v.index, pd.DatetimeIndex):
            tz = getattr(v.index, "tz", None)
            if tz is not None:
                ex_tz = str(tz)
                break
    global _EX_TZ
    _EX_TZ = ex_tz
    _state._EX_TZ = ex_tz
    return ex_tz


def _apply_pre_execution_hooks(
    trades_df: pd.DataFrame,
    *,
    base_cols: pd.DataFrame,
    price_data: TMapping[str, pd.Series],
    cfg: BacktestConfig,
    calendar: pd.DatetimeIndex,
    e0: pd.Timestamp,
    borrow_ctx: BorrowCtx | Any,
    market_data_panel: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    try:
        raw_yaml = getattr(cfg, "raw_yaml", {}) or {}
        trades_df, enf_rep = _perf_run(
            "borrow.enforcement",
            lambda: _apply_borrow_event_enforcement(
                trades_df,
                raw_yaml=raw_yaml if isinstance(raw_yaml, dict) else {},
                borrow_ctx=borrow_ctx,
            ),
        )
        logger.info(
            "Borrow enforcement: changed_exits=%d, buyin_penalties=%d",
            int(enf_rep.get("changed_exits", 0)),
            int(enf_rep.get("buyin_penalties", 0)),
        )
    except Exception as e:
        logger.warning("Borrow enforcement skipped: %s", e)

    trades_df = _ensure_exit_after_entry(trades_df, calendar)
    trades_df = _recompute_holding_days_inplace(trades_df)

    for c, default in (
        ("hard_exit", False),
        ("hard_exit_reason", ""),
        ("buyin_penalty_cost", 0.0),
    ):
        if c not in trades_df.columns:
            trades_df[c] = default

    return trades_df, None


def _apply_execution_hooks(
    trades_df: pd.DataFrame,
    *,
    base_cols: pd.DataFrame,
    price_data: TMapping[str, pd.Series],
    cfg: BacktestConfig,
    calendar: pd.DatetimeIndex,
    e0: pd.Timestamp,
    market_data_panel: pd.DataFrame | None,
    adv_map: Mapping[str, float] | None,
) -> tuple[pd.DataFrame, int, dict[str, int] | None, bool]:
    exec_rejected_count = 0
    exec_reject_reasons: dict[str, int] | None = None
    raw_yaml = getattr(cfg, "raw_yaml", {}) or {}
    strict_exec_hooks = bool(
        isinstance(raw_yaml, dict) and raw_yaml.get("_bo_require_execution_hooks")
    )

    exec_lob_cfg = getattr(cfg, "exec_lob", {}) or {}
    lob_enabled = True
    try:
        if isinstance(exec_lob_cfg, dict) and exec_lob_cfg.get("enabled") is False:
            lob_enabled = False
    except Exception:
        lob_enabled = True

    exec_light_cfg = getattr(cfg, "exec_light", {}) or {}
    light_enabled = True
    try:
        if isinstance(exec_light_cfg, dict) and exec_light_cfg.get("enabled") is False:
            light_enabled = False
    except Exception:
        light_enabled = True

    def _finalize_exec_overlay(df_in: pd.DataFrame) -> pd.DataFrame:
        nonlocal exec_rejected_count, exec_reject_reasons
        df_out = _restore_essential_columns(df_in, base_cols)
        df_out = _recompute_holding_days_inplace(df_out)

        if "exec_rejected" in df_out.columns:
            try:
                rej = df_out["exec_rejected"].astype(bool)
                exec_rejected_count = int(rej.sum())
                if "exec_reject_reason" in df_out.columns:
                    try:
                        vc = (
                            df_out.loc[rej, "exec_reject_reason"]
                            .fillna("")
                            .astype(str)
                            .value_counts()
                        )
                        exec_reject_reasons = {
                            str(k): int(v) for k, v in vc.to_dict().items() if str(k)
                        }
                    except Exception:
                        exec_reject_reasons = None
                if exec_rejected_count > 0:
                    df_out = df_out.loc[~rej].copy()
                    df_out = _restore_essential_columns(df_out, base_cols)
            except Exception:
                exec_rejected_count = 0
                exec_reject_reasons = None

        return df_out

    try:
        mode_exec_val = str(getattr(cfg, "exec_mode", "")).lower()
        if (
            mode_exec_val == "lob"
            and lob_enabled
            and _LOB_OK
            and callable(annotate_with_lob)
        ):
            lob_fn = cast(Any, annotate_with_lob)
            trades_df = _perf_run(
                "exec.lob",
                lambda: lob_fn(
                    trades_df,
                    price_data,
                    cfg,
                    market_data_panel=market_data_panel,
                    adv_map=adv_map,
                    calendar=calendar,
                ),
            )
            trades_df = _finalize_exec_overlay(trades_df)

        elif mode_exec_val == "lob" and not lob_enabled:
            logger.info(
                "LOB execution disabled (execution.lob.enabled=false) -> skipping exec.lob overlay."
            )
        elif (
            mode_exec_val == "light"
            and light_enabled
            and _LIGHT_OK
            and callable(annotate_with_light)
        ):
            light_fn = cast(Any, annotate_with_light)
            trades_df = _perf_run(
                "exec.light",
                lambda: light_fn(
                    trades_df,
                    price_data,
                    cfg,
                    market_data_panel=market_data_panel,
                    adv_map=adv_map,
                    calendar=calendar,
                ),
            )
            trades_df = _finalize_exec_overlay(trades_df)
        elif mode_exec_val == "light" and not light_enabled:
            logger.info(
                "Light execution disabled (execution.light.enabled=false) -> skipping exec.light overlay."
            )

    except Exception as e:
        if strict_exec_hooks:
            raise RuntimeError(f"Execution hooks failed in strict mode: {e}") from e
        try:
            trades_df.attrs["execution_hooks_failed"] = str(e)
        except Exception:
            pass
        logger.warning("Execution hooks failed (%s) - continuing without.", e)

    overlay_enabled = (
        bool(lob_enabled)
        if str(getattr(cfg, "exec_mode", "")).lower() == "lob"
        else bool(light_enabled)
    )
    return trades_df, exec_rejected_count, exec_reject_reasons, overlay_enabled


def _finalize_costs_and_risk(
    trades_df: pd.DataFrame,
    *,
    cfg: BacktestConfig,
    calendar: pd.DatetimeIndex,
    price_data: TMapping[str, pd.Series],
    borrow_ctx: BorrowCtx | Any,
    e0: pd.Timestamp,
    e1: pd.Timestamp,
    market_data_panel: pd.DataFrame | None,
    adv_map: Mapping[str, float] | None,
) -> pd.DataFrame:
    try:
        trades_df = _perf_run(
            "pnl.finalize",
            lambda: _finalize_costs_and_net(
                trades_df,
                calendar=calendar,
                price_data=price_data,
                borrow_ctx=borrow_ctx,
            ),
        )
    except Exception as e:
        logger.warning("PnL finalization failed -> continue without: %s", e)

    if cfg.risk_enabled and isinstance(cfg.risk_cfg, dict):
        trades_df, rrep = _perf_run(
            "risk.gating",
            lambda: _apply_risk_gating(
                trades_df,
                e0=e0,
                e1=e1,
                initial_capital=float(cfg.initial_capital),
                risk_cfg=cast(dict[str, Any], cfg.risk_cfg),
                price_data=price_data,
                market_data_panel=market_data_panel,
                adv_map=adv_map,
                settlement_lag_bars=int(getattr(cfg, "settlement_lag_bars", 0) or 0),
                calendar=calendar,
            ),
        )
        logger.info(
            "Risk gating: accepted=%d blocked=%d",
            rrep.get("accepted", 0),
            rrep.get("blocked", 0),
        )

    return trades_df


# ============================== Main engine ====================================
def backtest_portfolio(
    portfolio: Mapping[str, Mapping[str, Any]],
    price_data: TMapping[str, pd.Series],
    cfg: BacktestConfig | None = None,
    *,
    borrow_ctx: BorrowCtx | Any = None,
    market_data_panel: pd.DataFrame | None = None,
    adv_map: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core backtest. Returns (stats, trades_df).

    Ordering/logic guarantees (bug fix versus new-engine regression):
    - Per-pair trades are normalized/aliased BEFORE any drops.
    - TZ coercion follows a single policy derived from cfg/env/prices.
    - Market Rules, Borrow Enforcement, Execution annotations, Risk gating are preserved.
    - Calendar mapping is robust with policy + fallbacks; settlement lag applied once.
    """
    cfg = cfg or BacktestConfig()

    # Legacy default behaviour for mapping: map exits to PRIOR session unless overridden
    if not getattr(cfg, "calendar_mapping", None):
        try:
            setattr(cfg, "calendar_mapping", "prior")
        except Exception:
            pass

    if not hasattr(cfg, "raw_yaml") or not isinstance(getattr(cfg, "raw_yaml"), dict):
        try:
            setattr(cfg, "raw_yaml", {})
        except Exception:
            pass

    mode = "conservative"

    # --- Exchange tz resolution (single source of truth) -----------------------
    ex_tz = _resolve_engine_timezone(cfg, price_data)

    logger.info(
        "ENGINE | pairs=%d | price_symbols=%d | mode=%s | ex_tz=%s",
        len(portfolio),
        len(price_data),
        mode,
        _EX_TZ,
    )
    logger.info("ENGINE | tz_policy naive_is_utc=%s", _NAIVE_IS_UTC)

    # --- Calendar & eval window ------------------------------------------------
    calendar, e0, e1 = _build_calendar_and_window(cfg, price_data, ex_tz=ex_tz)

    # --- Collect & normalize portfolio trades ---------------------------------
    frames, dropped_outside_eval, hard_exit_count, total_trades_seen = (
        _collect_and_normalize_trades(
            portfolio,
            calendar=calendar,
            e0=e0,
            e1=e1,
            price_data=price_data,
        )
    )

    if not frames:
        stats = _flat_equity_stats(calendar, cfg=cfg, mode=mode, e0=e0, e1=e1)
        return stats, pd.DataFrame()

    if dropped_outside_eval > 0:
        logger.info(
            "ENGINE | dropped %d/%d trades outside eval window [%s..%s]",
            dropped_outside_eval,
            total_trades_seen,
            str(e0),
            str(e1),
        )
    if hard_exit_count > 0:
        logger.info("ENGINE | hard_exit(window_end)=%d", hard_exit_count)

    trades_df = pd.concat(frames, ignore_index=True)
    # Ensure exit strictly after entry -> next bar fallback
    trades_df = _ensure_exit_after_entry(trades_df, calendar)

    # essential snapshot for recovery after hooks
    essentials = [
        c
        for c in [
            "entry_date",
            "exit_date",
            "pair",
            "y_symbol",
            "x_symbol",
            "symbol",
            "asset",
            "ticker",
        ]
        if c in trades_df.columns
    ]
    base_cols = trades_df[essentials].copy()

    trades_df, _ = _apply_pre_execution_hooks(
        trades_df,
        base_cols=base_cols,
        price_data=price_data,
        cfg=cfg,
        calendar=calendar,
        e0=e0,
        borrow_ctx=borrow_ctx,
        market_data_panel=market_data_panel,
    )

    # ---- Execution annotations -------------------------------------------------
    # Supported execution overlays are LOB and light. Older modes remain removed.
    trades_df, exec_rejected_count, exec_reject_reasons, exec_overlay_enabled = (
        _apply_execution_hooks(
            trades_df,
            base_cols=base_cols,
            price_data=price_data,
            cfg=cfg,
            calendar=calendar,
            e0=e0,
            market_data_panel=market_data_panel,
            adv_map=adv_map,
        )
    )

    trades_df = _finalize_costs_and_risk(
        trades_df,
        cfg=cfg,
        calendar=calendar,
        price_data=price_data,
        borrow_ctx=borrow_ctx,
        e0=e0,
        e1=e1,
        market_data_panel=market_data_panel,
        adv_map=adv_map,
    )

    # ---- Eval subset (tz-hardened) --------------------------------------------
    trades_eval = _select_eval_trades(trades_df, e0=e0, e1=e1)

    # ---- MTM PnL on calendar -------------------------------------------------
    daily_pnl, daily_gross, mapped_rows, dropped = _map_trades_to_daily_pnl(
        trades_eval,
        calendar=calendar,
        cfg=cfg,
        price_data=price_data,
        borrow_ctx=borrow_ctx,
    )

    if dropped:
        logger.info(
            "MTM mapping dropped %d/%d trades (missing prices or timestamps).",
            dropped,
            len(trades_eval),
        )

    # ---- Equity & metrics ------------------------------------------------------
    stats, fin_info = _compute_equity_and_stats(
        daily_pnl,
        daily_gross,
        calendar=calendar,
        cfg=cfg,
        trades_eval=trades_eval,
    )

    stats.attrs.update(
        {
            "EquityFinal": float(stats["equity"].iloc[-1]),
            "EquityRawEnd": float(stats["equity"].iloc[-1]),
            "Sharpe": float(fin_info.get("sharpe", 0.0)),
            "CAGR": float(fin_info.get("cagr", float("nan"))),
            "MaxDrawdown": float(fin_info.get("max_drawdown", 0.0)),
            "WinRate": float(fin_info.get("win_rate", 0.0)),
            "NumTrades": int(len(trades_eval)),
            "mode": mode,
            "eval_window_start": e0.isoformat(),
            "eval_window_end": e1.isoformat(),
            "mapped_trades": int(mapped_rows),
            "pnl_mode": "mtm",
            "mtm_dropped_trades": int(dropped),
            "calendar_name": _calendar_name_from_cfg(cfg),
            "calendar_source": "exchange_calendars",
            "exec_mode": cfg.exec_mode,
            "exec_lob_enabled": bool(cfg.exec_mode == "lob" and exec_overlay_enabled),
            "exec_light_enabled": bool(
                cfg.exec_mode == "light" and exec_overlay_enabled
            ),
            "exec_rejected_count": int(exec_rejected_count),
            "settlement_lag_bars": int(getattr(cfg, "settlement_lag_bars", 0) or 0),
        }
    )
    for attr_name in (
        "exec_entry_blocked_count",
        "exec_delayed_entry_count",
        "exec_delayed_exit_count",
        "exec_forced_exit_count",
        "exec_regime_histogram",
    ):
        if attr_name in trades_df.attrs:
            stats.attrs[attr_name] = trades_df.attrs[attr_name]
    if exec_reject_reasons is not None:
        stats.attrs["exec_reject_reasons"] = dict(exec_reject_reasons)
    trades_df = _finalize_trade_columns(trades_df)

    return stats, trades_df


# ============================== YAML wrapper & legacy ===========================
def backtest_portfolio_with_yaml_cfg(
    portfolio: Mapping[str, Mapping[str, Any]],
    price_data: TMapping[str, pd.Series],
    yaml_cfg: dict[str, Any],
    *,
    borrow_ctx: BorrowCtx | Any = None,
    market_data_panel: pd.DataFrame | None = None,
    adv_map: Mapping[str, float] | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg_obj = make_config_from_yaml(yaml_cfg)
    try:
        if not hasattr(cfg_obj, "raw_yaml") or not isinstance(
            getattr(cfg_obj, "raw_yaml"), dict
        ):
            setattr(cfg_obj, "raw_yaml", yaml_cfg)
    except Exception:
        pass

    return backtest_portfolio(
        portfolio=portfolio,
        price_data=price_data,
        cfg=cfg_obj,
        borrow_ctx=borrow_ctx,
        market_data_panel=market_data_panel,
        adv_map=adv_map,
        **kwargs,
    )
