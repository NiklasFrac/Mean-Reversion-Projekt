from __future__ import annotations

import importlib
import logging
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd

logger = logging.getLogger("perf")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel("INFO")

# Optionales Kostenmodell (Batch 4)
try:
    _cm_mod = importlib.import_module("backtest.cost.cost_model")
    cm_compute_trade_cost: Any | None = getattr(_cm_mod, "compute_trade_cost", None)
    _COSTMODEL_AVAILABLE = cm_compute_trade_cost is not None
except Exception:
    cm_compute_trade_cost = None
    _COSTMODEL_AVAILABLE = False


# ======================================================================
#                      P n L   G R U N D L A G E N
# ======================================================================


def calculate_daily_pnl(signals: pd.Series, prices: pd.Series) -> pd.Series:
    """Lineare PnL-Zerlegung: Cashflow aus Positionswechsel + MTM aus Vorposition * ΔPreis."""
    if not signals.index.equals(prices.index):
        prices = prices.reindex(signals.index)
    else:
        prices = prices.copy()
    prices = prices.ffill()
    missing = prices.isna()
    pos_prev = signals.shift(1).fillna(0)
    delta = signals.diff().fillna(signals.iloc[0])
    cash_flow = -delta * prices
    mtm = pos_prev * prices.diff().fillna(0.0)
    pnl = cash_flow + mtm
    if bool(missing.any()):
        pnl = pnl.where(~missing, 0.0)
    pnl = pnl.fillna(0.0)
    pnl.name = "pnl"
    return pnl


def calculate_pair_daily_pnl(
    signals: pd.Series, price_y: pd.Series, price_x: pd.Series
) -> pd.Series:
    """Pairs-PnL: long y / short x (Signal +1) bzw. umgekehrt (Signal −1)."""
    pnl_y = calculate_daily_pnl(signals, price_y)
    pnl_x = calculate_daily_pnl(-signals, price_x)
    total = pnl_y.add(pnl_x, fill_value=0.0)
    total.name = "pnl"
    return total


# ======================================================================
#                         H A N D E L S K O S T E N
# ======================================================================


def apply_costs(
    signals: pd.Series,
    price_y: pd.Series,
    price_x: pd.Series,
    per_trade_cost: float,
    slippage_pct: float,
    adv_t1: float | None = None,
    adv_t2: float | None = None,
    min_fee_per_lot: float = 0.0,
    adv_coeff: float = 0.12,
) -> pd.Series:
    """
    Per-Trade Kosten auf Tagesebene verbuchen (nur an Signalkanten).
    Für unit-size Trades (1 Lot); Skalierung durch Strategiegröße extern.
    """
    signals = signals.ffill().fillna(0).astype(int)
    price_y = price_y.reindex(signals.index).ffill()
    price_x = price_x.reindex(signals.index).ffill()

    trade_flags = (signals != signals.shift()).astype(bool)
    trade_flags.iloc[0] = bool(signals.iloc[0] != 0)

    costs = pd.Series(0.0, index=signals.index, name="costs")
    adv_sum = None
    if adv_t1 or adv_t2:
        adv_sum = float((adv_t1 or 0.0) + (adv_t2 or 0.0)) or None

    for idx in costs.index[trade_flags]:
        py = float(price_y.at[idx])
        px = float(price_x.at[idx])
        if not np.isfinite(py) or not np.isfinite(px):
            continue
        if _COSTMODEL_AVAILABLE and cm_compute_trade_cost is not None:
            notional = abs(py) + abs(px)
            c = cm_compute_trade_cost(
                size=1,
                price_y=py,
                price_x=px,
                per_trade_fee=float(per_trade_cost),
                base_slippage_pct=float(slippage_pct),
                notional=float(notional),
                adv=float(adv_sum) if adv_sum else None,
                adv_coeff=float(adv_coeff),
                min_fee_per_lot=float(min_fee_per_lot),
            )
        else:
            # einfacher Fallback: linear + sqrt-impact
            notional = abs(py) + abs(px)
            lin = float(slippage_pct) * notional
            impact = (
                (adv_coeff * np.sqrt(notional / adv_sum) * notional)
                if adv_sum and adv_sum > 0
                else 0.0
            )
            c = float(per_trade_cost) * 2.0 + lin + impact
        costs.at[idx] = float(c)

    return costs.fillna(0.0)


def apply_costs_with_size(
    signals: pd.Series,  # {-1,0,+1} Position über Zeit
    price_y: pd.Series,
    price_x: pd.Series,
    size_ts: pd.Series,  # Stückzahl über Zeit (0 außerhalb der Position)
    per_trade_cost: float,
    slippage_pct: float,
    adv_t1: float | None = None,
    adv_t2: float | None = None,
    min_fee_per_lot: float = 0.0,
    adv_coeff: float = 0.12,
) -> pd.Series:
    # Index-Align
    signals = signals.ffill().fillna(0).astype(int)
    price_y = price_y.reindex(signals.index).ffill().astype(float)
    price_x = price_x.reindex(signals.index).ffill().astype(float)
    size_ts = size_ts.reindex(signals.index).fillna(0).astype(int)

    # Trades passieren nur bei Kanten
    trade_flags = (signals != signals.shift()).fillna(signals != 0)

    # Größe am Trade-Tag = aktuelle Positionsgröße
    edge_size = size_ts.where(trade_flags, 0).astype(float)

    # Vektorisierte Kostenkomponenten
    py = price_y.to_numpy(dtype=float, na_value=np.nan)
    px = price_x.to_numpy(dtype=float, na_value=np.nan)
    missing = np.isnan(py) | np.isnan(px)
    py = np.nan_to_num(py, nan=0.0)
    px = np.nan_to_num(px, nan=0.0)
    N = edge_size.to_numpy(dtype=float, na_value=0.0)
    notional = (np.abs(py) + np.abs(px)) * N

    base = float(per_trade_cost) * 2.0  # fix pro Roundtrip
    base_arr = np.where(N > 0, base + min_fee_per_lot * N, 0.0)
    lin = float(slippage_pct) * notional
    adv_sum = (adv_t1 or 0.0) + (adv_t2 or 0.0)
    if adv_sum > 0:
        impact = (
            float(adv_coeff) * np.sqrt(np.maximum(notional, 0.0) / adv_sum) * notional
        )
    else:
        impact = np.zeros_like(notional)

    costs = base_arr + lin + impact
    if missing.any():
        costs = np.where(missing, 0.0, costs)
    out = pd.Series(costs, index=signals.index, name="costs")
    out[~trade_flags] = 0.0
    return out.fillna(0.0)


def apply_execution_costs(
    signals: pd.Series,
    price_y: pd.Series,
    price_x: pd.Series,
    *,
    per_trade: float,
    fee_bps: float,
    per_share_fee: float,
    min_fee: float = 0.0,
    max_fee: float = 0.0,
    size_ts: pd.Series | None = None,
) -> pd.Series:
    """
    Deterministic execution-fee accrual for non-LOB or simplified BO paths.
    Fees are recognized on entry/exit edges and capped on the roundtrip.
    """
    signals = signals.ffill().fillna(0).astype(int)
    price_y = price_y.reindex(signals.index).ffill().astype(float)
    price_x = price_x.reindex(signals.index).ffill().astype(float)
    if size_ts is None:
        size = pd.Series(1.0, index=signals.index, dtype=float)
    else:
        size = (
            pd.to_numeric(size_ts.reindex(signals.index), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

    costs = pd.Series(0.0, index=signals.index, name="costs")
    open_trade: dict[str, Any] | None = None

    def _event_fee(*, py: float, px: float, qty: float) -> float:
        notional = abs(float(qty)) * (abs(float(py)) + abs(float(px)))
        fee = 0.0
        fee -= abs(float(per_trade)) * 2.0
        fee -= abs(float(qty)) * abs(float(per_share_fee)) * 2.0
        fee -= notional * abs(float(fee_bps)) * 1e-4
        return float(fee)

    def _apply_roundtrip(
        entry_idx: Any, exit_idx: Any, entry_fee_raw: float, exit_fee_raw: float
    ) -> None:
        total = float(entry_fee_raw + exit_fee_raw)
        if min_fee > 0.0 and total < 0.0:
            total = min(total, -abs(float(min_fee)))
        if max_fee > 0.0 and total < 0.0:
            total = max(total, -abs(float(max_fee)))
        denom = float(entry_fee_raw + exit_fee_raw)
        if denom != 0.0:
            scale = float(total) / denom
            costs.at[entry_idx] = costs.at[entry_idx] + float(entry_fee_raw) * scale
            costs.at[exit_idx] = costs.at[exit_idx] + float(exit_fee_raw) * scale
        else:
            costs.at[entry_idx] = costs.at[entry_idx] + 0.0
            costs.at[exit_idx] = costs.at[exit_idx] + 0.0

    prev = 0
    for idx in signals.index:
        curr = int(signals.at[idx])
        py = float(price_y.at[idx])
        px = float(price_x.at[idx])
        qty = float(abs(size.at[idx]) if idx in size.index else 1.0)
        if not np.isfinite(py) or not np.isfinite(px):
            prev = curr
            continue

        if prev == 0 and curr != 0:
            open_trade = {
                "entry_idx": idx,
                "entry_fee": _event_fee(py=py, px=px, qty=max(qty, 1.0)),
            }
        elif prev != 0 and curr == 0:
            if open_trade is not None:
                exit_fee = _event_fee(py=py, px=px, qty=max(qty, 1.0))
                _apply_roundtrip(
                    open_trade["entry_idx"],
                    idx,
                    float(open_trade["entry_fee"]),
                    float(exit_fee),
                )
                open_trade = None
        elif prev != 0 and curr != prev:
            if open_trade is not None:
                exit_fee = _event_fee(py=py, px=px, qty=max(qty, 1.0))
                _apply_roundtrip(
                    open_trade["entry_idx"],
                    idx,
                    float(open_trade["entry_fee"]),
                    float(exit_fee),
                )
            open_trade = {
                "entry_idx": idx,
                "entry_fee": _event_fee(py=py, px=px, qty=max(qty, 1.0)),
            }
        prev = curr

    if open_trade is not None:
        entry_idx = open_trade["entry_idx"]
        costs.at[entry_idx] = costs.at[entry_idx] + float(open_trade["entry_fee"])

    return costs.fillna(0.0)


# ======================================================================
#                   B O R R O W   ( v e k t o r i s i e r t )
# ======================================================================


def _to_series_rate(rate: float | pd.Series, index: pd.Index) -> pd.Series:
    """Hilfsfunktion: float → konstante Serie; Serie wird auf Index ausgerichtet."""
    if isinstance(rate, pd.Series):
        return rate.reindex(index).ffill().bfill().astype(float)
    return pd.Series(float(rate), index=index, dtype=float)


def accrue_borrow_series(
    short_shares: pd.Series,
    price: pd.Series,
    annual_rate: float | pd.Series,
    day_basis: float = 252.0,
) -> pd.Series:
    """
    Vektorisierte tägliche Borrow-Accruals (negativ, in Cash).
    - short_shares: >0 = Anzahl leerverkaufter Stücke, 0 sonst
    - price: Preis-Zeitreihe (Index wie short_shares)
    - annual_rate: Dezimal p.a. (z.B. 0.02) oder Serie
    - day_basis: 252 (Handelstage) oder 365
    Rückgabe: Serie "borrow_cost" (≤ 0)
    """
    idx = short_shares.index
    px = price.reindex(idx).ffill().bfill().astype(float)
    q = short_shares.reindex(idx).fillna(0.0).astype(float).clip(lower=0.0)
    ar = _to_series_rate(annual_rate, idx)
    daily_rate = ar / float(day_basis)
    borrow = -(q * px * daily_rate)
    return borrow.rename("borrow_cost").fillna(0.0)


def accrue_borrow_pair(
    signals: pd.Series,
    size_ts: pd.Series,
    price_y: pd.Series,
    price_x: pd.Series,
    annual_rate_y: float | pd.Series = 0.0,
    annual_rate_x: float | pd.Series = 0.0,
    day_basis: float = 252.0,
) -> pd.Series:
    """
    Borrow-Accrual für Pairs (Signal +1 => long y / short x; Signal −1 => short y / long x).
    - signals ∈ {-1,0,+1}, size_ts = Stückzahl pro Leg (gleiche Größe je Leg)
    - annual_rate_* sind p.a. (Dezimal) oder Serien (tagesvariabel)
    """
    idx = signals.index
    s = signals.ffill().fillna(0).astype(int)
    N = size_ts.reindex(idx).fillna(0).astype(float).clip(lower=0.0)

    # Short-Lots je Tag:
    q_y_short = N.where(s < 0, 0.0)  # Signal -1: y ist Short
    q_x_short = N.where(s > 0, 0.0)  # Signal +1: x ist Short

    b_y = accrue_borrow_series(q_y_short, price_y, annual_rate_y, day_basis)
    b_x = accrue_borrow_series(q_x_short, price_x, annual_rate_x, day_basis)
    return (b_y.add(b_x, fill_value=0.0)).rename("borrow_cost").fillna(0.0)


# ======================================================================
#                D R A W D O W N S   &   P E R F O R M A N C E
# ======================================================================


def compute_drawdowns(equity: pd.Series) -> tuple[pd.Series, float, int, int]:
    """Drawdown-Serie (negativ), max DD, max DD-Dauer (Tage), Recovery-Dauer (Tage)."""
    eq = equity.astype(float)
    cummax = eq.cummax()
    drawdown = eq / cummax.replace(0, np.nan) - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    is_dd = drawdown < 0
    grp = (~is_dd).cumsum()
    max_dd_dur = int(is_dd.groupby(grp).sum().max()) if not is_dd.empty else 0

    if max_dd < 0:
        dd_end = cast(Any, drawdown.idxmin())
        peak = cast(Any, eq.loc[:dd_end].idxmax())
        post = eq.loc[eq.index > dd_end]
        rec_idx = post[post >= eq.loc[peak]].first_valid_index()
        if rec_idx is not None:
            rec_dur = int((cast(pd.Timestamp, rec_idx) - cast(pd.Timestamp, peak)).days)
        else:
            rec_dur = 0
    else:
        rec_dur = 0
    return drawdown, max_dd, max_dd_dur, rec_dur


def compute_performance(
    pnl: pd.Series,
    initial_capital: float,
    periods_per_year: float = 252.0,
    trades_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    eq = pnl.cumsum().add(initial_capital)
    eq.name = "equity"
    rets = eq.pct_change().fillna(0.0)
    mu = float(rets.mean())
    sigma = float(rets.std(ddof=1))
    eps = 1e-12
    denom = sigma if abs(sigma) > eps else (eps if sigma >= 0 else -eps)
    sharpe = float(mu / denom * np.sqrt(periods_per_year))

    # CAGR robust
    if isinstance(eq.index, pd.DatetimeIndex) and len(eq) > 1:
        days = (eq.index[-1] - eq.index[0]).days
        years = max(1, days) / 365.25
    else:
        days = max(1, len(rets) - 1)
        years = days / periods_per_year
    final = float(eq.iloc[-1]) if len(eq) else float(initial_capital)
    total_ret = (final / initial_capital) - 1.0
    cagr = (
        (final / initial_capital) ** (1.0 / years) - 1.0
        if years and years > 0
        else float("nan")
    )

    dd_series, max_dd, max_dd_dur, rec_dur = compute_drawdowns(eq)

    if trades_df is not None and not trades_df.empty:
        num_trades = int(len(trades_df))
        win_rate = float((trades_df["net_pnl"] > 0).mean())
    else:
        pnl_nonzero = pnl[pnl != 0]
        num_trades = int(len(pnl_nonzero))
        win_rate = float((pnl_nonzero > 0).mean()) if num_trades > 0 else 0.0

    return {
        "equity": eq,
        "returns": rets,
        "total_return": total_ret,
        "equity_final": final,
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "max_dd_duration": max_dd_dur,
        "recovery_duration": rec_dur,
        "win_rate": win_rate,
        "num_trades": num_trades,
        "drawdown": dd_series,
    }


def bootstrap_sharpe_ci(
    pnl: pd.Series,
    initial_capital: float,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng=None,
):
    """Nonparametrische Bootstrap-CI des Sharpe (Daily)."""
    rng = np.random.default_rng(rng)
    eq = pnl.cumsum().add(initial_capital)
    ret = eq.pct_change().fillna(0.0)
    n = len(ret)
    if n <= 1 or float(ret.std(ddof=1)) == 0.0:
        return float("nan"), float("nan"), float("nan")
    sharpes = []
    for _ in range(int(n_boot)):
        sample = rng.choice(ret.values, size=n, replace=True)
        mu = float(np.mean(sample))
        sigma = float(np.std(sample, ddof=1))
        sharpes.append(0.0 if sigma == 0 else mu / sigma * np.sqrt(252.0))
    lower = float(np.quantile(sharpes, alpha / 2))
    upper = float(np.quantile(sharpes, 1 - alpha / 2))
    return float(np.mean(sharpes)), lower, upper


# ======================================================================
#                 P & L  E X P L A I N   &   R E P O R T S
# ======================================================================


def _ensure_cost_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sichert alle benötigten Kosten-/PnL-Spalten ab und füllt fehlende Werte.
    Kostenkomponenten dürfen beliebiges Vorzeichen haben (≤0 als Kosten empfohlen).
    """
    need_float = (
        "fees",
        "slippage_cost",
        "impact_cost",
        "borrow_cost",
        "total_costs",
        "net_pnl",
    )
    out = df.copy()
    for c in need_float:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
    # total_costs konsistent belegen, falls leer/0 aber Komponenten vorhanden
    comp = (
        out["fees"] + out["slippage_cost"] + out["impact_cost"] + out["borrow_cost"]
    ).astype(float)
    if "total_costs" not in df.columns or not np.isfinite(out["total_costs"]).any():
        out["total_costs"] = comp
    else:
        # falls vorhanden, dennoch NaNs/inf beheben
        out["total_costs"] = (
            pd.to_numeric(out["total_costs"], errors="coerce")
            .fillna(comp)
            .astype(float)
        )
    # gross_pnl rekonstruieren, falls nicht vorhanden
    if "gross_pnl" not in out.columns:
        out["gross_pnl"] = out["net_pnl"].astype(float) - out["total_costs"].astype(
            float
        )
    else:
        out["gross_pnl"] = pd.to_numeric(out["gross_pnl"], errors="coerce").fillna(
            out["net_pnl"].astype(float) - out["total_costs"].astype(float)
        )
    return out


def _maybe_derive_group_cols(
    df: pd.DataFrame, by: Sequence[str] | None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Erzeugt bekannte abgeleitete Gruppierungs-Spalten on-the-fly:
      'month'   → exit_date.to_period('M').start_time
      'week'    → exit_date.to_period('W').start_time
      'quarter' → exit_date.to_period('Q').start_time
    Andere Spaltennamen werden unverändert erwartet.
    """
    if not by:
        return df, []
    out = df.copy()
    want = [str(c) for c in by]
    # exit_date (oder Fallback) inferieren
    exit_col = None
    for cand in (
        "exit_date",
        "exit",
        "close_date",
        "close_time",
        "exit_dt",
        "close_dt",
    ):
        if cand in out.columns:
            exit_col = cand
            break
    if exit_col is not None:
        dt = pd.to_datetime(out[exit_col], errors="coerce")
        if "month" in want:
            out["month"] = dt.dt.to_period("M").dt.to_timestamp()
        if "week" in want:
            out["week"] = dt.dt.to_period("W").dt.start_time
        if "quarter" in want:
            out["quarter"] = dt.dt.to_period("Q").dt.to_timestamp()
    # nur tatsächlich vorhandene Spalten zurückgeben
    final_by = [c for c in want if c in out.columns]
    return out, final_by


def _win_rate_series(s: pd.Series) -> float:
    ser = pd.to_numeric(s, errors="coerce")
    if ser.empty:
        return 0.0
    return float(ser.gt(0).mean())


def pnl_explain(
    trades_df: pd.DataFrame,
    by: Sequence[str] | None = None,
    *,
    check_consistency: bool = True,
    atol: float = 1e-6,
) -> pd.DataFrame:
    """
    Aggregiert P&L gemäß Schema: gross → fees → slippage → impact → borrow → net.
    - `by`: optionale Gruppierung (z.B. ["pair"] oder ["month","pair"]).
            Erkannt/auto-abgeleitet: "month", "week", "quarter" (aus exit_date).
    - Liefert pro Gruppe Summen und Basis-KPIs (Trades, WinRate, Avg/Std Net).
    """
    if trades_df is None or trades_df.empty:
        cols = [
            "gross_pnl",
            "fees",
            "slippage_cost",
            "impact_cost",
            "borrow_cost",
            "total_costs",
            "net_pnl",
            "NumTrades",
            "WinRate",
            "AvgNet",
            "StdNet",
        ]
        return pd.DataFrame(columns=((list(by) if by else []) + cols))

    df0 = _ensure_cost_cols(trades_df)
    df, group_cols = _maybe_derive_group_cols(df0, by)
    if not group_cols:
        # Einzeilige Gesamtübersicht
        grp = df.assign(_ALL="ALL").groupby(["_ALL"], dropna=False, observed=False)
    else:
        grp = df.groupby(group_cols, dropna=False, observed=False)

    agg = grp.agg(
        gross_pnl=("gross_pnl", "sum"),
        fees=("fees", "sum"),
        slippage_cost=("slippage_cost", "sum"),
        impact_cost=("impact_cost", "sum"),
        borrow_cost=("borrow_cost", "sum"),
        total_costs=("total_costs", "sum"),
        net_pnl=("net_pnl", "sum"),
        NumTrades=("net_pnl", "size"),
        WinRate=("net_pnl", _win_rate_series),
        AvgNet=("net_pnl", "mean"),
        StdNet=("net_pnl", "std"),
    ).reset_index()

    # Konsistenz (optional): net ≈ gross + fees + slippage + impact + borrow
    if check_consistency and not agg.empty:
        lhs = agg["net_pnl"].astype(float)
        rhs = (
            agg["gross_pnl"]
            + agg["fees"]
            + agg["slippage_cost"]
            + agg["impact_cost"]
            + agg["borrow_cost"]
        ).astype(float)
        bad = (lhs - rhs).abs() > float(atol)
        if bool(bad.any()):
            # Nur Logging; kein Raise (Reporting soll nie crashen)
            n_bad = int(bad.sum())
            logger.warning(
                "pnl_explain: %d Gruppen mit inkonsistenter Summe (Toleranz %.1e).",
                n_bad,
                atol,
            )

    return agg


def _holding_days_series(df: pd.DataFrame) -> pd.Series:
    """Holding-Period in vollen Kalendertagen (≥0), robust gegen fehlende Datumswerte."""
    try:
        entry = df.get("entry_date")
        exit_ = df.get("exit_date")
        if entry is None or exit_ is None:
            return pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)
        e0 = pd.to_datetime(entry, errors="coerce")
        e1 = pd.to_datetime(exit_, errors="coerce")
        d = (e1.dt.normalize() - e0.dt.normalize()).dt.days
        return d.clip(lower=0).astype("Int64")
    except Exception:
        return pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)


def _maker_share_series(df: pd.DataFrame) -> pd.Series:
    """Maker-Share ∈ [0,1], falls maker_fills/taker_fills vorhanden; sonst NaN."""
    if "maker_fills" in df.columns and "taker_fills" in df.columns:
        m = pd.to_numeric(df["maker_fills"], errors="coerce").fillna(0.0)
        t = pd.to_numeric(df["taker_fills"], errors="coerce").fillna(0.0)
        den = (m + t).replace(0.0, np.nan)
        return (m / den).clip(0.0, 1.0)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _pick_notional_series(df: pd.DataFrame) -> pd.Series | None:
    """
    Wählt verfügbare Notional-Spalte für Size-Buckets:
      bevorzugt 'vx_notional' → 'gross_notional' → 'abs_notional' → 'notional'.
    None, wenn keine existiert.
    """
    for c in ("vx_notional", "gross_notional", "abs_notional", "notional"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return None


def make_bucket_reports(
    trades_df: pd.DataFrame,
    *,
    date_freq: str = "M",
    size_quantiles: int = 5,
    top_n_pairs: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Erzeugt Standard-Bucket-Reports als dict:
      - "by_month": Zeitbuckets nach exit_date (freq ∈ {"W","M","Q"})
      - "by_holding": Holding-Period-Buckets (0–1 / 2–5 / 6–10 / >10 Tage)
      - "by_size_q": Size-Quantile (Q{size_quantiles}) über Notional (falls vorhanden)
      - "by_liquidity": Maker-Share-Tertile (falls maker/taker_fills vorhanden)
      - "by_pair": Paar-Aggregation (optional Top-N Filter)
    Alle Tabellen enthalten P&L-Explain-Spalten + NumTrades/WinRate/Avg/Std.
    """
    reports: dict[str, pd.DataFrame] = {}
    if trades_df is None or trades_df.empty:
        return reports

    df = _ensure_cost_cols(trades_df)

    # --- Zeit-Bucket
    try:
        e1 = pd.to_datetime(df["exit_date"], errors="coerce")
        if date_freq.upper() not in {"W", "M", "Q"}:
            date_freq = "M"
        if date_freq.upper() == "W":
            df["_bucket_month"] = e1.dt.to_period("W").dt.start_time
        elif date_freq.upper() == "Q":
            df["_bucket_month"] = e1.dt.to_period("Q").dt.to_timestamp()
        else:
            df["_bucket_month"] = e1.dt.to_period("M").dt.to_timestamp()
        reports["by_month"] = pnl_explain(df, by=["_bucket_month"])
        reports["by_month"].rename(columns={"_bucket_month": "period"}, inplace=True)
    except Exception:
        pass

    # --- Holding-Period-Bucket
    try:
        hold = _holding_days_series(df)
        df["_holding_days"] = hold
        # Bins: 0–1, 2–5, 6–10, >10
        bins = [-0.1, 1, 5, 10, np.inf]
        labels = ["0-1", "2-5", "6-10", ">10"]
        df["_holding_bucket"] = pd.cut(
            pd.to_numeric(hold.fillna(-1)), bins=bins, labels=labels
        )
        reports["by_holding"] = pnl_explain(df, by=["_holding_bucket"])
        reports["by_holding"].rename(
            columns={"_holding_bucket": "holding_days"}, inplace=True
        )
    except Exception:
        pass

    # --- Size-Quantile-Bucket
    try:
        notion = _pick_notional_series(df)
        if notion is not None and notion.notna().any():
            q = pd.qcut(
                notion.rank(method="first"),
                q=size_quantiles,
                labels=[f"Q{i + 1}" for i in range(size_quantiles)],
            )
            df["_size_q"] = q
            reports["by_size_q"] = pnl_explain(df, by=["_size_q"])
            reports["by_size_q"].rename(
                columns={"_size_q": f"size_{size_quantiles}q"}, inplace=True
            )
    except Exception:
        pass

    # --- Liquidity/Maker-Share-Bucket
    try:
        ms = _maker_share_series(df)
        if ms.notna().any():
            tertiles = pd.qcut(ms.clip(0.0, 1.0), q=3, labels=["L", "M", "H"])
            df["_maker_tertile"] = tertiles
            reports["by_liquidity"] = pnl_explain(df, by=["_maker_tertile"])
            reports["by_liquidity"].rename(
                columns={"_maker_tertile": "maker_share_tertile"}, inplace=True
            )
    except Exception:
        pass

    # --- Pair-Bucket
    try:
        if "pair" in df.columns:
            rep = pnl_explain(df, by=["pair"])
            if (
                isinstance(top_n_pairs, int)
                and top_n_pairs > 0
                and len(rep) > top_n_pairs
            ):
                # Top-N nach |net_pnl|, Rest aggregieren
                rep = rep.sort_values("net_pnl", key=lambda s: s.abs(), ascending=False)
                top = rep.head(top_n_pairs).copy()
                rest = rep.iloc[top_n_pairs:].copy()
                if not rest.empty:
                    tail = pd.DataFrame(
                        {
                            "pair": ["__REST__"],
                            "gross_pnl": [rest["gross_pnl"].sum()],
                            "fees": [rest["fees"].sum()],
                            "slippage_cost": [rest["slippage_cost"].sum()],
                            "impact_cost": [rest["impact_cost"].sum()],
                            "borrow_cost": [rest["borrow_cost"].sum()],
                            "total_costs": [rest["total_costs"].sum()],
                            "net_pnl": [rest["net_pnl"].sum()],
                            "NumTrades": [int(rest["NumTrades"].sum())],
                            "WinRate": [
                                float(
                                    np.average(
                                        rest["WinRate"], weights=rest["NumTrades"]
                                    )
                                )
                                if (rest["NumTrades"] > 0).any()
                                else 0.0
                            ],
                            "AvgNet": [rest["AvgNet"].mean()],
                            "StdNet": [rest["StdNet"].mean()],
                        }
                    )
                    rep = pd.concat([top, tail], ignore_index=True)
                reports["by_pair"] = rep
            else:
                reports["by_pair"] = rep
    except Exception:
        pass

    return reports


def default_bucket_specs(trades_df: pd.DataFrame) -> dict[str, bool]:
    """
    Gibt an, welche Standard-Buckets auf Basis der vorhandenen Spalten sinnvoll sind.
    Rein informativ; die eigentliche Erstellung übernimmt make_bucket_reports(...).
    """
    has_exit = "exit_date" in trades_df.columns
    has_hold = "entry_date" in trades_df.columns and "exit_date" in trades_df.columns
    has_liq = "maker_fills" in trades_df.columns and "taker_fills" in trades_df.columns
    has_size = any(
        c in trades_df.columns
        for c in ("vx_notional", "gross_notional", "abs_notional", "notional")
    )
    has_pair = "pair" in trades_df.columns
    return {
        "by_month": bool(has_exit),
        "by_holding": bool(has_hold),
        "by_size_q": bool(has_size),
        "by_liquidity": bool(has_liq),
        "by_pair": bool(has_pair),
    }
