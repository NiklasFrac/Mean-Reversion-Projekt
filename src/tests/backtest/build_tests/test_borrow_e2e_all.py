# src/backtest/src/test_borrow_e2e_all.py
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import backtest_engine as be
import numpy as np
import pandas as pd

# ----------------------------- Hilfs-Typen/Kontext -----------------------------


@dataclass
class _BorrowCtx:
    day_basis: int = 252
    default_rate_annual: float = 0.01
    per_asset_rate_annual: Dict[str, float] = None
    rate_series_by_symbol: Dict[str, pd.Series] = None

    def __post_init__(self):
        if self.per_asset_rate_annual is None:
            self.per_asset_rate_annual = {}
        if self.rate_series_by_symbol is None:
            self.rate_series_by_symbol = {}


def _fake_resolve_borrow_rate(
    symbol: str, date: pd.Timestamp, ctx: _BorrowCtx
) -> float:
    """
    Priorität wie verlangt: Serie > per_asset > default.
    """
    sym = str(symbol).upper()
    d = pd.Timestamp(date).floor("D")
    s = (ctx.rate_series_by_symbol or {}).get(sym)
    if isinstance(s, pd.Series):
        try:
            v = s.loc[d]
            if pd.notna(v):
                return float(v)
        except KeyError:
            pass
    pa = (ctx.per_asset_rate_annual or {}).get(sym)
    if pa is not None:
        return float(pa)
    return float(ctx.default_rate_annual or 0.0)


# ----------------------------- Datenaufbau -------------------------------------


def _build_prices(dates: pd.DatetimeIndex) -> Mapping[str, pd.Series]:
    """
    Einfache Close-Serien für vier Symbole:
      X, Y: konstant
      Z: konstant 120
      W: linear steigend
    """
    X = pd.Series(50.0, index=dates, name="X")
    Y = pd.Series(100.0, index=dates, name="Y")
    Z = pd.Series(120.0, index=dates, name="Z")
    W = pd.Series(80.0 + np.arange(len(dates)), index=dates, name="W")
    return {"X": X, "Y": Y, "Z": Z, "W": W}


def _mk_trade(
    pair: str,
    y_sym: str,
    x_sym: str,
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
    y_units: int,
    x_units: int,
    net_pnl: float = 0.0,
    gross_pnl: float = 0.0,
) -> Dict[str, Any]:
    """
    Ein Trade-Dict im Format, das die Engine akzeptiert (Engine normalisiert selbst).
    Wir geben direkt y/x_units (signed) mit, damit Short-Legs zuverlässig erkennbar sind.
    """
    return {
        "pair": pair,
        "y_symbol": y_sym,
        "x_symbol": x_sym,
        "entry_date": pd.Timestamp(entry),
        "exit_date": pd.Timestamp(exit_),
        "y_units": int(y_units),
        "x_units": int(x_units),
        "net_pnl": float(net_pnl),
        "gross_pnl": float(gross_pnl),
    }


def _engine_calendar(
    price_map: Mapping[str, pd.Series], cfg_obj: be.BacktestConfig
) -> pd.DatetimeIndex:
    """
    Baue denselben Kalender/Eval-Fenster wie die Engine ihn intern verwendet.
    """
    raw_yaml = getattr(cfg_obj, "raw_yaml", {}) or {}
    data_cfg = raw_yaml.get("data", {}) if isinstance(raw_yaml, Mapping) else {}
    cal = be.build_trading_calendar(
        price_map, calendar_name=data_cfg.get("calendar_name", "XNYS")
    )
    e0, e1 = be._resolve_eval_window(cal, cfg_obj)  # type: ignore[attr-defined]
    cal = cal[(cal >= e0) & (cal <= e1)]
    return cal


# ----------------------------- Erwartungsrechner --------------------------------


def _close_on_or_before(
    price_map: Mapping[str, pd.Series], sym: str, d: pd.Timestamp
) -> float | None:
    s = price_map.get(sym)
    if not isinstance(s, pd.Series) or s.empty:
        return None
    s2 = s.loc[: pd.Timestamp(d)]
    if s2.empty:
        return None
    try:
        return float(s2.iloc[-1])
    except Exception:
        return None


def _expected_borrow_for_trade(
    row: pd.Series,
    cal: pd.DatetimeIndex,
    price_map: Mapping[str, pd.Series],
    ctx: _BorrowCtx,
) -> float:
    """
    Exakte Replikation der Engine-Konvention:
      - Halte-Tage = (entry, exit] ∩ cal
      - Kosten nur für Short-Legs (y oder x)
      - Tageskosten = -|units| * close_price(d) * (rate_annual(d) / day_basis)
      - close_price(d) = letzter bekannte Close ≤ d
    """
    try:
        t0 = pd.Timestamp(row["entry_date"]).floor("D")
        t1 = pd.Timestamp(row["exit_date"]).floor("D")
    except Exception:
        return 0.0
    if not (pd.notna(t0) and pd.notna(t1)) or t1 <= t0:
        return 0.0

    hold_days = cal[(cal > t0) & (cal <= t1)]
    if hold_days.empty:
        return 0.0

    total = 0.0

    def _leg(sym_col: str, units_col: str) -> Tuple[str | None, int]:
        sym = row.get(sym_col)
        sym = str(sym).upper() if isinstance(sym, str) and sym.strip() else None
        try:
            u = int(round(float(row.get(units_col, 0))))
        except Exception:
            u = 0
        return sym, u

    # y-Leg
    y_sym, y_u = _leg("y_symbol", "y_units")
    if y_sym and y_u < 0:
        for d in hold_days:
            px = _close_on_or_before(price_map, y_sym, d)
            if px is None or px <= 0:
                continue
            r = _fake_resolve_borrow_rate(y_sym, d, ctx)
            if r and r > 0:
                total += (
                    -abs(y_u)
                    * float(px)
                    * (float(r) / float(max(1, int(ctx.day_basis))))
                )

    # x-Leg
    x_sym, x_u = _leg("x_symbol", "x_units")
    if x_sym and x_u < 0:
        for d in hold_days:
            px = _close_on_or_before(price_map, x_sym, d)
            if px is None or px <= 0:
                continue
            r = _fake_resolve_borrow_rate(x_sym, d, ctx)
            if r and r > 0:
                total += (
                    -abs(x_u)
                    * float(px)
                    * (float(r) / float(max(1, int(ctx.day_basis))))
                )

    return float(total)


# ----------------------------- Sortierschlüssel ---------------------------------


def _row_key_tuple(r: pd.Series) -> Tuple[str, str, str, int, int]:
    p = str(r.get("pair", "")).strip()
    e = pd.Timestamp(r.get("entry_date")).isoformat()
    x = pd.Timestamp(r.get("exit_date")).isoformat()
    yu = int(round(float(r.get("y_units", 0)))) if pd.notna(r.get("y_units")) else 0
    xu = int(round(float(r.get("x_units", 0)))) if pd.notna(r.get("x_units")) else 0
    return (p, e, x, yu, xu)


def _add_key_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["_key"] = d.apply(_row_key_tuple, axis=1)
    return d.sort_values("_key").reset_index(drop=True)


# ----------------------------- Der große Test -----------------------------------


def test_borrow_integrity_all_paths():
    # 0) Engine-Borrow-Resolver durch Fake-Resolver ersetzen (Serie > per_asset > default)
    orig_resolve = getattr(be, "resolve_borrow_rate", None)
    be.resolve_borrow_rate = _fake_resolve_borrow_rate  # type: ignore[assignment]

    try:
        # 1) Daten & Borrow-Kontext
        dates = pd.bdate_range("2024-01-02", periods=7)  # D1..D7
        prices = _build_prices(dates)

        # Serie: X hat 20% an D3..D5 (sonst per-Asset/Default)
        ser_X = pd.Series(np.nan, index=dates)
        ser_X.loc[dates[2]] = 0.20
        ser_X.loc[dates[3]] = 0.20
        ser_X.loc[dates[4]] = 0.20

        ctx = _BorrowCtx(
            day_basis=252,
            default_rate_annual=0.01,
            per_asset_rate_annual={"X": 0.05, "W": 0.10},
            rate_series_by_symbol={"X": ser_X},
        )

        # 2) Portfolio (ohne preset-borrow)
        # A: Y/X, Short X, D2→D5, net=0.0
        A = _mk_trade("Y/X", "Y", "X", dates[1], dates[4], +10, -10, net_pnl=0.0)
        # B: Z/W, Short W, D3→D5, net=10.0
        B = _mk_trade("Z/W", "Z", "W", dates[2], dates[4], +5, -5, net_pnl=10.0)
        # C: Y/X, beide long (kein Borrow), D2→D3, net=0.0
        C = _mk_trade("Y/X", "Y", "X", dates[1], dates[2], +3, +3, net_pnl=0.0)

        port_no_bc = {
            "Y/X": {"trades": pd.DataFrame([A, C])},
            "Z/W": {"trades": pd.DataFrame([B])},
        }

        # 3) YAML-Builder
        def _mk_yaml(run_mode: str, exec_mode: str):
            y = {
                "data": {"calendar_name": "XNYS"},
                "backtest": {
                    "initial_capital": 100_000.0,
                    "run_mode": run_mode,
                    "calendar_mapping": "strict",
                    "settlement_lag_bars": 1,
                    "oos_years": 100,  # sicher für 'aktuell'
                },
                "execution": {"mode": exec_mode},
                "risk": {"enabled": False},
            }
            if run_mode == "konservativ":
                y["backtest"]["splits"] = {
                    "test": {
                        "start": str(dates[0].date()),
                        "end": str(dates[-1].date()),
                    }
                }
            return y

        # 4) Alle 4 Kombis prüfen
        combos = [
            ("aktuell", "vectorized"),
            ("aktuell", "lob"),
            ("konservativ", "vectorized"),
            ("konservativ", "lob"),
        ]

        for run_mode, exec_mode in combos:
            yaml_cfg = _mk_yaml(run_mode, exec_mode)
            cfg_obj = be.make_config_from_yaml(yaml_cfg)
            cal = _engine_calendar(prices, cfg_obj)

            # ---- Run 1: Engine berechnet borrow_cost selbst ----
            stats1, trades1 = be.backtest_portfolio_with_yaml_cfg(
                portfolio=port_no_bc,
                price_data=prices,
                yaml_cfg=yaml_cfg,
                borrow_ctx=ctx,
            )

            # Erwartung je Trade – WICHTIG: in derselben Sortierung wie Engine vergleichen
            raw = pd.concat(
                [port_no_bc["Y/X"]["trades"], port_no_bc["Z/W"]["trades"]],
                ignore_index=True,
            )
            t1s = _add_key_and_sort(trades1)
            raws = _add_key_and_sort(raw)

            exp_costs_sorted = []
            exp_net_sorted = []
            for _, r in raws.iterrows():
                exp_bc = _expected_borrow_for_trade(r, cal, prices, ctx)
                base_net = float(r.get("net_pnl", 0.0))
                exp_costs_sorted.append(float(exp_bc))
                exp_net_sorted.append(float(base_net) - float(exp_bc))

            # (i) borrow_cost passt exakt (Numerik: sehr eng)
            for i in range(len(raws)):
                got = float(t1s.loc[i, "borrow_cost"])
                exp = exp_costs_sorted[i]
                assert math.isclose(got, exp, rel_tol=1e-12, abs_tol=1e-12)

            # (ii) net_pnl = (vorheriges net_pnl) - borrow_cost  (da net_pnl im Input gesetzt war)
            for i in range(len(raws)):
                got_net = float(t1s.loc[i, "net_pnl"])
                exp_net = exp_net_sorted[i]
                assert math.isclose(got_net, exp_net, rel_tol=1e-12, abs_tol=1e-12)

            # ---- Run 2: preset borrow_cost (Engine darf NICHT neu berechnen) ----
            # preset = exakt unsere Erwartung aus oben
            port_with_bc = {
                "Y/X": {"trades": raws[raws["pair"].eq("Y/X")].copy()},
                "Z/W": {"trades": raws[raws["pair"].eq("Z/W")].copy()},
            }
            # Achtung: raws enthält bereits _key – nicht an Engine weitergeben
            for k in list(port_with_bc.keys()):
                df = (
                    port_with_bc[k]["trades"]
                    .drop(
                        columns=[
                            c
                            for c in ["_key"]
                            if c in port_with_bc[k]["trades"].columns
                        ]
                    )
                    .copy()
                )
                port_with_bc[k]["trades"] = df

            # Die erwarteten borrow_costs in derselben Reihenfolge in einen Vektor packen
            # und nach dem Merge wieder per Sort-Key vergleichen.
            preset_full = pd.concat(
                [port_with_bc["Y/X"]["trades"], port_with_bc["Z/W"]["trades"]],
                ignore_index=True,
            )
            preset_full = _add_key_and_sort(preset_full)
            preset_full["borrow_cost"] = pd.Series(
                exp_costs_sorted, index=preset_full.index, dtype=float
            )

            port_with_bc["Y/X"]["trades"] = preset_full[
                preset_full["pair"].eq("Y/X")
            ].copy()
            port_with_bc["Z/W"]["trades"] = preset_full[
                preset_full["pair"].eq("Z/W")
            ].copy()

            stats2, trades2 = be.backtest_portfolio_with_yaml_cfg(
                portfolio=port_with_bc,
                price_data=prices,
                yaml_cfg=yaml_cfg,
                borrow_ctx=ctx,
            )

            t2s = _add_key_and_sort(trades2)

            # (iii) Engine respektiert preset borrow_cost und subtrahiert korrekt vom net_pnl
            for i in range(len(preset_full)):
                got_bc = float(t2s.loc[i, "borrow_cost"])
                exp_bc = float(preset_full.loc[i, "borrow_cost"])
                assert math.isclose(got_bc, exp_bc, rel_tol=1e-12, abs_tol=1e-12)

                base_net = float(preset_full.loc[i, "net_pnl"])
                exp_net2 = base_net - exp_bc
                got_net2 = float(t2s.loc[i, "net_pnl"])
                assert math.isclose(got_net2, exp_net2, rel_tol=1e-12, abs_tol=1e-12)

    finally:
        # Resolver zurücksetzen
        if orig_resolve is not None:
            be.resolve_borrow_rate = orig_resolve  # type: ignore[assignment]
