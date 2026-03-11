# src/backtest/costs.py
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, TypedDict, TypeGuard, cast

import numpy as np
import pandas as pd

from backtest.config.execution_costs import resolve_execution_cost_spec

# Für Borrow-Hilfen & Units/Sides (nur intern in diesem Modul genutzt)
try:
    from .common import infer_side as _side_for_leg
    from .common import infer_units as _infer_units_for_leg
except Exception:  # pragma: no cover

    def _infer_units_for_leg(row: pd.Series, leg: str) -> int:
        try:
            # Fallback: übliche Spaltennamen; negatives Vorzeichen => short
            leg = leg.lower()
            for k in (
                f"units_{leg}",
                f"qty_{leg}",
                f"size_{leg}",
                f"{leg}_units",
                f"{leg}_qty",
            ):
                if k in row and pd.notna(row[k]):
                    return int(round(float(row[k])))
        except Exception:
            pass
        return 0

    def _side_for_leg(
        row: pd.Series,
        leg: str,
        *,
        default: Literal["buy", "sell"] = "buy",
    ) -> Literal["buy", "sell"]:
        try:
            n = _infer_units_for_leg(row, leg)
            return "sell" if n < 0 else default
        except Exception:
            return default


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("costs")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# =============================================================================
#                               P A R A M S
# =============================================================================


@dataclass(frozen=True)
class ExecParams:
    """
    Ausführungs-Parameter für vektorisierte (heuristische) Kosten/Slippage-Modelle.

    Einheiten:
      - bps-Angaben werden als Dezimal-Fraktionen gespeichert (1 bp = 0.0001).
      - Währungsbeträge in Cash-Einheiten.

    Felder:
      base_slippage      : Basis-Slippage (z.B. 0.0002 = 2 bps)
      adv_impact_k       : Linearer Impact-Koeffizient (impact_model='linear')
      sqrt_coefficient   : Faktor im sqrt-Modell (× Volatilität)
      power_alpha        : Exponent im Power-Modell (impact_model='power')
      impact_model       : 'sqrt' | 'linear' | 'power'
      impact_min_bps     : Mindest-Impact in bps (als Floor NUR auf den Impact-Anteil)
      max_slippage       : Obergrenze für die GESAMTE Slippage-Fraktion
      min_fee            : Mindestgebühr je Trade-Event (Währung)
      fill_prob_k        : Krümmung für partielle Fill-Heuristik
      min_fill_prob      : Untere Schranke der Fill-Heuristik

    Hinweise:
      - Im 'power'-Modell wird, falls gesetzt, `power_coefficient` als Skalenfaktor verwendet.
        Sonst fällt es aus Kompatibilitätsgründen auf `sqrt_coefficient` zurück.
    """

    base_slippage: float = 0.0002
    adv_impact_k: float = 0.5
    sqrt_coefficient: float = 0.1
    power_alpha: float = 0.6
    impact_model: str = "sqrt"  # 'sqrt' | 'linear' | 'power'
    impact_min_bps: float = 0.0  # nur Impact-Floor, in bps
    max_slippage: float = 0.02
    min_fee: float = 0.0
    fill_prob_k: float = 3.0
    min_fill_prob: float = 0.05
    # Optionaler, semantisch klarer Skalenfaktor für das Power-Modell
    power_coefficient: float | None = None

    def sanitized(self) -> ExecParams:
        """Clamp/normalisiert Werte in sinnvolle Bereiche und säubert Modell-String."""
        model = str(self.impact_model or "sqrt").strip().lower()
        if model not in {"sqrt", "linear", "power"}:
            model = "sqrt"
        # power_coefficient ist optional – clamp nur wenn nicht None
        pcoef = (
            None
            if self.power_coefficient is None
            else max(0.0, float(self.power_coefficient))
        )
        return replace(
            self,
            base_slippage=max(0.0, float(self.base_slippage)),
            adv_impact_k=max(0.0, float(self.adv_impact_k)),
            sqrt_coefficient=max(0.0, float(self.sqrt_coefficient)),
            power_alpha=max(0.0, float(self.power_alpha)),
            impact_model=model,
            impact_min_bps=max(0.0, float(self.impact_min_bps)),
            max_slippage=max(0.0, float(self.max_slippage)),
            min_fee=max(0.0, float(self.min_fee)),
            fill_prob_k=max(0.0, float(self.fill_prob_k)),
            min_fill_prob=float(min(1.0, max(0.0, float(self.min_fill_prob)))),
            power_coefficient=pcoef,
        )


DEFAULT_EXECUTION_PARAMS = ExecParams().sanitized()


def exec_params_from_cfg(cfg: Mapping[str, Any]) -> ExecParams:
    """
    Liest Parameter aus YAML:
      execution.heuristic: {
        base_slippage, sqrt_coefficient, adv_impact_k, power_alpha,
        impact_model, impact_min_bps, max_slippage, min_fee,
        fill_prob_k, min_fill_prob, power_coefficient?
      }
    Fallbacks: DEFAULT_EXECUTION_PARAMS.
    """
    try:
        execution_cfg = cast(Mapping[str, Any], cfg.get("execution") or {})
        exec_cfg = cast(Mapping[str, Any], execution_cfg.get("heuristic") or {})
    except Exception:
        exec_cfg = {}

    def _get(name: str, default: Any) -> Any:
        try:
            return exec_cfg.get(name, default)
        except Exception:
            return default

    # Optionaler Alias: falls power_coefficient gesetzt, nutzen wir ihn; sonst sqrt_coefficient.
    power_coeff_val = _get("power_coefficient", None)
    try:
        pc_float: float | None = (
            float(power_coeff_val) if power_coeff_val is not None else None
        )
    except Exception:
        pc_float = None

    params = ExecParams(
        base_slippage=float(
            _get("base_slippage", DEFAULT_EXECUTION_PARAMS.base_slippage)
        ),
        sqrt_coefficient=float(
            _get("sqrt_coefficient", DEFAULT_EXECUTION_PARAMS.sqrt_coefficient)
        ),
        adv_impact_k=float(_get("adv_impact_k", DEFAULT_EXECUTION_PARAMS.adv_impact_k)),
        power_alpha=float(_get("power_alpha", DEFAULT_EXECUTION_PARAMS.power_alpha)),
        impact_model=str(_get("impact_model", DEFAULT_EXECUTION_PARAMS.impact_model)),
        impact_min_bps=float(
            _get("impact_min_bps", DEFAULT_EXECUTION_PARAMS.impact_min_bps)
        ),
        max_slippage=float(_get("max_slippage", DEFAULT_EXECUTION_PARAMS.max_slippage)),
        min_fee=float(_get("min_fee", DEFAULT_EXECUTION_PARAMS.min_fee)),
        fill_prob_k=float(_get("fill_prob_k", DEFAULT_EXECUTION_PARAMS.fill_prob_k)),
        min_fill_prob=float(
            _get("min_fill_prob", DEFAULT_EXECUTION_PARAMS.min_fill_prob)
        ),
        power_coefficient=pc_float,
    )
    return params.sanitized()


# =============================================================================
#                       S L I P P A G E   &   C O S T S
# =============================================================================

# modernes Union-Alias (Ruff UP007)
Number = int | float | np.number


def _is_pos(n: Number | None) -> TypeGuard[Number]:
    """True, wenn n gesetzt, endlich und > 0.0 ist."""
    try:
        return (n is not None) and np.isfinite(n) and float(n) > 0.0
    except Exception:
        return False


def _to_float(n: Number) -> float:
    """Konvertiert Number robust zu float (für mypy explizit)."""
    return float(cast(float, n))


def _combine_adv(adv_y: Number | None, adv_x: Number | None) -> float | None:
    """ADV-Kombination als einfache Summe (Notional/Tag beider Legs)."""
    if _is_pos(adv_y) and _is_pos(adv_x):
        return _to_float(adv_y) + _to_float(adv_x)
    if _is_pos(adv_y):
        return _to_float(adv_y)
    if _is_pos(adv_x):
        return _to_float(adv_x)
    return None


def _combine_vol(vol_y: Number | None, vol_x: Number | None) -> float | None:
    """
    Kombinierte Tagesvolatilität (heuristisch):
      - beide vorhanden: RMS
      - eine vorhanden : diese
      - sonst          : None
    """
    has_y = _is_pos(vol_y)
    has_x = _is_pos(vol_x)
    if has_y and has_x:
        vy, vx = _to_float(cast(Number, vol_y)), _to_float(cast(Number, vol_x))
        return float(math.sqrt(0.5 * (vy * vy + vx * vx)))
    if has_y:
        return _to_float(cast(Number, vol_y))
    if has_x:
        return _to_float(cast(Number, vol_x))
    return None


@dataclass(frozen=True)
class SimulationResult:
    """
    Strukturierte Rückgabe von :func:`simulate_execution`.

    Alle Kostenfelder sind als positive Beträge in Basiswährung (gleiche Einheit
    wie Preise) angegeben.
    """

    requested_size: int
    filled_size: int
    fill_ratio: float
    fill_probability: float
    notional_requested: float
    notional_filled: float
    slippage_pct_total: float
    slippage_pct_base: float
    slippage_pct_impact: float
    slippage_cost_base: float
    slippage_cost_impact: float
    fixed_fee_cost: float
    min_fee_topup: float
    total_fee_cost: float
    total_cost: float

    @property
    def total_slippage_cost(self) -> float:
        return float(self.slippage_cost_base + self.slippage_cost_impact)

    @property
    def filled_notional_ratio(self) -> float:
        base = float(self.notional_requested)
        if base <= 0.0:
            return 0.0
        return float(self.notional_filled / base)


def square_root_impact(notional_ratio: float, coeff: float = 0.1) -> float:
    """√-Impact ~ coeff * sqrt(traded_notional / ADV). Rückgabe als Bruchteil."""
    if notional_ratio <= 0:
        return 0.0
    return float(coeff) * math.sqrt(float(notional_ratio))


def calc_adv_slippage(
    notional: Number,
    adv: Number | None,
    vol: Number | None,
    params: Mapping[str, Any] | ExecParams | None = None,
) -> float:
    """
    Slippage (Fraktion) in [0, max_slippage] als Funktion von Notional/ADV.
      - notional: komb. Notional (|px_y| + |px_x|) * size
      - adv     : ADV in gleicher Währungseinheit (kombiniert oder Single-ADV)
      - vol     : Tagesvolatilität (Fraktion), nur für 'sqrt' genutzt
      - params  : ExecParams oder Mapping (execution.heuristic.*), unterstützt 'power' & impact_min_bps

    Impact-Floor:
      - impact_min_bps wird nur auf den Impact-Anteil angewendet (nicht auf base_slippage).
    """
    ep = (
        params
        if isinstance(params, ExecParams)
        else exec_params_from_cfg({"execution": {"heuristic": dict(params or {})}})
    ).sanitized()

    n = _to_float(notional) if _is_pos(notional) else 0.0
    if n <= 0.0:
        return 0.0

    # ADV unbekannt → konservativer Baseline-Return
    if not _is_pos(adv):
        return float(min(ep.base_slippage, ep.max_slippage))

    frac = max(0.0, n / _to_float(adv))  # Notional/ADV
    impact = 0.0
    if ep.impact_model == "sqrt":
        v = _to_float(vol) if _is_pos(vol) else 1.0
        impact = ep.sqrt_coefficient * v * math.sqrt(frac)
    elif ep.impact_model == "linear":
        impact = ep.adv_impact_k * frac
    else:  # 'power'
        # Skalenfaktor: bevorzugt power_coefficient, sonst Backcompat via sqrt_coefficient
        scale = (
            float(ep.power_coefficient)
            if ep.power_coefficient is not None
            else float(ep.sqrt_coefficient)
        )
        impact = scale * (frac ** max(0.0, float(ep.power_alpha)))

    # Impact-Floor in bps nur auf den Impact-Anteil
    if ep.impact_min_bps > 0.0:
        impact = max(impact, ep.impact_min_bps / 10_000.0)

    slip = ep.base_slippage + max(0.0, impact)
    return float(min(max(slip, 0.0), ep.max_slippage))


def calc_trade_cost(
    size: Number,
    price_y: Number,
    price_x: Number,
    per_trade_fixed: Number,
    slippage_pct: Number,
    min_fee: Number = 0.0,
    *,
    charge_fixed_when_zero_fill: bool = True,
) -> float:
    """
    Gesamtkosten (Währung) für EIN Pair-Trade-Event (beide Legs):
      - Fixkosten : per_trade_fixed pro Leg → ×2
      - Slippage  : (|y|+|x|) * size * slippage_pct
      - min_fee   : Mindestgebühr, falls > 0

    Parameter:
      charge_fixed_when_zero_fill:
        True  (Default) → Fixkosten fallen immer an (Event-basiert).
        False → Fixkosten nur, wenn size > 0 (bei 0-Fill keine Fixkosten).
    """
    try:
        size_i = int(max(0, int(round(float(size)))))
    except Exception:
        size_i = 0

    fixed = float(per_trade_fixed) * 2.0
    if not charge_fixed_when_zero_fill and size_i == 0:
        fixed = 0.0

    slip_cost = (
        (abs(float(price_y)) + abs(float(price_x))) * size_i * float(slippage_pct)
    )
    total = fixed + slip_cost
    mf = float(min_fee)
    if mf > 0.0 and total < mf:
        total = mf
    return float(max(0.0, total))


def calc_pair_slippage_pct(
    size: Number,
    price_y: Number,
    price_x: Number,
    adv_y: Number | None,
    adv_x: Number | None,
    vol_y: Number | None,
    vol_x: Number | None,
    params: Mapping[str, Any] | ExecParams | None = None,
) -> float:
    """Komfort-Helper: Slippage-Fraktion, indem ADV & Vol pro Leg kombiniert werden."""
    notional = (abs(float(price_y)) + abs(float(price_x))) * max(0.0, float(size))
    adv_pair = _combine_adv(adv_y, adv_x)
    vol_pair = _combine_vol(vol_y, vol_x)
    return calc_adv_slippage(notional, adv_pair, vol_pair, params=params)


def partial_fill_probability(
    notional: Number,
    adv: Number | None,
    params: Mapping[str, Any] | ExecParams | None = None,
) -> float:
    """
    Heuristik für Füllwahrscheinlichkeit (Erwartungswert):
      p_full = max(min_fill_prob, 1 / (1 + (k * notional/adv)^2))
    Falls ADV unbekannt oder notional<=0 → 1.0
    """
    ep = (
        params
        if isinstance(params, ExecParams)
        else exec_params_from_cfg({"execution": {"heuristic": dict(params or {})}})
    ).sanitized()
    if not _is_pos(notional) or not _is_pos(adv):
        return 1.0
    frac = _to_float(notional) / _to_float(adv)
    val = 1.0 / (1.0 + (ep.fill_prob_k * frac) ** 2)
    return float(max(ep.min_fill_prob, min(1.0, val)))


def simulate_execution(
    size: Number,
    price_y: Number,
    price_x: Number,
    per_trade_fixed: Number,
    adv: Number | None,
    vol: Number | None,
    params: Mapping[str, Any] | ExecParams | None = None,
    *,
    charge_fixed_when_zero_fill: bool = True,
) -> SimulationResult:
    """
    Deterministische Erwartungswert-Simulation eines Pair-Trade-Events.

    RÜckgabe: :class:`SimulationResult` mit detaillierter Aufschlüsselung der
    Kostenkomponenten. Alle Beträge sind positiv (Kosten).
    """
    ep = (
        params
        if isinstance(params, ExecParams)
        else exec_params_from_cfg({"execution": {"heuristic": dict(params or {})}})
    ).sanitized()

    # Notional & Slippage-Fraktion
    try:
        size_i = int(max(0, int(round(float(size)))))
    except Exception:
        size_i = 0
    price_y_f = abs(float(price_y))
    price_x_f = abs(float(price_x))
    notional = (price_y_f + price_x_f) * size_i
    slippage_pct = calc_adv_slippage(notional, adv, vol, ep)

    # Erwartete Füllung
    p_fill = partial_fill_probability(notional, adv, ep)
    filled_size = int(math.floor(size_i * p_fill + 1e-9))
    fill_ratio = float(filled_size / size_i) if size_i > 0 else 0.0
    notional_filled = notional * fill_ratio

    # Slippage-Komponenten
    base_pct = float(min(ep.base_slippage, slippage_pct))
    impact_pct = float(max(0.0, slippage_pct - base_pct))
    slip_base_cost = notional_filled * base_pct
    slip_impact_cost = notional_filled * impact_pct

    # Gebühren (Fixkosten + Mindestfee-Top-Up)
    per_trade_fixed_f = float(per_trade_fixed)
    fixed_fee_cost = per_trade_fixed_f * 2.0
    if not charge_fixed_when_zero_fill and size_i == 0:
        fixed_fee_cost = 0.0

    min_fee = max(0.0, float(ep.min_fee))
    raw_total = float(fixed_fee_cost + slip_base_cost + slip_impact_cost)
    min_fee_topup = 0.0
    if min_fee > 0.0 and raw_total < min_fee:
        min_fee_topup = float(min_fee - raw_total)

    total_fee_cost = fixed_fee_cost + min_fee_topup
    total_cost = slip_base_cost + slip_impact_cost + total_fee_cost

    logger.debug(
        "simulate_execution: size=%s filled=%s notional=%.2f adv=%s vol=%s sl=%.5f "
        "base_pct=%.5f impact_pct=%.5f total=%.4f",
        size_i,
        filled_size,
        notional,
        adv,
        vol,
        slippage_pct,
        base_pct,
        impact_pct,
        total_cost,
    )

    return SimulationResult(
        requested_size=size_i,
        filled_size=filled_size,
        fill_ratio=fill_ratio,
        fill_probability=float(p_fill),
        notional_requested=float(notional),
        notional_filled=float(notional_filled),
        slippage_pct_total=float(slippage_pct),
        slippage_pct_base=float(base_pct),
        slippage_pct_impact=float(impact_pct),
        slippage_cost_base=float(slip_base_cost),
        slippage_cost_impact=float(slip_impact_cost),
        fixed_fee_cost=float(fixed_fee_cost),
        min_fee_topup=float(min_fee_topup),
        total_fee_cost=float(total_fee_cost),
        total_cost=float(total_cost),
    )


def _coalesce(*vals: Any, default: Any = None) -> Any:
    for v in vals:
        if v is None:
            continue
        try:
            if isinstance(v, (float, int)) and not np.isfinite(v):
                continue
        except Exception:
            pass
        return v
    return default


def _price_for_leg(row: pd.Series, leg: str) -> float | None:
    """Preispräferenz: exec_entry_vwap_* → entry_price_* → price_* → py_/px_-Aliase."""
    leg = leg.lower()
    candidates = [f"exec_entry_vwap_{leg}", f"entry_price_{leg}", f"price_{leg}"]
    candidates += (
        ["py_entry", "py_exit", "py"] if leg == "y" else ["px_entry", "px_exit", "px"]
    )
    for cand in candidates:
        if cand in row and pd.notna(row[cand]):
            try:
                v = float(row[cand])
                return v if np.isfinite(v) and v > 0 else None
            except Exception:
                continue
    return None


def _liquidity_flag(row: pd.Series, leg: str) -> Literal["maker", "taker"]:
    """Maker/Taker-Flag pro Leg (Default: taker). String oder Bool (True ~ maker)."""
    leg = leg.lower()
    for cand in (
        f"liquidity_{leg}",
        f"liq_flag_{leg}",
        f"{leg}_liquidity",
        f"{leg}_maker",
    ):
        v = row.get(cand)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"maker", "taker"}:
                return cast(Literal["maker", "taker"], s)
        if isinstance(v, (int, bool)):
            return "maker" if bool(v) else "taker"
    return "taker"


class FeeCfg(TypedDict, total=False):
    maker_bps: float
    taker_bps: float
    maker_per_share: float
    taker_per_share: float
    min_fee: float
    max_fee: float


def _maker_taker_fees_for_leg(
    notional: float, shares: float, is_maker: bool, fee_cfg: FeeCfg | Mapping[str, Any]
) -> float:
    """
    Gebühren pro Leg. fee_cfg (optional):
      - maker_bps / taker_bps (in bps; Rebates → negative Werte)
      - maker_per_share / taker_per_share (Währung/Aktie)
    Rückgabe: signierte Kosten (Kosten ≤ 0; Rebate ≥ 0).
    """
    role = "maker" if is_maker else "taker"
    bps = float(_coalesce(fee_cfg.get(f"{role}_bps"), 0.0))
    per_share = float(_coalesce(fee_cfg.get(f"{role}_per_share"), 0.0))
    fee = -(bps / 1e4) * float(max(notional, 0.0)) - per_share * float(max(shares, 0.0))
    return float(fee)


def _borrow_for_row(
    row: pd.Series, *, bps_annual: float, day_basis: int = 365, short_only: bool = False
) -> float:
    """
    Borrow-Accrual (≤ 0).
    Falls short_only=False → auf Bruttonotional; sonst nur auf Short-Notional (aus Seite).
    Erwartet entry_date/exit_date; nutzt exec_entry_vwap_* bzw. entry_* / px/py.
    """
    try:
        d0 = pd.Timestamp(row["entry_date"]).normalize()
        d1 = pd.Timestamp(row["exit_date"]).normalize()
        days = max(0, int((d1 - d0).days)) + 1
    except Exception:
        days = 0
    if days <= 0 or bps_annual <= 0:
        return 0.0

    uy = _infer_units_for_leg(row, "y")
    ux = _infer_units_for_leg(row, "x")
    py = _price_for_leg(row, "y") or 0.0
    px = _price_for_leg(row, "x") or 0.0
    ny = abs(uy * py)
    nx = abs(ux * px)

    if not short_only:
        gross = float(ny + nx)
        return (
            -abs(gross) * (bps_annual / 1e4) * (float(days) / float(max(1, day_basis)))
        )

    sy = _side_for_leg(row, "y")
    sx = _side_for_leg(row, "x")
    short_notional = 0.0
    if sy == "sell":
        short_notional += ny
    if sx == "sell":
        short_notional += nx
    return (
        -abs(short_notional)
        * (bps_annual / 1e4)
        * (float(days) / float(max(1, day_basis)))
    )


# =============================================================================
#  Vektorisiertes Gebühren-API (Maker/Taker-Rebates, Tiers, Auction-Fees)
# =============================================================================


class _FeeTier(TypedDict, total=False):
    metric: Literal["notional", "shares"]  # Default: 'notional'
    thresh: float  # Schwellenwert (>= 0)
    bps: float | None
    per_share: float | None


class _RoleFee(TypedDict, total=False):
    bps: float
    per_share: float
    tiers: list[_FeeTier]  # optional


class _Caps(TypedDict, total=False):
    min_fee: float
    max_fee: float


class _AuctionSide(TypedDict, total=False):
    bps: float
    per_share: float


class _AuctionFees(TypedDict, total=False):
    open: _AuctionSide
    close: _AuctionSide


class VenueFeeProfile(TypedDict, total=False):
    maker: _RoleFee
    taker: _RoleFee
    caps: _Caps
    auction: _AuctionFees
    commission_per_share: float
    clearing_per_trade: float
    clearing_min_per_trade: float


class FeeSchedule(TypedDict, total=False):
    venues: dict[str, VenueFeeProfile]
    # Optional: globale Aufschläge (z.B. Broker/Clearing), werden additiv zu venue-bps/per-share gerechnet
    global_bps: float
    global_per_share: float
    global_commission_per_share: float
    global_clearing_per_trade: float
    global_clearing_min_per_trade: float


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _pick_tier(
    values: np.ndarray, tiers: list[_FeeTier], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Liefert vektorisiert (bps, per_share) je Row anhand von tiers und gegebener Kennzahl.
    - Step-Funktion: wählt die höchste Stufe mit thresh <= value.
    - Fehlende Felder in Tier -> ignorieren (None).
    """
    if not tiers:
        return (
            np.full(values.shape, np.nan, dtype=float),
            np.full(values.shape, np.nan, dtype=float),
        )

    # sort by thresh ascending
    tiers_sorted = sorted(tiers, key=lambda t: _as_float(t.get("thresh"), 0.0))
    thresholds = np.array(
        [_as_float(t.get("thresh"), 0.0) for t in tiers_sorted], dtype=float
    )
    bps_arr = np.array(
        [_as_float(t.get("bps"), np.nan) for t in tiers_sorted], dtype=float
    )
    ps_arr = np.array(
        [_as_float(t.get("per_share"), np.nan) for t in tiers_sorted], dtype=float
    )

    # index of rightmost thresh <= value
    idx = np.searchsorted(thresholds, values, side="right") - 1
    idx = np.clip(idx, 0, len(thresholds) - 1)

    eff_bps = bps_arr[idx]
    eff_ps = ps_arr[idx]
    # Falls ein Tier None enthält → NaN bleibt stehen; der Aufrufer mischt Basiswerte und Tierwerte
    return eff_bps, eff_ps


def normalize_fee_schedule_from_cfg(cfg: Mapping[str, Any]) -> FeeSchedule:
    """
    Extrahiert ein robustes Gebührenschema aus dem YAML:
      - venues.<VID>.fees.*   (+ Alias maker_rebate_bs → maker_rebate_bps)
      - execution.fees.default / venue_overrides als Fallback
      - execution.fallback.fees als globaler Rückfall
      - auctions.* und costs.* als globale Zusätze
    """
    venues_cfg = cast(Mapping[str, Any], (cfg.get("venues") or {}))
    execution_cfg = cast(Mapping[str, Any], (cfg.get("execution") or {}))
    execution_fees_cfg = cast(Mapping[str, Any], execution_cfg.get("fees") or {})
    default_fee_cfg = cast(Mapping[str, Any], execution_fees_cfg.get("default") or {})
    fee_overrides_cfg = cast(
        Mapping[str, Any], execution_fees_cfg.get("venue_overrides") or {}
    )
    fallback_fee_cfg = cast(
        Mapping[str, Any], (execution_cfg.get("fallback") or {}).get("fees") or {}
    )

    out: dict[str, VenueFeeProfile] = {}

    # globale Zusätze (Broker-Aufschläge etc.)
    spec = resolve_execution_cost_spec(cfg)
    global_bps = _as_float(
        spec.fee_bps if spec.mode == "light" and spec.enabled else 0.0, 0.0
    )
    global_ps = _as_float(
        spec.per_share_fee if spec.mode == "light" and spec.enabled else 0.0, 0.0
    )
    global_commission_ps = _as_float(
        execution_fees_cfg.get("global_commission_per_share"), 0.0
    )
    global_clearing_per_trade = _as_float(
        execution_fees_cfg.get("global_clearing_per_trade"), 0.0
    )
    global_clearing_min = _as_float(
        execution_fees_cfg.get("global_clearing_min_per_trade"), 0.0
    )

    # Auction Defaults
    auc_global = cast(
        Mapping[str, Any],
        (((cfg.get("backtest") or {}).get("auctions") or {}).get("fees") or {}),
    )
    auc_profiles = cast(Mapping[str, Any], auc_global.get("venue_profiles") or {})

    for vid, vmeta in venues_cfg.items():
        if not isinstance(vmeta, Mapping) or vid.lower() in {
            "default",
            "aliases",
            "symbol_suffix_map",
            "symbol_venue_map",
        }:
            continue

        fees = cast(Mapping[str, Any], vmeta.get("fees") or {})
        override_fee = cast(Mapping[str, Any], fee_overrides_cfg.get(vid) or {})

        def _resolve_fee(key: str, default: float = 0.0) -> float:
            return _as_float(
                _coalesce(
                    fees.get(key),
                    override_fee.get(key),
                    default_fee_cfg.get(key),
                    fallback_fee_cfg.get(key),
                    default,
                ),
                default,
            )

        maker_bps = _resolve_fee("maker_bps", 0.0)
        taker_bps = _resolve_fee("taker_bps", 0.0)
        maker_rebate_bps = _as_float(
            _coalesce(
                fees.get("maker_rebate_bps"),
                fees.get("maker_rebate_bs"),
                override_fee.get("maker_rebate_bps"),
                override_fee.get("maker_rebate_bs"),
                default_fee_cfg.get("maker_rebate_bps"),
                default_fee_cfg.get("maker_rebate_bs"),
                0.0,
            ),
            0.0,
        )
        if maker_rebate_bps != 0.0:
            maker_bps = maker_bps + maker_rebate_bps

        maker_ps = _resolve_fee("maker_per_share", 0.0)
        taker_ps = _resolve_fee("taker_per_share", 0.0)
        commission_ps = _resolve_fee("commission_per_share", 0.0)

        min_fee = _as_float(
            _coalesce(
                fees.get("min_fee"),
                override_fee.get("min_fee"),
                default_fee_cfg.get("min_fee"),
                0.0,
            ),
            0.0,
        )
        max_fee = _as_float(
            _coalesce(
                fees.get("max_fee"),
                override_fee.get("max_fee"),
                default_fee_cfg.get("max_fee"),
                0.0,
            ),
            0.0,
        )
        clearing_per_trade = _resolve_fee("clearing_per_trade", 0.0)
        clearing_min_per_trade = _resolve_fee("clearing_min_per_trade", 0.0)

        maker_tiers: list[_FeeTier] = []
        taker_tiers: list[_FeeTier] = []
        for key, bucket in (("maker_tiers", maker_tiers), ("taker_tiers", taker_tiers)):
            raw = vmeta.get(key) or fees.get(key)
            if isinstance(raw, list):
                for t in raw:
                    if not isinstance(t, Mapping):
                        continue
                    bucket.append(
                        _FeeTier(
                            metric=str(_coalesce(t.get("metric"), "notional")).lower()
                            in {"shares", "share", "qty"}
                            and "shares"
                            or "notional",
                            thresh=_as_float(t.get("thresh"), 0.0),
                            bps=None
                            if t.get("bps") is None
                            else _as_float(t.get("bps"), 0.0),
                            per_share=None
                            if t.get("per_share") is None
                            else _as_float(t.get("per_share"), 0.0),
                        )
                    )

        auc_local = cast(Mapping[str, Any], vmeta.get("auction_fees") or {})
        auc_open = {
            "bps": _as_float(
                _coalesce(
                    auc_local.get("open_bps"),
                    (auc_profiles.get(vid) or {}).get("open", {}).get("bps"),
                ),
                0.0,
            ),
            "per_share": _as_float(auc_local.get("open_per_share"), 0.0),
        }
        auc_close = {
            "bps": _as_float(
                _coalesce(
                    auc_local.get("close_bps"),
                    (auc_profiles.get(vid) or {}).get("close", {}).get("bps"),
                ),
                0.0,
            ),
            "per_share": _as_float(auc_local.get("close_per_share"), 0.0),
        }

        auc_open_t = cast(_AuctionSide, auc_open)
        auc_close_t = cast(_AuctionSide, auc_close)
        out[str(vid)] = VenueFeeProfile(
            maker=_RoleFee(bps=maker_bps, per_share=maker_ps, tiers=maker_tiers or []),
            taker=_RoleFee(bps=taker_bps, per_share=taker_ps, tiers=taker_tiers or []),
            caps=_Caps(min_fee=min_fee, max_fee=max_fee),
            auction=_AuctionFees(open=auc_open_t, close=auc_close_t),
            commission_per_share=commission_ps,
            clearing_per_trade=clearing_per_trade,
            clearing_min_per_trade=clearing_min_per_trade,
        )

    return FeeSchedule(
        venues=out,
        global_bps=global_bps,
        global_per_share=global_ps,
        global_commission_per_share=global_commission_ps,
        global_clearing_per_trade=global_clearing_per_trade,
        global_clearing_min_per_trade=global_clearing_min,
    )


def compute_costs(
    fills: pd.DataFrame,
    fee_schedule: FeeSchedule | Mapping[str, Any],
    borrow: Mapping[str, Any] | None = None,
    auction_flags: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Vektorisiertes Gebühren-API für venue-/child-granulare Fills.

    Rückgabe-Spalten:
      - fee             (signiert, Kosten ≤ 0, Rebates ≥ 0)  [inkl. Auction-Overrides]
      - fees            (Alias von 'fee' für Downstream-Konsistenz)
      - auctions_cost   (nur der inkrementelle Auction-Anteil; ≤ 0)
      - slippage_cost   (falls nicht vorhanden → 0)
      - impact_cost     (falls nicht vorhanden → 0)
      - borrow_cost     (immer 0 – Borrow wird außerhalb gerechnet)
      - total_costs     (= fee + slippage_cost + impact_cost)
      - diagnostics: 'blended_bps','per_share_eff','auction_applied','fee_bps_equiv'
    """
    if fills is None or len(fills) == 0:
        return pd.DataFrame(
            columns=[
                "fee",
                "fees",
                "auctions_cost",
                "slippage_cost",
                "impact_cost",
                "borrow_cost",
                "total_costs",
                "blended_bps",
                "per_share_eff",
                "commission_per_share",
                "clearing_per_trade",
                "clearing_min_per_trade",
                "auction_applied",
                "fee_bps_equiv",
            ]
        )

    df = fills.copy()

    # ---- venue / schedule ---------------------------------------------------
    sched = cast(
        FeeSchedule,
        fee_schedule if isinstance(fee_schedule, Mapping) else dict(fee_schedule),
    )
    venues = sched.get("venues", {})
    global_bps = _as_float(sched.get("global_bps", 0.0), 0.0)
    global_ps = _as_float(sched.get("global_per_share", 0.0), 0.0)
    global_commission = _as_float(sched.get("global_commission_per_share", 0.0), 0.0)
    global_clearing_pt = _as_float(sched.get("global_clearing_per_trade", 0.0), 0.0)
    global_clearing_min = _as_float(
        sched.get("global_clearing_min_per_trade", 0.0), 0.0
    )

    # ---- notional & shares --------------------------------------------------
    n = len(df)
    notional = np.zeros(n, dtype=float)
    shares = np.zeros(n, dtype=float)

    if "notional" in df.columns:
        notional = (
            pd.to_numeric(df["notional"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    else:
        if {"qty", "vwap"}.issubset(df.columns):
            q = (
                pd.to_numeric(df["qty"], errors="coerce")
                .fillna(0.0)
                .abs()
                .to_numpy(dtype=float)
            )
            px = (
                pd.to_numeric(df["vwap"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            notional = q * px
        else:
            qy = (
                pd.to_numeric(cast(pd.Series, df.get("qty_y", 0.0)), errors="coerce")
                .fillna(0.0)
                .abs()
                .to_numpy(dtype=float)
            )
            qx = (
                pd.to_numeric(cast(pd.Series, df.get("qty_x", 0.0)), errors="coerce")
                .fillna(0.0)
                .abs()
                .to_numpy(dtype=float)
            )
            py = (
                pd.to_numeric(
                    cast(pd.Series, df.get("px_y", df.get("price_y", 0.0))),
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            pxv = (
                pd.to_numeric(
                    cast(pd.Series, df.get("px_x", df.get("price_x", 0.0))),
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            notional = qy * py + qx * pxv

    if "shares" in df.columns:
        shares = (
            pd.to_numeric(df["shares"], errors="coerce")
            .fillna(0.0)
            .abs()
            .to_numpy(dtype=float)
        )
    else:
        if "qty" in df.columns:
            shares = (
                pd.to_numeric(df["qty"], errors="coerce")
                .fillna(0.0)
                .abs()
                .to_numpy(dtype=float)
            )
        else:
            sy = (
                pd.to_numeric(cast(pd.Series, df.get("qty_y", 0.0)), errors="coerce")
                .fillna(0.0)
                .abs()
                .to_numpy(dtype=float)
            )
            sx = (
                pd.to_numeric(cast(pd.Series, df.get("qty_x", 0.0)), errors="coerce")
                .fillna(0.0)
                .abs()
                .to_numpy(dtype=float)
            )
            shares = sy + sx

    commission_ps = np.zeros(n, dtype=float)
    clearing_per_trade_arr = np.zeros(n, dtype=float)
    clearing_min_arr = np.zeros(n, dtype=float)

    # ---- maker share --------------------------------------------------------
    mshare = np.full(n, 0.5, dtype=float)
    if "maker_share_eff" in df.columns:
        mshare = (
            pd.to_numeric(df["maker_share_eff"], errors="coerce")
            .fillna(0.5)
            .to_numpy(dtype=float)
        )
        mshare = np.clip(mshare, 0.0, 1.0)
    else:
        if "maker_flag" in df.columns:
            mf = df["maker_flag"].astype(bool).to_numpy()
            mshare = np.where(mf, 1.0, 0.0)
        elif "role" in df.columns:
            rr = df["role"].astype(str).str.lower().to_numpy()
            mshare = np.where(rr == "maker", 1.0, 0.0)

    # ---- auction flags ------------------------------------------------------
    is_open = np.zeros(n, dtype=bool)
    is_close = np.zeros(n, dtype=bool)

    if "is_open_auction" in df.columns:
        is_open = df["is_open_auction"].astype(bool).to_numpy()
    if "is_close_auction" in df.columns:
        is_close = df["is_close_auction"].astype(bool).to_numpy()
    if "order_kind" in df.columns:
        ok = df["order_kind"].astype(str).str.upper()
        is_open = np.logical_or(is_open, ok.eq("MOO").to_numpy())
        is_close = np.logical_or(is_close, ok.eq("MOC").to_numpy())
    if auction_flags:
        if "is_open_auction" in auction_flags:
            is_open = np.logical_or(
                is_open, np.asarray(auction_flags["is_open_auction"], dtype=bool)
            )
        if "is_close_auction" in auction_flags:
            is_close = np.logical_or(
                is_close, np.asarray(auction_flags["is_close_auction"], dtype=bool)
            )

    # ---- per-venue eff. maker/taker bps/per-share (inkl. Tiers) -------------
    venue_col = df.get("venue")
    venues_rows = (
        (venue_col.astype(str) if venue_col is not None else pd.Series([""] * n))
        .fillna("")
        .astype(str)
    )

    eff_maker_bps = np.zeros(n, dtype=float)
    eff_taker_bps = np.zeros(n, dtype=float)
    eff_maker_ps = np.zeros(n, dtype=float)
    eff_taker_ps = np.zeros(n, dtype=float)
    cap_min = np.zeros(n, dtype=float)
    cap_max = np.zeros(n, dtype=float)

    for vid, idx in venues_rows.groupby(venues_rows).groups.items():
        sel = np.asarray(list(idx), dtype=int)
        prof = venues.get(str(vid), {})

        maker_base_bps = _as_float(((prof.get("maker") or {}).get("bps")), 0.0)
        taker_base_bps = _as_float(((prof.get("taker") or {}).get("bps")), 0.0)
        maker_base_ps = _as_float(((prof.get("maker") or {}).get("per_share")), 0.0)
        taker_base_ps = _as_float(((prof.get("taker") or {}).get("per_share")), 0.0)

        mk_tiers = cast(list[_FeeTier], (prof.get("maker") or {}).get("tiers") or [])
        tk_tiers = cast(list[_FeeTier], (prof.get("taker") or {}).get("tiers") or [])

        if mk_tiers:
            if any((t.get("metric") or "notional") == "shares" for t in mk_tiers):
                bps_sh, ps_sh = _pick_tier(shares[sel], mk_tiers, "shares")
                bps_no, ps_no = _pick_tier(notional[sel], mk_tiers, "notional")
                m_bps_eff = np.where(np.isfinite(bps_sh), bps_sh, bps_no)
                m_ps_eff = np.where(np.isfinite(ps_sh), ps_sh, ps_no)
            else:
                m_bps_eff, m_ps_eff = _pick_tier(notional[sel], mk_tiers, "notional")
            m_bps_eff = np.where(np.isfinite(m_bps_eff), m_bps_eff, maker_base_bps)
            m_ps_eff = np.where(np.isfinite(m_ps_eff), m_ps_eff, maker_base_ps)
        else:
            m_bps_eff = np.full(sel.shape, maker_base_bps, dtype=float)
            m_ps_eff = np.full(sel.shape, maker_base_ps, dtype=float)

        if tk_tiers:
            if any((t.get("metric") or "notional") == "shares" for t in tk_tiers):
                bps_sh, ps_sh = _pick_tier(shares[sel], tk_tiers, "shares")
                bps_no, ps_no = _pick_tier(notional[sel], tk_tiers, "notional")
                t_bps_eff = np.where(np.isfinite(bps_sh), bps_sh, bps_no)
                t_ps_eff = np.where(np.isfinite(ps_sh), ps_sh, ps_no)
            else:
                t_bps_eff, t_ps_eff = _pick_tier(notional[sel], tk_tiers, "notional")
            t_bps_eff = np.where(np.isfinite(t_bps_eff), t_bps_eff, taker_base_bps)
            t_ps_eff = np.where(np.isfinite(t_ps_eff), t_ps_eff, taker_base_ps)
        else:
            t_bps_eff = np.full(sel.shape, taker_base_bps, dtype=float)
            t_ps_eff = np.full(sel.shape, taker_base_ps, dtype=float)

        eff_maker_bps[sel] = m_bps_eff
        eff_taker_bps[sel] = t_bps_eff
        eff_maker_ps[sel] = m_ps_eff
        eff_taker_ps[sel] = t_ps_eff
        commission_ps[sel] = _as_float((prof.get("commission_per_share")), 0.0)
        clearing_per_trade_arr[sel] = _as_float((prof.get("clearing_per_trade")), 0.0)
        clearing_min_arr[sel] = _as_float((prof.get("clearing_min_per_trade")), 0.0)

        cap_min[sel] = _as_float((prof.get("caps") or {}).get("min_fee"), 0.0)
        cap_max[sel] = _as_float((prof.get("caps") or {}).get("max_fee"), 0.0)

    commission_ps = commission_ps + float(global_commission)
    clearing_per_trade_arr = clearing_per_trade_arr + float(global_clearing_pt)
    clearing_min_arr = np.maximum(clearing_min_arr, float(global_clearing_min))

    # ---- blended (vor Auction-Overrides) ------------------------------------
    blended_bps_base = (
        mshare * eff_maker_bps + (1.0 - mshare) * eff_taker_bps
    ) + float(global_bps)
    per_share_base = (mshare * eff_maker_ps + (1.0 - mshare) * eff_taker_ps) + float(
        global_ps
    )

    # ---- auctions override (ersetzt blended_* bei Open/Close) ----------------
    blended_bps = blended_bps_base.copy()
    per_share_eff = per_share_base.copy()
    auction_applied = np.zeros(n, dtype="U5")  # "", "open", "close"

    if np.any(is_open) or np.any(is_close):
        for vid, idx in venues_rows.groupby(venues_rows).groups.items():
            sel = np.asarray(list(idx), dtype=int)
            prof = venues.get(str(vid), {})
            auc = prof.get("auction") or {}

            if np.any(is_open[sel]):
                open_bps = _as_float((auc.get("open") or {}).get("bps"), 0.0) + float(
                    global_bps
                )
                open_ps = _as_float(
                    (auc.get("open") or {}).get("per_share"), 0.0
                ) + float(global_ps)
                m = np.logical_and(is_open, venues_rows.to_numpy() == vid)
                blended_bps[m] = open_bps
                per_share_eff[m] = open_ps
                auction_applied[m] = "open"

            if np.any(is_close[sel]):
                close_bps = _as_float((auc.get("close") or {}).get("bps"), 0.0) + float(
                    global_bps
                )
                close_ps = _as_float(
                    (auc.get("close") or {}).get("per_share"), 0.0
                ) + float(global_ps)
                m = np.logical_and(is_close, venues_rows.to_numpy() == vid)
                blended_bps[m] = close_bps
                per_share_eff[m] = close_ps
                auction_applied[m] = "close"

    # ---- fees (signiert) + Caps ---------------------------------------------
    per_share_base_total = per_share_base + commission_ps
    per_share_eff_total = per_share_eff + commission_ps
    fee_cash_base = -(blended_bps_base / 10_000.0) * notional - (
        per_share_base_total * shares
    )  # ohne Auction-Override
    fee_cash = -(blended_bps / 10_000.0) * notional - (
        per_share_eff_total * shares
    )  # mit  Auction-Override
    auctions_cost = (
        fee_cash - fee_cash_base
    )  # rein inkrementeller Auction-Anteil (≤ 0 oder 0)

    # Caps deckeln NUR Kosten (negativ). Rebates (positiv) NICHT deckeln.
    have_cost = fee_cash < 0.0
    # Min-Fee (Floor): nicht „über“ -min_fee hinaus
    fee_cash = np.where(
        np.logical_and(have_cost, cap_min > 0.0),
        np.maximum(fee_cash, -np.abs(cap_min)),
        fee_cash,
    )
    # Max-Fee (Ceiling): nicht „unter“ -max_fee hinaus
    fee_cash = np.where(
        np.logical_and(have_cost, cap_max > 0.0),
        np.maximum(
            fee_cash, -np.abs(cap_max)
        ),  # <-- BUGFIX: vorher fälschlich np.minimum(...)
        fee_cash,
    )

    # ---- slippage/impact ----------------------------------------------------
    def _col_or_zero(name: str) -> np.ndarray:
        col = df.get(name, None)
        if col is None:
            ser = pd.Series(0.0, index=df.index)
        else:
            ser = pd.Series(col, index=df.index)
        return pd.to_numeric(ser, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    slip = _col_or_zero("slippage_cost")
    imp = _col_or_zero("impact_cost")

    borrow_cost = np.zeros(
        n, dtype=float
    )  # hier explizit 0 (globaler Borrow liegt außerhalb)

    total = fee_cash + slip + imp

    # Diagnostics
    with np.errstate(divide="ignore", invalid="ignore"):
        fee_bps_equiv = np.where(notional > 0.0, (fee_cash / notional) * 10_000.0, 0.0)

    out = pd.DataFrame(
        {
            "fee": fee_cash.astype(float),
            "fees": fee_cash.astype(float),  # Alias für Downstream (TCA-Parität)
            "auctions_cost": auctions_cost.astype(float),
            "slippage_cost": slip.astype(float),
            "impact_cost": imp.astype(float),
            "borrow_cost": borrow_cost.astype(float),
            "total_costs": total.astype(float),
            # Diagnostics:
            "blended_bps": blended_bps.astype(float),
            "per_share_eff": per_share_eff_total.astype(float),
            "commission_per_share": commission_ps.astype(float),
            "clearing_per_trade": clearing_per_trade_arr.astype(float),
            "clearing_min_per_trade": clearing_min_arr.astype(float),
            "auction_applied": auction_applied,
            "fee_bps_equiv": fee_bps_equiv.astype(float),
        },
        index=df.index,
    )
    return out


# =============================================================================
# Post-LOB Kosten (Fees + Borrow) mit optionalem Explain-Split
# =============================================================================


def compute_post_lob_costs(
    trades_df: pd.DataFrame, cfg: Mapping[str, Any]
) -> pd.DataFrame:
    """
    Admin-Kosten *nach* LOB/VWAP-Ausführung.
    Gibt DataFrame mit ["fees","fees_entry","fees_exit","borrow_cost","slippage_cost","impact_cost","total_costs"] zurück.
    Alle Kosten sind signiert:
      - Kosten ≤ 0
      - Rebates ≥ 0

    Single-source-of-truth policy in this codebase:
      - Borrow wird NICHT hier gerechnet (BorrowContext in der Engine).
      - Slippage/Impact werden NICHT hier gerechnet (LOB annotator).
      - Diese Funktion liefert ausschließlich Fees (auf Entry-Notional), plus 0-Spalten
        für die anderen Komponenten zur Kompatibilität.
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(
            columns=[
                "fees",
                "fees_entry",
                "fees_exit",
                "borrow_cost",
                "slippage_cost",
                "impact_cost",
                "total_costs",
            ],
            index=getattr(trades_df, "index", None),
        ).fillna(0.0)

    post: dict[str, Any] = dict(cfg or {})
    if "post_costs" in post and isinstance(post["post_costs"], Mapping):
        post = dict(post["post_costs"])

    per_trade = float(_coalesce(post.get("per_trade"), 0.0))
    fee_cfg: FeeCfg = {
        "maker_bps": float(_coalesce(post.get("maker_bps"), 0.0)),
        "taker_bps": float(_coalesce(post.get("taker_bps"), 0.0)),
        "maker_per_share": float(_coalesce(post.get("maker_per_share"), 0.0)),
        "taker_per_share": float(_coalesce(post.get("taker_per_share"), 0.0)),
        "min_fee": float(_coalesce(post.get("min_fee"), 0.0)),
        "max_fee": float(_coalesce(post.get("max_fee"), 0.0)),
    }

    n = len(trades_df)
    fees_out = np.zeros(n, dtype=float)
    fees_entry_out = np.zeros(n, dtype=float)
    fees_exit_out = np.zeros(n, dtype=float)

    for i in range(n):
        row = trades_df.iloc[i]

        uy = _infer_units_for_leg(row, "y")
        ux = _infer_units_for_leg(row, "x")

        # Prefer executed VWAPs (entry/exit), else fall back to mid/close prices.
        py0 = float(
            _coalesce(
                row.get("exec_entry_vwap_y"),
                row.get("entry_price_y"),
                row.get("price_y"),
                0.0,
            )
        )
        px0 = float(
            _coalesce(
                row.get("exec_entry_vwap_x"),
                row.get("entry_price_x"),
                row.get("price_x"),
                0.0,
            )
        )
        py1 = float(
            _coalesce(
                row.get("exec_exit_vwap_y"),
                row.get("exit_price_y"),
                row.get("price_y"),
                py0,
            )
        )
        px1 = float(
            _coalesce(
                row.get("exec_exit_vwap_x"),
                row.get("exit_price_x"),
                row.get("price_x"),
                px0,
            )
        )

        # Liquidity: prefer per-event flags if present.
        ly0 = (
            str(
                _coalesce(row.get("liquidity_entry_y"), row.get("liquidity_y"), "taker")
            )
            .strip()
            .lower()
        )
        lx0 = (
            str(
                _coalesce(row.get("liquidity_entry_x"), row.get("liquidity_x"), "taker")
            )
            .strip()
            .lower()
        )
        ly1 = (
            str(_coalesce(row.get("liquidity_exit_y"), row.get("liquidity_y"), "taker"))
            .strip()
            .lower()
        )
        lx1 = (
            str(_coalesce(row.get("liquidity_exit_x"), row.get("liquidity_x"), "taker"))
            .strip()
            .lower()
        )

        # Total executed notional across entry+exit (4 execution events for pair trades).
        ny0 = abs(float(uy) * py0)
        nx0 = abs(float(ux) * px0)
        ny1 = abs(float(uy) * py1)
        nx1 = abs(float(ux) * px1)

        fy0 = _maker_taker_fees_for_leg(ny0, uy, ly0 == "maker", fee_cfg)
        fx0 = _maker_taker_fees_for_leg(nx0, ux, lx0 == "maker", fee_cfg)
        fy1 = _maker_taker_fees_for_leg(ny1, uy, ly1 == "maker", fee_cfg)
        fx1 = _maker_taker_fees_for_leg(nx1, ux, lx1 == "maker", fee_cfg)

        # per_trade is treated as per-execution-event cost (entry_y, entry_x, exit_y, exit_x).
        per_trade_entry = -abs(per_trade) * 2.0
        per_trade_exit = -abs(per_trade) * 2.0
        f_entry = float(fy0 + fx0 + per_trade_entry)
        f_exit = float(fy1 + fx1 + per_trade_exit)
        f_total = float(f_entry + f_exit)

        # Floor/Cap nur auf Kosten (nicht auf Rebates)
        min_fee = float(fee_cfg.get("min_fee", 0.0) or 0.0)
        if min_fee > 0 and f_total < 0:
            f_total = min(f_total, -abs(min_fee))
        max_fee = float(fee_cfg.get("max_fee", 0.0) or 0.0)
        if max_fee > 0 and f_total < 0:
            f_total = max(f_total, -abs(max_fee))

        fees_out[i] = float(f_total)
        denom = float(f_entry + f_exit)
        if denom != 0.0:
            scale = float(f_total) / denom
            fees_entry_out[i] = float(f_entry) * scale
            fees_exit_out[i] = float(f_exit) * scale
        else:
            fees_entry_out[i] = 0.0
            fees_exit_out[i] = 0.0

    zero = np.zeros(n, dtype=float)
    return pd.DataFrame(
        {
            "fees": fees_out.astype(float),
            "fees_entry": fees_entry_out.astype(float),
            "fees_exit": fees_exit_out.astype(float),
            "borrow_cost": zero,
            "slippage_cost": zero,
            "impact_cost": zero,
            "total_costs": fees_out.astype(float),
        },
        index=trades_df.index,
    )


# =============================================================================
#                                  __all__
# =============================================================================

__all__ = [
    # Params
    "ExecParams",
    "DEFAULT_EXECUTION_PARAMS",
    "exec_params_from_cfg",
    # Core slippage/costs
    "square_root_impact",
    "calc_adv_slippage",
    "calc_trade_cost",
    "calc_pair_slippage_pct",
    "partial_fill_probability",
    "simulate_execution",
    "SimulationResult",
    # Legacy & post-trade
    "compute_post_lob_costs",
    # Neues vektorisiertes Gebühren-API
    "normalize_fee_schedule_from_cfg",
    "compute_costs",
    # Types (optional export)
    "FeeCfg",
    "VenueFeeProfile",
    "FeeSchedule",
]
