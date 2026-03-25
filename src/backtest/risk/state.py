from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .policy import (
    RiskExposureSnapshot,
    RiskPolicy,
    build_risk_policy,
    check_pair_admission,
    short_availability_reason,
)

__all__ = ["PortfolioRiskState"]


def _norm_symbol(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    return s or None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


@dataclass(frozen=True)
class _OpenPairRecord:
    pair: str
    y_sym: str | None
    x_sym: str | None
    ny: float
    nx: float


class PortfolioRiskState:
    def __init__(
        self, initial_capital: float, cfg: Mapping[str, Any] | None = None
    ) -> None:
        cfg_dict = dict(cfg or {})
        self.policy: RiskPolicy = build_risk_policy(
            risk_cfg=cfg_dict,
            backtest_cfg=cfg_dict,
            execution_cfg=cfg_dict,
        )
        self.current_capital = max(0.0, _safe_float(initial_capital, 0.0))
        self._open_pairs: list[_OpenPairRecord] = []

    def update_capital(self, capital: float) -> None:
        cap = _safe_float(capital, self.current_capital)
        if cap > 0.0:
            self.current_capital = float(cap)

    def _cap_base(self) -> float:
        return max(1e-9, float(self.current_capital))

    def _gross_exposure(self) -> float:
        return float(sum(abs(rec.ny) + abs(rec.nx) for rec in self._open_pairs))

    def _net_exposure(self) -> float:
        return float(sum(rec.ny + rec.nx for rec in self._open_pairs))

    def _per_name_gross(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for rec in self._open_pairs:
            if rec.y_sym:
                out[rec.y_sym] = float(out.get(rec.y_sym, 0.0) + abs(rec.ny))
            if rec.x_sym:
                out[rec.x_sym] = float(out.get(rec.x_sym, 0.0) + abs(rec.nx))
        return out

    def _positions_per_symbol(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self._open_pairs:
            seen: set[str] = set()
            for sym in (rec.y_sym, rec.x_sym):
                if sym is None or sym in seen:
                    continue
                out[sym] = int(out.get(sym, 0) + 1)
                seen.add(sym)
        return out

    def _record_for_pair(
        self,
        pair: str,
        leg_symbols: tuple[str | None, str | None],
        leg_notionals: tuple[float, float],
    ) -> _OpenPairRecord:
        y_sym, x_sym = (_norm_symbol(leg_symbols[0]), _norm_symbol(leg_symbols[1]))
        ny = _safe_float(leg_notionals[0], 0.0)
        nx = _safe_float(leg_notionals[1], 0.0)
        return _OpenPairRecord(
            pair=str(pair),
            y_sym=y_sym,
            x_sym=x_sym,
            ny=float(ny),
            nx=float(nx),
        )

    def short_availability_pair_reason(
        self,
        *,
        leg_symbols: tuple[str | None, str | None],
        leg_notionals: tuple[float, float],
        leg_units: tuple[float, float],
        leg_entry_prices: tuple[float | None, float | None],
        leg_adv_usd: tuple[float | None, float | None],
        block_on_missing: bool = True,
    ) -> str:
        return short_availability_reason(
            heuristic=self.policy.short_heuristic,
            leg_symbols=leg_symbols,
            leg_notionals=leg_notionals,
            leg_units=leg_units,
            leg_entry_prices=leg_entry_prices,
            leg_adv_usd=leg_adv_usd,
            block_on_missing=block_on_missing,
        )

    def _snapshot(self) -> RiskExposureSnapshot:
        return RiskExposureSnapshot(
            open_positions=len(self._open_pairs),
            gross_exposure=self._gross_exposure(),
            net_exposure=self._net_exposure(),
            per_name_gross=self._per_name_gross(),
            positions_per_symbol=self._positions_per_symbol(),
        )

    def can_open_pair(
        self,
        pair: str,
        leg_symbols: tuple[str | None, str | None],
        leg_notionals: tuple[float, float],
    ) -> bool:
        del pair
        return check_pair_admission(
            policy=self.policy,
            capital=self._cap_base(),
            snapshot=self._snapshot(),
            leg_symbols=leg_symbols,
            leg_notionals=leg_notionals,
        )

    def register_open_pair(
        self,
        pair: str,
        leg_symbols: tuple[str | None, str | None],
        leg_notionals: tuple[float, float],
    ) -> None:
        self._open_pairs.append(self._record_for_pair(pair, leg_symbols, leg_notionals))

    def register_close_pair(
        self,
        pair: str,
        leg_symbols: tuple[str | None, str | None] | None = None,
        leg_notionals: tuple[float, float] | None = None,
    ) -> None:
        del leg_notionals
        pair_s = str(pair)
        rec_idx = None
        for i, rec in enumerate(self._open_pairs):
            if rec.pair != pair_s:
                continue
            if leg_symbols is not None:
                want = (_norm_symbol(leg_symbols[0]), _norm_symbol(leg_symbols[1]))
                have = (rec.y_sym, rec.x_sym)
                if want != have:
                    continue
            rec_idx = i
            break
        if rec_idx is None:
            return
        self._open_pairs.pop(rec_idx)
