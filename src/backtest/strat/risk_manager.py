from __future__ import annotations

import math
from typing import Any

from backtest.risk_policy import build_risk_policy, is_short_leg


def _norm_sym(value: Any) -> str | None:
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


class RiskManager:
    def __init__(
        self, initial_capital: float, cfg: dict[str, Any] | None = None
    ) -> None:
        self.cfg = dict(cfg or {})
        self.policy = build_risk_policy(
            risk_cfg=self.cfg,
            backtest_cfg=self.cfg,
            execution_cfg=self.cfg,
        )
        self.short_availability_heuristic = self.policy.short_heuristic
        self.current_capital = max(0.0, _safe_float(initial_capital, 0.0))
        self._open_pairs: list[dict[str, Any]] = []

    def update_capital(self, capital: float) -> None:
        cap = _safe_float(capital, self.current_capital)
        if cap > 0.0:
            self.current_capital = float(cap)

    def _cap_base(self) -> float:
        return max(1e-9, float(self.current_capital))

    def _gross_exposure(self) -> float:
        return float(sum(abs(p["ny"]) + abs(p["nx"]) for p in self._open_pairs))

    def _net_exposure(self) -> float:
        return float(sum(p["ny"] + p["nx"] for p in self._open_pairs))

    def _per_name_gross(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for rec in self._open_pairs:
            y_sym = rec.get("y_sym")
            x_sym = rec.get("x_sym")
            if y_sym:
                out[str(y_sym)] = float(
                    out.get(str(y_sym), 0.0) + abs(float(rec["ny"]))
                )
            if x_sym:
                out[str(x_sym)] = float(
                    out.get(str(x_sym), 0.0) + abs(float(rec["nx"]))
                )
        return out

    def _positions_per_symbol(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self._open_pairs:
            seen: set[str] = set()
            for sym in (_norm_sym(rec.get("y_sym")), _norm_sym(rec.get("x_sym"))):
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
    ) -> dict[str, Any]:
        y_sym, x_sym = (_norm_sym(leg_symbols[0]), _norm_sym(leg_symbols[1]))
        ny = _safe_float(leg_notionals[0], 0.0)
        nx = _safe_float(leg_notionals[1], 0.0)
        return {
            "pair": str(pair),
            "y_sym": y_sym,
            "x_sym": x_sym,
            "ny": float(ny),
            "nx": float(nx),
        }

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
        pol = self.short_availability_heuristic
        if not bool(getattr(pol, "enabled", False)):
            return ""
        min_price = float(getattr(pol, "min_price", 0.0))
        min_adv_usd = float(getattr(pol, "min_adv_usd", 0.0))
        needs_price = min_price > 0.0
        needs_adv = min_adv_usd > 0.0

        for sym, notional, units, price, adv in zip(
            leg_symbols,
            leg_notionals,
            leg_units,
            leg_entry_prices,
            leg_adv_usd,
            strict=False,
        ):
            if not is_short_leg(
                signed_notional=_safe_float(notional), units=_safe_float(units)
            ):
                continue
            price_f = _safe_float(price, float("nan"))
            adv_f = _safe_float(adv, float("nan"))
            if needs_price and not math.isfinite(price_f):
                if block_on_missing:
                    return f"short_price_missing:{_norm_sym(sym) or 'UNKNOWN'}"
            elif needs_price and price_f < min_price:
                return f"short_price:{_norm_sym(sym) or 'UNKNOWN'}"

            if needs_adv and not math.isfinite(adv_f):
                if block_on_missing and bool(getattr(pol, "block_on_missing", True)):
                    return f"short_adv_missing:{_norm_sym(sym) or 'UNKNOWN'}"
            elif needs_adv and adv_f < min_adv_usd:
                return f"short_adv:{_norm_sym(sym) or 'UNKNOWN'}"
        return ""

    def can_open_pair(
        self,
        pair: str,
        leg_symbols: tuple[str | None, str | None],
        leg_notionals: tuple[float, float],
    ) -> bool:
        exp = self.policy.exposure
        sizing = self.policy.sizing
        cap = self._cap_base()
        rec = self._record_for_pair(pair, leg_symbols, leg_notionals)
        gross_add = abs(rec["ny"]) + abs(rec["nx"])
        if exp.strict and (gross_add <= 0.0 or not math.isfinite(gross_add)):
            return False
        if gross_add <= 0.0:
            return True

        if exp.max_open_positions is not None and len(self._open_pairs) >= int(
            exp.max_open_positions
        ):
            return False

        if exp.max_positions_per_symbol is not None:
            counts = self._positions_per_symbol()
            seen: set[str] = set()
            for sym in (rec["y_sym"], rec["x_sym"]):
                if sym is None or sym in seen:
                    continue
                if int(counts.get(sym, 0)) + 1 > int(exp.max_positions_per_symbol):
                    return False
                seen.add(sym)

        if (
            float(sizing.max_trade_pct) > 0.0
            and gross_add > float(sizing.max_trade_pct) * cap
        ):
            return False

        if self._gross_exposure() + gross_add > float(exp.max_gross_exposure) * cap:
            return False

        net_after = self._net_exposure() + float(rec["ny"]) + float(rec["nx"])
        if abs(net_after) > float(exp.max_net_exposure) * cap:
            return False

        if exp.max_per_name_pct is not None:
            per_name = self._per_name_gross()
            if (
                rec["y_sym"] is not None
                and float(per_name.get(rec["y_sym"], 0.0) + abs(rec["ny"]))
                > float(exp.max_per_name_pct) * cap
            ):
                return False
            if (
                rec["x_sym"] is not None
                and float(per_name.get(rec["x_sym"], 0.0) + abs(rec["nx"]))
                > float(exp.max_per_name_pct) * cap
            ):
                return False

        return True

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
            if str(rec.get("pair")) != pair_s:
                continue
            if leg_symbols is not None:
                want = (_norm_sym(leg_symbols[0]), _norm_sym(leg_symbols[1]))
                have = (_norm_sym(rec.get("y_sym")), _norm_sym(rec.get("x_sym")))
                if want != have:
                    continue
            rec_idx = i
            break
        if rec_idx is None:
            return

        self._open_pairs.pop(rec_idx)

    def can_open(self, pair: str, notional: float) -> bool:
        signed = _safe_float(notional, 0.0)
        return self.can_open_pair(pair, (None, None), (signed, 0.0))

    def register_open(self, pair: str, signed_notional: float = 0.0) -> None:
        self.register_open_pair(pair, (None, None), (signed_notional, 0.0))
