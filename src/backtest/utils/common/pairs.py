from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "parse_pair_symbols",
    "normalize_pairs_input",
]


_PAIR_SEPS: tuple[str, ...] = ("::", "/", "-", "_", "|", ":", " ")


def _norm_sym(val: Any, *, upper: bool) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    return s.upper() if upper else s


def parse_pair_symbols(
    pair_val: Any, *, upper: bool = True
) -> tuple[str | None, str | None]:
    """
    Parse a pair string into (left, right) symbols.
    Supports multiple separators; returns (symbol, None) when no separator is found.
    """
    if pair_val is None:
        return None, None
    s = str(pair_val).strip()
    if not s:
        return None, None
    for sep in _PAIR_SEPS:
        if sep in s:
            left_raw, right_raw = s.split(sep, 1)
            left = _norm_sym(left_raw, upper=upper)
            right = _norm_sym(right_raw, upper=upper)
            return left, right
    sym = _norm_sym(s, upper=upper)
    return (sym, None) if sym else (None, None)


def normalize_pairs_input(obj: Any, *, upper: bool = True) -> dict[str, dict[str, str]]:
    """
    Normalize various input forms into:
        dict(pair -> {'t1':..., 't2':...})
    """
    out: dict[str, dict[str, str]] = {}

    def _from_value(v: Any) -> tuple[str | None, str | None]:
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return _norm_sym(v[0], upper=upper), _norm_sym(v[1], upper=upper)
        if isinstance(v, Mapping):
            for a_k, b_k in (
                ("t1", "t2"),
                ("y", "x"),
                ("left", "right"),
                ("asset1", "asset2"),
                ("t1_ticker", "t2_ticker"),
            ):
                a = v.get(a_k)
                b = v.get(b_k)
                if a is not None and b is not None:
                    return _norm_sym(a, upper=upper), _norm_sym(b, upper=upper)
        if isinstance(v, str):
            return parse_pair_symbols(v, upper=upper)
        return None, None

    if isinstance(obj, Mapping):
        for k, v in obj.items():
            t1, t2 = _from_value(v)
            if (t1 is None or t2 is None) and isinstance(k, str):
                kt1, kt2 = parse_pair_symbols(k, upper=upper)
                t1 = t1 or kt1
                t2 = t2 or kt2
            if t1 and t2:
                out[str(k)] = {"t1": t1, "t2": t2}
        return out

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for i, elt in enumerate(obj):
            t1, t2 = _from_value(elt)
            if t1 and t2:
                out[f"pair_{i}"] = {"t1": t1, "t2": t2}
        return out

    return out
