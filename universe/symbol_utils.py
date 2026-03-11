from __future__ import annotations

from typing import Any, Iterable

from universe.checkpoint import norm_symbol


def normalize_symbols(symbols: Iterable[Any], *, unique: bool = False) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = norm_symbol(raw)
        if not sym:
            continue
        if unique:
            if sym in seen:
                continue
            seen.add(sym)
        out.append(sym)
    return out
