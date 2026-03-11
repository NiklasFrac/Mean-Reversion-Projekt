"""Small utility helpers used across analysis modules."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


def _guard(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@contextmanager
def _stage(timings: dict[str, float], name: str) -> Iterator[None]:
    t0 = time.time()
    try:
        yield
    finally:
        timings[name] = round(time.time() - t0, 3)


def _pct_label(thr: float) -> str:
    s = f"{thr:.3f}".rstrip("0").rstrip(".")
    return f"pct>={s}"


def _canon_pair(a: Any, b: Any) -> tuple[str, str]:
    a_s, b_s = str(a).strip(), str(b).strip()
    _guard(a_s != b_s, f"Pair legs identical: {a_s}")
    if a_s > b_s:
        a_s, b_s = b_s, a_s
    return a_s, b_s


def parse_pair(pair_in: Any) -> tuple[str, str]:
    if isinstance(pair_in, (list, tuple)) and len(pair_in) >= 2:
        return _canon_pair(pair_in[0], pair_in[1])
    if isinstance(pair_in, str):
        s = pair_in.strip()
        for sep in ("-", "/", ",", ":"):
            if sep in s:
                a, b = s.split(sep, 1)
                return _canon_pair(a, b)
        parts = s.split()
        if len(parts) >= 2:
            return _canon_pair(parts[0], parts[1])
    raise ValueError(f"Cannot parse pair identifier into two tickers: {pair_in}")


__all__ = ["_guard", "_stage", "_pct_label", "_canon_pair", "parse_pair"]
