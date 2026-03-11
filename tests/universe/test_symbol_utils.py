from __future__ import annotations

from universe.symbol_utils import normalize_symbols


def test_normalize_symbols_preserves_order_and_optionally_dedupes():
    src = ["aaa", "BBB", "aaa", "brk.b"]
    assert normalize_symbols(src, unique=False) == ["AAA", "BBB", "AAA", "BRK-B"]
    assert normalize_symbols(src, unique=True) == ["AAA", "BBB", "BRK-B"]
