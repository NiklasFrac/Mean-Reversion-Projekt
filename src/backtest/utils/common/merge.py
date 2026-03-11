from __future__ import annotations

from typing import Any, Mapping


def deep_merge(a: Mapping[str, Any], b: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
