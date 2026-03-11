from __future__ import annotations

import pandas as pd

__all__ = ["validate_prices_wide"]


def validate_prices_wide(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    if df.empty:
        return {"checks": {"rows": 0, "cols": 0}}
    numeric = df.apply(pd.to_numeric, errors="coerce")
    checks: dict[str, int] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "nonpositive_prices": int((numeric <= 0).sum().sum()) if numeric.size else 0,
        "duplicate_index": int(df.index.duplicated().sum()),
        # Non-numeric payloads are treated as malformed and counted as missing.
        "na_total": int(numeric.isna().sum().sum()),
        "monotonic_index": int(df.index.is_monotonic_increasing),
    }
    return {"checks": checks}
