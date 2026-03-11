from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from universe.checkpoint import norm_symbol

UniverseSchema = Literal["symbols", "interval"]


def _canon_symbol(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    raw = str(value).strip()
    if not raw:
        return ""
    return norm_symbol(raw)


def _as_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(cast(Any, value))
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _resolve_schema(columns: list[str], schema: str) -> UniverseSchema:
    lowered = {c.strip().lower() for c in columns}
    if schema != "auto":
        if schema not in {"symbols", "interval"}:
            raise ValueError(f"Unsupported universe schema: {schema}")
        return cast(UniverseSchema, schema)
    if {"symbol", "start_date", "end_date"}.issubset(lowered):
        return "interval"
    if "ticker" in lowered or "symbol" in lowered:
        return "symbols"
    raise ValueError("Could not infer universe schema from columns.")


def _prepare_col_for_nan_mask(out: pd.DataFrame, col: object) -> None:
    """Upcast dtypes that cannot represent NaN before masking operations."""
    dtype = out[col].dtype
    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
        out[col] = out[col].astype("float64")


def load_universe(
    path: str | Path, *, schema: str = "auto"
) -> tuple[UniverseSchema, pd.DataFrame]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Universe file not found: {src}")

    df = pd.read_csv(src)
    if df.empty:
        resolved = _resolve_schema(list(df.columns), schema)
        return resolved, df

    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key == "ticker":
            rename[col] = "symbol"
        elif key in {"symbol", "start_date", "end_date"}:
            rename[col] = key
    if rename:
        df = df.rename(columns=rename)

    resolved = _resolve_schema(list(df.columns), schema)

    if resolved == "symbols":
        if "symbol" not in df.columns:
            raise ValueError("symbols schema requires column 'symbol' or 'ticker'.")
        out = pd.DataFrame({"symbol": df["symbol"].map(_canon_symbol)})
        out = (
            out[out["symbol"] != ""]
            .drop_duplicates(subset=["symbol"])
            .reset_index(drop=True)
        )
        return resolved, out

    if not {"symbol", "start_date", "end_date"}.issubset(df.columns):
        raise ValueError("interval schema requires columns: symbol,start_date,end_date")
    out = df[["symbol", "start_date", "end_date"]].copy()
    out["symbol"] = out["symbol"].map(_canon_symbol)
    out = out[out["symbol"] != ""]
    out["start_date"] = out["start_date"].map(_as_timestamp)
    out["end_date"] = out["end_date"].map(_as_timestamp)
    out = out.sort_values(["symbol", "start_date"], na_position="first").reset_index(
        drop=True
    )
    return resolved, out


def apply_universe(
    prices: pd.DataFrame, schema: UniverseSchema, universe_df: pd.DataFrame
) -> pd.DataFrame:
    if prices is None:
        raise ValueError("prices must be a DataFrame")
    out = prices.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.sort_index()

    idx = pd.to_datetime(out.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx
    nan_ready_cols: set[object] = set()

    def _ensure_nan_ready(col: object) -> None:
        if col in nan_ready_cols:
            return
        _prepare_col_for_nan_mask(out, col)
        nan_ready_cols.add(col)

    cols_by_symbol: dict[str, list[object]] = {}
    for col in out.columns:
        cols_by_symbol.setdefault(_canon_symbol(col), []).append(col)

    if schema == "symbols":
        allowed_symbols = {
            _canon_symbol(sym)
            for sym in universe_df.get("symbol", pd.Series(dtype="string")).tolist()
            if _canon_symbol(sym)
        }
        for col in out.columns:
            if _canon_symbol(col) not in allowed_symbols:
                _ensure_nan_ready(col)
                out.loc[:, col] = float("nan")
        return out

    if schema != "interval":
        raise ValueError(f"Unsupported universe schema: {schema}")

    # Interval schema should still enforce symbol membership first:
    # symbols absent from the universe definition must be fully masked.
    allowed_symbols = {
        _canon_symbol(sym)
        for sym in universe_df.get("symbol", pd.Series(dtype="string")).tolist()
        if _canon_symbol(sym)
    }
    for col in out.columns:
        if _canon_symbol(col) not in allowed_symbols:
            _ensure_nan_ready(col)
            out.loc[:, col] = float("nan")

    for symbol, rows in universe_df.groupby("symbol", dropna=False):
        sym = _canon_symbol(symbol)
        if not sym:
            continue
        symbol_cols = cols_by_symbol.get(sym, [])
        if not symbol_cols:
            continue
        allowed_mask = pd.Series(False, index=out.index)
        for _, row in rows.iterrows():
            start_ts = _as_timestamp(row.get("start_date"))
            end_ts = _as_timestamp(row.get("end_date"))
            mask = pd.Series(True, index=out.index)
            if start_ts is not None:
                mask &= out.index >= start_ts
            if end_ts is not None:
                mask &= out.index <= end_ts
            allowed_mask |= mask
        for col in symbol_cols:
            _ensure_nan_ready(col)
            out.loc[~allowed_mask, col] = float("nan")
    return out
