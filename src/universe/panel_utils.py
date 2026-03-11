from __future__ import annotations

import pandas as pd


def coerce_utc_naive_index(index: pd.Index, *, normalize: bool) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        try:
            idx = idx.tz_convert("UTC")
        except Exception:
            pass
        idx = idx.tz_localize(None)
    if normalize:
        idx = idx.normalize()
    return idx


def collapse_duplicate_index_rows(df: pd.DataFrame) -> pd.DataFrame:
    if not df.index.duplicated().any():
        return df
    # GroupBy.last keeps the last non-null per column for each timestamp.
    return df.groupby(level=0).last()


def merge_duplicate_columns_prefer_non_null(df: pd.DataFrame) -> pd.DataFrame:
    cols_idx = pd.Index(df.columns)
    if not cols_idx.duplicated().any():
        return df
    merged = pd.DataFrame(index=df.index)
    for col in cols_idx.unique():
        same = df.loc[:, cols_idx == col]
        if isinstance(same, pd.Series):
            same = same.to_frame()
        if same.shape[1] == 1:
            merged[col] = same.iloc[:, 0]
            continue
        # Keep the first non-null value across duplicate columns so retry
        # columns can repair NaN-only originals deterministically.
        merged_col = same.iloc[:, 0]
        for pos in range(1, same.shape[1]):
            miss = merged_col.isna()
            if miss.any():
                merged_col = merged_col.where(~miss, same.iloc[:, pos])
        merged[col] = merged_col
    return merged
