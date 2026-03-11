from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class QualityIssue:
    code: str
    message: str
    severity: str  # "info" | "warn" | "error"
    examples: list[dict[str, Any]]


def _latest_run_outputs_dir(root: Path) -> Path:
    by_run = root / "runs" / "data" / "by_run"
    if not by_run.is_dir():
        raise FileNotFoundError(f"Run-scoped output root not found: {by_run}")
    candidates = [
        p / "outputs"
        for p in by_run.iterdir()
        if p.is_dir()
        and (p / "outputs").is_dir()
        and (p / "outputs" / "universe_manifest.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No run-scoped outputs found under {by_run}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate repository root from {start} (expected pyproject.toml + runs/)."
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a JSON object at top level.")
    return payload


def _load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} did not contain a DataFrame (got {type(df)})")
    return df


def _load_volume(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} did not contain a DataFrame (got {type(df)})")
    return df


def _as_utc_ts(epoch_seconds: float | int | None) -> str | None:
    if epoch_seconds is None or (
        isinstance(epoch_seconds, float) and math.isnan(epoch_seconds)
    ):
        return None
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).isoformat()


def _field_panel(
    prices: pd.DataFrame, tickers: Iterable[str], field: str
) -> FloatArray:
    cols = [f"{t}_{field}" for t in tickers]
    missing = [c for c in cols if c not in prices.columns]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} expected OHLC columns, e.g. {missing[:5]}"
        )
    return np.asarray(prices.loc[:, cols].to_numpy(dtype="float64", copy=False))


def _nan_fraction(arr: FloatArray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.isnan(arr).mean())


def _safe_log_returns(close: FloatArray) -> FloatArray:
    # close shape: (n_dates, n_tickers)
    # returns shape: (n_dates - 1, n_tickers)
    prev = close[:-1, :]
    nxt = close[1:, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(nxt / prev)
    r[~np.isfinite(r)] = np.nan
    return np.asarray(r, dtype=np.float64)


def _top_examples(
    mask: FloatArray | npt.NDArray[np.bool_],
    tickers: list[str],
    index: pd.DatetimeIndex,
    open_: FloatArray | None = None,
    high: FloatArray | None = None,
    low: FloatArray | None = None,
    close: FloatArray | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    where = np.argwhere(mask)
    examples: list[dict[str, Any]] = []
    for i, (row, col) in enumerate(where):
        if i >= limit:
            break
        ex: dict[str, Any] = {"ticker": tickers[int(col)], "ts": str(index[int(row)])}
        if open_ is not None:
            ex["open"] = float(open_[int(row), int(col)])
        if high is not None:
            ex["high"] = float(high[int(row), int(col)])
        if low is not None:
            ex["low"] = float(low[int(row), int(col)])
        if close is not None:
            ex["close"] = float(close[int(row), int(col)])
        examples.append(ex)
    return examples


def check_outputs(outputs_dir: Path) -> tuple[dict[str, Any], list[QualityIssue]]:
    manifest = _read_json(outputs_dir / "universe_manifest.json")
    run_id = manifest.get("run_id")

    p_tickers_final = outputs_dir / "tickers_final.txt"
    p_tickers_csv = outputs_dir / "tickers_universe.csv"
    p_tickers_ext = outputs_dir / "tickers_universe_ext.csv"
    p_prices = outputs_dir / "raw_prices.pkl"
    p_volume = outputs_dir / "raw_volume.pkl"
    p_fund = outputs_dir / "fundamentals_universe.parquet"
    p_adv_filtered = outputs_dir / "adv_map_usd_filtered.csv"

    tickers_final = [
        t.strip()
        for t in p_tickers_final.read_text(encoding="utf-8").splitlines()
        if t.strip()
    ]
    tickers_csv = pd.read_csv(p_tickers_csv)["ticker"].astype(str).tolist()
    tickers = sorted(set(tickers_final))

    issues: list[QualityIssue] = []

    if len(tickers_final) != len(tickers):
        issues.append(
            QualityIssue(
                code="TICKERS_DUPLICATES",
                severity="error",
                message=f"tickers_final.txt contains duplicates: {len(tickers_final)} lines but {len(tickers)} unique.",
                examples=[{"duplicates": len(tickers_final) - len(tickers)}],
            )
        )

    if sorted(tickers_csv) != tickers:
        missing_in_csv = sorted(set(tickers) - set(tickers_csv))
        extra_in_csv = sorted(set(tickers_csv) - set(tickers))
        issues.append(
            QualityIssue(
                code="TICKERS_MISMATCH",
                severity="error",
                message="tickers_universe.csv and tickers_final.txt disagree.",
                examples=[
                    {
                        "missing_in_csv": missing_in_csv[:25],
                        "extra_in_csv": extra_in_csv[:25],
                    },
                ],
            )
        )

    prices = _load_prices(p_prices)
    volume = _load_volume(p_volume)

    if not isinstance(prices.index, pd.DatetimeIndex) or not isinstance(
        volume.index, pd.DatetimeIndex
    ):
        issues.append(
            QualityIssue(
                code="INDEX_NOT_DATETIME",
                severity="error",
                message="Price/volume indices are expected to be DatetimeIndex.",
                examples=[
                    {
                        "prices_index": str(type(prices.index)),
                        "volume_index": str(type(volume.index)),
                    },
                ],
            )
        )

    # Index sanity / vendor quirks (time-of-day, weekends)
    price_idx = prices.index
    vol_idx = volume.index
    if not price_idx.is_monotonic_increasing or not price_idx.is_unique:
        issues.append(
            QualityIssue(
                code="PRICE_INDEX_ORDER",
                severity="error",
                message="raw_prices.pkl index must be unique and sorted ascending.",
                examples=[
                    {
                        "is_monotonic": bool(price_idx.is_monotonic_increasing),
                        "is_unique": bool(price_idx.is_unique),
                    }
                ],
            )
        )
    if not vol_idx.is_monotonic_increasing or not vol_idx.is_unique:
        issues.append(
            QualityIssue(
                code="VOLUME_INDEX_ORDER",
                severity="error",
                message="raw_volume.pkl index must be unique and sorted ascending.",
                examples=[
                    {
                        "is_monotonic": bool(vol_idx.is_monotonic_increasing),
                        "is_unique": bool(vol_idx.is_unique),
                    }
                ],
            )
        )
    if not price_idx.equals(vol_idx):
        issues.append(
            QualityIssue(
                code="INDEX_MISMATCH",
                severity="error",
                message="Price and volume indices must match exactly.",
                examples=[
                    {
                        "prices_min": str(price_idx.min()),
                        "prices_max": str(price_idx.max()),
                    },
                    {
                        "volume_min": str(vol_idx.min()),
                        "volume_max": str(vol_idx.max()),
                    },
                ],
            )
        )
    tod = price_idx.strftime("%H:%M:%S").value_counts()
    if len(tod) > 1:
        issues.append(
            QualityIssue(
                code="MIXED_TIME_OF_DAY",
                severity="warn",
                message="Date index has multiple times-of-day (timezone/DST quirk).",
                examples=[{"time_of_day_counts": tod.head(5).to_dict()}],
            )
        )
    norm_dates = price_idx.normalize()
    norm_counts = norm_dates.value_counts()
    n_dates_multiple_rows = int((norm_counts > 1).sum())
    if n_dates_multiple_rows:
        # This is a serious alignment issue: when panels are combined, multiple timestamps for the
        # same "day" create near-empty duplicate rows and inflate missingness ~50%.
        row_nan = prices.isna().mean(axis=1)
        sparse = prices.loc[row_nan > 0.99]
        sparse_tickers: list[str] = []
        if not sparse.empty:
            active_cols = sparse.notna().sum(axis=0)
            active_cols = active_cols[active_cols > 0].index.astype(str)
            sparse_tickers = sorted({c.rsplit("_", 1)[0] for c in active_cols})

        # Diagnostic: if we collapse per normalized day, how much missingness remains?
        # (This is what downstream consumers typically want for daily research.)
        def _combine_group(g: pd.DataFrame) -> pd.Series:
            if len(g) == 1:
                return g.iloc[0]
            out = g.iloc[0]
            for i in range(1, len(g)):
                out = out.combine_first(g.iloc[i])
            return out

        prices_clean = (
            prices.sort_index().groupby(norm_dates, sort=True).apply(_combine_group)
        )
        prices_clean.index = pd.DatetimeIndex(prices_clean.index)
        close_cols = [
            c for c in prices_clean.columns.astype(str) if c.endswith("_close")
        ]
        close_cov_min = float("nan")
        close_cov_full = None
        if close_cols:
            close_cov = 1.0 - prices_clean[close_cols].isna().mean(axis=0)
            close_cov_min = float(close_cov.min())
            close_cov_full = int((close_cov == 1.0).sum())

        sample_days = norm_counts[norm_counts > 1].head(5).index.tolist()
        examples: list[dict[str, Any]] = []
        for day in sample_days:
            rows = prices.loc[norm_dates == day]
            rows_nan = row_nan.loc[norm_dates == day]
            examples.append(
                {
                    "day": str(pd.Timestamp(day).date()),
                    "timestamps": [str(ts) for ts in rows.index.tolist()],
                    "row_nan_frac": [float(x) for x in rows_nan.tolist()],
                }
            )
        if sparse_tickers:
            examples.append(
                {
                    "tickers_with_data_in_sparse_rows": sparse_tickers,
                    "note": "These tickers likely have a different timezone/metadata source causing index misalignment.",
                }
            )
        examples.append(
            {
                "collapsed_daily_rows": int(prices_clean.shape[0]),
                "collapsed_close_min_coverage": close_cov_min,
                "collapsed_close_n_full_coverage": close_cov_full,
            }
        )
        issues.append(
            QualityIssue(
                code="DUPLICATE_DATES",
                severity="error",
                message=(
                    "Found multiple rows per trading day (index differs only by time-of-day); "
                    "this indicates timezone normalization mismatch across tickers and makes the "
                    "raw panels not directly usable without collapsing to daily dates."
                ),
                examples=examples,
            )
        )
    weekend = price_idx[price_idx.dayofweek >= 5]
    if len(weekend) > 0:
        issues.append(
            QualityIssue(
                code="WEEKEND_ROWS",
                severity="warn",
                message="Found weekend rows in price panel (unexpected for daily US equities).",
                examples=[
                    {
                        "n_weekend_rows": int(len(weekend)),
                        "sample": [str(x) for x in weekend[:5]],
                    }
                ],
            )
        )

    # Columns / ticker alignment
    price_tickers = sorted({c.rsplit("_", 1)[0] for c in prices.columns})
    if price_tickers != tickers:
        issues.append(
            QualityIssue(
                code="PRICE_TICKERS_MISMATCH",
                severity="error",
                message="raw_prices.pkl columns do not match tickers_final.txt.",
                examples=[
                    {
                        "missing_in_prices": sorted(set(tickers) - set(price_tickers))[
                            :25
                        ],
                        "extra_in_prices": sorted(set(price_tickers) - set(tickers))[
                            :25
                        ],
                    }
                ],
            )
        )
    if sorted(volume.columns.astype(str).tolist()) != tickers:
        issues.append(
            QualityIssue(
                code="VOLUME_TICKERS_MISMATCH",
                severity="error",
                message="raw_volume.pkl columns do not match tickers_final.txt.",
                examples=[
                    {
                        "missing_in_volume": sorted(
                            set(tickers) - set(volume.columns.astype(str))
                        )[:25],
                        "extra_in_volume": sorted(
                            set(volume.columns.astype(str)) - set(tickers)
                        )[:25],
                    }
                ],
            )
        )

    # Core OHLC checks
    tickers_list = tickers  # sorted list
    open_ = _field_panel(prices, tickers_list, "open")
    high = _field_panel(prices, tickers_list, "high")
    low = _field_panel(prices, tickers_list, "low")
    close = _field_panel(prices, tickers_list, "close")

    nan_open = _nan_fraction(open_)
    nan_high = _nan_fraction(high)
    nan_low = _nan_fraction(low)
    nan_close = _nan_fraction(close)
    nan_ohlc = nan_open + nan_high + nan_low + nan_close
    if nan_ohlc > 0.0:
        # Note: we still compute deeper stats below.
        issues.append(
            QualityIssue(
                code="NAN_OHLC_PRESENT",
                severity="warn",
                message="OHLC panel contains NaNs.",
                examples=[
                    {
                        "nan_frac_open": nan_open,
                        "nan_frac_high": nan_high,
                        "nan_frac_low": nan_low,
                        "nan_frac_close": nan_close,
                    }
                ],
            )
        )
    if nan_close > 0.05:
        issues.append(
            QualityIssue(
                code="EXCESSIVE_NANS_CLOSE",
                severity="error",
                message="Close panel NaN share is too high for a research-grade daily dataset.",
                examples=[{"nan_frac_close": nan_close}],
            )
        )

    neg_price = (open_ < 0) | (high < 0) | (low < 0) | (close < 0)
    n_neg_price = int(np.count_nonzero(neg_price))
    if n_neg_price:
        issues.append(
            QualityIssue(
                code="NEGATIVE_PRICES",
                severity="error",
                message=f"Found {n_neg_price} negative OHLC values.",
                examples=_top_examples(
                    neg_price,
                    tickers_list,
                    price_idx,
                    open_=open_,
                    high=high,
                    low=low,
                    close=close,
                ),
            )
        )

    # OHLC invariants: low <= open/close <= high and low <= high
    # Use a small tolerance to avoid flagging floating rounding noise.
    eps = 1e-8
    high_lt_low = high < (low - eps)
    open_outside = (open_ < (low - eps)) | (open_ > (high + eps))
    close_outside = (close < (low - eps)) | (close > (high + eps))
    n_viol = int(np.count_nonzero(high_lt_low | open_outside | close_outside))
    if n_viol:
        issues.append(
            QualityIssue(
                code="OHLC_VIOLATIONS",
                severity="warn",
                message=f"Found {n_viol} OHLC invariant violations (vendor quirks / bad bars).",
                examples=_top_examples(
                    (high_lt_low | open_outside | close_outside),
                    tickers_list,
                    price_idx,
                    open_=open_,
                    high=high,
                    low=low,
                    close=close,
                ),
            )
        )

    # Return outliers (close-to-close)
    rets = _safe_log_returns(close)
    max_abs_ret = (
        float(np.nanmax(np.abs(rets)))
        if np.isfinite(np.nanmax(np.abs(rets)))
        else float("nan")
    )
    extreme = np.abs(rets) > math.log(2.0)  # > ~100% move (close-to-close)
    n_extreme = int(np.count_nonzero(extreme))
    if n_extreme:
        issues.append(
            QualityIssue(
                code="EXTREME_RETURNS",
                severity="warn",
                message=f"Found {n_extreme} close-to-close moves > 100% (log-return threshold).",
                examples=_top_examples(
                    extreme, tickers_list, price_idx[1:], close=close[1:, :]
                ),
            )
        )

    # Volume sanity
    vol = volume.loc[:, tickers_list].to_numpy(dtype="float64", copy=False)
    nan_vol = _nan_fraction(vol)
    neg_vol = vol < 0
    n_neg_vol = int(np.count_nonzero(neg_vol))
    if n_neg_vol:
        issues.append(
            QualityIssue(
                code="NEGATIVE_VOLUME",
                severity="error",
                message=f"Found {n_neg_vol} negative volume values.",
                examples=_top_examples(neg_vol, tickers_list, vol_idx, limit=10),
            )
        )
    zero_vol = vol == 0
    n_zero_vol = int(np.count_nonzero(zero_vol))
    if n_zero_vol:
        issues.append(
            QualityIssue(
                code="ZERO_VOLUME_ROWS",
                severity="warn",
                message=f"Found {n_zero_vol} zero-volume rows (halts/holidays/vendor fill).",
                examples=_top_examples(zero_vol, tickers_list, vol_idx, limit=10),
            )
        )
    mask = ~np.isnan(vol)
    frac_nonint = (
        float(np.mean(~np.isclose(vol[mask], np.round(vol[mask]), atol=1e-6, rtol=0)))
        if bool(mask.any())
        else 0.0
    )
    if frac_nonint > 0.0:
        issues.append(
            QualityIssue(
                code="NON_INTEGER_VOLUME",
                severity="warn",
                message="Volume contains non-integer values (should typically be integer shares).",
                examples=[{"fraction_non_integer": frac_nonint}],
            )
        )

    # CSV sanity for extended tickers
    ext = pd.read_csv(p_tickers_ext)
    if ext.shape[0] != len(tickers_list):
        issues.append(
            QualityIssue(
                code="TICKERS_EXT_ROWCOUNT",
                severity="error",
                message="tickers_universe_ext.csv rowcount does not match tickers_final.txt.",
                examples=[
                    {
                        "tickers_ext_rows": int(ext.shape[0]),
                        "tickers_final": len(tickers_list),
                    }
                ],
            )
        )
    if "ticker" in ext.columns:
        ext_tickers = ext["ticker"].astype(str).tolist()
        if sorted(ext_tickers) != tickers_list:
            issues.append(
                QualityIssue(
                    code="TICKERS_EXT_MISMATCH",
                    severity="error",
                    message="tickers_universe_ext.csv tickers do not match tickers_final.txt.",
                    examples=[
                        {
                            "missing_in_ext": sorted(
                                set(tickers_list) - set(ext_tickers)
                            )[:25],
                            "extra_in_ext": sorted(
                                set(ext_tickers) - set(tickers_list)
                            )[:25],
                        }
                    ],
                )
            )

    # Range checks on core columns if present
    if "float_pct" in ext.columns:
        bad_float = (ext["float_pct"].astype(float) < 0) | (
            ext["float_pct"].astype(float) > 1.05
        )
        if bool(bad_float.any()):
            bad = (
                ext.loc[bad_float, ["ticker", "float_pct"]]
                .head(10)
                .to_dict(orient="records")
            )
            issues.append(
                QualityIssue(
                    code="FLOAT_PCT_RANGE",
                    severity="warn",
                    message="float_pct outside [0, 1] (allowing small tolerance).",
                    examples=bad,
                )
            )
    for col in [
        "price",
        "market_cap",
        "volume",
        "free_float_shares",
        "free_float_mcap",
    ]:
        if col in ext.columns:
            s = pd.to_numeric(ext[col], errors="coerce")
            if bool((s < 0).any()):
                bad = ext.loc[s < 0, ["ticker", col]].head(10).to_dict(orient="records")
                issues.append(
                    QualityIssue(
                        code=f"NEGATIVE_{col.upper()}",
                        severity="error",
                        message=f"{col} contains negative values.",
                        examples=bad,
                    )
                )

    # Fundamentals: core missingness / negativity on the source snapshot
    fund = pd.read_parquet(p_fund)
    core_fund_cols = ["price", "market_cap", "volume", "dollar_adv", "float_pct"]
    missing_core = {}
    for col in core_fund_cols:
        if col in fund.columns:
            missing_core[col] = float(fund[col].isna().mean())
    if any(v > 0.0 for v in missing_core.values()):
        issues.append(
            QualityIssue(
                code="FUNDAMENTALS_MISSINGNESS",
                severity="warn",
                message="Fundamentals snapshot has missing core fields (expected for some tickers).",
                examples=[missing_core],
            )
        )
    for col in ["price", "market_cap", "volume", "dollar_adv"]:
        if col in fund.columns:
            s = pd.to_numeric(fund[col], errors="coerce")
            if bool((s < 0).any()):
                bad = fund.loc[s < 0, [col]].head(10)
                neg_examples: list[dict[str, Any]] = [
                    {"ticker": str(idx), col: float(val)}
                    for idx, val in bad[col].items()
                ]
                issues.append(
                    QualityIssue(
                        code=f"FUNDAMENTALS_NEG_{col.upper()}",
                        severity="error",
                        message=f"Fundamentals contains negative {col}.",
                        examples=neg_examples,
                    )
                )

    # ADV filtered: enforce threshold and no NaNs
    adv = pd.read_csv(p_adv_filtered)
    if bool(adv.isna().any().any()):
        issues.append(
            QualityIssue(
                code="ADV_NANS",
                severity="warn",
                message="adv_map_usd_filtered.csv contains NaNs.",
                examples=[adv.isna().mean().to_dict()],
            )
        )
    if "dollar_adv_hist" in adv.columns:
        below = adv["dollar_adv_hist"].astype(float) < 1_000_000.0
        if bool(below.any()):
            examples = (
                adv.loc[below, ["ticker", "dollar_adv_hist"]]
                .head(10)
                .to_dict(orient="records")
            )
            issues.append(
                QualityIssue(
                    code="ADV_BELOW_THRESHOLD",
                    severity="error",
                    message="Filtered ADV contains values below configured threshold (1,000,000).",
                    examples=examples,
                )
            )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "outputs_dir": str(outputs_dir),
        "manifest_timestamp": manifest.get("timestamp"),
        "data_policy": manifest.get("extra", {}).get("data_policy", {}),
        "counts": {
            "n_tickers_final": len(tickers_list),
            "prices_shape": tuple(prices.shape),
            "volume_shape": tuple(volume.shape),
            "fundamentals_shape": tuple(fund.shape),
            "adv_filtered_shape": tuple(adv.shape),
        },
        "index": {
            "start": str(price_idx.min()),
            "end": str(price_idx.max()),
            "n_rows": int(len(price_idx)),
            "time_of_day_counts_top5": tod.head(5).to_dict(),
            "unique_days": int(norm_dates.nunique()),
            "days_with_multiple_rows": n_dates_multiple_rows,
        },
        "ohlc": {
            "nan_frac_open": nan_open,
            "nan_frac_high": nan_high,
            "nan_frac_low": nan_low,
            "nan_frac_close": nan_close,
            "n_ohlc_violations": n_viol,
            "max_abs_log_return": max_abs_ret,
            "n_extreme_returns_gt_100pct": n_extreme,
        },
        "volume": {
            "nan_frac": nan_vol,
            "n_negative": n_neg_vol,
            "n_zero": n_zero_vol,
            "fraction_non_integer": frac_nonint,
        },
        "fundamentals_missingness": missing_core,
        "manifest_git_commit": manifest.get("git_commit"),
        "manifest_cfg_hash": manifest.get("cfg_hash"),
        "manifest_cfg_path": manifest.get("cfg_path"),
        "manifest_monitoring": manifest.get("monitoring"),
        "manifest_unadjusted_coverage": manifest.get("extra", {}).get(
            "unadjusted_download"
        ),
        "manifest_cache_info": {
            "fundamentals_cache_mtime_utc": manifest.get("extra", {})
            .get("fundamentals_provenance", {})
            .get("cache_store_mtime_utc"),
            "fundamentals_cache_ttl_days": manifest.get("extra", {})
            .get("fundamentals_provenance", {})
            .get("cache_ttl_days"),
            "vendor_cache_root": manifest.get("extra", {})
            .get("data_policy", {})
            .get("vendor", None),
        },
    }

    return summary, issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quality checks for run-scoped universe outputs."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Path to a run outputs folder (…/runs/data/by_run/<run>/outputs). Defaults to latest run.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write JSON report.",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__))
    outputs_dir = args.outputs_dir or _latest_run_outputs_dir(repo_root)
    summary, issues = check_outputs(outputs_dir)

    report = {
        "summary": summary,
        "issues": [
            {
                "code": i.code,
                "severity": i.severity,
                "message": i.message,
                "examples": i.examples,
            }
            for i in issues
        ],
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
