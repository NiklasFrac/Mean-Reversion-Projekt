# src/backtest/scripts/verify_borrow_inputs.py
#!/usr/bin/env python3
"""
Borrow Inputs Verifier (Inline/YAML)
===================================
Validates borrow inputs defined directly in the backtest YAML (no files).

Supported keys:
  - borrow.per_asset_rate_annual: {SYMBOL: rate_annual, ...} or
    borrow.per_asset_rates:       [{"symbol": "...", "rate_annual": ...}, ...]
  - borrow.rates:                 [{"date": "...", "symbol": "...", "rate_annual": ...}, ...]  (long)
  - borrow.rate_series_by_symbol: {SYMBOL: {date: rate_annual, ...} | [{"date": "...", "rate_annual": ...}, ...]}
  - borrow.availability:          [{"date": "...", "symbol": "...", "available": 0/1}, ...]

Legacy (disabled): borrow.*_csv / borrow.availability_path

In addition, data.prices_path is loaded to check coverage against the price symbols.

Usage (repo root):
  python src/backtest/scripts/verify_borrow_inputs.py
Optional:
  BACKTEST_CONFIG=runs/configs/config_backtest.yaml python src/backtest/scripts/verify_borrow_inputs.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pandas as pd

from backtest.utils.tz import coerce_series_to_tz, to_naive_day

try:
    pass
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required: pip install pyyaml") from e


from backtest.utils.common.io import load_yaml_dict as _load_yaml_dict

RATE_MIN, RATE_MAX = 0.0001, 0.50  # 1 bps .. 50% (wide, conservative)
PER_ASSET_MIN, PER_ASSET_MAX = 0.0025, 0.10  # 25 bps .. 10%


def _status(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


def _discover_config() -> Path | None:
    for env in ("BACKTEST_CONFIG", "STRAT_CONFIG"):
        v = os.environ.get(env)
        if v:
            p = Path(v)
            if p.exists():
                return p
    for cand in (
        Path("runs/configs/config_backtest.yaml"),
        Path("backtest/configs/strat_new.yaml"),
        Path("configs/config_backtest.yaml"),
        Path("config_backtest.yaml"),
    ):
        if cand.exists():
            return cand
    return None


def _load_yaml(p: Path) -> dict[str, Any]:
    return _load_yaml_dict(p)


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve(base: Path, p: str | None) -> Path | None:
    if not p:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp)


def _load_prices(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".pkl", ".pickle"):
        obj = pd.read_pickle(path)
        if not isinstance(obj, pd.DataFrame):
            raise ValueError("Pickle does not contain a DataFrame")
        df = obj
    elif suf == ".parquet":
        df = pd.read_parquet(path)
    elif suf == ".csv":
        df = pd.read_csv(path)
        if "date" in df.columns:
            df = df.set_index("date")
        idx_utc = coerce_series_to_tz(
            pd.Series(df.index), "UTC", naive_is_utc=True, utc_hint="auto"
        )
        df.index = pd.DatetimeIndex(idx_utc)
    else:
        raise ValueError(f"Unsupported price file extension: {suf}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Prices must have a DatetimeIndex")
    if df.empty:
        raise ValueError("Prices are empty")
    return df.sort_index()


def _norm_dates(s: pd.Series) -> pd.Series:
    return to_naive_day(pd.to_datetime(s, errors="coerce"))


def _parse_per_asset_any(obj: Any) -> dict[str, float]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        out: dict[str, float] = {}
        for k, v in obj.items():
            sym = str(k or "").strip().upper()
            if not sym:
                continue
            vv = pd.to_numeric(v, errors="coerce")
            if pd.notna(vv):
                out[sym] = float(vv)
        return out
    if isinstance(obj, list):
        out_list: dict[str, float] = {}
        for row in obj:
            if not isinstance(row, dict):
                continue
            sym = (
                str(row.get("symbol") or row.get("ticker") or row.get("asset") or "")
                .strip()
                .upper()
            )
            if not sym:
                continue
            vv = pd.to_numeric(
                cast(Any, row.get("rate_annual", row.get("rate"))), errors="coerce"
            )
            if pd.notna(vv):
                out_list[sym] = float(vv)
        return out_list
    return {}


def check_per_asset(per_asset_obj: Any, prices: pd.DataFrame) -> bool:
    d = _parse_per_asset_any(per_asset_obj)
    if not d:
        _status("OK", "per-asset: not set (optional).")
        return True

    ok = True
    rates = pd.Series(d, dtype=float)
    rmin, rmed, rmax = float(rates.min()), float(rates.median()), float(rates.max())
    _status(
        "OK",
        f"per-asset: rows={len(rates)}, min/med/max={rmin:.4f}/{rmed:.4f}/{rmax:.4f}",
    )

    if not ((rates >= PER_ASSET_MIN) & (rates <= PER_ASSET_MAX)).all():
        bad = int((~((rates >= PER_ASSET_MIN) & (rates <= PER_ASSET_MAX))).sum())
        _status(
            "WARN",
            f"{bad} rates outside the expected range [{PER_ASSET_MIN:.4f},{PER_ASSET_MAX:.4f}].",
        )
        ok = False

    missing = set(prices.columns.astype(str)) - set(rates.index.astype(str))
    if missing:
        sample = sorted(list(missing))[:5]
        _status(
            "WARN",
            f"{len(missing)} symbols are missing from the per-asset set (e.g. {sample} ...)",
        )
        ok = False

    return ok


def _parse_rates_long(obj: Any) -> pd.DataFrame:
    if obj is None or not isinstance(obj, list):
        return pd.DataFrame(columns=["date", "symbol", "rate_annual"])
    rows = [r for r in obj if isinstance(r, dict)]
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "rate_annual"])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "rate_annual"])
    cols = {c.lower(): c for c in df.columns.astype(str)}
    date_col = (
        cols.get("date") or cols.get("day") or cols.get("ts") or cols.get("timestamp")
    )
    sym_col = cols.get("symbol") or cols.get("ticker") or cols.get("asset")
    rate_col = (
        cols.get("rate_annual")
        or cols.get("rate")
        or cols.get("borrow_rate_annual")
        or cols.get("borrow_rate")
    )
    if not (date_col and sym_col and rate_col):
        return pd.DataFrame(columns=["date", "symbol", "rate_annual"])
    out = df[[date_col, sym_col, rate_col]].copy()
    out.columns = ["date", "symbol", "rate_annual"]
    out["date"] = _norm_dates(out["date"])
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["rate_annual"] = pd.to_numeric(out["rate_annual"], errors="coerce")
    out = out.dropna(subset=["date", "symbol", "rate_annual"])
    return out


def _parse_rate_series_by_symbol(obj: Any) -> dict[str, pd.Series]:
    if obj is None or not isinstance(obj, dict):
        return {}
    out: dict[str, pd.Series] = {}
    for sym0, v in obj.items():
        sym = str(sym0 or "").strip().upper()
        if not sym:
            continue
        if isinstance(v, dict):
            s = pd.Series(
                list(v.values()), index=_norm_dates(pd.Series(list(v.keys())))
            )
        elif isinstance(v, list):
            dates: list[Any] = []
            vals: list[Any] = []
            for row in v:
                if isinstance(row, dict):
                    dates.append(row.get("date", row.get("day", row.get("ts"))))
                    vals.append(row.get("rate_annual", row.get("rate")))
                elif isinstance(row, (list, tuple)) and len(row) >= 2:
                    dates.append(row[0])
                    vals.append(row[1])
            s = pd.Series(vals, index=_norm_dates(pd.Series(dates)))
        else:
            continue
        s = pd.to_numeric(s, errors="coerce")
        s = s.loc[~s.index.isna()].dropna().astype(float).sort_index()
        if not s.empty:
            out[sym] = s.groupby(level=0).last()
    return out


def check_rates(
    *,
    rates_long_obj: Any,
    rate_series_by_symbol_obj: Any,
    prices: pd.DataFrame,
    default_rate: float,
) -> bool:
    ok = True
    df_long = _parse_rates_long(rates_long_obj)
    rs_map = _parse_rate_series_by_symbol(rate_series_by_symbol_obj)

    if df_long.empty and not rs_map:
        _status(
            "OK",
            f"rates: not set (falling back to default_rate_annual={default_rate:.4f}).",
        )
        if default_rate <= 0:
            _status(
                "WARN",
                "default_rate_annual is 0 => borrow_cost will effectively always be 0.",
            )
            ok = False
        return ok

    if not df_long.empty:
        vals = df_long["rate_annual"]
        _status("OK", f"rates (long): rows={len(df_long)}")
        if vals.isna().any():
            _status("FAIL", "NaN in borrow.rates.rate_annual")
            ok = False
        if not ((vals >= RATE_MIN) & (vals <= RATE_MAX)).all():
            _status("WARN", "borrow.rates contains values outside [1bp, 50%].")
            ok = False
        miss_sym = set(prices.columns.astype(str)) - set(df_long["symbol"].astype(str))
        if miss_sym:
            sample = sorted(list(miss_sym))[:5]
            _status(
                "WARN",
                f"borrow.rates is missing {len(miss_sym)} symbols (e.g. {sample} ...)",
            )
            ok = False

    if rs_map:
        lens = [len(s) for s in rs_map.values()]
        _status(
            "OK",
            f"rate_series_by_symbol: symbols={len(rs_map)}, median_len={int(pd.Series(lens).median()) if lens else 0}",
        )
        all_vals = (
            pd.concat([s for s in rs_map.values()], ignore_index=True)
            if rs_map
            else pd.Series(dtype=float)
        )
        if (
            not all_vals.empty
            and not ((all_vals >= RATE_MIN) & (all_vals <= RATE_MAX)).all()
        ):
            _status(
                "WARN",
                "borrow.rate_series_by_symbol contains values outside [1bp, 50%].",
            )
            ok = False
        miss_sym = set(prices.columns.astype(str)) - set(rs_map.keys())
        if miss_sym:
            sample = sorted(list(miss_sym))[:5]
            _status(
                "WARN",
                f"rate_series_by_symbol is missing {len(miss_sym)} symbols (e.g. {sample} ...)",
            )
            ok = False

    return ok


def _parse_availability(obj: Any) -> pd.DataFrame:
    if obj is None or not isinstance(obj, list):
        return pd.DataFrame(columns=["date", "symbol", "available"])
    rows = [r for r in obj if isinstance(r, dict)]
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "available"])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "available"])
    cols = {c.lower(): c for c in df.columns.astype(str)}
    date_col = (
        cols.get("date")
        or cols.get("day")
        or cols.get("dt")
        or cols.get("ts")
        or cols.get("timestamp")
    )
    sym_col = cols.get("symbol") or cols.get("ticker") or cols.get("asset")
    av_col = (
        cols.get("available")
        or cols.get("avail")
        or cols.get("is_available")
        or cols.get("borrowable")
    )
    if not (date_col and sym_col and av_col):
        return pd.DataFrame(columns=["date", "symbol", "available"])
    out = df[[date_col, sym_col, av_col]].copy()
    out.columns = ["date", "symbol", "available"]
    out["date"] = _norm_dates(out["date"])
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    av = pd.to_numeric(out["available"], errors="coerce")
    if av.isna().all():
        av = (
            out["available"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": 1,
                    "t": 1,
                    "yes": 1,
                    "y": 1,
                    "1": 1,
                    "false": 0,
                    "f": 0,
                    "no": 0,
                    "n": 0,
                    "0": 0,
                }
            )
            .fillna(0)
        )
    out["available"] = av.fillna(0.0).astype(float)
    out = out.dropna(subset=["date", "symbol"])
    return out


def check_availability(av_obj: Any, prices: pd.DataFrame) -> bool:
    df = _parse_availability(av_obj)
    if df.empty:
        _status("OK", "availability: not set (optional, only for enforcement).")
        return True

    ok = True
    vals = pd.to_numeric(df["available"], errors="coerce")
    if vals.isna().any():
        _status("FAIL", "NaN in borrow.availability.available")
        ok = False
    uniq = set(vals.dropna().astype(int).unique().tolist())
    if not uniq.issubset({0, 1}):
        _status("FAIL", f"availability contains values outside {{0,1}}: {sorted(uniq)}")
        ok = False
    expected = prices.shape[0] * prices.shape[1]
    if len(df) != expected:
        _status(
            "WARN",
            f"availability size {len(df)} != expected {expected} (dates*assets)",
        )
        ok = False
    share0 = float((vals == 0).mean())
    _status("OK", f"availability: rows={len(df)}, share_unavailable={share0:.2%}")
    try:
        full_block = int((df.groupby("symbol")["available"].sum() == 0).sum())
        if full_block > 0:
            _status("WARN", f"{full_block} symbols are unavailable on all days.")
            ok = False
    except Exception:
        pass
    return ok


def main() -> int:
    print("=== Borrow Inputs Verifier (Inline) ===")
    cfgp = _discover_config()
    if not cfgp:
        _status(
            "FAIL",
            "No configuration file found (ENV or runs/configs/config_backtest.yaml)",
        )
        return 2

    print(f"Config: {cfgp}")
    cfg = _load_yaml(cfgp)
    base = Path.cwd()

    prices_path = _resolve(base, _get(cfg, "data.prices_path"))
    if not prices_path or not prices_path.exists():
        _status("FAIL", f"prices_path is invalid: {prices_path}")
        return 2

    default_rate = float(_get(cfg, "borrow.default_rate_annual", 0.0) or 0.0)
    print(f"Prices:   {prices_path}")
    print(
        f"defaults: default_rate_annual={default_rate:.4f}, day_basis={_get(cfg, 'borrow.day_basis', 252)}"
    )

    try:
        prices = _load_prices(prices_path)
    except Exception as e:
        _status("FAIL", f"Failed to load prices: {e}")
        return 2

    print(f"Prices shape: dates={prices.shape[0]}, assets={prices.shape[1]}")

    ok_all = True
    ok_all &= check_per_asset(
        _get(cfg, "borrow.per_asset_rate_annual", _get(cfg, "borrow.per_asset_rates")),
        prices,
    )
    ok_all &= check_rates(
        rates_long_obj=_get(cfg, "borrow.rates"),
        rate_series_by_symbol_obj=_get(cfg, "borrow.rate_series_by_symbol"),
        prices=prices,
        default_rate=default_rate,
    )
    ok_all &= check_availability(
        _get(cfg, "borrow.availability", _get(cfg, "borrow.availability_long")), prices
    )

    print("-")
    if ok_all:
        _status("PASS", "Borrow inputs look consistent and plausible.")
        return 0
    _status("WARN", "There are issues to review. See details above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
