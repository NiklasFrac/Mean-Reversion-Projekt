from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger("download")


def run_download(cfg_path: str | Path) -> None:
    cfg = _load_cfg(cfg_path)
    symbols, dropped = _read_symbols(cfg)
    raw, missing = _download_close(symbols, cfg["download"])
    dropped += [{"ticker": t, "reason": "download_missing"} for t in missing]
    filled, quality_drops = _quality(raw, cfg["quality"])
    dropped += quality_drops

    _write_prices(raw, cfg["output"]["raw_close"])
    _write_drops(dropped, cfg["output"]["dropped_tickers"])
    if filled.shape[1] < int(cfg["quality"]["min_final_tickers"]):
        raise ValueError(f"Only {filled.shape[1]} final tickers after quality")
    _write_prices(filled, cfg["output"]["filled_close"])
    logger.info(
        "done raw=%d final=%d dropped=%d", raw.shape[1], filled.shape[1], len(dropped)
    )


def _load_cfg(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key in ("input", "download", "quality", "output"):
        if key not in cfg:
            raise ValueError(f"Missing config section: {key}")
    for key in ("screener_path", "symbol_col"):
        if not cfg["input"].get(key):
            raise ValueError(f"Missing input.{key}")
    for key in ("start", "end"):
        datetime.strptime(str(cfg["download"].get(key, "")), "%Y-%m-%d")
    for key in ("raw_close", "filled_close", "dropped_tickers"):
        if not cfg["output"].get(key):
            raise ValueError(f"Missing output.{key}")
    return cfg


def _read_symbols(cfg: dict) -> tuple[list[str], list[dict[str, str]]]:
    inp = cfg["input"]
    df = pd.read_csv(inp["screener_path"], dtype=str)
    if inp["symbol_col"] not in df:
        raise ValueError(f"Missing symbol column: {inp['symbol_col']}")
    df["_ticker"] = df[inp["symbol_col"]].map(_symbol)
    df = df[df["_ticker"].ne("")]

    dropped: list[dict[str, str]] = []
    df, etf_drops = _apply_etf_filter(df, inp.get("etf_filter") or {})
    dropped += etf_drops
    df, security_drops = _apply_security_filter(
        df, inp.get("security_filter") or {}
    )
    dropped += security_drops

    return list(dict.fromkeys(df["_ticker"])), dropped


def _symbol(value) -> str:
    return str(value).strip().upper().replace(".", "-").replace("/", "-")


def _find_etf_col(columns) -> str | None:
    names = {"etf", "is_etf", "fund", "is_fund", "type", "security_type"}
    by_lower = {str(c).lower(): c for c in columns}
    for name in names:
        if name in by_lower:
            return by_lower[name]
    return None


def _apply_etf_filter(
    df: pd.DataFrame, filt: dict
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    if not filt.get("enabled", False):
        return df, []
    col = filt.get("column") or _find_etf_col(df.columns)
    if col is None:
        logger.warning("ETF filter skipped: no hard ETF column found")
        return df, []
    values = {str(v).strip().lower() for v in filt.get("exclude_values", [])}
    mask = df[col].fillna("").str.strip().str.lower().isin(values)
    return _drop_masked(df, mask, "etf_filtered")


def _apply_security_filter(
    df: pd.DataFrame, filt: dict
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    if not filt.get("enabled", False):
        return df, []

    mask = pd.Series(False, index=df.index)
    name_patterns = filt.get("exclude_name_patterns") or []
    if name_patterns:
        name_col = filt.get("name_column") or "Name"
        if name_col not in df:
            logger.warning("security filter skipped name patterns: missing %s", name_col)
        else:
            mask |= _regex_mask(df[name_col], name_patterns, "exclude_name_patterns")

    symbol_patterns = filt.get("exclude_symbol_patterns") or []
    if symbol_patterns:
        mask |= _regex_mask(df["_ticker"], symbol_patterns, "exclude_symbol_patterns")

    for col, values in (filt.get("exclude_column_values") or {}).items():
        if col not in df:
            logger.warning("security filter skipped column values: missing %s", col)
            continue
        blocked = {str(v).strip().lower() for v in values}
        mask |= df[col].fillna("").str.strip().str.lower().isin(blocked)

    return _drop_masked(df, mask, "security_filtered")


def _regex_mask(series: pd.Series, patterns: list[str], config_key: str) -> pd.Series:
    out = pd.Series(False, index=series.index)
    text = series.fillna("").astype(str)
    for pattern in patterns:
        try:
            out |= text.str.contains(pattern, case=False, regex=True, na=False)
        except re.error as exc:
            raise ValueError(f"Invalid regex in security_filter.{config_key}: {pattern}") from exc
    return out


def _drop_masked(
    df: pd.DataFrame, mask: pd.Series, reason: str
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    dropped = [{"ticker": t, "reason": reason} for t in df.loc[mask, "_ticker"]]
    return df.loc[~mask], dropped


def _download_close(symbols: list[str], cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    got, missing = [], []
    size = int(cfg.get("batch_size", 100))
    total_batches = (len(symbols) + size - 1) // size
    for batch_no, i in enumerate(range(0, len(symbols), size), start=1):
        batch = symbols[i : i + size]
        data = pd.DataFrame()
        for attempt in range(int(cfg.get("retries", 2)) + 1):
            try:
                data = yf.download(
                    batch,
                    start=cfg["start"],
                    end=cfg["end"],
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    group_by="column",
                )
                if not data.empty:
                    break
            except Exception:
                if attempt >= int(cfg.get("retries", 2)):
                    raise
            time.sleep(float(cfg.get("pause_seconds", 1.0)))
        close = _close_frame(data, batch)
        got.append(close)
        missing += [t for t in batch if t not in close.columns]
        logger.info(
            "batch %d/%d processed=%d/%d loaded=%d missing_total=%d",
            batch_no,
            total_batches,
            min(i + size, len(symbols)),
            len(symbols),
            close.shape[1],
            len(missing),
        )
    raw = pd.concat(got, axis=1) if got else pd.DataFrame()
    raw = raw.loc[:, ~raw.columns.duplicated()]
    raw.index = pd.to_datetime(raw.index, utc=True).normalize()
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    return raw, missing


def _close_frame(data: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        else:
            close = data.xs("Close", axis=1, level=1)
    elif "Close" in data:
        close = data[["Close"]].rename(columns={"Close": batch[0]})
    else:
        close = data
    close.columns = [str(c).upper() for c in close.columns]
    return close.reindex(columns=[t for t in batch if t in close.columns])


def _quality(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    keep, dropped = {}, []
    min_cov = float(cfg["min_coverage"])
    max_gap = int(cfg["max_gap"])
    for ticker in raw.columns:
        s = pd.to_numeric(raw[ticker], errors="coerce")
        reason = _drop_reason(s, min_cov, max_gap)
        if reason:
            dropped.append({"ticker": ticker, "reason": reason})
        else:
            keep[ticker] = s.ffill()
    return pd.DataFrame(keep, index=raw.index), dropped


def _drop_reason(s: pd.Series, min_cov: float, max_gap: int) -> str | None:
    if s.isna().all():
        return "all_nan"
    if (s.dropna() <= 0).any():
        return "nonpositive_price"
    if s.notna().mean() < min_cov:
        return "coverage_below_threshold"
    if pd.isna(s.iloc[0]) or pd.isna(s.iloc[-1]):
        return "edge_gap"
    if _max_gap(s) > max_gap:
        return "gap_too_large"
    if s.ffill().isna().any():
        return "remaining_nan_after_fill"
    return None


def _max_gap(s: pd.Series) -> int:
    mask = s.isna()
    if not mask.any():
        return 0
    groups = mask.ne(mask.shift()).cumsum()
    return int(mask.groupby(groups).sum().max())


def _write_prices(df: pd.DataFrame, path: str | Path) -> None:
    out = df.copy()
    out.index = out.index.strftime("%Y-%m-%d")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index_label="date")


def _write_drops(rows: list[dict[str, str]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["ticker", "reason"]).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="runs/configs/config_download.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    run_download(args.cfg)


if __name__ == "__main__":
    main()
