from __future__ import annotations

import datetime as dt
import json
import logging
import os
import platform
import shutil as _sh
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from universe.coercion import cfg_bool
from universe.fs_atomic import atomic_replace_or_raise
from universe.monitoring import logger, prom_set_final
from universe.storage import ArtifactPaths, resolve_artifact_paths

__all__ = [
    "atomic_write_text",
    "copy_optional",
    "git_commit",
    "now_utc_iso",
    "persist_universe_run_artifacts",
    "write_tickers_final_txt",
    "write_manifest",
    "write_markdown_report",
    "write_universe_csv",
    "write_universe_ext_csv",
]


def now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.tz_convert("UTC").isoformat()
        return value.isoformat()
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(str(v) for v in value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def git_commit() -> str | None:
    try:
        rev = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return rev or None
    except Exception:
        return None


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _atomic_replace(tmp, path)


def _atomic_replace(tmp: Path, dst: Path, *, attempts: int = 8) -> None:
    atomic_replace_or_raise(
        tmp,
        dst,
        attempts=attempts,
        replace_fn=os.replace,
        sleep_fn=time.sleep,
    )


def copy_optional(src: Path | None, dst: Path, *, label: str) -> None:
    log = logging.getLogger("runner_universe")
    if src is None:
        return
    try:
        if not src.exists():
            log.warning("Copy skipped (%s missing): %s", label, src)
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        _sh.copy2(str(src), str(dst))
    except Exception as e:
        log.warning("Copy failed for %s (%s -> %s): %s", label, src, dst, e)


def write_csv_atomic(
    rows: Iterable[Iterable[Any]], path: Path, header: list[str] | None = None
) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)
    _atomic_replace(tmp, path)


def write_universe_csv(df: pd.DataFrame, path: Path) -> None:
    rows = ([t] for t in df.index.tolist())
    write_csv_atomic(rows, path, header=["ticker"])
    logger.info("Wrote tickers: %s (n=%d)", path, len(df))
    prom_set_final(len(df))


def write_universe_ext_csv(
    df_universe: pd.DataFrame,
    df_fundamentals: pd.DataFrame,
    path: Path,
    *,
    adv_window: int | None = None,
) -> None:
    cols = [
        "ticker",
        "stable_id",
        "currency",
        "issuer_id",
        "price",
        "price_snapshot",
        "price_filter_basis",
        "market_cap",
        "volume",
        "float_pct",
        "float_quality",
        "free_float_shares",
        "free_float_mcap",
        "dollar_adv_filter_value",
        "dollar_adv_snapshot",
        "dollar_adv_filter_basis",
        "dividend",
        "is_etf",
        "is_etn",
        "is_baby_bond",
        "is_adr",
        "is_trust",
        "sector",
        "industry",
        "country",
        "shares_out",
        "exchange_code",
        "market",
        "quote_type",
        "adv_window_used",
    ]
    if df_universe.empty:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=cols).to_csv(tmp, index=False)
        _atomic_replace(tmp, path)
        logger.info("Wrote extended tickers: %s (n=0)", path)
        return

    idx = df_universe.index.astype(str)
    meta = df_fundamentals.reindex(idx)

    def _meta(col: str) -> pd.Series:
        if col in meta.columns:
            return meta[col]
        return pd.Series([pd.NA] * len(idx), index=idx)

    def _universe(col: str) -> pd.Series:
        if col in df_universe.columns:
            return df_universe[col]
        return pd.Series([pd.NA] * len(idx), index=idx)

    def _issuer(row: pd.Series) -> str:
        raw_name = row.get("long_name")
        name = (
            str(raw_name).strip()
            if raw_name is not None and not pd.isna(raw_name)
            else ""
        )
        if not name:
            name = str(row.name)
        raw_country = row.get("country")
        country = (
            str(raw_country).strip()
            if raw_country is not None and not pd.isna(raw_country)
            else "unknown"
        )
        return f"{name.lower()} | {country.lower()}"

    out = pd.DataFrame(index=idx)
    out["ticker"] = idx
    out["stable_id"] = idx.map(lambda x: f"YF:{x}")
    out["currency"] = _meta("currency")
    if {"long_name", "country"}.issubset(meta.columns) and not meta.empty:
        out["issuer_id"] = meta.apply(_issuer, axis=1)
    else:
        out["issuer_id"] = pd.Series(
            [f"{sym.lower()} | unknown" for sym in idx], index=idx
        )
    price_snapshot = pd.to_numeric(_universe("price"), errors="coerce")
    price_filter = pd.to_numeric(
        _universe("price_eff")
        if "price_eff" in df_universe.columns
        else price_snapshot,
        errors="coerce",
    )
    price_basis = pd.Series("snapshot", index=idx, dtype="string")
    if "price_warmup_med" in df_universe.columns:
        warm_mask = pd.to_numeric(
            _universe("price_warmup_med"), errors="coerce"
        ).notna()
        price_basis = price_basis.mask(warm_mask, "warmup_median")
    out["price"] = price_filter
    out["price_snapshot"] = price_snapshot
    out["price_filter_basis"] = price_basis.astype(str)
    out["market_cap"] = _universe("market_cap")
    out["volume"] = (
        _universe("volume_eff")
        if "volume_eff" in df_universe.columns
        else _universe("volume")
    )
    out["float_pct"] = _universe("float_pct")
    out["float_quality"] = _meta("float_quality")
    out["free_float_shares"] = _universe("free_float_shares")
    out["free_float_mcap"] = _universe("free_float_mcap")
    dadv_snapshot = pd.to_numeric(_universe("dollar_adv"), errors="coerce")
    dadv_filter = pd.to_numeric(
        _universe("dollar_adv_eff")
        if "dollar_adv_eff" in df_universe.columns
        else dadv_snapshot,
        errors="coerce",
    )
    dadv_basis = pd.Series("snapshot", index=idx, dtype="string")
    if "dollar_adv_hist" in df_universe.columns:
        hist_mask = pd.to_numeric(_universe("dollar_adv_hist"), errors="coerce").notna()
        dadv_basis = dadv_basis.mask(hist_mask, "warmup_hist")
    out["dollar_adv_filter_value"] = dadv_filter
    out["dollar_adv_snapshot"] = dadv_snapshot
    out["dollar_adv_filter_basis"] = dadv_basis.astype(str)
    out["dividend"] = _universe("dividend").fillna(False)
    out["is_etf"] = _universe("is_etf").fillna(False)

    def _bool_meta(col: str) -> pd.Series:
        series = pd.Series(_meta(col), dtype="boolean")
        return series.fillna(False)

    out["is_etn"] = _bool_meta("is_etn")
    out["is_baby_bond"] = _bool_meta("is_baby_bond")
    out["is_adr"] = _bool_meta("is_adr")
    out["is_trust"] = _bool_meta("is_trust")
    out["sector"] = _universe("sector")
    out["industry"] = _universe("industry")
    out["country"] = _universe("country")
    out["shares_out"] = _universe("shares_out")
    out["exchange_code"] = _meta("exchange_code")
    out["market"] = _meta("market")
    out["quote_type"] = _meta("quote_type")
    out["adv_window_used"] = adv_window if adv_window is not None else ""

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(tmp, index=False)
    _atomic_replace(tmp, path)
    logger.info("Wrote extended tickers: %s (n=%d)", path, len(out))


def write_manifest(
    manifest_path: Path,
    cfg_path: Path,
    cfg_hash: str | None,
    run_id: str,
    n_initial: int,
    n_final: int,
    monitoring: Mapping[str, Any],
    extra: Mapping[str, Any] | dict[str, Any],
    *,
    schema_version: str,
) -> None:
    payload = {
        "schema_version": schema_version,
        "timestamp": now_utc_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "cfg_path": str(cfg_path),
        "cfg_hash": cfg_hash,
        "git_commit": git_commit(),
        "run_id": run_id,
        "n_tickers_initial": int(n_initial),
        "n_tickers_final": int(n_final),
        "monitoring": monitoring,
        "extra": extra,
    }
    atomic_write_text(
        manifest_path,
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
    )
    logger.info("Wrote manifest: %s", manifest_path)


def write_tickers_final_txt(path: Path, tickers: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join([str(t).strip().upper() for t in tickers if str(t).strip()]) + "\n"
    atomic_write_text(path, text)


def _nan_share(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    if df is None or df.empty:
        return out
    for c in cols:
        if c in df.columns:
            out[c] = float(pd.to_numeric(df[c], errors="coerce").isna().mean())
    return out


def persist_universe_run_artifacts(
    *,
    cfg_path: Path,
    cfg_hash: str | None,
    run_id: str,
    universe_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    out_tickers: Path,
    out_tickers_ext: Path | None,
    out_manifest: Path,
    adv_csv: Path | None,
    adv_csv_filtered: Path | None,
    tickers_final: list[str],
    df_fundamentals: pd.DataFrame,
    df_universe: pd.DataFrame,
    monitoring: Mapping[str, Any],
    stats: Mapping[str, Any],
    artifact_paths: ArtifactPaths | None = None,
) -> None:
    """
    Best-effort persistence for auditability:
    - Optional immutable run-scoped outputs under runs/data/by_run/<run_id>_<cfg_hash8>/

    This function should never raise; it logs warnings instead.
    """
    try:
        paths = artifact_paths or resolve_artifact_paths(
            universe_cfg=universe_cfg,
            runtime_cfg=runtime_cfg,
            data_cfg=data_cfg,
        )
        screener_path: Path | None = None
        try:
            from universe.exchange_source import get_last_screener_meta

            meta = dict(get_last_screener_meta())
            sp = meta.get("path")
            if isinstance(sp, str) and sp:
                screener_path = Path(sp)
        except Exception:
            screener_path = None

        canonical_prices = paths.raw_prices_cache
        canonical_vols = paths.volume_path
        warmup_unadj = paths.raw_prices_unadj_warmup_cache
        raw_unadj = paths.raw_prices_unadj_cache
        raw_vol_unadj = paths.raw_volume_unadj_cache
        fundamentals_store = paths.fundamentals_out
        nan_raw = _nan_share(
            df_fundamentals, ["price", "market_cap", "volume", "float_pct"]
        )
        nan_filt = _nan_share(
            df_universe, ["price", "market_cap", "volume", "float_pct"]
        )

        if cfg_bool(runtime_cfg, "persist_run_scoped_outputs", False):
            base = paths.run_scoped_outputs_dir
            tag = (cfg_hash or "nohash")[:8]
            run_root = base / f"{run_id}_{tag}"
            rs_inputs = run_root / "inputs"
            rs_outputs = run_root / "outputs"
            rs_inputs.mkdir(parents=True, exist_ok=True)
            rs_outputs.mkdir(parents=True, exist_ok=True)

            copy_optional(
                cfg_path if cfg_path.exists() else None,
                rs_inputs / cfg_path.name,
                label="config",
            )
            if screener_path is not None:
                copy_optional(
                    screener_path, rs_inputs / screener_path.name, label="screener_csv"
                )

            for src in [
                out_tickers,
                out_manifest,
                canonical_prices,
                canonical_vols,
                warmup_unadj,
                raw_unadj,
                raw_vol_unadj,
            ]:
                copy_optional(
                    src if src.exists() else None, rs_outputs / src.name, label=src.name
                )
            if out_tickers_ext is not None:
                copy_optional(
                    out_tickers_ext if out_tickers_ext.exists() else None,
                    rs_outputs / out_tickers_ext.name,
                    label="tickers_ext_csv",
                )
            if fundamentals_store is not None:
                copy_optional(
                    fundamentals_store if fundamentals_store.exists() else None,
                    rs_outputs / fundamentals_store.name,
                    label="fundamentals_store",
                )
            if adv_csv is not None:
                copy_optional(
                    adv_csv if adv_csv.exists() else None,
                    rs_outputs / adv_csv.name,
                    label="adv_csv",
                )
            if adv_csv_filtered is not None:
                copy_optional(
                    adv_csv_filtered if adv_csv_filtered.exists() else None,
                    rs_outputs / adv_csv_filtered.name,
                    label="adv_csv_filtered",
                )

            write_tickers_final_txt(rs_outputs / "tickers_final.txt", tickers_final)
            write_markdown_report(
                out=run_root / "report.md",
                cfg_path=cfg_path,
                manifest_path=out_manifest,
                stats=stats,
                nan_raw=nan_raw,
                nan_filt=nan_filt,
                sample_tickers=tickers_final[:20],
                failed_tickers=list(monitoring.get("failed", []) or []),
            )
    except Exception as e:
        logger.warning("Run-scoped persistence failed: %s", e)


def write_markdown_report(
    out: Path,
    cfg_path: Path,
    manifest_path: Path,
    stats: Mapping[str, Any],
    nan_raw: Mapping[str, float],
    nan_filt: Mapping[str, float],
    sample_tickers: list[str],
    failed_tickers: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Universe Run Report\n")
    lines.append(f"- Config: `{cfg_path}`")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- Timestamp (UTC): {now_utc_iso()}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Tickers (loaded): {stats.get('n_tickers_total', 0)}")
    lines.append(f"- Fundamentals OK: {stats.get('n_fundamentals_ok', 0)}")
    lines.append(f"- Failed fundamentals: {stats.get('n_failed', 0)}")
    lines.append(f"- Final universe: {stats.get('n_filtered', 0)}")
    lines.append("")
    policy = stats.get("data_policy")
    if isinstance(policy, dict) and policy:
        lines.append("## Data policy")
        for key in sorted(policy):
            val = policy.get(key)
            if isinstance(val, (dict, list)):
                try:
                    rendered = json.dumps(val, ensure_ascii=False, sort_keys=True)
                except Exception:
                    rendered = str(val)
            else:
                rendered = str(val)
            lines.append(f"- {key}: {rendered}")
        lines.append("")
    dq = stats.get("data_quality")
    if isinstance(dq, dict) and dq:
        lines.append("## Data quality")
        ohlc = dq.get("ohlc")
        if isinstance(ohlc, dict):
            lines.append(f"- OHLC violations: {ohlc.get('violations', 0)}")
            examples = ohlc.get("examples") or []
            if isinstance(examples, list) and examples:
                rendered_examples: list[str] = []
                for ex in examples[:5]:
                    if not isinstance(ex, dict):
                        continue
                    tkr = ex.get("ticker")
                    ts = ex.get("ts")
                    if tkr is None or ts is None:
                        continue
                    rendered_examples.append(f"{tkr}@{ts}")
                if rendered_examples:
                    lines.append(
                        f"- Examples (up to 5): {', '.join(rendered_examples)}"
                    )
        lines.append("")
    gap_info = stats.get("price_funda_gap")
    if isinstance(gap_info, dict):
        lines.append("## Price vs fundamentals gap check")
        lines.append(
            f"- Threshold: >{gap_info.get('threshold')}x or <1/{gap_info.get('threshold')}"
        )
        lines.append(
            f"- Outliers: high={gap_info.get('high_count', 0)}, low={gap_info.get('low_count', 0)}"
        )
        high_ex = gap_info.get("high_examples") or []
        low_ex = gap_info.get("low_examples") or []
        if high_ex:
            lines.append(
                "- High examples: "
                + ", ".join(f"{e['ticker']}:{e['ratio']:.2f}" for e in high_ex[:5])
            )
        if low_ex:
            lines.append(
                "- Low examples: "
                + ", ".join(f"{e['ticker']}:{e['ratio']:.2f}" for e in low_ex[:5])
            )
        lines.append("")
    if nan_raw:
        lines.append("## NaN shares (raw)")
        for k, v in nan_raw.items():
            lines.append(f"- {k}: {v:.2%}")
        lines.append("")
    if nan_filt:
        lines.append("## NaN shares (filtered)")
        for k, v in nan_filt.items():
            lines.append(f"- {k}: {v:.2%}")
        lines.append("")

    def _chunk(seq: list[str], size: int) -> Iterable[list[str]]:
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    if sample_tickers:
        total = len(sample_tickers)
        cap = 40 if total <= 2000 else 20
        subset = [str(t).upper() for t in sample_tickers[:cap]]
        lines.append(f"## Sample final tickers (showing {len(subset)} of {total})")
        for group in _chunk(subset, 10):
            lines.append(", ".join(group))
        if total > cap:
            lines.append(f"... {total - cap} more not shown")
        lines.append("")
    if failed_tickers:
        total_fail = len(failed_tickers)
        cap_fail = 30
        subset_fail = [str(t).upper() for t in failed_tickers[:cap_fail]]
        lines.append(
            f"## Failed fundamentals (showing {len(subset_fail)} of {total_fail})"
        )
        for group in _chunk(subset_fail, 10):
            lines.append(", ".join(group))
        if total_fail > cap_fail:
            lines.append(f"... {total_fail - cap_fail} more not shown")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
