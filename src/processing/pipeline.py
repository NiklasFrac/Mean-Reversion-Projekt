from __future__ import annotations

import csv
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd

from .config_loader import load_config
from .filtering_stage import (
    EVENT_COLUMNS,
    finalize_after_vendor,
    prepare_inputs,
    run_filtering_stages,
)
from .input_mode import resolve_processing_inputs
from .io_atomic import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_pickle,
    collect_runtime_context,
    file_hash,
    make_manifest,
)
from .logging_utils import logger
from .processing_primitives import process_and_fill_prices
from .quality_helpers import validate_prices_wide
from .raw_loader import UniversePanelBundle, load_raw_prices_from_universe
from .timebase import build_tradable_mask, ensure_ny_index
from .vendor_guards import apply_vendor_guards

__all__ = ["main"]


def _utc_run_id(prefix: str, cfg_hash: str | None) -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    h = (cfg_hash or "nocfg")[:8]
    return f"{prefix}-{ts}-{h}"


def _sync_latest_from_run(run_path: Path, latest_path: Path) -> None:
    """
    Make latest_path a stable copy of run_path (atomic replace).
    Prefer hardlinks when possible; fall back to copy.
    """
    if not run_path.exists():
        return
    try:
        if run_path.resolve() == latest_path.resolve():
            return
    except Exception:
        pass

    latest_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    tmp = latest_path.with_name(f"{latest_path.name}.tmp.{token}")
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        if tmp.exists():
            tmp.unlink()
    try:
        os.link(run_path, tmp)
    except Exception:
        shutil.copy2(run_path, tmp)
    os.replace(tmp, latest_path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _sync_latest_outputs_transactional(sync_pairs: list[tuple[Path, Path]]) -> None:
    """
    Transaction-like multi-file sync from run-scoped artefacts to latest paths.
    - Stage all temp copies first.
    - Replace latest targets only after staging succeeded.
    - Roll back replaced targets if a later replace fails.
    """
    if not sync_pairs:
        return

    token = uuid4().hex
    staged: list[tuple[Path, Path, Path]] = []
    replaced: list[tuple[Path, Path, bool]] = []

    try:
        for run_path, latest_path in sync_pairs:
            if not run_path.exists():
                logger.warning(
                    "Skipping latest sync because run-scoped artefact is missing: %s",
                    run_path,
                )
                continue
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = latest_path.with_name(f"{latest_path.name}.tmp.{token}")
            bak = latest_path.with_name(f"{latest_path.name}.bak.{token}")
            _safe_unlink(tmp)
            _safe_unlink(bak)
            try:
                os.link(run_path, tmp)
            except Exception:
                shutil.copy2(run_path, tmp)
            staged.append((latest_path, tmp, bak))

        for latest_path, tmp, bak in staged:
            had_existing = latest_path.exists()
            if had_existing:
                os.replace(latest_path, bak)
            try:
                os.replace(tmp, latest_path)
            except Exception:
                if had_existing and bak.exists():
                    os.replace(bak, latest_path)
                _safe_unlink(tmp)
                raise
            replaced.append((latest_path, bak, had_existing))

        for _, bak, had_existing in replaced:
            if had_existing:
                _safe_unlink(bak)
    except Exception:
        for latest_path, bak, had_existing in reversed(replaced):
            try:
                if had_existing and bak.exists():
                    os.replace(bak, latest_path)
                elif not had_existing:
                    _safe_unlink(latest_path)
            except Exception:
                pass
        for _, tmp, bak in staged:
            _safe_unlink(tmp)
            _safe_unlink(bak)
        raise


def _ensure_unique_output_basenames(paths: dict[str, Path]) -> None:
    """
    Run-scoped outputs are stored under one flat directory and therefore keyed by basename.
    Reject configs where different logical outputs map to the same filename.
    """
    by_base: dict[str, list[str]] = {}
    for logical_name, path in paths.items():
        by_base.setdefault(path.name, []).append(logical_name)

    duplicates = {base: names for base, names in by_base.items() if len(names) > 1}
    if not duplicates:
        return

    conflicts: list[str] = []
    for base in sorted(duplicates):
        names = sorted(duplicates[base])
        members = "; ".join(f"{name}={paths[name]}" for name in names)
        conflicts.append(f"{base}: {members}")
    raise ValueError(
        "Configured output paths must have distinct basenames for run-scoped outputs. "
        f"Conflicts: {', '.join(conflicts)}"
    )


def _snapshot_config_inputs(
    *, cfg_path: Path, cfg: dict[str, Any], inputs_dir: Path
) -> dict[str, Any]:
    inputs_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = inputs_dir / "config_resolved.json"
    atomic_write_json(cfg, resolved_path)

    source_copy: Path | None = None
    if cfg_path.exists():
        source_copy = inputs_dir / f"config_source{cfg_path.suffix or '.yaml'}"
        try:
            atomic_write_bytes(cfg_path.read_bytes(), source_copy)
        except Exception:
            source_copy = inputs_dir / "config_source.yaml"
            atomic_write_bytes(cfg_path.read_bytes(), source_copy)

    return {
        "resolved_config_json": str(resolved_path),
        "resolved_config_sha1": file_hash(resolved_path),
        "source_config_path": str(cfg_path),
        "source_config_copy": str(source_copy) if source_copy else None,
        "source_config_copy_sha1": file_hash(source_copy) if source_copy else None,
    }


def _find_project_root(
    *, cfg_source_path: Path | None = None, cwd: Path | None = None
) -> Path:
    starts: list[Path] = []
    if cfg_source_path is not None:
        try:
            starts.append(cfg_source_path.expanduser().resolve().parent)
        except Exception:
            starts.append(cfg_source_path.expanduser().parent)
    base_cwd = cwd if cwd is not None else Path.cwd()
    starts.append(base_cwd)
    if base_cwd.name == "src":
        starts.append(base_cwd.parent)

    seen: set[str] = set()
    for start in starts:
        cur = start if start.is_dir() else start.parent
        for candidate in [cur, *cur.parents]:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if (candidate / "pyproject.toml").is_file() or (
                candidate / ".git"
            ).exists():
                return candidate

    for start in starts:
        cur = start if start.is_dir() else start.parent
        if cur.name == "configs" and cur.parent.name == "runs":
            return cur.parent.parent
        if cur.name == "configs":
            return cur.parent

    fallback = starts[0] if starts else Path.cwd()
    if fallback.name == "src":
        return fallback.parent
    return fallback if fallback.is_dir() else fallback.parent


def _resolve_path_from_root(
    raw_path: Any, *, project_root: Path, allow_glob: bool = False
) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p
    combined = project_root / p
    if allow_glob and any(ch in str(raw_path) for ch in "*?[]"):
        return combined
    try:
        return combined.resolve()
    except Exception:
        return combined


def _file_uri(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        return f"file:{resolved.as_posix()}"
    return resolved.as_uri()


def _resolve_mlflow_tracking_uri(
    uri: Any, *, project_root: Path, is_windows: bool | None = None
) -> str | None:
    normalized = _normalize_mlflow_tracking_uri(uri, is_windows=is_windows)
    if normalized is None:
        return None
    low = normalized.lower()
    if low.startswith("sqlite:///"):
        db_path_raw = normalized[len("sqlite:///") :]
        if not db_path_raw:
            return normalized
        db_path = Path(db_path_raw)
        if not db_path.is_absolute():
            db_path = _resolve_path_from_root(db_path, project_root=project_root)
        return f"sqlite:///{db_path.as_posix()}"
    if low.startswith("file:"):
        file_path_raw = normalized[5:]
        if not file_path_raw:
            return normalized
        file_path = Path(file_path_raw)
        if not file_path.is_absolute():
            return _file_uri(
                _resolve_path_from_root(file_path, project_root=project_root)
            )
    return normalized


def _default_mlflow_tracking_uri(*, project_root: Path | None = None) -> str:
    root = (
        _find_project_root(cwd=Path.cwd())
        if project_root is None
        else Path(project_root).resolve()
    )
    meta_dir = root / "runs" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    db_path = (meta_dir / "mlflow.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"


def _default_mlflow_artifact_uri(*, project_root: Path) -> str:
    artifact_dir = (project_root / "runs" / "mlruns").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return _file_uri(artifact_dir)


def _normalize_mlflow_tracking_uri(
    uri: Any, *, is_windows: bool | None = None
) -> str | None:
    raw = str(uri).strip() if uri is not None else ""
    if not raw:
        return None
    if is_windows is None:
        is_windows = os.name == "nt"
    if not is_windows:
        return raw

    low = raw.lower()
    if low.startswith("file:///"):
        tail = raw[8:]
        if len(tail) >= 3 and tail[0] == "/" and tail[1].isalpha() and tail[2] == ":":
            tail = tail[1:]
        return f"file:{tail}"
    if low.startswith("file://"):
        tail = raw[7:]
        if len(tail) >= 2 and tail[1] == ":":
            return f"file:{tail}"
        return raw
    if "://" not in raw and len(raw) >= 2 and raw[1] == ":":
        return f"file:{raw}"
    return raw


def _resolve_config_paths(cfg: dict[str, Any], *, project_root: Path) -> dict[str, Any]:
    dcfg = cast(dict[str, Any], cfg.get("data", {}) or {})
    for key in (
        "dir",
        "out_dir",
        "pinned_universe_outputs_dir",
        "filled_prices_path",
        "filled_prices_exec_path",
        "filled_prices_panel_exec_path",
        "removed_symbols_path",
        "diagnostics_path",
        "manifest_path",
        "adv_out_path",
    ):
        raw_value = dcfg.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            dcfg[key] = str(
                _resolve_path_from_root(raw_value, project_root=project_root)
            )
    for key in ("raw_prices_globs", "raw_volume_globs"):
        raw_values = dcfg.get(key)
        if isinstance(raw_values, list):
            dcfg[key] = [
                str(
                    _resolve_path_from_root(
                        value, project_root=project_root, allow_glob=True
                    )
                )
                for value in raw_values
                if str(value).strip()
            ]
    cfg["data"] = dcfg

    dp = cast(dict[str, Any], cfg.get("data_processing", {}) or {})
    cal_cfg = cast(dict[str, Any], dp.get("calendars", {}) or {})
    csv_map = cal_cfg.get("symbol_calendar_csv")
    if isinstance(csv_map, str) and csv_map.strip():
        cal_cfg["symbol_calendar_csv"] = str(
            _resolve_path_from_root(csv_map, project_root=project_root)
        )
    if cal_cfg:
        dp["calendars"] = cal_cfg

    qcfg = cast(dict[str, Any], dp.get("quality_flags", {}) or {})
    qflags_path = qcfg.get("path")
    if isinstance(qflags_path, str) and qflags_path.strip():
        qcfg["path"] = str(
            _resolve_path_from_root(qflags_path, project_root=project_root)
        )
    if qcfg:
        dp["quality_flags"] = qcfg

    vendor_cfg = cast(dict[str, Any], dp.get("vendor_guards", {}) or {})
    bad_rows_cfg = cast(dict[str, Any], vendor_cfg.get("bad_rows", {}) or {})
    bad_rows_path = bad_rows_cfg.get("path")
    if isinstance(bad_rows_path, str) and bad_rows_path.strip():
        bad_rows_cfg["path"] = str(
            _resolve_path_from_root(bad_rows_path, project_root=project_root)
        )
    if bad_rows_cfg:
        vendor_cfg["bad_rows"] = bad_rows_cfg
    if vendor_cfg:
        dp["vendor_guards"] = vendor_cfg

    event_cfg = cast(dict[str, Any], dp.get("event_log", {}) or {})
    event_path = event_cfg.get("path")
    if isinstance(event_path, str) and event_path.strip():
        event_cfg["path"] = str(
            _resolve_path_from_root(event_path, project_root=project_root)
        )
    if event_cfg:
        dp["event_log"] = event_cfg

    cfg["data_processing"] = dp

    mcfg = cast(dict[str, Any], cfg.get("mlflow", {}) or {})
    if "tracking_uri" in mcfg:
        mcfg["tracking_uri"] = _resolve_mlflow_tracking_uri(
            mcfg.get("tracking_uri"), project_root=project_root
        )
    cfg["mlflow"] = mcfg
    return cfg


def _build_outputs_payload(
    *,
    run_exec_path: Path,
    run_panel_exec_path: Path,
    run_removed_path: Path,
    run_diag_path: Path,
    run_manifest_path: Path,
    run_adv_out: Path,
    run_event_log_path: Path,
    exec_path: Path,
    panel_exec_path: Path,
    removed_path: Path,
    diag_path: Path,
    manifest_path: Path,
    adv_out: Path,
    event_log_path: Path,
    event_log_written: bool,
) -> dict[str, dict[str, str | None]]:
    event_run = str(run_event_log_path) if event_log_written else None
    event_latest = str(event_log_path) if event_log_written else None
    return {
        "run_scoped": {
            "exec": str(run_exec_path),
            "panel_exec": str(run_panel_exec_path),
            "removed": str(run_removed_path),
            "diagnostics": str(run_diag_path),
            "manifest": str(run_manifest_path),
            "adv_map": str(run_adv_out),
            "processing_events": event_run,
        },
        "latest": {
            "exec": str(exec_path),
            "panel_exec": str(panel_exec_path),
            "removed": str(removed_path),
            "diagnostics": str(diag_path),
            "manifest": str(manifest_path),
            "adv_map": str(adv_out),
            "processing_events": event_latest,
        },
    }


def _build_processing_summary(
    *,
    exec_filled: pd.DataFrame,
    exec_removed: list[str],
    exec_diag: dict[str, dict[str, Any]] | None,
    grid_mode: str,
    calendar: str,
    vendor_tz: str,
    vendor_tz_policy: str,
    rth_only: bool,
    adv_window: int,
    adv_mode: str,
    adv_stat: str,
    adv_price_source: str,
    max_gap_bars_cfg_val: Any,
    max_gap_bars: int,
) -> dict[str, Any]:
    kept = int(exec_filled.shape[1] if not exec_filled.empty else 0)
    removed = int(len(exec_removed))
    exec_diag_map: dict[str, dict[str, Any]] = exec_diag or {}
    mean_non_na_pct = (
        float(np.mean([d.get("non_na_pct", 0.0) for d in exec_diag_map.values()]))
        if exec_diag_map
        else 0.0
    )
    max_longest_gap_kept = int(
        max(
            [
                (exec_diag_map.get(sym, {}).get("longest_gap", 0))
                for sym in exec_filled.columns
            ],
            default=0,
        )
    )
    sum_outliers_flagged = int(
        sum(d.get("outliers_flagged", 0) for d in exec_diag_map.values())
    )
    return {
        "kept": kept,
        "removed": removed,
        "grid_mode": grid_mode,
        "calendar": calendar,
        "vendor_tz": vendor_tz,
        "vendor_tz_policy": vendor_tz_policy,
        "rth_only": bool(rth_only),
        "adv_window": int(adv_window),
        "adv_mode": str(adv_mode),
        "adv_stat": str(adv_stat),
        "adv_price_source": str(adv_price_source),
        "mean_non_na_pct": float(mean_non_na_pct),
        "max_longest_gap_kept": max_longest_gap_kept,
        "sum_outliers_flagged": sum_outliers_flagged,
        "max_gap_bars_config": max_gap_bars_cfg_val,
        "max_gap_bars_applied": max_gap_bars,
    }


def _tradable_mask_key(
    *, index: pd.DatetimeIndex, calendar_code: str, rth_only: bool
) -> tuple[str, bool, int, str | None, str | None, str]:
    if index.empty:
        return (str(calendar_code), bool(rth_only), 0, None, None, str(index.tz))
    return (
        str(calendar_code),
        bool(rth_only),
        int(len(index)),
        str(index[0]),
        str(index[-1]),
        str(index.tz),
    )


def _get_tradable_mask_cached(
    *,
    index: pd.DatetimeIndex,
    calendar_code: str,
    rth_only: bool,
    cache: dict[
        tuple[str, bool, int, str | None, str | None, str],
        tuple[pd.DatetimeIndex, pd.Series],
    ],
) -> pd.Series:
    key = _tradable_mask_key(
        index=index, calendar_code=calendar_code, rth_only=bool(rth_only)
    )
    cached = cache.get(key)
    if cached is not None:
        cached_idx, cached_mask = cached
        if cached_idx.equals(index):
            return cached_mask
    mask = build_tradable_mask(
        index,
        calendar_code=str(calendar_code),
        rth_only=bool(rth_only),
    )
    cache[key] = (index, mask)
    return mask


def _normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Light-weight config normalization for config key shape.
    - data_processing.calendar -> calendars.default
    """
    dp = cfg.get("data_processing", {}) or {}
    cal = dp.get("calendar")
    cal_cfg = (
        (dp.get("calendars") or {}) if isinstance(dp.get("calendars"), dict) else {}
    )
    if cal and not cal_cfg.get("default"):
        logger.warning(
            "data_processing.calendar is deprecated; setting calendars.default=%s", cal
        )
        cal_cfg["default"] = cal
    if cal_cfg:
        dp["calendars"] = cal_cfg
    cfg["data_processing"] = dp
    return cfg


def _resolve_vendor_tz(
    *,
    config_vendor_tz: str,
    upstream_vendor_tz: str | None,
    policy_raw: Any,
) -> tuple[str, str]:
    """
    Resolve effective vendor timezone from config + upstream metadata policy.

    Policies:
    - upstream_override: use upstream value when present (legacy behavior).
    - config_wins: keep config value even if upstream disagrees.
    """
    policy = str(policy_raw or "upstream_override").strip().lower()
    allowed = {"upstream_override", "config_wins"}
    if policy not in allowed:
        raise ValueError(
            "data_processing.vendor_tz_policy must be one of "
            f"{sorted(allowed)}; got {policy_raw!r}."
        )

    upstream = str(upstream_vendor_tz or "").strip() or None
    config = str(config_vendor_tz or "").strip() or "UTC"
    if upstream is None:
        return config, policy

    if policy == "upstream_override":
        if upstream != config:
            logger.warning(
                "Overriding data_processing.vendor_tz=%s with upstream universe policy raw_index_naive_tz=%s",
                config,
                upstream,
            )
        return upstream, policy

    if policy == "config_wins":
        if upstream != config:
            logger.warning(
                "Ignoring upstream raw_index_naive_tz=%s because vendor_tz_policy=config_wins; using data_processing.vendor_tz=%s",
                upstream,
                config,
            )
        return config, policy
    return config, policy


def _reconcile_ohlc(
    panel_df: pd.DataFrame,
    *,
    eps_abs: float = 1.0e-12,
    eps_rel: float = 1.0e-8,
) -> pd.DataFrame:
    """
    Enforce OHLC invariants on a MultiIndex panel (symbol, field) with epsilon tolerance.
    - Only correct violations larger than epsilon.
    - Never impute NaNs (missing stays missing).
    - For hard_cross rows, bounds are widened from available OHLC values.
    """
    if panel_df is None or panel_df.empty:
        return panel_df
    if not isinstance(panel_df.columns, pd.MultiIndex):
        return panel_df

    out = panel_df.copy()
    fields = set(out.columns.get_level_values(1))

    def _get(field: str) -> pd.DataFrame | None:
        if field not in fields:
            return None
        return cast(pd.DataFrame, out.xs(field, level="field", axis=1))

    close = _get("close")
    high = _get("high")
    low = _get("low")
    opn = _get("open")
    if close is None:
        return out

    def _eps_from(*parts: pd.DataFrame | None) -> pd.DataFrame:
        valid = [p for p in parts if p is not None]
        if not valid:
            return pd.DataFrame(index=close.index, columns=close.columns, data=eps_abs)
        arrs = [np.abs(v.to_numpy(dtype=float)) for v in valid]
        stacked = np.stack(arrs, axis=2)
        # Avoid RuntimeWarning("All-NaN slice encountered") by mapping non-finite
        # values to zero scale before reduction.
        stacked = np.where(np.isfinite(stacked), stacked, 0.0)
        scale = np.max(stacked, axis=2)
        eps = eps_abs + eps_rel * scale
        return pd.DataFrame(eps, index=valid[0].index, columns=valid[0].columns)

    if high is not None:
        common = close.columns.intersection(high.columns)
        if not common.empty:
            c = close[common]
            h = high[common]
            eps = _eps_from(c, h)
            mask = c.notna() & h.notna() & ((c - h) > eps)
            h = h.where(~mask, c)
            for sym in common:
                out[(sym, "high")] = h[sym]
            high = cast(pd.DataFrame, out.xs("high", level="field", axis=1))

    if low is not None:
        common = close.columns.intersection(low.columns)
        if not common.empty:
            c = close[common]
            low_common = low[common]
            eps = _eps_from(c, low_common)
            mask = c.notna() & low_common.notna() & ((low_common - c) > eps)
            low_common = low_common.where(~mask, c)
            for sym in common:
                out[(sym, "low")] = low_common[sym]
            low = cast(pd.DataFrame, out.xs("low", level="field", axis=1))

    if opn is not None:
        o = opn.copy()
        if high is not None:
            common_hi = o.columns.intersection(high.columns)
            if not common_hi.empty:
                o_hi = o[common_hi]
                h_hi = high[common_hi]
                eps_hi = _eps_from(o_hi, h_hi)
                mask_hi = o_hi.notna() & h_hi.notna() & ((o_hi - h_hi) > eps_hi)
                o.loc[:, common_hi] = o_hi.where(~mask_hi, h_hi)
        if low is not None:
            common_lo = o.columns.intersection(low.columns)
            if not common_lo.empty:
                o_lo = o[common_lo]
                l_lo = low[common_lo]
                eps_lo = _eps_from(o_lo, l_lo)
                mask_lo = o_lo.notna() & l_lo.notna() & ((l_lo - o_lo) > eps_lo)
                o.loc[:, common_lo] = o_lo.where(~mask_lo, l_lo)
        for sym in o.columns:
            out[(sym, "open")] = o[sym]
        opn = cast(pd.DataFrame, out.xs("open", level="field", axis=1))

    if high is not None and low is not None:
        common = high.columns.intersection(low.columns)
        if not common.empty:
            h = high[common]
            low_common = low[common]
            eps = _eps_from(h, low_common)
            cross_mask = h.notna() & low_common.notna() & ((low_common - h) > eps)
            if bool(cross_mask.to_numpy().any()):
                for sym in common:
                    sym_mask = cross_mask[sym]
                    if not bool(sym_mask.any()):
                        continue
                    pieces: list[pd.Series] = [h[sym], low_common[sym]]
                    if sym in close.columns:
                        pieces.append(close[sym])
                    if opn is not None and sym in opn.columns:
                        pieces.append(opn[sym])
                    bounds = pd.concat(pieces, axis=1)
                    fallback_high = bounds.max(axis=1, skipna=True)
                    fallback_low = bounds.min(axis=1, skipna=True)
                    h.loc[sym_mask, sym] = fallback_high.loc[sym_mask]
                    low_common.loc[sym_mask, sym] = fallback_low.loc[sym_mask]
                for sym in common:
                    out[(sym, "high")] = h[sym]
                    out[(sym, "low")] = low_common[sym]

    return out


def _load_expected_symbols(
    input_dir: Path, *, require_readable: bool = False
) -> set[str] | None:
    p = input_dir / "tickers_universe.csv"
    if not p.is_file():
        if require_readable:
            raise FileNotFoundError(
                f"strict_inputs=true requires tickers_universe.csv under {input_dir}."
            )
        return None
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        if not rows or not any(rows):
            if require_readable:
                raise ValueError(
                    f"tickers_universe.csv under {input_dir} is empty/unreadable."
                )
            return None
        header0 = rows[0][0].strip().lower() if rows[0] else ""
        start = 1 if header0 in {"ticker", "symbol"} else 0
        out = {r[0].strip() for r in rows[start:] if r and r[0].strip()}
        if out:
            return out
        if require_readable:
            raise ValueError(
                f"tickers_universe.csv under {input_dir} contains no symbol rows."
            )
        return None
    except Exception as exc:
        if require_readable:
            raise RuntimeError(
                f"Failed to parse tickers_universe.csv under {input_dir}: {exc}"
            ) from exc
        return None


def _validate_symbol_contract(
    *,
    expected_symbols: set[str],
    prices: pd.DataFrame,
    volume: pd.DataFrame | None,
    context: str,
) -> None:
    price_cols = {str(c).strip() for c in prices.columns if str(c).strip()}
    missing_prices = sorted(expected_symbols - price_cols)
    extra_prices = sorted(price_cols - expected_symbols)
    if missing_prices or extra_prices:
        raise RuntimeError(
            "Symbol contract mismatch for prices in "
            f"{context}: missing={missing_prices[:10]} (n={len(missing_prices)}), "
            f"extra={extra_prices[:10]} (n={len(extra_prices)})."
        )

    if volume is None or volume.empty:
        return
    vol_cols = {str(c).strip() for c in volume.columns if str(c).strip()}
    missing_vol = sorted(expected_symbols - vol_cols)
    extra_vol = sorted(vol_cols - expected_symbols)
    if missing_vol or extra_vol:
        raise RuntimeError(
            "Symbol contract mismatch for volume in "
            f"{context}: missing={missing_vol[:10]} (n={len(missing_vol)}), "
            f"extra={extra_vol[:10]} (n={len(extra_vol)})."
        )


def main(cfg_path: Path | None = None) -> None:
    cfg_path = (
        Path(cfg_path)
        if cfg_path
        else Path("runs") / "configs" / "config_processing.yaml"
    )
    cfg_loaded, cfg_source_path = load_config(cfg_path, return_source=True)
    cfg = _normalize_config(cfg_loaded)
    project_root = _find_project_root(
        cfg_source_path=cfg_source_path, cwd=Path.cwd()
    ).resolve()
    cfg = _resolve_config_paths(cfg, project_root=project_root)
    cfg_sha1 = file_hash(cfg_source_path)
    try:
        if cfg_source_path.resolve() != Path(cfg_path).resolve():
            logger.warning(
                "Config loaded from %s (requested=%s).",
                cfg_source_path,
                cfg_path,
            )
    except Exception:
        pass

    try:
        lvl = (
            cfg.get("data_processing", {}).get("logging_level", "INFO") or "INFO"
        ).upper()
        logger.setLevel(getattr(logging, lvl, logging.INFO))
    except Exception:
        pass

    data_dir = Path(cfg.get("data", {}).get("dir", "runs/data"))
    filled_prices_legacy = cfg.get("data", {}).get("filled_prices_path")
    out_dir_default = (
        Path(filled_prices_legacy).parent if filled_prices_legacy else Path("runs/data")
    )
    out_dir = Path(cfg.get("data", {}).get("out_dir", out_dir_default))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Immutable per-run snapshot under runs/data/by_run/<run_id>/outputs/processed/*
    run_id = _utc_run_id("PRC", cfg_sha1)
    run_root = data_dir / "by_run" / run_id
    run_inputs_dir = run_root / "inputs"
    run_outputs_dir = run_root / "outputs" / "processed"
    config_snapshot = _snapshot_config_inputs(
        cfg_path=cfg_source_path, cfg=cfg, inputs_dir=run_inputs_dir
    )
    run_outputs_dir.mkdir(parents=True, exist_ok=True)

    base_fill = Path(filled_prices_legacy) if filled_prices_legacy else None
    exec_path = Path(
        cfg.get("data", {}).get(
            "filled_prices_exec_path",
            str(base_fill if base_fill else out_dir / "filled_prices_exec.parquet"),
        )
    )
    panel_exec_path = Path(
        cfg.get("data", {}).get(
            "filled_prices_panel_exec_path",
            str(out_dir / "filled_prices_panel_exec.parquet"),
        )
    )
    removed_path = Path(
        cfg.get("data", {}).get(
            "removed_symbols_path",
            str(
                (
                    base_fill.with_name(base_fill.stem + "_removed.pkl")
                    if base_fill
                    else out_dir / "filled_removed.pkl"
                )
            ),
        )
    )
    diag_path = Path(
        cfg.get("data", {}).get(
            "diagnostics_path",
            str(
                (
                    base_fill.with_name(base_fill.stem + ".diag.json")
                    if base_fill
                    else out_dir / "filled.diag.json"
                )
            ),
        )
    )
    manifest_path = Path(
        cfg.get("data", {}).get(
            "manifest_path",
            str(
                (
                    base_fill.with_name(base_fill.stem + "_manifest.json")
                    if base_fill
                    else out_dir / "filled_manifest.json"
                )
            ),
        )
    )
    dp = cfg.get("data_processing", {}) or {}
    adv_out = Path(
        cfg.get("data", {}).get("adv_out_path", str(out_dir / "adv_map.pkl"))
    )
    adv_mode = str(dp.get("adv_mode", "shares") or "shares")
    adv_stat = str(dp.get("adv_stat", "mean") or "mean")
    adv_price_source_cfg = str(dp.get("adv_price_source", "") or "").strip().lower()
    adv_mode_n = str(adv_mode or "").strip().lower()
    adv_price_source = (
        adv_price_source_cfg
        if adv_price_source_cfg
        else (
            "raw_unadjusted" if adv_mode_n in {"dollar", "usd", "notional"} else "exec"
        )
    )
    vendor_cfg = cast(dict[str, Any], dp.get("vendor_guards", {}) or {})
    event_cfg = cast(
        dict[str, Any],
        dp.get("event_log", cast(dict[str, Any], vendor_cfg.get("output", {}) or {}))
        or {},
    )
    event_log_enabled = bool(
        event_cfg.get("enabled", event_cfg.get("write_event_log", True))
    )
    event_log_path = Path(
        str(event_cfg.get("path", str(out_dir / "processing_events.parquet")))
    )
    stage1_cfg = cast(dict[str, Any], dp.get("stage1", {}) or {})
    stage2_cfg = cast(dict[str, Any], dp.get("stage2", {}) or {})
    reverse_split_cfg = cast(dict[str, Any], dp.get("reverse_split_v2", {}) or {})
    adv_gates_cfg = cast(dict[str, Any], dp.get("adv_gates", {}) or {})
    panel_reconcile_cfg = cast(dict[str, Any], dp.get("panel_reconcile", {}) or {})

    configured_outputs: dict[str, Path] = {
        "exec": exec_path,
        "panel_exec": panel_exec_path,
        "removed": removed_path,
        "diagnostics": diag_path,
        "manifest": manifest_path,
        "adv_map": adv_out,
    }
    if event_log_enabled:
        configured_outputs["processing_events"] = event_log_path
    _ensure_unique_output_basenames(configured_outputs)

    run_exec_path = run_outputs_dir / exec_path.name
    run_panel_exec_path = run_outputs_dir / panel_exec_path.name
    run_removed_path = run_outputs_dir / removed_path.name
    run_diag_path = run_outputs_dir / diag_path.name
    run_manifest_path = run_outputs_dir / manifest_path.name
    run_adv_out = run_outputs_dir / adv_out.name
    run_event_log_path = run_outputs_dir / event_log_path.name
    parquet_comp = cast(
        str | None, cfg.get("data_processing", {}).get("parquet_compression", "zstd")
    )
    keep_pct = float(
        stage2_cfg.get("keep_pct_threshold", dp.get("keep_pct_threshold", 0.8))
    )
    n_jobs = int(dp.get("n_jobs", 1))
    grid_mode = str(dp.get("grid_mode", "leader"))
    cal_cfg = (cfg.get("data_processing", {}) or {}).get("calendars", {}) or {}
    calendar = str(cal_cfg.get("default", "XNYS"))
    vendor_tz_config = str(dp.get("vendor_tz", "UTC") or "UTC")
    vendor_tz_policy_raw = dp.get("vendor_tz_policy", "upstream_override")
    max_start_na = int(stage2_cfg.get("max_start_na", dp.get("max_start_na", 3)))
    max_end_na = int(stage2_cfg.get("max_end_na", dp.get("max_end_na", 3)))
    outlier_cfg = cast(
        dict[str, Any],
        dp.get(
            "outliers",
            {"enabled": True, "zscore": 8.0, "window": 21, "use_log_returns": True},
        ),
    )
    rth_only = bool(dp.get("rth_only", True))

    caps_cfg = dp.get("return_caps", {}) or {}
    st_cfg = dp.get("staleness", {}) or {"enabled": False}
    fill_cfg = dp.get("filling", {}) or {}
    causal_only = bool(fill_cfg.get("causal_only", False))
    hard_drop = bool(fill_cfg.get("hard_drop", True))

    dcfg = cfg.get("data", {}) or {}
    strict_inputs = bool(dcfg.get("strict_inputs", False))
    price_globs_fallback = list(dcfg.get("raw_prices_globs", []) or [])
    volume_globs_fallback = list(dcfg.get("raw_volume_globs", []) or [])
    resolved_inputs = resolve_processing_inputs(
        data_cfg=dcfg,
        base_data_dir=data_dir,
        price_globs_fallback=price_globs_fallback,
        volume_globs_fallback=volume_globs_fallback,
    )
    input_dir = resolved_inputs.input_dir
    price_globs = resolved_inputs.price_globs
    volume_globs = resolved_inputs.volume_globs
    universe_meta = (
        resolved_inputs.universe_meta
        if isinstance(resolved_inputs.universe_meta, dict)
        else {}
    )
    data_policy = (
        universe_meta.get("data_policy") if isinstance(universe_meta, dict) else {}
    )
    universe_artifacts = (
        universe_meta.get("artifacts") if isinstance(universe_meta, dict) else {}
    )

    def _resolve_upstream_artifact_path(raw_path: Any) -> Path | None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        p = Path(raw_path)
        if p.is_absolute():
            return p
        # In run-scoped modes, avoid drifting to mutable workspace artefacts.
        if resolved_inputs.mode in {"run_latest", "run_pinned"}:
            if resolved_inputs.universe_manifest_path is not None:
                manifest_candidate = resolved_inputs.universe_manifest_path.parent / p
                if manifest_candidate.exists():
                    return manifest_candidate
            candidate = input_dir / p
            if candidate.exists():
                return candidate
            # Compatibility fallback for manifests that store repo-relative paths
            # while run-scoped persistence copies artefacts by basename.
            basename_candidate = input_dir / p.name
            if basename_candidate.exists():
                return basename_candidate
            return basename_candidate
        root_candidate = project_root / p
        if root_candidate.exists():
            return root_candidate
        if resolved_inputs.universe_manifest_path is not None:
            manifest_candidate = resolved_inputs.universe_manifest_path.parent / p
            if manifest_candidate.exists():
                return manifest_candidate
        return input_dir / p

    upstream_vendor_tz: str | None = None
    if isinstance(data_policy, dict):
        upstream_vendor_tz = (
            str(
                data_policy.get("raw_index_naive_tz")
                or data_policy.get("raw_index_timezone")
                or data_policy.get("naive_index_timezone")
                or ""
            ).strip()
            or None
        )
    vendor_tz, vendor_tz_policy = _resolve_vendor_tz(
        config_vendor_tz=vendor_tz_config,
        upstream_vendor_tz=upstream_vendor_tz,
        policy_raw=vendor_tz_policy_raw,
    )
    raw_loaded = load_raw_prices_from_universe(
        input_dir,
        price_globs=price_globs,
        volume_globs=volume_globs,
        include_bundle=True,
    )
    if isinstance(raw_loaded, tuple) and len(raw_loaded) == 3:
        df_prices_raw, df_vol_raw, used_paths = raw_loaded
        panel_bundle = UniversePanelBundle()
    else:
        df_prices_raw, df_vol_raw, panel_bundle, used_paths = raw_loaded

    if strict_inputs and resolved_inputs.mode in {"run_latest", "run_pinned"}:
        expected_symbols = _load_expected_symbols(input_dir, require_readable=True)
        if expected_symbols is None:
            raise RuntimeError(
                "strict_inputs=true requires a non-empty tickers_universe.csv "
                f"under {input_dir}."
            )
        _validate_symbol_contract(
            expected_symbols=expected_symbols,
            prices=df_prices_raw,
            volume=df_vol_raw,
            context=f"input_mode={resolved_inputs.mode} input_dir={input_dir}",
        )

    df_prices_raw = ensure_ny_index(df_prices_raw, vendor_tz=vendor_tz)

    max_gap_bars_cfg_raw = stage2_cfg.get("max_gap_bars", dp.get("max_gap_bars"))
    if max_gap_bars_cfg_raw is None:
        raise ValueError(
            "data_processing.stage2.max_gap_bars is required (or legacy data_processing.max_gap_bars)."
        )
    try:
        max_gap_bars_cfg_val = float(max_gap_bars_cfg_raw)
    except Exception as exc:
        raise ValueError(
            "data_processing.max_gap_bars must be numeric (bars-only processing config)."
        ) from exc
    max_gap_bars = int(math.ceil(max_gap_bars_cfg_val))
    max_gap_bars = max(1, max_gap_bars)

    qcfg = dp.get("quality_flags", {}) or {}
    if bool(qcfg.get("enabled", False)):
        p_flags = Path(qcfg.get("path", ""))
        if not p_flags.exists():
            raise FileNotFoundError(
                f"Quality flags enabled but file not found: {p_flags}"
            )
        try:
            flags_df = (
                pd.read_parquet(p_flags)
                if p_flags.suffix == ".parquet"
                else pd.read_csv(p_flags)
            )
            ts_col_cfg = str(qcfg.get("ts_col", "ts") or "ts").strip().lower()
            sym_col_cfg = (
                str(qcfg.get("symbol_col", "symbol") or "symbol").strip().lower()
            )
            flag_col_cfg = str(qcfg.get("flag_col", "flag") or "flag").strip().lower()
            flags_cols_lower = [str(c).strip().lower() for c in flags_df.columns]
            flags_colmap = {str(c): str(c).strip().lower() for c in flags_df.columns}
            flags_df = flags_df.rename(columns=flags_colmap)
            flags_cols_set = set(flags_df.columns)
            if qcfg.get("format", "auto") in ("auto", "long") and {
                ts_col_cfg,
                sym_col_cfg,
                flag_col_cfg,
            }.issubset(flags_cols_set):
                if (
                    ts_col_cfg != "ts"
                    or sym_col_cfg != "symbol"
                    or flag_col_cfg != "flag"
                ):
                    flags_df = flags_df.rename(
                        columns={
                            ts_col_cfg: "ts",
                            sym_col_cfg: "symbol",
                            flag_col_cfg: "flag",
                        }
                    )
                # Keep raw timestamps as-is and let ensure_ny_index apply the same
                # vendor_tz/daily semantics as price inputs for exact alignment.
                flags_df["ts"] = pd.to_datetime(flags_df["ts"], errors="coerce")
                flags_df = flags_df.pivot(index="ts", columns="symbol", values="flag")
            elif qcfg.get("format", "auto") in ("long",):
                raise ValueError(
                    "quality_flags format=long but required columns are missing "
                    f"(expected: {ts_col_cfg}, {sym_col_cfg}, {flag_col_cfg}; got: {sorted(flags_cols_lower)})"
                )
            flags_df = ensure_ny_index(flags_df, vendor_tz=vendor_tz).reindex(
                df_prices_raw.index
            )
            if bool(qcfg.get("invert_meaning", False)):
                flags_df = ~flags_df.astype(bool)
            df_prices_raw = df_prices_raw.mask(flags_df.fillna(False))
            logger.info("Applied vendor quality flags from %s", p_flags)
        except Exception as e:
            raise RuntimeError(
                f"Failed to apply quality flags from {p_flags}: {e}"
            ) from e

    panel_fields_raw = panel_bundle.fields or {}
    aligned_panel_fields: dict[str, pd.DataFrame] = {}
    for field_name, field_df in panel_fields_raw.items():
        try:
            aligned = ensure_ny_index(field_df, vendor_tz=vendor_tz).reindex(
                df_prices_raw.index
            )
            aligned.columns = pd.Index(map(str, aligned.columns))
            aligned_panel_fields[field_name] = aligned.astype(float)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to align panel field '{field_name}': {exc}"
            ) from exc

    volume_vendor: pd.DataFrame | None = None
    if df_vol_raw is not None and not df_vol_raw.empty:
        volume_vendor = ensure_ny_index(df_vol_raw, vendor_tz=vendor_tz).reindex(
            df_prices_raw.index
        )
    elif aligned_panel_fields.get("volume") is not None:
        volume_vendor = aligned_panel_fields["volume"]
    df_vol_raw = volume_vendor

    panel_fields: dict[str, pd.DataFrame] = aligned_panel_fields
    if "close" not in panel_fields:
        panel_fields["close"] = df_prices_raw.copy()

    volume_for_processing: pd.DataFrame | None = None
    if df_vol_raw is not None and not df_vol_raw.empty:
        try:
            volume_for_processing = df_vol_raw.copy().astype(float)
        except Exception as exc:
            logger.warning("Volume preparation failed, using vendor volume: %s", exc)
            volume_for_processing = df_vol_raw.astype(float)
        volume_for_processing = volume_for_processing.reindex(df_prices_raw.index)

    sym_cal: dict[str, str] = {}
    symbol_cal_enabled_raw = cal_cfg.get("symbol_calendar_csv_enabled", True)
    if isinstance(symbol_cal_enabled_raw, str):
        symbol_cal_enabled = symbol_cal_enabled_raw.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    else:
        symbol_cal_enabled = bool(symbol_cal_enabled_raw)
    csv_map = cal_cfg.get("symbol_calendar_csv")
    if not symbol_cal_enabled:
        logger.info(
            "symbol_calendar_csv loading disabled via calendars.symbol_calendar_csv_enabled=false"
        )
    elif csv_map:
        csv_path = Path(str(csv_map))
        if csv_path.exists():
            try:
                m = pd.read_csv(csv_path)
                m.columns = [str(c).strip().lower() for c in m.columns]
                if {"symbol", "calendar_code"}.issubset(set(m.columns)):
                    sym_cal = {
                        str(row["symbol"]): str(row["calendar_code"])
                        for _, row in m.iterrows()
                        if str(row.get("symbol", "")).strip()
                    }
                    logger.info("Loaded symbol calendar map (%d).", len(sym_cal))
                else:
                    logger.warning(
                        "symbol_calendar_csv missing required columns symbol/calendar_code: %s",
                        csv_path,
                    )
            except Exception as e:
                logger.warning("Failed to load symbol_calendar_csv: %s", e)
        else:
            logger.warning("symbol_calendar_csv not found: %s", csv_path)

    prepared = prepare_inputs(
        close_raw=df_prices_raw,
        panel_fields=panel_fields,
        volume=volume_for_processing,
        grid_mode=grid_mode,
        calendar_code=calendar,
        rth_only=bool(rth_only),
        symbol_calendar_map=sym_cal,
    )
    pre_q_raw = prepared.pre_q_raw

    vg_result = apply_vendor_guards(
        df_exec_raw=prepared.close_raw,
        panel_fields=prepared.panel_fields,
        volume_for_processing=prepared.volume,
        config=vendor_cfg,
    )
    vendor_events = vg_result.anomalies

    finalized = finalize_after_vendor(
        close_after_vendor=vg_result.df_exec_raw,
        panel_fields_after_vendor=vg_result.panel_fields,
        volume_after_vendor=vg_result.volume_for_processing,
        grid_mode=grid_mode,
        calendar_code=calendar,
        rth_only=bool(rth_only),
        symbol_calendar_map=sym_cal,
    )
    pre_q_exec = finalized.pre_q_exec

    close_raw_unadj: pd.DataFrame | None = None
    unadj_path_used: str | None = None
    if adv_price_source in {"raw_unadjusted", "unadjusted", "raw_unadj"} or bool(
        reverse_split_cfg.get("enabled", True)
    ):
        from .raw_loader import _extract_panel_from_suffixes, _load_any_prices

        p_unadj_manifest = (
            _resolve_upstream_artifact_path(universe_artifacts.get("prices_unadjusted"))
            if isinstance(universe_artifacts, dict)
            else None
        )
        cand_paths = [
            input_dir / "raw_prices_unadj.pkl",
            input_dir / "raw_prices_unadj.parquet",
            input_dir / "raw_prices_unadj.csv",
            p_unadj_manifest,
        ]
        p_unadj = next((p for p in cand_paths if p is not None and p.is_file()), None)
        if p_unadj is not None:
            raw_unadj_any = _load_any_prices(p_unadj)
            _, _, close_unadj = _extract_panel_from_suffixes(raw_unadj_any)
            close_raw_unadj = close_unadj if close_unadj is not None else raw_unadj_any
            close_raw_unadj = ensure_ny_index(close_raw_unadj, vendor_tz=vendor_tz)
            unadj_path_used = str(p_unadj)

    filtering = run_filtering_stages(
        finalized=finalized,
        stage1_cfg=stage1_cfg,
        stage2_cfg={**stage2_cfg, "max_gap_bars": max_gap_bars},
        reverse_split_cfg=reverse_split_cfg,
        caps_cfg=caps_cfg,
        outlier_cfg=outlier_cfg,
        staleness_cfg=st_cfg,
        close_raw_unadj=close_raw_unadj,
        strict_inputs=bool(strict_inputs),
    )

    def _event_frame(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        out = df.copy()
        for col in EVENT_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
        return out[EVENT_COLUMNS]

    def _event_row(
        *,
        ts: pd.Timestamp,
        symbol: str,
        stage: str,
        field: str,
        rule: str,
        severity: str,
        action: str,
        source: str = "auto",
        metric_value: float | None = None,
        threshold: float | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        return {
            "ts": ts,
            "symbol": symbol,
            "stage": stage,
            "field": field,
            "rule": rule,
            "severity": severity,
            "action": action,
            "source": source,
            "value_open": np.nan,
            "value_high": np.nan,
            "value_low": np.nan,
            "value_close": np.nan,
            "value_volume": np.nan,
            "metric_value": metric_value,
            "threshold": threshold,
            "note": note,
        }

    extra_event_rows: list[dict[str, Any]] = []
    exec_filled, fill_removed, exec_diag = process_and_fill_prices(
        filtering.close,
        max_gap=max_gap_bars,
        keep_pct_threshold=keep_pct,
        n_jobs=n_jobs,
        grid_mode=grid_mode,
        calendar_code=calendar,
        max_start_na=max_start_na,
        max_end_na=max_end_na,
        outlier_cfg={"enabled": False},
        rth_only=rth_only,
        symbol_calendar_map=sym_cal,
        caps_cfg={"enabled": False},
        staleness_cfg={"enabled": False},
        volume_df=filtering.volume,
        causal_only=causal_only,
        hard_drop=hard_drop,
    )
    if exec_filled.empty and not filtering.close.empty:
        exec_filled = pd.DataFrame(index=filtering.close.index)

    fill_removed = sorted(set(map(str, fill_removed)))
    ts_event = (
        cast(pd.DatetimeIndex, filtering.close.index)[-1]
        if len(filtering.close.index)
        else pd.Timestamp.utcnow()
    )
    for sym in fill_removed:
        diag_sym = exec_diag.get(sym, {})
        reason = str(diag_sym.get("reason", "fill_backstop_drop"))
        metric_val_raw = diag_sym.get("post_longest_gap", diag_sym.get("longest_gap"))
        metric_val = (
            float(metric_val_raw)
            if metric_val_raw is not None and np.isfinite(metric_val_raw)
            else None
        )
        extra_event_rows.append(
            _event_row(
                ts=ts_event,
                symbol=sym,
                stage="fill",
                field="close",
                rule=reason,
                severity="error",
                action="drop_symbol",
                metric_value=metric_val,
                threshold=float(max_gap_bars),
            )
        )

    exec_removed = sorted(set(filtering.removed_symbols) | set(fill_removed))
    stages_summary: dict[str, Any] = dict(filtering.stages or {})
    stages_summary["fill"] = {
        "removed": fill_removed,
        "kept_count": int(exec_filled.shape[1]),
        "removed_count": int(len(fill_removed)),
    }

    panel_exec_frames: dict[str, pd.DataFrame] = {}
    panel_field_results: dict[str, Any] = {}
    panel_exec_df: pd.DataFrame | None = None

    target_index = cast(pd.DatetimeIndex, exec_filled.index)
    target_symbols = list(map(str, exec_filled.columns))
    if not exec_filled.empty:
        panel_exec_frames["close"] = exec_filled.copy()
    panel_field_results["close"] = {
        "kept": int(exec_filled.shape[1]),
        "removed": exec_removed,
        "mode": "fill_backstop",
    }

    for field_name, field_df in (filtering.panel_fields or {}).items():
        f = str(field_name).strip().lower()
        if f in {"close", "adj_close", "volume"}:
            continue
        if field_df is None or field_df.empty:
            panel_field_results[f] = {
                "kept": 0,
                "removed": target_symbols,
                "mode": "align_only",
                "error": "empty_field_input",
            }
            continue
        aligned = field_df.reindex(target_index)
        aligned = aligned.reindex(columns=target_symbols)
        panel_exec_frames[f] = aligned.astype(float)
        panel_field_results[f] = {
            "kept": int(aligned.shape[1]),
            "removed": [],
            "mode": "align_only",
        }

    volume_panel_src: pd.DataFrame | None = None
    if filtering.volume is not None and not filtering.volume.empty:
        volume_panel_src = filtering.volume
    elif "volume" in (filtering.panel_fields or {}):
        volume_panel_src = filtering.panel_fields["volume"]
    if volume_panel_src is not None and not volume_panel_src.empty:
        vol_panel = volume_panel_src.reindex(target_index).reindex(
            columns=target_symbols
        )
        panel_exec_frames["volume"] = vol_panel.astype(float)
        panel_field_results["volume"] = {
            "kept": int(vol_panel.shape[1]),
            "removed": [],
            "mode": "align_only",
        }

    snapshots: dict[str, Any] = dict(filtering.snapshots or {})
    snapshots["post_close_anchor"] = {
        "rows": int(len(target_index)),
        "kept_count": int(exec_filled.shape[1]),
        "removed_count": int(len(exec_removed)),
        "symbols": target_symbols,
    }

    reconcile_eps_abs = float(panel_reconcile_cfg.get("eps_abs", 1.0e-12))
    reconcile_eps_rel = float(panel_reconcile_cfg.get("eps_rel", 1.0e-8))
    if panel_exec_frames:
        try:
            panel_exec_df = pd.concat(panel_exec_frames, axis=1)
            panel_exec_df.columns = panel_exec_df.columns.set_names(["field", "symbol"])
            panel_exec_df = panel_exec_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
            panel_exec_df = _reconcile_ohlc(
                panel_exec_df,
                eps_abs=reconcile_eps_abs,
                eps_rel=reconcile_eps_rel,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build panel execution dataframe: {exc}"
            ) from exc
    snapshots["post_ohlc"] = {
        "rows": int(panel_exec_df.shape[0]) if panel_exec_df is not None else 0,
        "kept_count": int(exec_filled.shape[1]),
        "has_panel_exec": bool(panel_exec_df is not None and not panel_exec_df.empty),
    }
    stages_summary["panel"] = {
        "align_only": True,
        "fields": sorted(panel_exec_frames.keys()),
        "reconcile_eps_abs": reconcile_eps_abs,
        "reconcile_eps_rel": reconcile_eps_rel,
    }

    adv_map: dict[str, Any] | None = None
    adv_map_written = False
    adv_price_path_used: str | None = None
    adv_window = int(dp.get("adv_window", 21))
    adv_min_valid_ratio = float(adv_gates_cfg.get("min_valid_ratio", 0.80))
    adv_min_total_windows = int(adv_gates_cfg.get("min_total_windows_for_adv_gate", 20))
    adv_max_invalid_ratio = float(adv_gates_cfg.get("max_invalid_window_ratio", 0.35))
    adv_gate_metrics: dict[str, dict[str, Any]] = {}

    try:
        if exec_filled.empty:
            raise RuntimeError(
                "ADV build failed: final universe empty before ADV build."
            )

        volume_for_adv = panel_exec_frames.get("volume")
        if volume_for_adv is None or volume_for_adv.empty:
            raise RuntimeError(
                "ADV build failed: volume panel is empty after filtering."
            )
        volume_for_adv = volume_for_adv.reindex(exec_filled.index).reindex(
            columns=exec_filled.columns
        )

        price_for_adv = exec_filled.reindex(columns=volume_for_adv.columns)
        if adv_mode_n in {"dollar", "usd", "notional"} and adv_price_source in {
            "raw_unadjusted",
            "unadjusted",
            "raw_unadj",
        }:
            if close_raw_unadj is None or close_raw_unadj.empty:
                msg = (
                    f"adv_price_source={adv_price_source} requires raw_prices_unadj.* under {input_dir} "
                    "(and/or manifest artifacts.prices_unadjusted)."
                )
                if strict_inputs:
                    raise FileNotFoundError(msg)
                logger.warning("%s Falling back to exec_filled for ADV.", msg)
            else:
                unadj = close_raw_unadj.reindex(exec_filled.index).reindex(
                    columns=exec_filled.columns
                )
                price_for_adv = unadj
                adv_price_path_used = unadj_path_used
        elif adv_price_source not in {"exec", "execution", "exec_filled", "filled"}:
            logger.warning(
                "Unknown adv_price_source=%s; using exec_filled.", adv_price_source
            )

        from . import liquidity as _liquidity

        if hasattr(_liquidity, "build_adv_map_with_gates"):
            try:
                adv_map_all, adv_gate_metrics = _liquidity.build_adv_map_with_gates(
                    price_for_adv,
                    volume_for_adv,
                    window=adv_window,
                    adv_mode=adv_mode,
                    stat=adv_stat,
                    min_valid_ratio=adv_min_valid_ratio,
                    min_total_windows_for_adv_gate=adv_min_total_windows,
                    max_invalid_window_ratio=adv_max_invalid_ratio,
                )
            except TypeError:
                adv_map_all = _liquidity.build_adv_map_from_price_and_volume(
                    price_for_adv,
                    volume_for_adv,
                    window=adv_window,
                    adv_mode=adv_mode,
                    stat=adv_stat,
                )
                adv_gate_metrics = {}
        else:
            adv_map_all = _liquidity.build_adv_map_from_price_and_volume(
                price_for_adv,
                volume_for_adv,
                window=adv_window,
                adv_mode=adv_mode,
                stat=adv_stat,
            )
            adv_gate_metrics = {}

        if not isinstance(adv_map_all, dict):
            raise RuntimeError("ADV build failed: adv builder did not return dict.")

        adv_kept_symbols = sorted(set(map(str, adv_map_all.keys())))
        adv_dropped_symbols = sorted(
            set(map(str, exec_filled.columns)) - set(adv_kept_symbols)
        )
        ts_adv = (
            cast(pd.DatetimeIndex, exec_filled.index)[-1]
            if len(exec_filled.index)
            else pd.Timestamp.utcnow()
        )
        for sym in adv_dropped_symbols:
            metric = adv_gate_metrics.get(sym, {})
            invalid_ratio = metric.get("invalid_window_ratio")
            invalid_ratio_f = (
                float(invalid_ratio)
                if invalid_ratio is not None and np.isfinite(invalid_ratio)
                else None
            )
            extra_event_rows.append(
                _event_row(
                    ts=ts_adv,
                    symbol=sym,
                    stage="adv",
                    field="volume",
                    rule="adv_gate_failed",
                    severity="error",
                    action="drop_symbol",
                    metric_value=invalid_ratio_f,
                    threshold=adv_max_invalid_ratio,
                    note=(
                        f"min_valid_ratio={adv_min_valid_ratio},min_total_windows={adv_min_total_windows}"
                    ),
                )
            )
            exec_diag.setdefault(sym, {})["reason"] = "adv_gate_failed"

        if adv_dropped_symbols:
            exec_filled = exec_filled.drop(columns=adv_dropped_symbols, errors="ignore")
            exec_removed = sorted(set(exec_removed) | set(adv_dropped_symbols))
            if panel_exec_df is not None and not panel_exec_df.empty:
                keep_cols = [
                    c
                    for c in panel_exec_df.columns
                    if str(c[0]) in set(exec_filled.columns)
                ]
                panel_exec_df = panel_exec_df.loc[:, keep_cols]
            panel_field_results["close"]["removed"] = exec_removed

        adv_map = {
            str(sym): cast(dict[str, Any], payload)
            for sym, payload in adv_map_all.items()
            if str(sym) in set(exec_filled.columns)
        }
        if not adv_map:
            raise RuntimeError("ADV map is empty after applying ADV gates.")
        if exec_filled.empty:
            raise RuntimeError("ADV gates removed all symbols from final universe.")

        atomic_write_pickle(adv_map, run_adv_out)
        adv_map_written = True
    except Exception as e:
        raise RuntimeError(f"ADV build failed: {e}") from e

    stages_summary["adv"] = {
        "min_valid_ratio": adv_min_valid_ratio,
        "min_total_windows_for_adv_gate": adv_min_total_windows,
        "max_invalid_window_ratio": adv_max_invalid_ratio,
        "kept_count": int(len(adv_map or {})),
        "dropped_count": int(
            len(
                [
                    s
                    for s, d in exec_diag.items()
                    if str(d.get("reason", "")).strip() == "adv_gate_failed"
                ]
            )
        ),
    }

    events_parts: list[pd.DataFrame] = []
    vendor_events_df = _event_frame(vendor_events)
    if not vendor_events_df.empty:
        events_parts.append(vendor_events_df)
    filtering_events_df = _event_frame(filtering.events)
    if not filtering_events_df.empty:
        events_parts.append(filtering_events_df)
    if extra_event_rows:
        extra_events_df = _event_frame(pd.DataFrame(extra_event_rows))
        if not extra_events_df.empty:
            events_parts.append(extra_events_df)
    processing_events = _event_frame(
        pd.concat(events_parts, ignore_index=True) if events_parts else None
    )

    sync_pairs: list[tuple[Path, Path]] = []

    def _queue_latest_sync(run_path: Path, latest_path: Path) -> None:
        if run_path.exists():
            sync_pairs.append((run_path, latest_path))
            return
        logger.warning(
            "Run-scoped artefact not found after write; skipping latest sync for %s -> %s",
            run_path,
            latest_path,
        )

    def _write_df_run(df: pd.DataFrame, run_path: Path) -> None:
        suffix = run_path.suffix.lower()
        if suffix in (".pkl", ".pickle"):
            atomic_write_pickle(df, run_path)
        else:
            atomic_write_parquet(df, run_path, compression=parquet_comp)

    panel_exec_written = False
    if panel_exec_df is not None and not panel_exec_df.empty:
        _write_df_run(panel_exec_df, run_panel_exec_path)
        panel_exec_written = True
        _queue_latest_sync(run_panel_exec_path, panel_exec_path)

    _write_df_run(exec_filled, run_exec_path)
    _queue_latest_sync(run_exec_path, exec_path)
    atomic_write_pickle(sorted(set(exec_removed)), run_removed_path)
    _queue_latest_sync(run_removed_path, removed_path)
    event_log_written = False
    if event_log_enabled:
        atomic_write_parquet(
            processing_events, run_event_log_path, compression=parquet_comp
        )
        event_log_written = True
        _queue_latest_sync(run_event_log_path, event_log_path)

    try:
        env_ctx = collect_runtime_context(
            pip_freeze=bool(dp.get("pip_freeze", True)),
            project_root=project_root,
        )
    except TypeError:
        try:
            env_ctx = collect_runtime_context(
                pip_freeze=bool(dp.get("pip_freeze", True))
            )
        except TypeError:
            # Backwards compatibility with monkeypatched collect_runtime_context
            env_ctx = collect_runtime_context()
    post_quality = (
        validate_prices_wide(exec_filled)
        if not exec_filled.empty
        else {"checks": {"rows": 0, "cols": 0}}
    )
    processing_summary = _build_processing_summary(
        exec_filled=exec_filled,
        exec_removed=exec_removed,
        exec_diag=exec_diag,
        grid_mode=grid_mode,
        calendar=calendar,
        vendor_tz=vendor_tz,
        vendor_tz_policy=vendor_tz_policy,
        rth_only=bool(rth_only),
        adv_window=int(dp.get("adv_window", 21)),
        adv_mode=str(adv_mode),
        adv_stat=str(adv_stat),
        adv_price_source=str(adv_price_source),
        max_gap_bars_cfg_val=max_gap_bars_cfg_val,
        max_gap_bars=max_gap_bars,
    )
    filling_payload = {"causal_only": bool(causal_only), "hard_drop": bool(hard_drop)}
    vendor_tz_resolution = {
        "policy": vendor_tz_policy,
        "config": vendor_tz_config,
        "upstream": upstream_vendor_tz,
        "effective": vendor_tz,
    }
    run_payload = {"run_id": run_id, "run_root": str(run_root)}
    outputs_payload = _build_outputs_payload(
        run_exec_path=run_exec_path,
        run_panel_exec_path=run_panel_exec_path,
        run_removed_path=run_removed_path,
        run_diag_path=run_diag_path,
        run_manifest_path=run_manifest_path,
        run_adv_out=run_adv_out,
        run_event_log_path=run_event_log_path,
        exec_path=exec_path,
        panel_exec_path=panel_exec_path,
        removed_path=removed_path,
        diag_path=diag_path,
        manifest_path=manifest_path,
        adv_out=adv_out,
        event_log_path=event_log_path,
        event_log_written=event_log_written,
    )

    by_stage: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    if not processing_events.empty:
        by_stage = {
            str(k): int(v)
            for k, v in processing_events["stage"].value_counts(dropna=False).items()
        }
        by_rule = {
            str(k): int(v)
            for k, v in processing_events["rule"].value_counts(dropna=False).items()
        }
    events_summary = {
        "total_events": int(processing_events.shape[0]),
        "by_stage": by_stage,
        "by_rule": by_rule,
    }

    diag_payload: dict[str, Any] = {
        "schema_version": 3,
        "quality": {"pre_raw": pre_q_raw, "pre_exec": pre_q_exec, "post": post_quality},
        "processing": processing_summary,
        "exec_diag": exec_diag,
        "stages": stages_summary,
        "snapshots": snapshots,
        "events": {"summary": events_summary},
        "env": env_ctx,
        "filling": filling_payload,
        "inputs": {
            "input_mode": resolved_inputs.mode,
            "input_dir": str(resolved_inputs.input_dir),
            "price_globs": list(price_globs),
            "volume_globs": list(volume_globs),
            "used_paths": dict(used_paths or {}),
            "universe_manifest_path": str(resolved_inputs.universe_manifest_path)
            if resolved_inputs.universe_manifest_path
            else None,
            "universe_meta": resolved_inputs.universe_meta,
            "strict_inputs": bool(dcfg.get("strict_inputs", False)),
            "allow_fallback_to_legacy": bool(
                dcfg.get("allow_fallback_to_legacy", False)
            ),
            "vendor_tz_resolution": vendor_tz_resolution,
        },
        "run": run_payload,
        "config_snapshot": config_snapshot,
        "outputs": outputs_payload,
    }
    if panel_field_results:
        diag_payload["panel_fields"] = panel_field_results
    atomic_write_json(diag_payload, run_diag_path)
    _queue_latest_sync(run_diag_path, Path(diag_path))

    prices_path = used_paths.get("prices")
    volume_path = used_paths.get("volume")
    panel_path = used_paths.get("panel")
    manifest_inputs: dict[str, Path | None] = {
        "raw_prices": Path(prices_path) if prices_path else None,
        "raw_volume": Path(volume_path) if volume_path else None,
        "raw_prices_panel": Path(panel_path) if panel_path else None,
    }
    manifest = make_manifest(
        Path(cfg_source_path),
        manifest_inputs,
        extra={
            "env": env_ctx,
            "filling": filling_payload,
            "event_log_written": bool(event_log_written),
            "panel_fields_available": sorted((filtering.panel_fields or {}).keys()),
            "panel_exec_written": bool(
                panel_exec_df is not None and not panel_exec_df.empty
            ),
            "processing": processing_summary,
            "stages": stages_summary,
            "snapshots": snapshots,
            "events_summary": events_summary,
            "vendor_tz_resolution": vendor_tz_resolution,
            "run": run_payload,
            "config_snapshot": config_snapshot,
            "outputs": outputs_payload,
        },
    )
    atomic_write_json(manifest, run_manifest_path)
    _queue_latest_sync(run_manifest_path, Path(manifest_path))
    if adv_map_written:
        _queue_latest_sync(run_adv_out, adv_out)

    _sync_latest_outputs_transactional(sync_pairs)
    if adv_map_written:
        logger.info("Saved adv_map: run=%s | latest=%s", run_adv_out, adv_out)
        if adv_price_path_used:
            logger.info("ADV price source: raw_unadjusted (%s)", adv_price_path_used)
    if panel_exec_written:
        logger.info(
            "Saved panel execution data: run=%s | latest=%s",
            run_panel_exec_path,
            panel_exec_path,
        )

    mcfg = cast(dict[str, Any], cfg.get("mlflow", {}) or {})
    if bool(mcfg.get("enabled", False)):
        try:
            import mlflow

            raw_tracking_uri = mcfg.get("tracking_uri")
            tracking_uri = _resolve_mlflow_tracking_uri(
                raw_tracking_uri, project_root=project_root
            )
            if tracking_uri is None:
                tracking_uri = _default_mlflow_tracking_uri(project_root=project_root)
                logger.info(
                    "MLflow tracking_uri not configured; defaulting to sqlite backend: %s",
                    tracking_uri,
                )
            elif str(raw_tracking_uri).strip() != tracking_uri:
                logger.info(
                    "Normalized MLflow tracking_uri: %s -> %s",
                    raw_tracking_uri,
                    tracking_uri,
                )
            if tracking_uri.lower().startswith("file:"):
                logger.warning(
                    "MLflow filesystem backend URI detected (%s). "
                    "Filesystem backend is deprecated in recent MLflow versions; "
                    "prefer sqlite:///... or a server tracking URI.",
                    tracking_uri,
                )
            mlflow.set_tracking_uri(tracking_uri)
            experiment_name = str(mcfg.get("experiment_name", "") or "").strip()
            effective_experiment_name = experiment_name or "processing"
            desired_artifact_uri = _default_mlflow_artifact_uri(
                project_root=project_root
            )
            existing_experiment = None
            get_experiment_by_name = getattr(mlflow, "get_experiment_by_name", None)
            if callable(get_experiment_by_name):
                existing_experiment = get_experiment_by_name(effective_experiment_name)
            if existing_experiment is None:
                create_experiment = getattr(mlflow, "create_experiment", None)
                if callable(create_experiment):
                    try:
                        create_experiment(
                            effective_experiment_name,
                            artifact_location=desired_artifact_uri,
                        )
                    except Exception:
                        # Another process may have created it concurrently.
                        pass
            elif getattr(existing_experiment, "artifact_location", None) and str(
                existing_experiment.artifact_location
            ).rstrip("/\\") != desired_artifact_uri.rstrip("/\\"):
                logger.warning(
                    "MLflow experiment %s already exists with artifact_location=%s "
                    "(desired=%s). Existing location will be used.",
                    effective_experiment_name,
                    existing_experiment.artifact_location,
                    desired_artifact_uri,
                )
            set_experiment = getattr(mlflow, "set_experiment", None)
            if callable(set_experiment):
                set_experiment(effective_experiment_name)
            with mlflow.start_run(
                run_name=cast(str, mcfg.get("run_name", "processing_run"))
            ):
                mlflow.log_params(
                    {
                        "grid_mode": grid_mode,
                        "calendar": calendar,
                        "rth_only": rth_only,
                        "max_gap_bars": max_gap_bars,
                        "keep_pct": keep_pct,
                        "causal_only": bool(causal_only),
                    }
                )
                kept_val = float(exec_filled.shape[1] if not exec_filled.empty else 0)
                mlflow.log_metrics(
                    {
                        "exec_kept": kept_val,
                        "exec_removed": float(len(exec_removed)),
                        "kept": kept_val,
                    }
                )
                if run_exec_path.exists():
                    mlflow.log_artifact(str(run_exec_path))
                if run_diag_path.exists():
                    mlflow.log_artifact(str(run_diag_path))
                if run_manifest_path.exists():
                    mlflow.log_artifact(str(run_manifest_path))
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)

    logger.info(
        "Data processing complete. Kept=%d | grid=%s | tz=America/New_York | run_id=%s",
        int(processing_summary["kept"]),
        grid_mode,
        run_id,
    )
