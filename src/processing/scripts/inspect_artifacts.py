# src/processing/scripts/inspect_artifacts.py
from __future__ import annotations

import argparse
import json
from glob import glob as _glob
from pathlib import Path
from typing import Any

import pandas as pd


def _read_df_any(p: Path) -> pd.DataFrame:
    """Reads a DataFrame from pkl/parquet/csv. Raises on missing files."""
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix in {".pkl", ".p"}:
        obj = pd.read_pickle(p)
        if isinstance(obj, pd.Series):
            return obj.to_frame()
        if isinstance(obj, pd.DataFrame):
            return obj
        raise TypeError("Pickle did not contain a pandas DataFrame/Series.")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    # CSV: versuche Index+Dates, fallback plain
    try:
        return pd.read_csv(p, index_col=0, parse_dates=True)
    except Exception:
        return pd.read_csv(p)


def _latest(glob_pattern: str) -> Path | None:
    """
    Returns the newest file for the pattern. Supports absolute/relative patterns.
    (Path.glob cannot handle absolute patterns on Windows -> use glob() instead).
    """
    pat = Path(glob_pattern)
    if pat.is_absolute():
        matches = [Path(s) for s in _glob(str(pat))]
    else:
        matches = [Path(s) for s in _glob(str(Path.cwd() / glob_pattern))]
    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _as_path(raw: Any, *, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    try:
        p = Path(str(raw))
    except Exception:
        return None
    if p.is_absolute():
        return p
    if p.exists():
        return p
    rel = base_dir / p
    if rel.exists():
        return rel
    return p


def _search_dirs(data_dir: Path) -> list[Path]:
    out: list[Path] = []
    processed = data_dir / "processed"
    if processed.is_dir():
        out.append(processed)
    out.append(data_dir)
    return out


def load_universe_paths(data_dir: Path) -> dict[str, Path | None]:
    """
    Finds the raw data path (raw_prices) from the manifest or via fallback.
    """
    uni_manifest = data_dir / "universe_manifest.json"
    raw_prices: Path | None = None

    if uni_manifest.exists():
        try:
            j = json.loads(uni_manifest.read_text(encoding="utf-8"))
            raw_prices_val = (
                j.get("raw_prices")
                or j.get("artifacts", {}).get("raw_prices")
                or j.get("inputs", {}).get("raw_prices", {}).get("path")
            )
            if raw_prices_val:
                raw_prices = Path(raw_prices_val)
        except Exception:
            raw_prices = None

    if not raw_prices:
        raw_prices = (
            _latest(str(data_dir / "raw_prices.*.pkl"))
            or _latest(str(data_dir / "raw_prices.pkl"))
            or _latest(str(data_dir / "raw_prices.parquet"))
        )

    return {"raw_prices": raw_prices}


def load_processing_paths(data_dir: Path) -> dict[str, Path | None]:
    """
    Returns paths to the current processing artifacts.
    Prefers runs/data/processed/* and uses manifest hints when available.
    Legacy names are still recognized as a fallback.
    """
    dirs = _search_dirs(data_dir)

    manifest: Path | None = None
    for d in dirs:
        manifest_candidate = d / "filled_manifest.json"
        if manifest_candidate.exists():
            manifest = manifest_candidate
            break
    if manifest is None:
        for d in dirs:
            manifest_candidate = d / "filled_data_manifest.json"
            if manifest_candidate.exists():
                manifest = manifest_candidate
                break
    if manifest is None:
        for d in dirs:
            latest_candidate = _latest(str(d / "*_manifest.json"))
            if latest_candidate:
                manifest = latest_candidate
                break

    exec_path: Path | None = None
    panel_exec: Path | None = None
    removed: Path | None = None
    diag: Path | None = None
    adv_map: Path | None = None

    if manifest and manifest.exists():
        try:
            man = json.loads(manifest.read_text(encoding="utf-8"))
            latest_out = (
                ((man.get("extra") or {}).get("outputs") or {}).get("latest")
            ) or {}
            exec_path = _as_path(latest_out.get("exec"), base_dir=manifest.parent)
            panel_exec = _as_path(
                latest_out.get("panel_exec"), base_dir=manifest.parent
            )
            removed = _as_path(latest_out.get("removed"), base_dir=manifest.parent)
            diag = _as_path(latest_out.get("diagnostics"), base_dir=manifest.parent)
            adv_map = _as_path(latest_out.get("adv_map"), base_dir=manifest.parent)
        except Exception:
            pass

    # Manifest paths can become stale after file moves/cleanup. In that case we
    # must still run the local fallback discovery.
    if exec_path is not None and not exec_path.exists():
        exec_path = None
    if panel_exec is not None and not panel_exec.exists():
        panel_exec = None
    if removed is not None and not removed.exists():
        removed = None
    if diag is not None and not diag.exists():
        diag = None
    if adv_map is not None and not adv_map.exists():
        adv_map = None

    if exec_path is None:
        for d in dirs:
            direct = d / "filled_prices_exec.parquet"
            if direct.exists():
                exec_path = direct
                break
    if exec_path is None:
        for d in dirs:
            exec_path = (
                _latest(str(d / "filled_prices_exec*.parquet"))
                or _latest(str(d / "filled_prices_exec*.pkl"))
                or _latest(str(d / "filled_data.parquet"))
                or _latest(str(d / "filled_data.pkl"))
                or _latest(str(d / "filled_*.parquet"))
                or _latest(str(d / "filled_*.pkl"))
            )
            if exec_path is not None:
                break

    if diag is None:
        for d in dirs:
            direct = d / "filled.diag.json"
            if direct.exists():
                diag = direct
                break
    if diag is None:
        for d in dirs:
            direct = d / "filled_data.diag.json"
            if direct.exists():
                diag = direct
                break
    if diag is None:
        for d in dirs:
            latest_candidate = _latest(str(d / "*.diag.json"))
            if latest_candidate:
                diag = latest_candidate
                break

    if removed is None:
        for d in dirs:
            direct = d / "filled_removed.pkl"
            if direct.exists():
                removed = direct
                break
    if removed is None:
        for d in dirs:
            latest_candidate = _latest(str(d / "*removed*.pkl"))
            if latest_candidate:
                removed = latest_candidate
                break

    if panel_exec is None:
        for d in dirs:
            direct = d / "filled_prices_panel_exec.parquet"
            if direct.exists():
                panel_exec = direct
                break

    if adv_map is None:
        for d in dirs:
            direct = d / "adv_map.pkl"
            if direct.exists():
                adv_map = direct
                break

    return {
        "exec": exec_path if exec_path and exec_path.exists() else None,
        "filled": exec_path if exec_path and exec_path.exists() else None,
        "panel_exec": panel_exec if panel_exec and panel_exec.exists() else None,
        "removed": removed if removed and removed.exists() else None,
        "diag": diag if diag and diag.exists() else None,
        "manifest": manifest if manifest and manifest.exists() else None,
        "adv_map": adv_map if adv_map and adv_map.exists() else None,
    }


def summarize_df(df: pd.DataFrame | None, name: str) -> dict[str, Any]:
    if df is None or df.empty:
        return {"name": name, "rows": 0, "cols": 0, "mean_nan_pct": 1.0}
    m = float(df.isna().mean().mean())
    return {
        "name": name,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "mean_nan_pct": m,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="runs/data", help="Path to the data directory")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)

    # ---- Check universe ----
    uni = load_universe_paths(data_dir)
    raw_p = uni["raw_prices"]
    if not raw_p:
        print("[UNIVERSE] No raw_prices.* found under", data_dir)
        raw_df = None
    else:
        raw_df = _read_df_any(raw_p)
        if isinstance(raw_df, pd.Series):
            raw_df = raw_df.to_frame()
        print(f"[UNIVERSE] raw_prices: {raw_p} | shape={raw_df.shape}")

    uni_sum = summarize_df(raw_df, "universe_raw")

    # ---- Check processing ----
    pr = load_processing_paths(data_dir)
    exec_p = pr["exec"]
    if not exec_p:
        print("[PROCESSING] No filled_prices_exec.* found under", data_dir)
        filled_df = None
    else:
        filled_df = _read_df_any(exec_p)
        print(f"[PROCESSING] exec: {exec_p} | shape={filled_df.shape}")

    proc_sum = summarize_df(filled_df, "processing_filled")

    # ---- Diag/Manifest lesen (optional) ----
    grid_mode = fill_mode = None
    kept_count = removed_count = None
    schema_version = None
    stage_keys: list[str] = []
    event_count: int | None = None
    if pr.get("diag"):
        try:
            d = json.loads(Path(pr["diag"]).read_text(encoding="utf-8"))  # type: ignore[arg-type]
            schema_version = d.get("schema_version")
            proc = d.get("processing") or {}
            kept_count = proc.get("kept")
            removed_count = proc.get("removed")
            grid_mode = proc.get("grid_mode")
            stages = d.get("stages") or {}
            if isinstance(stages, dict):
                stage_keys = sorted(str(k) for k in stages.keys())
            ev = (d.get("events") or {}).get("summary") if isinstance(d, dict) else None
            if isinstance(ev, dict):
                try:
                    event_count = int(ev.get("total_events", 0))
                except Exception:
                    event_count = None
            fill_mode = None
            filling = d.get("filling") or {}
            if isinstance(filling, dict):
                fill_mode = (
                    f"causal_only={bool(filling.get('causal_only', False))},"
                    f"hard_drop={bool(filling.get('hard_drop', False))}"
                )
            print(
                "[PROCESSING] diag: "
                f"schema={schema_version} kept={kept_count} removed={removed_count} "
                f"grid={grid_mode} fill={fill_mode} stages={stage_keys} events={event_count}"
            )
        except Exception:
            pass
    if pr.get("manifest"):
        try:
            m = json.loads(Path(pr["manifest"]).read_text(encoding="utf-8"))  # type: ignore[arg-type]
            inp = m.get("inputs", {})
            raw_used = (inp.get("raw_prices") or {}).get("path")
            if raw_used:
                print(f"[PROCESSING] manifest.inputs.raw_prices = {raw_used}")
        except Exception:
            pass

    # ---- Urteil ----
    raw_cols = uni_sum["cols"]
    filled_cols = proc_sum["cols"]

    if raw_cols == 0:
        verdict = "LIKELY_UNIVERSE_ISSUE"
        reason = "Universe did not find/create a usable raw_prices cache."
    elif filled_cols == 0 and raw_cols > 0:
        verdict = "LIKELY_PROCESSING_ISSUE"
        reason = (
            "Processing produced no output despite an existing universe cache."
        )
    elif raw_cols <= 10 and filled_cols <= raw_cols:
        verdict = "LIKELY_UNIVERSE_ISSUE"
        reason = f"Universe produced only {raw_cols} tickers (very few)."
    elif raw_cols > 50 and filled_cols <= max(5, int(0.2 * raw_cols)):
        verdict = "LIKELY_PROCESSING_ISSUE"
        reason = (
            f"Large drop: raw_cols={raw_cols} -> filled_cols={filled_cols} "
            "(check grid_mode/keep_pct)."
        )
    else:
        verdict = "NO_CLEAR_FAULT"
        reason = (
            f"raw_cols={raw_cols}, filled_cols={filled_cols}. "
            "Drop is within the expected range or mixed."
        )

    print(f"\n=== VERDICT: {verdict} ===")
    print(f"Reason: {reason}")
    if grid_mode or fill_mode:
        print(f"Notes: grid_mode={grid_mode}, fill_mode={fill_mode}")

    try:
        if raw_df is not None:
            print("\n[UNIVERSE] Beispiel-Ticker:", list(raw_df.columns[:10]))
        if filled_df is not None:
            print("[PROCESSING] Beispiel-Ticker:", list(filled_df.columns[:10]))
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
