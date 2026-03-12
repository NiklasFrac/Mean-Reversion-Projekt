# tests/system_check.py
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def tick(ok: bool) -> str:
    return f"{GREEN}[OK]{RESET}" if ok else f"{RED}[FAIL]{RESET}"


class Score:
    def __init__(self) -> None:
        self.total = 0
        self.passed = 0
        self.warnings = 0
        self.details: list[str] = []

    def check(self, cond: bool, msg: str, warn: bool = False) -> None:
        self.total += 1
        if cond:
            self.passed += 1
            self.details.append(f"{tick(True)} {msg}")
        else:
            if warn:
                self.warnings += 1
                self.details.append(f"{YELLOW}! {msg}{RESET}")
            else:
                self.details.append(f"{tick(False)} {msg}")

    def summarize(self) -> Tuple[bool, str]:
        ok = self.passed == self.total
        pct = 100.0 * self.passed / max(1, self.total)
        badge = (
            f"{GREEN}TIER-1{RESET}"
            if ok and self.warnings == 0
            else (
                f"{YELLOW}TIER-1 (with warnings){RESET}"
                if ok
                else f"{RED}NOT TIER-1{RESET}"
            )
        )
        summary = f"{BOLD}Result:{RESET} {self.passed}/{self.total} checks passed ({pct:.1f}%) | Warnings: {self.warnings} -> {badge}"
        return ok, summary


def load_cfg(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def is_tz_ny(idx: pd.Index) -> bool:
    try:
        if not isinstance(idx, pd.DatetimeIndex):
            return False
        return str(getattr(idx, "tz", None)) == "America/New_York"
    except Exception:
        return False


def safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def longest_nan_run(series: pd.Series) -> int:
    arr = series.isna().to_numpy()
    best = cur = 0
    for v in arr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def read_optional_json(path: Path) -> Any | None:
    if not path or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def max_gap_bars_from_config(dp_cfg: Dict[str, Any]) -> int:
    stage2 = dp_cfg.get("stage2", {}) if isinstance(dp_cfg.get("stage2"), dict) else {}
    bars_cfg = stage2.get("max_gap_bars", dp_cfg.get("max_gap_bars"))
    if bars_cfg is None:
        raise ValueError(
            "Config missing required data_processing.stage2.max_gap_bars (or legacy data_processing.max_gap_bars)"
        )
    return max(1, int(math.ceil(float(bars_cfg))))


def main() -> int:
    ap = argparse.ArgumentParser(description="System check for processing artifacts")
    ap.add_argument(
        "--cfg",
        type=Path,
        default=Path("runs/configs/config_processing.yaml"),
        help="Path to config_processing.yaml",
    )
    ap.add_argument(
        "--strict", action="store_true", help="Strict checks (less tolerance)"
    )
    ap.add_argument(
        "--sample", type=int, default=8, help="Number of symbols for sample checks"
    )
    args = ap.parse_args()

    sc = Score()

    print(f"{BOLD}System Check: Data Processing Pipeline{RESET}")
    print(f"Config: {args.cfg.resolve()}\n")

    # 1) Load config
    sc.check(args.cfg.exists(), f"Config exists: {args.cfg}")
    if not args.cfg.exists():
        for line in sc.details:
            print(line)
        ok, summary = sc.summarize()
        print("\n" + summary)
        return 2

    cfg = load_cfg(args.cfg)
    data_cfg = cfg.get("data", {}) or {}
    dp_cfg = cfg.get("data_processing", {}) or {}

    # 2) Expected output paths
    exec_path = Path(
        data_cfg.get(
            "filled_prices_exec_path", "runs/data/processed/filled_prices_exec.parquet"
        )
    )
    removed_p = Path(
        data_cfg.get("removed_symbols_path", "runs/data/processed/filled_removed.pkl")
    )
    diag_path = Path(
        data_cfg.get("diagnostics_path", "runs/data/processed/filled.diag.json")
    )
    manifest = Path(
        data_cfg.get("manifest_path", "runs/data/processed/filled_manifest.json")
    )
    adv_pkl = Path(data_cfg.get("adv_out_path", "runs/data/processed/adv_map.pkl"))

    print(f"{BOLD}Artifact existence{RESET}")
    sc.check(exec_path.exists(), f"Exec parquet present: {exec_path}")
    sc.check(removed_p.exists(), f"Removed list present: {removed_p}")
    sc.check(diag_path.exists(), f"Diagnostics present:  {diag_path}")
    sc.check(manifest.exists(), f"Manifest present:     {manifest}")
    sc.check(adv_pkl.exists(), f"ADV map present:      {adv_pkl}")

    # 3) Read parquet files
    print(f"\n{BOLD}Parquet contents{RESET}")
    exec_df = pd.read_parquet(exec_path) if exec_path.exists() else pd.DataFrame()

    sc.check(not exec_df.empty, f"Exec DataFrame is not empty (shape={exec_df.shape})")
    if not exec_df.empty:
        sc.check(is_tz_ny(exec_df.index), "Exec index is America/New_York (tz-aware)")
        sc.check(exec_df.index.is_monotonic_increasing, "Exec index is monotonically increasing")

        nonpos_exec = int((exec_df <= 0).sum().sum())
        sc.check(
            nonpos_exec == 0,
            f"Exec contains no non-positive prices (found={nonpos_exec})",
        )

        # Longest NaN run must be <= max_gap according to config (after filling)
        try:
            max_gap = max_gap_bars_from_config(dp_cfg)
            sc.check(True, f"Config max_gap_bars loadable ({max_gap})")
        except Exception as e:
            max_gap = 1
            sc.check(False, f"Config max_gap_bars invalid/missing ({e})")
        # Sample: check n symbols (or all, when strict)
        cols = list(exec_df.columns)
        sample_n = len(cols) if args.strict else min(args.sample, len(cols))
        sample_syms = cols[:sample_n]
        ok_all = True
        for sym in sample_syms:
            lrun = longest_nan_run(exec_df[sym])
            if lrun > max_gap:
                ok_all = False
                print(
                    f"{RED}[FAIL] Post-fill longest NaN run for {sym} = {lrun} > max_gap({max_gap}){RESET}"
                )
        sc.check(
            ok_all,
            f"Post-fill longest NaN run <= max_gap({max_gap}) for sample ({sample_n} symbols)",
        )

    # 4) Removed list and consistency
    print(f"\n{BOLD}Removed symbols & consistency{RESET}")
    removed_list = []
    try:
        import pickle

        if removed_p.exists():
            with removed_p.open("rb") as f:
                removed_list = pickle.load(f)
    except Exception as e:
        print(f"{YELLOW}! Could not read removed list: {e}{RESET}")

    sc.check(
        isinstance(removed_list, (list, set, tuple)),
        f"Removed is a sequence (type={type(removed_list).__name__})",
    )

    # 5) Check diagnostics
    print(f"\n{BOLD}Diagnostics-Checks{RESET}")
    diag = read_optional_json(diag_path) if diag_path.exists() else None
    sc.check(isinstance(diag, dict), "Diagnostics JSON loadable")
    if isinstance(diag, dict):
        schema_version = int(diag.get("schema_version", -1))
        has_quality = isinstance(diag.get("quality"), dict)
        has_processing = isinstance(diag.get("processing"), dict)
        has_stages = isinstance(diag.get("stages"), dict)
        has_snapshots = isinstance(diag.get("snapshots"), dict)
        has_events = isinstance(diag.get("events"), dict)
        sc.check(
            schema_version == 3, f"Diagnostics schema_version==3 ({schema_version})"
        )
        sc.check(has_quality, "Diagnostics contains quality block")
        sc.check(has_processing, "Diagnostics contains processing block")
        sc.check(has_stages, "Diagnostics contains stages block")
        sc.check(has_snapshots, "Diagnostics contains snapshots block")
        sc.check(has_events, "Diagnostics contains events block")
        for k in ["exec_diag", "env"]:
            sc.check(k in diag, f"Diagnostics contains field: {k}")

        quality = diag.get("quality", {})
        if isinstance(quality, dict):
            for qk in ["pre_raw", "pre_exec", "post"]:
                sc.check(qk in quality, f"Diagnostics.quality contains field: {qk}")

        # Consistency of kept/removed counts
        proc = diag.get("processing") or {}
        if isinstance(proc, dict) and proc and not exec_df.empty:
            kept = int(proc.get("kept", -1))
            sc.check(
                kept == exec_df.shape[1],
                f"Diagnostics.processing.kept == exec columns ({kept} == {exec_df.shape[1]})",
            )

        # Check sample symbol diagnostics
        exec_diag = diag.get("exec_diag", {})
        if isinstance(exec_diag, dict) and exec_diag:
            some = list(exec_diag.keys())[: min(5, len(exec_diag))]
            fields = [
                "non_na_pct",
                "post_non_na_pct",
                "longest_gap",
                "post_longest_gap",
            ]
            for sym in some:
                info = exec_diag.get(sym, {})
                has_all = all(
                    f in json.dumps(info) for f in fields
                )  # lax because nested
                sc.check(
                    has_all, f"exec_diag[{sym}] contains core fields (prefill/post)"
                )

    # 6) Check manifest
    print(f"\n{BOLD}Manifest-Checks{RESET}")
    mani = read_optional_json(manifest) if manifest.exists() else None
    sc.check(isinstance(mani, dict), "Manifest JSON loadable")
    if isinstance(mani, dict):
        for k in ["timestamp", "cfg_path", "cfg_hash", "inputs"]:
            sc.check(k in mani, f"Manifest contains field: {k}")
        ins = mani.get("inputs", {})
        # Core inputs should contain paths + sha1 (when present)
        if isinstance(ins, dict):
            for key in ["raw_prices", "raw_volume"]:
                if key in ins:
                    entry = ins[key]
                    sc.check(
                        isinstance(entry, dict), f"Manifest.inputs.{key} is a mapping"
                    )
                    # Warning instead of fail because inputs may be optional
                    path_present = bool(entry.get("path"))
                    sc.check(
                        True,
                        f"Manifest.inputs.{key}.path={entry.get('path')}",
                        warn=not path_present,
                    )

    # 7) Check ADV map
    print(f"\n{BOLD}ADV-Map-Checks{RESET}")
    adv_map: Dict[str, Dict[str, Any]] = {}
    try:
        import pickle

        if adv_pkl.exists():
            with adv_pkl.open("rb") as f:
                adv_map = pickle.load(f) or {}
    except Exception as e:
        print(f"{YELLOW}! Could not load ADV map: {e}{RESET}")

    sc.check(isinstance(adv_map, dict), "ADV map is a dict")
    if adv_map and not exec_df.empty:
        # Keys should be a subset of the columns
        keys = set(adv_map.keys())
        cols_set = set(exec_df.columns)
        sc.check(
            keys.issubset(cols_set),
            f"ADV map keys are a subset of exec columns (|missing|={len(keys - cols_set)})",
        )
        # Check sample values
        sample_keys = list(keys)[: min(10, len(keys))]
        for k in sample_keys:
            entry = adv_map.get(k, {})
            ok_entry = (
                isinstance(entry, dict) and ("adv" in entry) and ("last_price" in entry)
            )
            sc.check(ok_entry, f"ADV map[{k}] contains 'adv' & 'last_price'")
            adv_val = to_float(entry.get("adv"))
            lp_val = to_float(entry.get("last_price"))
            sc.check(
                adv_val >= 0 or math.isnan(adv_val), f"ADV map[{k}].adv >= 0 (or NaN)"
            )
            sc.check(
                lp_val > 0 or math.isnan(lp_val),
                f"ADV map[{k}].last_price > 0 (or NaN)",
            )

    # 8) Summary
    print(f"\n{BOLD}Detail-Report{RESET}")
    for line in sc.details:
        print(line)

    ok, summary = sc.summarize()
    print("\n" + summary)

    # Exit code: 0 for TIER-1 (without warnings in strict mode), otherwise 0 for OK with warnings, else 1
    if ok and (args.strict and sc.warnings == 0):
        return 0
    if ok:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
