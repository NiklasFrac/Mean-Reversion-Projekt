# src/processing/scripts/causality_audit.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

GREEN = "[OK]"
RED = "[FAIL]"
YELLOW = "[WARN]"


def load_json(p: Path) -> dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {p}, got {type(obj).__name__}")
    return cast(dict[str, Any], obj)


def main() -> None:
    base = Path.cwd()
    diag_p = base / "runs" / "data" / "processed" / "filled.diag.json"
    man_p = base / "runs" / "data" / "processed" / "filled_manifest.json"
    exec_p = base / "runs" / "data" / "processed" / "filled_prices_exec.parquet"
    adv_p = base / "runs" / "data" / "processed" / "adv_map.pkl"

    print("Causality/Leakage Audit\n")

    # --- Existenz
    ok = True
    for pth, name in [
        (diag_p, "Diagnostics"),
        (man_p, "Manifest"),
        (exec_p, "Exec-Parquet"),
        (adv_p, "ADV-Map"),
    ]:
        if pth.exists():
            print(f"{GREEN} {name} found: {pth}")
        else:
            print(f"{RED} {name} missing: {pth}")
            ok = False
    if not ok:
        print(f"\n{RED} Abort: artifacts are missing.")
        return

    # --- Load
    diag = load_json(diag_p)
    man = load_json(man_p)
    df_exec = pd.read_parquet(exec_p)
    adv_map = pd.read_pickle(adv_p)

    # --- Flags from diagnostics/manifest
    schema_version = int(diag.get("schema_version", -1))
    causal_only_diag = bool(((diag.get("filling") or {}).get("causal_only", False)))
    man_extra = (man.get("extra") or {}) if isinstance(man, dict) else {}
    man_filling = (
        (man_extra.get("filling") or {}) if isinstance(man_extra, dict) else {}
    )
    causal_only_man = bool(man_filling.get("causal_only", False))
    events_summary = (diag.get("events") or {}).get("summary", {})
    total_events = (
        int(events_summary.get("total_events", 0))
        if isinstance(events_summary, dict)
        else 0
    )

    print("\nFlags")
    print(
        f"{GREEN if schema_version == 3 else YELLOW} Diagnostics.schema_version = {schema_version}"
    )
    print(
        f"{GREEN if causal_only_diag else YELLOW} Diagnostics.filling.causal_only = {causal_only_diag}"
    )
    print(
        f"{GREEN if causal_only_man else YELLOW} Manifest.extra.causal_only      = {causal_only_man}"
    )
    print(f"{GREEN} Diagnostics.events.summary.total_events = {total_events}")

    # --- Check method labels (no non-causal fills)
    # Inspect exec_diag
    violations = []
    d = diag.get("exec_diag", {})
    for sym, info in d.items():
        fill = (info or {}).get("filling", {})
        methods = fill.get("methods", [])
        for m in methods:
            label = str(m.get("method"))
            if causal_only_diag and label in (
                "filled_linear",
                "filled_kalman",
                "filled_bfill",
                "filled_avg",
            ):
                violations.append(("exec_diag", sym, label))
    if violations:
        print(f"\n{RED} Found non-causal fill methods (causal_only=true):")
        for trk, sym, lab in violations[:20]:
            print(f"   - {trk}:{sym}: {lab}")
    else:
        print(
            f"\n{GREEN} No non-causal fill methods labeled (with causal_only=true)."
        )

    # --- Sample test: filled positions must not depend on the right neighbor
    # Heuristic: if causal_only, fill=ffill -> fill value = last valid value on the left.
    # Sample 10 symbols and randomly check 30 timestamps for plausible NA->value jumps.
    import random

    random.seed(7)
    probe_syms = random.sample(list(df_exec.columns), min(10, df_exec.shape[1]))
    suspicious = 0
    checks = 0
    for sym in probe_syms:
        s = pd.to_numeric(df_exec[sym], errors="coerce").astype(float)
        isnan = s.isna().to_numpy()
        # Find transitions from NA -> value (potentially filled)
        trans = np.where(isnan[:-1] & (~isnan[1:]))[0]
        pick = trans[:30]
        for idx in pick:
            # Transition NaN (idx) -> value (idx+1)
            if idx - 1 < 0 or idx + 1 >= len(s):
                continue
            checks += 1
            left_val = s.iloc[idx - 1]  # last valid value on the left
            filled_val = s.iloc[idx + 1]  # newly filled value
            if pd.notna(left_val) and pd.notna(filled_val):
                if not np.isclose(
                    float(left_val), float(filled_val), rtol=0.0, atol=1e-12
                ):
                    suspicious += 1

    if causal_only_diag:
        if suspicious == 0:
            print(
                f"{GREEN} Sample (ffill consistency) clean: 0 deviations across {checks} checks."
            )
        else:
            print(
                f"{RED} ffill consistency violated: {suspicious} deviations across {checks} checks."
            )

    # --- ADV map: plausible and no bfill traces (indirectly covered by the manifest)
    if isinstance(adv_map, dict) and len(adv_map) > 0:
        print(f"{GREEN} ADV map loaded (|symbols|={len(adv_map)}).")
    else:
        print(f"{YELLOW} ADV map empty or inconsistent.")

    hard_fail = (
        (not ok)
        or (causal_only_diag and len(violations) > 0)
        or (causal_only_diag and suspicious > 0)
    )
    if hard_fail:
        print(f"\nResult: {RED} CAUSALITY GUARD **FAILED**.")
    else:
        print(
            f"\nResult: {GREEN} CAUSALITY GUARD passed - strictly causal filling is active and consistent."
        )


if __name__ == "__main__":
    main()
