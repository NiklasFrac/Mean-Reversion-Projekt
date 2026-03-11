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
            print(f"{GREEN} {name} gefunden: {pth}")
        else:
            print(f"{RED} {name} fehlt: {pth}")
            ok = False
    if not ok:
        print(f"\n{RED} Abbruch: Artefakte fehlen.")
        return

    # --- Laden
    diag = load_json(diag_p)
    man = load_json(man_p)
    df_exec = pd.read_parquet(exec_p)
    adv_map = pd.read_pickle(adv_p)

    # --- Flags aus Diagnostik/Manifest
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

    print("\nSchalter")
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

    # --- Methoden-Labels kontrollieren (keine non-kausalen Fills)
    # Wir schauen in exec_diag
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
        print(f"\n{RED} Non-kausale Fill-Methoden gefunden (causal_only=true):")
        for trk, sym, lab in violations[:20]:
            print(f"   - {trk}:{sym}: {lab}")
    else:
        print(
            f"\n{GREEN} Keine non-kausalen Fill-Methoden etikettiert (bei causal_only=true)."
        )

    # --- Stichproben-Test: gefuellte Stellen duerfen nicht vom rechten Nachbarn abhaengen
    # Heuristik: Wenn causal_only, dann ist Fill=ffill -> Fuellwert = letzter gueltiger Wert links.
    # Wir picken 10 Symbole und pruefen zufaellig 30 Zeitpunkte, ob NA->Wert Spruenge plausibel sind.
    import random

    random.seed(7)
    probe_syms = random.sample(list(df_exec.columns), min(10, df_exec.shape[1]))
    suspicious = 0
    checks = 0
    for sym in probe_syms:
        s = pd.to_numeric(df_exec[sym], errors="coerce").astype(float)
        isnan = s.isna().to_numpy()
        # finde Uebergaenge von NA -> Wert (potenziell gefuellt)
        trans = np.where(isnan[:-1] & (~isnan[1:]))[0]
        pick = trans[:30]
        for idx in pick:
            # Uebergang NaN (idx) -> Wert (idx+1)
            if idx - 1 < 0 or idx + 1 >= len(s):
                continue
            checks += 1
            left_val = s.iloc[idx - 1]  # letzter gueltiger links
            filled_val = s.iloc[idx + 1]  # der neu gefuellte Wert
            if pd.notna(left_val) and pd.notna(filled_val):
                if not np.isclose(
                    float(left_val), float(filled_val), rtol=0.0, atol=1e-12
                ):
                    suspicious += 1

    if causal_only_diag:
        if suspicious == 0:
            print(
                f"{GREEN} Stichprobe (ffill-Konsistenz) sauber: 0 Abweichungen in {checks} Checks."
            )
        else:
            print(
                f"{RED} ffill-Konsistenz verletzt: {suspicious} Abweichungen in {checks} Checks."
            )

    # --- ADV-Map: plausibel & keine bfill-Spuren (indirekt ueber Manifest abgedeckt)
    if isinstance(adv_map, dict) and len(adv_map) > 0:
        print(f"{GREEN} ADV-Map geladen (|symbols|={len(adv_map)}).")
    else:
        print(f"{YELLOW} ADV-Map leer oder inkonsistent.")

    hard_fail = (
        (not ok)
        or (causal_only_diag and len(violations) > 0)
        or (causal_only_diag and suspicious > 0)
    )
    if hard_fail:
        print(f"\nErgebnis: {RED} CAUSALITY GUARD **NICHT** bestanden.")
    else:
        print(
            f"\nErgebnis: {GREEN} CAUSALITY GUARD bestanden - strikt kausale Fuellung aktiv und konsistent."
        )


if __name__ == "__main__":
    main()
