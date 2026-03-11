# check_lob_exec.py
import glob
from pathlib import Path

import pandas as pd


def main():
    root = Path.cwd()
    paths = sorted(glob.glob(str(root / "results" / "trades_te_*.csv")))
    if not paths:
        paths = sorted(glob.glob(str(root / "results" / "wf" / "trades_te_*.csv")))
    if not paths:
        print("❗ Keine trades_te_*.csv gefunden (results/ oder results/wf/).")
        return

    total_rows = 0
    total_exec_cols = set()
    total_filled = {}

    print(f"Gefundene Dateien: {len(paths)}")
    for p in paths:
        df = pd.read_csv(p, index_col=0)
        total_rows += len(df)
        exec_cols = [c for c in df.columns if c.startswith("exec_")]
        total_exec_cols.update(exec_cols)

        print(f"\n— {p}  rows={len(df)}  exec_cols={len(exec_cols)}")
        for c in sorted(exec_cols):
            n = int(df[c].notna().sum())
            if c not in total_filled:
                total_filled[c] = 0
            total_filled[c] += n
            print(f"   {c:25s} filled={n}")

        if "lob_net_pnl" in df.columns:
            n_lob = int(df["lob_net_pnl"].notna().sum())
            print(f"   lob_net_pnl              filled={n_lob}")

    print("\n===== Zusammenfassung =====")
    print(f"Gesamtzeilen (TE): {total_rows}")
    print(f"Verschiedene exec_-Spalten: {len(total_exec_cols)}")
    filled_sum = sum(total_filled.values())
    print(f"Summe gefüllter exec_-Felder: {filled_sum}")
    if filled_sum > 0:
        print("✅ LOB-Ausführung aktiv: exec_-Annotationen vorhanden.")
    else:
        print(
            "⚠️  Keine gefüllten exec_-Felder gefunden. YAML prüfen (execution.mode: lob) oder Trade-Felder (Leg/Preise) fehlen."
        )


if __name__ == "__main__":
    main()
