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
        print("No trades_te_*.csv found (results/ or results/wf/).")
        return

    total_rows = 0
    total_exec_cols = set()
    total_filled = {}

    print(f"Found files: {len(paths)}")
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

    print("\n===== Summary =====")
    print(f"Total rows (TE): {total_rows}")
    print(f"Distinct exec_ columns: {len(total_exec_cols)}")
    filled_sum = sum(total_filled.values())
    print(f"Sum of filled exec_ fields: {filled_sum}")
    if filled_sum > 0:
        print("LOB execution active: exec_ annotations present.")
    else:
        print(
            "No filled exec_ fields found. Check YAML (execution.mode: lob) or missing trade fields (legs/prices)."
        )


if __name__ == "__main__":
    main()
