from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else 100.0 * n / d


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/check_maker_mix.py <trades_csv>")
        return 2

    rows = load(Path(sys.argv[1]))
    counts: Counter[tuple[str, str]] = Counter()
    keys = ["y_entry", "y_exit", "x_entry", "x_exit"]

    for row in rows:
        counts[("y_entry", row["liquidity_y_entry"])] += 1
        counts[("y_exit", row["liquidity_y_exit"])] += 1
        counts[("x_entry", row["liquidity_x_entry"])] += 1
        counts[("x_exit", row["liquidity_x_exit"])] += 1

    totals = {
        key: sum(value for kk, value in counts.items() if kk[0] == key) for key in keys
    }
    for key in keys:
        maker = counts[(key, "maker")]
        taker = counts[(key, "taker")]
        print(
            f"{key:8s} | maker={maker:3d} ({pct(maker, totals[key]):5.1f}%)  "
            f"taker={taker:3d} ({pct(taker, totals[key]):5.1f}%)"
        )

    # If maker_frac columns are populated, check their averages.
    try:
        import statistics

        mk = [float(row.get("maker_frac_entry_y", 0.0)) for row in rows]
        print(f"avg maker_frac_entry_y: {statistics.mean(mk):.3f}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
