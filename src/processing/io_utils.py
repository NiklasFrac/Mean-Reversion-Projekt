from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger("io_utils")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def read_tickers_file(path: Path) -> list[str]:
    """
    Unterstuetzt:
      - CSV mit Spalte 'Symbol' oder 'ticker' (case-insensitive)
      - generisches CSV: erste Spalte
      - Plain-Text: ein Ticker je Zeile
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tickers file not found: {p}")
    try:
        df = pd.read_csv(p)
        cols_lower = {c.lower(): c for c in df.columns}
        col = cols_lower.get("symbol") or cols_lower.get("ticker")
        if col:
            s = df[col].astype(str).str.strip().str.upper()
            return [t for t in s.dropna().unique().tolist() if t]
        col0 = df.columns[0]
        s = df[col0].astype(str).str.strip().str.upper()
        return [t for t in s.dropna().unique().tolist() if t]
    except Exception as e:  # pragma: no cover - fallback
        logger.info("CSV parse fallback: %s", e)
        with p.open("r", encoding="utf-8") as f:
            lines = [line.strip().upper() for line in f if line.strip()]
            out = [ln.split()[-1] for ln in lines]
            seen: dict[str, int] = {}
            for t in out:
                if t not in seen:
                    seen[t] = 1
            return list(seen.keys())


def write_csv(
    path: Path, rows: Iterable[Sequence[object]], header: Sequence[object] | None = None
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)
