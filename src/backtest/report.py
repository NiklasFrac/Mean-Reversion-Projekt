from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from backtest.config import config_dict
from backtest.engine import BacktestResult
from backtest.wf_debug import build_wf_debug


def write_report(
    result: BacktestResult,
    out_dir: str | Path,
    *,
    cfg: Any,
    windows: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    bo_trials: pd.DataFrame,
    bo_best: list[dict[str, Any]],
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.daily.reset_index(names="date").to_csv(out / "daily.csv", index=False)
    result.positions.reset_index(names="date").to_csv(
        out / "positions.csv", index=False
    )
    result.weights.reset_index(names="date").to_csv(out / "weights.csv", index=False)
    result.trades.to_csv(out / "trades.csv", index=False)
    windows.to_csv(out / "windows.csv", index=False)
    selected_pairs.to_csv(out / "selected_pairs.csv", index=False)
    bo_trials.to_csv(out / "bo_trials.csv", index=False)
    (out / "wf_debug.json").write_text(
        json.dumps(
            build_wf_debug(result, windows, selected_pairs, bo_trials, bo_best, cfg),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (out / "summary.json").write_text(
        json.dumps(result.summary, indent=2), encoding="utf-8"
    )
    (out / "bo_best.json").write_text(
        json.dumps(bo_best, indent=2, default=str), encoding="utf-8"
    )
    (out / "config_used.yaml").write_text(
        yaml.safe_dump(config_dict(cfg), sort_keys=False), encoding="utf-8"
    )
    _plots(result.daily, out)


def _plots(daily: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        for col, name in (("equity", "equity.png"), ("drawdown", "drawdown.png")):
            ax = daily[col].plot(title=col)
            ax.figure.tight_layout()
            ax.figure.savefig(out / name, dpi=130)
            plt.close(ax.figure)
    except Exception:
        pass
