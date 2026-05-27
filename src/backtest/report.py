from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from backtest.config import config_dict
from backtest.engine import BacktestResult


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
    result.daily[["equity"]].reset_index(names="date").to_csv(
        out / "equity.csv", index=False
    )
    result.positions.reset_index(names="date").to_csv(
        out / "positions.csv", index=False
    )
    result.weights.reset_index(names="date").to_csv(out / "weights.csv", index=False)
    result.trades.to_csv(out / "trades.csv", index=False)
    windows.to_csv(out / "windows.csv", index=False)
    selected_pairs.to_csv(out / "selected_pairs.csv", index=False)
    bo_trials.to_csv(out / "bo_trials.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(result.summary, indent=2), encoding="utf-8"
    )
    (out / "bo_best.json").write_text(
        json.dumps(bo_best, indent=2, default=str), encoding="utf-8"
    )
    (out / "config_used.yaml").write_text(
        yaml.safe_dump(config_dict(cfg), sort_keys=False), encoding="utf-8"
    )
    _plots(result.daily, windows, cfg.markov.enabled, out)


def _plots(daily: pd.DataFrame, windows: pd.DataFrame, markov: bool, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        for col, name in (("equity", "equity.png"), ("drawdown", "drawdown.png")):
            ax = daily[col].plot(title=col)
            ax.figure.tight_layout()
            ax.figure.savefig(out / name, dpi=130)
            plt.close(ax.figure)

        cols = ["entry_z", "exit_z", "stop_z"]
        if markov:
            cols += ["min_revert_prob", "horizon_days"]
        cols = [col for col in cols if col in windows]
        if cols:
            data = windows.set_index("window")[cols].astype(float)
            data = (data - data.min()) / (data.max() - data.min()).replace(0, 1) * 100
            ax = data.plot(marker="o", title="params")
            ax.set_xlabel("fold")
            ax.set_ylabel("normiert 0-100")
            ax.set_ylim(0, 100)
            ax.figure.tight_layout()
            ax.figure.savefig(out / "params.png", dpi=130)
            plt.close(ax.figure)
    except Exception:
        pass
