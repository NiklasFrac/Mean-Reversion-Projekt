from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.config.types import ReportingConfig
from backtest.reporting.tearsheet import summarize_stats, write_tearsheet
from backtest.utils.io import write_json


def report_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "report"


def debug_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "debug"


def _summary_payload(
    eq: pd.Series, trades_df: pd.DataFrame | None = None
) -> dict[str, Any]:
    stats = summarize_stats(eq, trades_df=trades_df)
    if stats.empty:
        return {}
    return cast(dict[str, Any], stats.to_dict(orient="records")[0])


def _cast_number(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if np.isfinite(out):
        return out
    return None


def _median_or_none(series: pd.Series) -> float | None:
    vals = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if vals.empty:
        return None
    return float(vals.median())


def _iqr_or_none(series: pd.Series) -> float | None:
    vals = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if vals.empty:
        return None
    return float(vals.quantile(0.75) - vals.quantile(0.25))


def _write_placeholder_plot(path: Path, *, title: str, body: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, body, ha="center", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _refit_value(item: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _write_test_equity(report_root: Path, eq: pd.Series) -> None:
    df = pd.DataFrame({"date": eq.index, "equity": eq.to_numpy(dtype=float)})
    df.to_csv(report_root / "test_equity.csv", index=False)


def _write_test_summary(
    report_root: Path,
    *,
    eq: pd.Series,
    trades_df: pd.DataFrame,
    window_rows: pd.DataFrame | None,
) -> dict[str, Any]:
    payload = _summary_payload(eq, trades_df=trades_df)
    if window_rows is not None and not window_rows.empty:
        wdf = window_rows.copy()
        payload.update(
            {
                "window_count": int(len(wdf)),
                "positive_window_rate": float(
                    (pd.to_numeric(wdf["total_return"], errors="coerce") > 0.0).mean()
                )
                if "total_return" in wdf.columns
                else 0.0,
                "window_sharpe_median": _median_or_none(wdf["sharpe"])
                if "sharpe" in wdf.columns
                else None,
                "window_return_median": _median_or_none(wdf["total_return"])
                if "total_return" in wdf.columns
                else None,
                "worst_window_return": (
                    float(pd.to_numeric(wdf["total_return"], errors="coerce").min())
                    if "total_return" in wdf.columns
                    and pd.to_numeric(wdf["total_return"], errors="coerce")
                    .notna()
                    .any()
                    else None
                ),
            }
        )
        wdf.to_csv(report_root / "test_window_summary.csv", index=False)
    write_json(report_root / "test_summary.json", payload)
    return payload


def _write_test_tearsheet(
    report_root: Path,
    *,
    eq: pd.Series,
    trades_df: pd.DataFrame,
    cfg: ReportingConfig,
) -> None:
    if not cfg.test_tearsheet.enabled:
        return
    write_tearsheet(
        eq,
        stats_df=None,
        out_dir=report_root / "test_tearsheet",
        trades_df=trades_df,
        dpi=cfg.test_tearsheet.dpi,
    )


def _cv_score_series(cv_scores: pd.DataFrame | None) -> pd.Series:
    if cv_scores is None or cv_scores.empty:
        return pd.Series(dtype=float)
    score_col = (
        "score"
        if "score" in cv_scores.columns
        else "oos_score"
        if "oos_score" in cv_scores.columns
        else None
    )
    if score_col is None:
        return pd.Series(dtype=float)
    return (
        pd.to_numeric(cv_scores[score_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _write_train_cv_csv(
    report_root: Path, cv_scores: pd.DataFrame | None
) -> pd.DataFrame:
    cols = ["wf_i", "fold_id", "score", "selection_metric", "component"]
    if cv_scores is None or cv_scores.empty:
        empty = pd.DataFrame(columns=cols)
        empty.to_csv(report_root / "train_cv_scores.csv", index=False)
        return empty
    df = cv_scores.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan if col != "selection_metric" and col != "component" else ""
    df = df.loc[:, cols]
    df.to_csv(report_root / "train_cv_scores.csv", index=False)
    return df


def _write_train_cv_plot(
    report_root: Path, *, cv_scores: pd.DataFrame, dpi: int
) -> None:
    path = report_root / "train_cv_scores.png"
    if cv_scores.empty:
        _write_placeholder_plot(
            path,
            title="Train CV Scores",
            body="CV not available",
            dpi=dpi,
        )
        return

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 4))
    if (
        "wf_i" in cv_scores.columns
        and pd.to_numeric(cv_scores["wf_i"], errors="coerce").notna().any()
    ):
        plot_df = cv_scores.copy()
        plot_df["wf_i"] = pd.to_numeric(plot_df["wf_i"], errors="coerce")
        plot_df["score"] = pd.to_numeric(plot_df["score"], errors="coerce")
        groups = [
            plot_df.loc[plot_df["wf_i"] == wf_i, "score"].dropna().to_numpy(dtype=float)
            for wf_i in sorted(plot_df["wf_i"].dropna().unique())
        ]
        labels = [
            f"WF-{int(wf_i):03d}" for wf_i in sorted(plot_df["wf_i"].dropna().unique())
        ]
        if groups:
            ax.boxplot(groups, tick_labels=labels, showfliers=True)
            for pos, values in enumerate(groups, start=1):
                if values.size:
                    ax.scatter(np.full(values.size, pos), values, alpha=0.6, s=18)
            ax.set_title("Train CV Scores by Walkforward Window")
            ax.set_ylabel("Score")
        else:
            _write_placeholder_plot(
                path, title="Train CV Scores", body="CV not available", dpi=dpi
            )
            plt.close(fig)
            return
    else:
        vals = (
            pd.to_numeric(cv_scores["score"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        ax.plot(np.arange(len(vals)), vals, marker="o", linewidth=1.2)
        ax.axhline(float(np.median(vals)), linestyle="--", linewidth=1.0)
        ax.set_title("Train CV Scores")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _train_refit_summary(refits: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not refits:
        return {}
    if len(refits) == 1:
        summary_raw = _refit_value(refits[0], "summary")
        summary = dict(summary_raw or {})
        summary["window_count"] = 1
        summary["aggregate_method"] = "single"
        return summary

    rows = []
    for row in refits:
        summary = _refit_value(row, "summary")
        if isinstance(summary, Mapping):
            rows.append(dict(summary))
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "window_count": int(len(rows)),
        "aggregate_method": "median_per_window",
    }
    for col in ("sharpe", "cagr", "max_drawdown", "hit_rate", "num_trades"):
        if col not in df.columns:
            continue
        val = _median_or_none(df[col])
        if val is None:
            continue
        if col == "num_trades":
            out[col] = int(round(val))
        else:
            out[col] = float(val)
    return out


def _write_train_refit_equity_csv(
    report_root: Path,
    *,
    refits: list[Mapping[str, Any]],
) -> pd.DataFrame:
    if not refits:
        empty = pd.DataFrame(columns=["series", "wf_i", "step", "equity_norm"])
        empty.to_csv(report_root / "train_refit_equity.csv", index=False)
        return empty

    if len(refits) == 1:
        eq = _refit_value(refits[0], "equity")
        if isinstance(eq, pd.Series) and not eq.empty:
            s = pd.to_numeric(eq, errors="coerce").dropna()
            start = float(s.iloc[0]) if not s.empty else 1.0
            df = pd.DataFrame(
                {
                    "date": s.index,
                    "equity": s.to_numpy(dtype=float),
                    "equity_norm": (s / start).to_numpy(dtype=float)
                    if start != 0.0
                    else np.ones(len(s)),
                }
            )
            df.to_csv(report_root / "train_refit_equity.csv", index=False)
            return df
    rows: list[dict[str, Any]] = []
    window_curves: list[pd.Series] = []
    for item in refits:
        eq = _refit_value(item, "equity")
        wf_i = _refit_value(item, "wf_i")
        if not isinstance(eq, pd.Series) or eq.empty:
            continue
        s = pd.to_numeric(eq, errors="coerce").dropna()
        if s.empty:
            continue
        base = float(s.iloc[0])
        if base == 0.0:
            continue
        norm = (s / base).astype(float).reset_index(drop=True)
        norm.index = pd.Index(range(len(norm)), name="step")
        window_curves.append(norm)
        for step, value in norm.items():
            rows.append(
                {
                    "series": "window",
                    "wf_i": int(wf_i) if wf_i is not None and str(wf_i) != "" else None,
                    "step": int(step),
                    "equity_norm": float(value),
                }
            )
    if window_curves:
        overlay = pd.concat(window_curves, axis=1)
        median = overlay.median(axis=1, skipna=True)
        for step, value in median.items():
            rows.append(
                {
                    "series": "median",
                    "wf_i": None,
                    "step": int(step),
                    "equity_norm": float(value),
                }
            )
    df = pd.DataFrame(rows, columns=["series", "wf_i", "step", "equity_norm"])
    df.to_csv(report_root / "train_refit_equity.csv", index=False)
    return df


def _write_train_refit_plot(
    report_root: Path, *, equity_df: pd.DataFrame, dpi: int
) -> None:
    path = report_root / "train_refit_equity.png"
    if equity_df.empty:
        _write_placeholder_plot(
            path, title="Train Refit Equity", body="Train refit not available", dpi=dpi
        )
        return

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 4))
    if {"date", "equity_norm"}.issubset(equity_df.columns):
        ax.plot(
            pd.to_datetime(equity_df["date"], errors="coerce"),
            pd.to_numeric(equity_df["equity_norm"], errors="coerce"),
            linewidth=1.5,
        )
        ax.set_title("Train Refit Equity")
        ax.set_ylabel("Normalized Equity")
    else:
        plot_df = equity_df.copy()
        plot_df["step"] = pd.to_numeric(plot_df["step"], errors="coerce")
        plot_df["equity_norm"] = pd.to_numeric(plot_df["equity_norm"], errors="coerce")
        for wf_i, grp in plot_df.loc[plot_df["series"] == "window"].groupby(
            "wf_i", dropna=False
        ):
            _ = wf_i
            ax.plot(
                grp["step"], grp["equity_norm"], linewidth=1.0, alpha=0.25, color="0.4"
            )
        median = plot_df.loc[plot_df["series"] == "median"]
        if not median.empty:
            ax.plot(
                median["step"], median["equity_norm"], linewidth=2.0, color="tab:blue"
            )
        ax.set_title("Train Refit Equity Overlay")
        ax.set_xlabel("Relative Step")
        ax.set_ylabel("Normalized Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _gini_from_abs(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").abs().dropna().to_numpy(dtype=float)
    if arr.size == 0 or np.all(arr == 0.0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    gini = (n + 1 - 2 * (cum / cum[-1]).sum()) / n
    return float(max(0.0, min(1.0, gini)))


def _collect_train_trades(refits: list[Mapping[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item in refits:
        trades = _refit_value(item, "trades")
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            frames.append(trades.copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _write_train_pair_concentration_csv(
    report_root: Path,
    *,
    refits: list[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = ["pair", "net_pnl", "abs_net_pnl", "share_abs", "n_trades"]
    trades = _collect_train_trades(refits)
    if (
        trades.empty
        or "pair" not in trades.columns
        or "net_pnl" not in trades.columns
    ):
        empty = pd.DataFrame(columns=cols)
        empty.to_csv(report_root / "train_pair_concentration.csv", index=False)
        payload = {
            "available": False,
            "n_pairs": 0,
            "n_trades": 0,
            "top5_abs_share": None,
            "gini_abs": None,
        }
        write_json(report_root / "train_pair_concentration.json", payload)
        return empty, payload

    df = trades.copy()
    df["net_pnl"] = pd.to_numeric(df["net_pnl"], errors="coerce")
    df = df.dropna(subset=["net_pnl"])
    grouped = (
        df.groupby("pair", dropna=False)
        .agg(net_pnl=("net_pnl", "sum"), n_trades=("net_pnl", "size"))
        .reset_index()
    )
    if grouped.empty:
        empty = pd.DataFrame(columns=cols)
        empty.to_csv(report_root / "train_pair_concentration.csv", index=False)
        payload = {
            "available": False,
            "n_pairs": 0,
            "n_trades": int(len(df)),
            "top5_abs_share": None,
            "gini_abs": None,
        }
        write_json(report_root / "train_pair_concentration.json", payload)
        return empty, payload

    grouped["abs_net_pnl"] = grouped["net_pnl"].abs()
    abs_total = float(grouped["abs_net_pnl"].sum())
    grouped["share_abs"] = (
        grouped["abs_net_pnl"] / abs_total if abs_total > 0.0 else 0.0
    )
    grouped = grouped.sort_values(
        ["abs_net_pnl", "net_pnl"], ascending=[False, False]
    ).reset_index(drop=True)
    grouped = grouped.loc[:, cols]
    grouped.to_csv(report_root / "train_pair_concentration.csv", index=False)

    payload = {
        "available": True,
        "n_pairs": int(len(grouped)),
        "n_trades": int(len(df)),
        "top5_abs_share": (
            float(grouped["share_abs"].head(5).sum()) if abs_total > 0.0 else 0.0
        ),
        "gini_abs": _gini_from_abs(grouped["net_pnl"]),
        "top_pairs": grouped.head(10).to_dict(orient="records"),
    }
    write_json(report_root / "train_pair_concentration.json", payload)
    return grouped, payload


def _write_train_pair_concentration_plot(
    report_root: Path,
    *,
    concentration_df: pd.DataFrame,
    summary: Mapping[str, Any],
    dpi: int,
) -> None:
    path = report_root / "train_pair_concentration.png"
    if concentration_df.empty:
        _write_placeholder_plot(
            path,
            title="Train Pair Concentration",
            body="Train pair concentration not available",
            dpi=dpi,
        )
        return

    plot_df = concentration_df.head(10).copy().iloc[::-1]
    plot_df["net_pnl"] = pd.to_numeric(plot_df["net_pnl"], errors="coerce").fillna(0.0)
    colors = np.where(plot_df["net_pnl"] >= 0.0, "tab:green", "tab:red")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(plot_df["pair"].astype(str), plot_df["net_pnl"], color=colors, alpha=0.85)
    ax.axvline(0.0, color="0.25", linewidth=1.0)
    ax.set_title("Train Pair PnL Concentration")
    ax.set_xlabel("Net PnL")
    ax.set_ylabel("Pair")
    ax.grid(True, axis="x", alpha=0.25)

    txt = (
        f"pairs={int(summary.get('n_pairs', 0))}   "
        f"trades={int(summary.get('n_trades', 0))}   "
        f"top5 abs share={float(summary.get('top5_abs_share', 0.0)):.1%}   "
        f"gini(abs)={float(summary.get('gini_abs', 0.0)):.2f}"
    )
    fig.text(0.01, 0.01, txt, ha="left", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _score_col_from_trials(df: pd.DataFrame) -> str | None:
    for col in ("score", "oos_score", "sharpe"):
        if col in df.columns:
            return col
    return None


def _write_train_param_stability_csv(
    report_root: Path,
    *,
    bo_trials: pd.DataFrame | None,
) -> pd.DataFrame:
    cols = [
        "wf_i",
        "component",
        "model_id",
        "score",
        "entry_z",
        "exit_z",
        "stop_z",
        "min_revert_prob",
        "horizon_days",
    ]
    if bo_trials is None or bo_trials.empty:
        empty = pd.DataFrame(columns=cols)
        empty.to_csv(report_root / "train_param_stability.csv", index=False)
        return empty

    df = bo_trials.copy()
    if "fold_id" in df.columns:
        fold_num = pd.to_numeric(df["fold_id"], errors="coerce")
        if fold_num.isna().any():
            df = df.loc[fold_num.isna()].copy()
    score_col = _score_col_from_trials(df)
    if score_col is None or "params_json" not in df.columns:
        empty = pd.DataFrame(columns=cols)
        empty.to_csv(report_root / "train_param_stability.csv", index=False)
        return empty

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        params_raw = row.get("params_json")
        try:
            params = json.loads(str(params_raw))
        except Exception:
            continue
        if not isinstance(params, Mapping):
            continue
        score = _cast_number(row.get(score_col))
        if score is None:
            continue
        rows.append(
            {
                "wf_i": _cast_number(row.get("wf_i")),
                "component": str(row.get("component", "")),
                "model_id": str(row.get("model_id", "")),
                "score": float(score),
                "entry_z": _cast_number(params.get("entry_z")),
                "exit_z": _cast_number(params.get("exit_z")),
                "stop_z": _cast_number(params.get("stop_z")),
                "min_revert_prob": _cast_number(params.get("min_revert_prob")),
                "horizon_days": _cast_number(params.get("horizon_days")),
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(report_root / "train_param_stability.csv", index=False)
    return out


def _plot_param_surface_signal(ax: plt.Axes, df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["entry_z", "exit_z", "score"]).copy()
    if plot_df.empty:
        ax.axis("off")
        ax.set_title("Signal Surface")
        ax.text(0.5, 0.5, "Signal BO not available", ha="center", va="center")
        return
    stop = pd.to_numeric(plot_df["stop_z"], errors="coerce").fillna(
        pd.to_numeric(plot_df["stop_z"], errors="coerce").median()
    )
    stop_min = float(stop.min()) if not stop.empty else 1.0
    stop_span = float(stop.max() - stop.min()) if not stop.empty else 0.0
    sizes = 60.0 + 120.0 * ((stop - stop_min) / stop_span) if stop_span > 0 else 90.0
    sc = ax.scatter(
        plot_df["entry_z"],
        plot_df["exit_z"],
        c=plot_df["score"],
        s=sizes,
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )
    best = plot_df.loc[plot_df["score"].idxmax()]
    ax.scatter(
        [best["entry_z"]],
        [best["exit_z"]],
        marker="*",
        s=220,
        c="gold",
        edgecolors="black",
        linewidths=0.8,
    )
    ax.set_title("Signal Surface")
    ax.set_xlabel("entry_z")
    ax.set_ylabel("exit_z")
    ax.grid(True, alpha=0.25)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("score")


def _plot_param_surface_markov(ax: plt.Axes, df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["min_revert_prob", "horizon_days", "score"]).copy()
    if plot_df.empty:
        ax.axis("off")
        ax.set_title("Markov Surface")
        ax.text(0.5, 0.5, "Markov BO not available", ha="center", va="center")
        return
    sc = ax.scatter(
        plot_df["min_revert_prob"],
        plot_df["horizon_days"],
        c=plot_df["score"],
        s=95,
        cmap="plasma",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )
    best = plot_df.loc[plot_df["score"].idxmax()]
    ax.scatter(
        [best["min_revert_prob"]],
        [best["horizon_days"]],
        marker="*",
        s=220,
        c="gold",
        edgecolors="black",
        linewidths=0.8,
    )
    ax.set_title("Markov Surface")
    ax.set_xlabel("min_revert_prob")
    ax.set_ylabel("horizon_days")
    ax.grid(True, alpha=0.25)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("score")


def _write_train_param_stability_plot(
    report_root: Path,
    *,
    stability_df: pd.DataFrame,
    dpi: int,
) -> None:
    path = report_root / "train_param_stability.png"
    if stability_df.empty:
        _write_placeholder_plot(
            path,
            title="Train Parameter Stability",
            body="BO parameter stability not available",
            dpi=dpi,
        )
        return

    signal_df = stability_df.loc[stability_df["component"] == "theta_sig"].copy()
    markov_df = stability_df.loc[stability_df["component"] == "theta_markov"].copy()
    n_panels = int(bool(not signal_df.empty)) + int(bool(not markov_df.empty))
    if n_panels == 0:
        _write_placeholder_plot(
            path,
            title="Train Parameter Stability",
            body="BO parameter stability not available",
            dpi=dpi,
        )
        return

    plt.style.use("default")
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    axes_arr = np.atleast_1d(axes)
    pos = 0
    if not signal_df.empty:
        _plot_param_surface_signal(axes_arr[pos], signal_df)
        pos += 1
    if not markov_df.empty:
        _plot_param_surface_markov(axes_arr[pos], markov_df)
    wf_count = int(
        pd.to_numeric(stability_df["wf_i"], errors="coerce").dropna().nunique()
    )
    fig.suptitle("Train Parameter Stability", y=1.02)
    fig.text(
        0.01,
        0.01,
        f"candidates={int(len(stability_df))}   windows={wf_count if wf_count > 0 else 1}",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _write_train_selection_summary(
    report_root: Path,
    *,
    refits: list[Mapping[str, Any]],
    cv_scores: pd.DataFrame,
) -> dict[str, Any]:
    cv_series = _cv_score_series(cv_scores)
    refit_summary = _train_refit_summary(refits)
    selection_metric = None
    if not cv_scores.empty and "selection_metric" in cv_scores.columns:
        vals = [
            str(v).strip()
            for v in cv_scores["selection_metric"].tolist()
            if str(v).strip()
        ]
        if vals:
            selection_metric = vals[0]

    payload = {
        "cv_available": bool(not cv_series.empty),
        "cv": {
            "selection_metric": selection_metric or "sharpe",
            "fold_count": int(len(cv_series)),
            "score_median": float(cv_series.median()) if not cv_series.empty else None,
            "score_iqr": _iqr_or_none(cv_series),
            "score_min": float(cv_series.min()) if not cv_series.empty else None,
            "score_max": float(cv_series.max()) if not cv_series.empty else None,
            "window_count_with_cv": (
                int(
                    pd.to_numeric(cv_scores["wf_i"], errors="coerce").dropna().nunique()
                )
                if not cv_scores.empty
                and "wf_i" in cv_scores.columns
                and pd.to_numeric(cv_scores["wf_i"], errors="coerce").notna().any()
                else (1 if not cv_series.empty else 0)
            ),
        },
        "refit": refit_summary,
        "headline": {
            "sharpe": refit_summary.get("sharpe"),
            "cagr": refit_summary.get("cagr"),
            "max_drawdown": refit_summary.get("max_drawdown"),
            "hit_rate": refit_summary.get("hit_rate"),
            "num_trades": refit_summary.get("num_trades"),
        },
    }
    write_json(report_root / "train_selection_summary.json", payload)
    return payload


def write_core_report(
    out_dir: Path,
    *,
    reporting_cfg: ReportingConfig,
    test_eq: pd.Series,
    test_trades: pd.DataFrame,
    train_refits: list[Mapping[str, Any]],
    cv_scores: pd.DataFrame | None = None,
    bo_trials: pd.DataFrame | None = None,
    window_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    report_root = report_dir(out_dir)
    report_root.mkdir(parents=True, exist_ok=True)

    test_eq_clean = pd.to_numeric(test_eq, errors="coerce")
    test_eq_clean.name = "equity"
    if isinstance(test_eq_clean.index, pd.DatetimeIndex):
        test_eq_clean = test_eq_clean.dropna().sort_index()

    _write_test_equity(report_root, test_eq_clean)
    test_trades.to_csv(report_root / "test_trades.csv", index=False)
    test_summary = _write_test_summary(
        report_root,
        eq=test_eq_clean,
        trades_df=test_trades,
        window_rows=window_rows,
    )
    _write_test_tearsheet(
        report_root, eq=test_eq_clean, trades_df=test_trades, cfg=reporting_cfg
    )

    cv_df = _write_train_cv_csv(report_root, cv_scores)
    _write_train_cv_plot(
        report_root, cv_scores=cv_df, dpi=reporting_cfg.test_tearsheet.dpi
    )
    train_equity_df = _write_train_refit_equity_csv(report_root, refits=train_refits)
    _write_train_refit_plot(
        report_root, equity_df=train_equity_df, dpi=reporting_cfg.test_tearsheet.dpi
    )
    train_pair_df, train_pair_summary = _write_train_pair_concentration_csv(
        report_root, refits=train_refits
    )
    _write_train_pair_concentration_plot(
        report_root,
        concentration_df=train_pair_df,
        summary=train_pair_summary,
        dpi=reporting_cfg.test_tearsheet.dpi,
    )
    train_stability_df = _write_train_param_stability_csv(
        report_root, bo_trials=bo_trials
    )
    _write_train_param_stability_plot(
        report_root,
        stability_df=train_stability_df,
        dpi=reporting_cfg.test_tearsheet.dpi,
    )
    train_summary = _write_train_selection_summary(
        report_root, refits=train_refits, cv_scores=cv_df
    )

    return {
        "report_dir": str(report_root),
        "test_summary": test_summary,
        "train_summary": train_summary,
    }
