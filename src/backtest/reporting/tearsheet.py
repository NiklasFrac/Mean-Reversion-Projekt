from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter

__all__ = [
    "TearsheetConfig",
    "compute_drawdown",
    "compute_returns",
    "summarize_stats",
    "trade_hit_rate",
    "trade_count",
    "write_tearsheet",
]

TRADING_DAYS: int = 252
RETURN_EPS: float = 1e-12


@dataclass(frozen=True)
class TearsheetConfig:
    dpi: int = 150
    theme: str = "default"
    monthly_heatmap_min_months: int = 3


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def _validate_equity(eq: pd.Series) -> pd.Series:
    if not isinstance(eq, pd.Series):
        raise TypeError("eq must be a pandas Series")

    out = eq.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            try:
                idx = pd.to_datetime(out.index, errors="coerce", format="mixed")
            except TypeError:
                idx = pd.to_datetime(out.index, errors="coerce")
        except Exception as exc:
            raise TypeError("eq index must be datetime-like or convertible") from exc
        if pd.isna(idx).all():
            raise TypeError("eq index must be datetime-like or convertible")
        out.index = idx

    try:
        out = pd.to_numeric(out, errors="coerce")
    except Exception:
        out = out.apply(lambda v: pd.to_numeric(v, errors="coerce"))

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.name = out.name or "equity"
    return out


def compute_returns(eq: pd.Series, kind: str = "pct") -> pd.Series:
    s = _validate_equity(eq)
    if s.empty:
        return s.copy()
    if kind == "pct":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = cast(pd.Series, s.pct_change())
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        out.name = "returns_pct"
        return out
    if kind == "diff":
        out = cast(pd.Series, s.diff()).fillna(0.0).astype(float)
        out.name = "pnl_abs"
        return out
    raise ValueError("kind must be 'pct' or 'diff'")


def compute_drawdown(eq: pd.Series) -> pd.Series:
    s = _validate_equity(eq)
    if s.empty:
        return s.copy()
    peak = s.cummax()
    dd = (s / peak - 1.0).astype(float)
    dd.name = "drawdown"
    return dd


def trade_count(trades_df: pd.DataFrame | None) -> int:
    if trades_df is None or trades_df.empty or "net_pnl" not in trades_df.columns:
        return 0
    pnl = pd.to_numeric(trades_df["net_pnl"], errors="coerce")
    return int(pnl.notna().sum())


def trade_hit_rate(trades_df: pd.DataFrame | None) -> float:
    if trades_df is None or trades_df.empty or "net_pnl" not in trades_df.columns:
        return 0.0
    pnl = pd.to_numeric(trades_df["net_pnl"], errors="coerce").dropna()
    if pnl.empty:
        return 0.0
    return float((pnl > 0.0).mean())


def _year_fraction(start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max(int((end - start).days), 1)
    return float(days / 365.25)


def _nonzero(x: float, eps: float = RETURN_EPS) -> float:
    return x if abs(x) > eps else math.copysign(eps, x if x != 0.0 else 1.0)


def _downside_std(returns: pd.Series) -> float:
    neg = returns[returns < 0.0]
    if neg.empty:
        return 0.0
    return float(neg.std(ddof=1))


def _compound(returns: pd.Series) -> float:
    return float(np.prod(1.0 + returns.to_numpy(dtype=float, na_value=np.nan)))


def summarize_stats(
    eq: pd.Series,
    *,
    trades_df: pd.DataFrame | None = None,
    rf: float = 0.0,
    benchmark: Optional[pd.Series] = None,
) -> pd.DataFrame:
    _ = benchmark
    s = _validate_equity(eq)
    if s.empty:
        return pd.DataFrame()

    returns = compute_returns(s, kind="pct")
    start = cast(pd.Timestamp, s.index[0])
    end = cast(pd.Timestamp, s.index[-1])
    years = _year_fraction(start, end)
    total_return = _compound(returns) - 1.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    ann_vol = std * math.sqrt(TRADING_DAYS)
    ann_return = mean * TRADING_DAYS
    rf_daily = float(rf) / TRADING_DAYS
    excess_daily = returns - rf_daily
    sharpe = float(
        excess_daily.mean() / _nonzero(float(excess_daily.std(ddof=1)))
    ) * math.sqrt(TRADING_DAYS)
    sortino = float(
        (excess_daily.mean() * TRADING_DAYS)
        / _nonzero(_downside_std(excess_daily) * math.sqrt(TRADING_DAYS))
    )

    drawdown = compute_drawdown(s)
    max_drawdown = float(drawdown.min())
    calmar = (
        float(cagr / abs(_nonzero(max_drawdown)))
        if max_drawdown < 0.0
        else float("inf")
    )

    num_trades = trade_count(trades_df)
    hit_rate = trade_hit_rate(trades_df)

    data = {
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "n_days": int(len(s)),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "hit_rate": float(hit_rate),
        "num_trades": int(num_trades),
    }
    df = pd.DataFrame([data])
    df.attrs["_labels"] = {
        "start_date": "Start",
        "end_date": "End",
        "n_days": "Days",
        "total_return": "Total Return",
        "cagr": "CAGR",
        "ann_return": "Ann. Return",
        "ann_vol": "Ann. Vol",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "max_drawdown": "Max DD",
        "calmar": "Calmar",
        "hit_rate": "Hit Rate",
        "num_trades": "Trades",
    }
    return df


def _save_fig(fig: plt.Figure, out: Path, *, dpi: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def _fmt_pct(x: float, _pos: int) -> str:
    return f"{x:.0%}"


def _fmt_date(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))


def _plot_equity(eq: pd.Series, out_dir: Path, cfg: TearsheetConfig) -> None:
    eq_norm = (eq / float(eq.iloc[0])).astype(float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        eq_norm.index, eq_norm.to_numpy(dtype=float, na_value=np.nan), linewidth=1.6
    )
    ax.set_title("Equity Curve (normalized)")
    ax.grid(True, alpha=0.3)
    _fmt_date(ax)
    _save_fig(fig, out_dir / "equity_curve.png", dpi=cfg.dpi)


def _plot_drawdown(dd: pd.Series, out_dir: Path, cfg: TearsheetConfig) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    vals = dd.to_numpy(dtype=float, na_value=np.nan)
    ax.fill_between(dd.index, vals, 0.0, alpha=0.5)
    ax.set_title("Underwater Drawdown")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pct))
    ax.grid(True, alpha=0.3)
    _fmt_date(ax)
    _save_fig(fig, out_dir / "drawdown_underwater.png", dpi=cfg.dpi)


def _plot_monthly_heatmap(
    r_pct: pd.Series, out_dir: Path, cfg: TearsheetConfig
) -> None:
    if r_pct.empty:
        return
    idx = cast(pd.DatetimeIndex, r_pct.index)
    # PeriodIndex is tz-naive, so make the month bucketing explicit first.
    heatmap_idx = idx.tz_localize(None) if idx.tz is not None else idx
    months_seen = int(heatmap_idx.to_period("M").nunique())
    if months_seen < int(cfg.monthly_heatmap_min_months):
        return
    monthly = (1.0 + r_pct).groupby([heatmap_idx.year, heatmap_idx.month]).prod() - 1.0
    years = sorted(set(heatmap_idx.year))
    heat = np.full((len(years), 12), np.nan, dtype=float)
    for yi, year in enumerate(years):
        for month in range(1, 13):
            val = monthly.get((year, month), np.nan)
            heat[yi, month - 1] = float(val) if not pd.isna(val) else np.nan

    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.36 * len(years)))
    im = ax.imshow(heat, aspect="auto", interpolation="nearest")
    ax.set_title("Monthly Return Heatmap")
    ax.set_yticks(range(len(years)), [str(y) for y in years])
    ax.set_xticks(
        range(12),
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    for yi in range(len(years)):
        for mi in range(12):
            val = heat[yi, mi]
            if not np.isnan(val):
                ax.text(mi, yi, f"{val:.1%}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03, format=FuncFormatter(_fmt_pct))
    _save_fig(fig, out_dir / "monthly_heatmap.png", dpi=cfg.dpi)


def _stats_table_image(
    stats: pd.DataFrame, out_dir: Path, cfg: TearsheetConfig
) -> None:
    if stats.empty:
        return
    labels = cast(Mapping[str, str], stats.attrs.get("_labels", {}))
    preferred_order = [
        "start_date",
        "end_date",
        "n_days",
        "total_return",
        "cagr",
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "num_trades",
    ]
    row = cast(dict[str, Any], stats.iloc[0].to_dict())

    def _fmt_val(key: str, value: Any) -> str:
        if key in {"start_date", "end_date"}:
            return str(value)
        try:
            num = float(value)
        except Exception:
            return str(value)
        if key in {"n_days", "num_trades"}:
            return f"{num:.0f}"
        if key in {"sharpe", "sortino", "calmar"}:
            return f"{num:.2f}"
        return f"{num:.2%}"

    display = [
        (labels.get(key, key), _fmt_val(key, row[key]))
        for key in preferred_order
        if key in row
    ]
    fig, ax = plt.subplots(figsize=(8, 0.55 + 0.35 * len(display)))
    ax.axis("off")
    table = ax.table(
        cellText=[[key, value] for key, value in display],
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.2)
    ax.set_title("Summary Statistics", pad=10)
    _save_fig(fig, out_dir / "stats_table.png", dpi=cfg.dpi)


def write_tearsheet(
    eq: pd.Series,
    stats_df: Optional[pd.DataFrame],
    out_dir: Path,
    *,
    trades_df: pd.DataFrame | None = None,
    rf: float = 0.0,
    dpi: int = 150,
    benchmark: Optional[pd.Series] = None,
    rolling_window: int = 252,
    use_log_scale: bool = False,
    include_pdf: bool = False,
    save_svg: bool = False,
    phase_filter: Optional[str] = None,
) -> None:
    _ = stats_df
    _ = benchmark
    _ = rolling_window
    _ = use_log_scale
    _ = include_pdf
    _ = save_svg
    _ = phase_filter
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = _validate_equity(eq)
    if s.empty:
        return

    cfg = TearsheetConfig(dpi=int(dpi))
    plt.style.use(cfg.theme)
    stats = summarize_stats(s, trades_df=trades_df, rf=rf)
    stats.to_csv(out_dir / "stats.csv", index=False)
    payload = stats.to_dict(orient="records")[0] if not stats.empty else {}
    (out_dir / "stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    returns = compute_returns(s, kind="pct")
    drawdown = compute_drawdown(s)
    _plot_equity(s, out_dir, cfg)
    _plot_drawdown(drawdown, out_dir, cfg)
    _plot_monthly_heatmap(returns, out_dir, cfg)
    _stats_table_image(stats, out_dir, cfg)
