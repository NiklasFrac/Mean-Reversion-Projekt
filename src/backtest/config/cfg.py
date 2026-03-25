"""
Strict backtest config parsing and serialization.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar, cast

import pandas as pd

from backtest.utils.io import load_yaml_dict

from .types import (
    AppConfig,
    BOConfig,
    BOCVConfig,
    BOFastConfig,
    BORealisticConfig,
    BacktestSettings,
    BorrowAvailabilityPoint,
    BorrowConfig,
    BorrowEnforcementConfig,
    BorrowRatePoint,
    DataConfig,
    ExecutionConfig,
    ExecutionFeesConfig,
    HalfLifeConfig,
    LOBExecutionConfig,
    LOBFillModelConfig,
    LOBLiquidityConfig,
    LOBOrderFlowConfig,
    LOBOrderFlowSideConfig,
    LOBPostCostConfig,
    LOBStressModelConfig,
    LightExecutionConfig,
    MarkovFilterConfig,
    PairPrefilterConfig,
    ReportingConfig,
    ReportingTearsheetConfig,
    RiskConfig,
    RuntimeConfig,
    ShortAvailabilityHeuristicConfig,
    SignalConfig,
    SplitWindowConfig,
    SpreadZscoreConfig,
    StrategyConfig,
    WalkforwardConfig,
    WindowRangeConfig,
)

__all__ = [
    "AppConfig",
    "config_to_dict",
    "load_config",
    "parse_config",
]


_T = TypeVar("_T")

_ROOT_KEYS = {
    "runtime",
    "data",
    "backtest",
    "risk",
    "strategy",
    "signal",
    "spread_zscore",
    "markov_filter",
    "pair_prefilter",
    "borrow",
    "execution",
    "reporting",
    "bo",
    "cv",
}


def _as_dict(value: Any, *, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{section} must be a mapping")
    return dict(value)


def _expect_keys(section: str, raw: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(raw.keys()) - set(allowed))
    if extra:
        joined = ", ".join(f"{section}.{key}" for key in extra)
        raise KeyError(f"Unknown config keys: {joined}")


def _require_sections(cfg: Mapping[str, Any], sections: Sequence[str]) -> None:
    missing = [name for name in sections if name not in cfg]
    if missing:
        raise KeyError("Missing required config sections: " + ", ".join(missing))


def _to_int(value: Any, *, section: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise TypeError(f"{section} must be an int") from exc


def _to_int_opt(value: Any, *, section: str) -> int | None:
    if value is None:
        return None
    return _to_int(value, section=section)


def _to_float(value: Any, *, section: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise TypeError(f"{section} must be a float") from exc
    if not pd.notna(out):
        raise ValueError(f"{section} must be finite")
    return float(out)


def _to_str(value: Any, *, section: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{section} must be a string")
    out = value.strip()
    if not out:
        raise ValueError(f"{section} must not be empty")
    return out


def _to_bool(value: Any, *, section: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{section} must be a bool")


def _to_literal(value: Any, *, section: str, allowed: Sequence[_T]) -> _T:
    if value not in allowed:
        joined = ", ".join(repr(v) for v in allowed)
        raise ValueError(f"{section} must be one of ({joined})")
    return cast(_T, value)


def _to_str_opt(value: Any, *, section: str) -> str | None:
    if value is None:
        return None
    return _to_str(value, section=section)


def _to_float_opt(value: Any, *, section: str) -> float | None:
    if value is None:
        return None
    return _to_float(value, section=section)


def _to_range(
    value: Any,
    *,
    section: str,
    cast_item: type[float] | type[int],
) -> tuple[float, float] | tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{section} must be a two-element list")
    if cast_item is int:
        lo = _to_int(value[0], section=f"{section}[0]")
        hi = _to_int(value[1], section=f"{section}[1]")
        return (int(lo), int(hi))
    lo_f = _to_float(value[0], section=f"{section}[0]")
    hi_f = _to_float(value[1], section=f"{section}[1]")
    return (float(lo_f), float(hi_f))


def _normalize_symbol(symbol: Any, *, section: str) -> str:
    return _to_str(symbol, section=section).upper()


def _parse_runtime(raw: Mapping[str, Any]) -> RuntimeConfig:
    _expect_keys("runtime", raw, {"seed", "require_execution_hooks"})
    return RuntimeConfig(
        seed=_to_int(raw.get("seed", 42), section="runtime.seed"),
        require_execution_hooks=_to_bool(
            raw.get("require_execution_hooks", False),
            section="runtime.require_execution_hooks",
        ),
    )


def _parse_data(raw: Mapping[str, Any]) -> DataConfig:
    _expect_keys(
        "data",
        raw,
        {
            "input_mode",
            "analysis_meta_path",
            "prices_path",
            "pairs_path",
            "adv_map_path",
            "calendar_name",
            "prefer_col",
        },
    )
    input_mode = _to_literal(
        raw.get("input_mode", "explicit"),
        section="data.input_mode",
        allowed=("explicit", "analysis_meta"),
    )
    return DataConfig(
        input_mode=input_mode,
        analysis_meta_path=_to_str_opt(
            raw.get("analysis_meta_path"), section="data.analysis_meta_path"
        ),
        prices_path=_to_str(raw.get("prices_path", ""), section="data.prices_path"),
        pairs_path=_to_str(raw.get("pairs_path", ""), section="data.pairs_path"),
        adv_map_path=_to_str_opt(raw.get("adv_map_path"), section="data.adv_map_path"),
        calendar_name=_to_str(
            raw.get("calendar_name", "XNYS"), section="data.calendar_name"
        ),
        prefer_col=_to_str(raw.get("prefer_col", "close"), section="data.prefer_col"),
    )


def _parse_split_window(section: str, raw: Mapping[str, Any]) -> SplitWindowConfig:
    _expect_keys(section, raw, {"start", "end", "entry_end", "exit_end"})
    return SplitWindowConfig(
        start=_to_str(raw.get("start"), section=f"{section}.start"),
        end=_to_str(raw.get("end"), section=f"{section}.end"),
        entry_end=_to_str_opt(raw.get("entry_end"), section=f"{section}.entry_end"),
        exit_end=_to_str_opt(raw.get("exit_end"), section=f"{section}.exit_end"),
    )


def _validate_splits(splits: Mapping[str, SplitWindowConfig]) -> None:
    if "train" not in splits or "test" not in splits:
        raise KeyError("backtest.splits must include train and test")

    train = splits["train"]
    test = splits["test"]
    tr0 = pd.to_datetime(train.start)
    tr1 = pd.to_datetime(train.end)
    te0 = pd.to_datetime(test.start)
    te1 = pd.to_datetime(test.end)
    if not (tr0 <= tr1 < te0 <= te1):
        raise ValueError(
            "backtest.splits must satisfy train.start <= train.end < test.start <= test.end"
        )

    if "analysis" in splits:
        analysis = splits["analysis"]
        an0 = pd.to_datetime(analysis.start)
        an1 = pd.to_datetime(analysis.end)
        if not (an0 <= an1 < tr0):
            raise ValueError(
                "backtest.splits.analysis must satisfy analysis.start <= analysis.end < train.start"
            )


def _parse_backtest(raw: Mapping[str, Any]) -> BacktestSettings:
    _expect_keys(
        "backtest",
        raw,
        {
            "initial_capital",
            "risk_per_trade",
            "execution_lag_bars",
            "calendar_mapping",
            "strict_calendar_only",
            "annualization_factor",
            "settlement_lag_bars",
            "range",
            "walkforward",
            "splits",
        },
    )
    range_raw = _as_dict(raw.get("range"), section="backtest.range")
    _expect_keys("backtest.range", range_raw, {"start", "end", "analysis_cfg_path"})
    range_cfg = WindowRangeConfig(
        start=_to_str_opt(range_raw.get("start"), section="backtest.range.start"),
        end=_to_str_opt(range_raw.get("end"), section="backtest.range.end"),
        analysis_cfg_path=_to_str_opt(
            range_raw.get("analysis_cfg_path"),
            section="backtest.range.analysis_cfg_path",
        ),
    )

    wf_raw = _as_dict(raw.get("walkforward"), section="backtest.walkforward")
    _expect_keys(
        "backtest.walkforward",
        wf_raw,
        {
            "enabled",
            "train_mode",
            "initial_train_months",
            "test_months",
            "step_months",
        },
    )
    walkforward = WalkforwardConfig(
        enabled=_to_bool(
            wf_raw.get("enabled", False), section="backtest.walkforward.enabled"
        ),
        train_mode=_to_literal(
            wf_raw.get("train_mode", "rolling"),
            section="backtest.walkforward.train_mode",
            allowed=("expanding", "rolling"),
        ),
        initial_train_months=_to_int(
            wf_raw.get("initial_train_months", 18),
            section="backtest.walkforward.initial_train_months",
        ),
        test_months=_to_int(
            wf_raw.get("test_months", 3), section="backtest.walkforward.test_months"
        ),
        step_months=_to_int(
            wf_raw.get("step_months", 3), section="backtest.walkforward.step_months"
        ),
    )

    splits_raw = _as_dict(raw.get("splits"), section="backtest.splits")
    splits: dict[str, SplitWindowConfig] = {}
    for name, value in splits_raw.items():
        splits[name] = _parse_split_window(
            f"backtest.splits.{name}",
            _as_dict(value, section=f"backtest.splits.{name}"),
        )
    if splits:
        _validate_splits(splits)

    return BacktestSettings(
        initial_capital=_to_float(
            raw.get("initial_capital", 1_000_000.0),
            section="backtest.initial_capital",
        ),
        risk_per_trade=_to_float(
            raw.get("risk_per_trade", 0.01), section="backtest.risk_per_trade"
        ),
        execution_lag_bars=_to_int(
            raw.get("execution_lag_bars", 1),
            section="backtest.execution_lag_bars",
        ),
        calendar_mapping=_to_literal(
            raw.get("calendar_mapping", "prior"),
            section="backtest.calendar_mapping",
            allowed=("prior", "strict", "next"),
        ),
        strict_calendar_only=_to_bool(
            raw.get("strict_calendar_only", False),
            section="backtest.strict_calendar_only",
        ),
        annualization_factor=_to_int_opt(
            raw.get("annualization_factor"),
            section="backtest.annualization_factor",
        ),
        settlement_lag_bars=_to_int(
            raw.get("settlement_lag_bars", 0),
            section="backtest.settlement_lag_bars",
        ),
        range=range_cfg,
        walkforward=walkforward,
        splits=splits,
    )


def _parse_short_heuristic(raw: Mapping[str, Any]) -> ShortAvailabilityHeuristicConfig:
    _expect_keys(
        "risk.short_availability_heuristic",
        raw,
        {"enabled", "min_price", "min_adv_usd", "block_on_missing"},
    )
    return ShortAvailabilityHeuristicConfig(
        enabled=_to_bool(
            raw.get("enabled", False),
            section="risk.short_availability_heuristic.enabled",
        ),
        min_price=_to_float(
            raw.get("min_price", 0.0),
            section="risk.short_availability_heuristic.min_price",
        ),
        min_adv_usd=_to_float(
            raw.get("min_adv_usd", 0.0),
            section="risk.short_availability_heuristic.min_adv_usd",
        ),
        block_on_missing=_to_bool(
            raw.get("block_on_missing", True),
            section="risk.short_availability_heuristic.block_on_missing",
        ),
    )


def _parse_risk(raw: Mapping[str, Any]) -> RiskConfig:
    _expect_keys(
        "risk",
        raw,
        {
            "enabled",
            "max_open_positions",
            "max_trade_pct",
            "max_gross_exposure",
            "max_net_exposure",
            "max_per_name_pct",
            "max_positions_per_symbol",
            "require_shortable_flag",
            "cap_by_availability",
            "strict",
            "short_availability_heuristic",
        },
    )
    short_heuristic = _parse_short_heuristic(
        _as_dict(
            raw.get("short_availability_heuristic"),
            section="risk.short_availability_heuristic",
        )
    )
    max_open_positions = _to_int_opt(
        raw.get("max_open_positions"), section="risk.max_open_positions"
    )
    max_positions_per_symbol = _to_int_opt(
        raw.get("max_positions_per_symbol"), section="risk.max_positions_per_symbol"
    )
    return RiskConfig(
        enabled=_to_bool(raw.get("enabled", False), section="risk.enabled"),
        max_open_positions=(
            max_open_positions if max_open_positions and max_open_positions > 0 else None
        ),
        max_trade_pct=_to_float(
            raw.get("max_trade_pct", 0.10), section="risk.max_trade_pct"
        ),
        max_gross_exposure=_to_float(
            raw.get("max_gross_exposure", 2.0), section="risk.max_gross_exposure"
        ),
        max_net_exposure=_to_float(
            raw.get("max_net_exposure", 1.0), section="risk.max_net_exposure"
        ),
        max_per_name_pct=_to_float_opt(
            raw.get("max_per_name_pct"), section="risk.max_per_name_pct"
        ),
        max_positions_per_symbol=(
            max_positions_per_symbol
            if max_positions_per_symbol and max_positions_per_symbol > 0
            else None
        ),
        require_shortable_flag=_to_bool(
            raw.get("require_shortable_flag", True),
            section="risk.require_shortable_flag",
        ),
        cap_by_availability=_to_bool(
            raw.get("cap_by_availability", True),
            section="risk.cap_by_availability",
        ),
        strict=_to_bool(raw.get("strict", False), section="risk.strict"),
        short_availability_heuristic=short_heuristic,
    )


def _parse_strategy(raw: Mapping[str, Any]) -> StrategyConfig:
    _expect_keys(
        "strategy", raw, {"name", "pair_z_window_as_volatility_window", "params"}
    )
    params_raw = raw.get("params", {})
    if not isinstance(params_raw, Mapping):
        raise TypeError("strategy.params must be a mapping")
    return StrategyConfig(
        name=_to_str(raw.get("name", "baseline"), section="strategy.name"),
        pair_z_window_as_volatility_window=_to_bool(
            raw.get("pair_z_window_as_volatility_window", False),
            section="strategy.pair_z_window_as_volatility_window",
        ),
        params=dict(params_raw),
    )


def _parse_signal(raw: Mapping[str, Any]) -> SignalConfig:
    _expect_keys(
        "signal",
        raw,
        {
            "entry_z",
            "exit_z",
            "stop_z",
            "max_hold_days",
            "cooldown_days",
            "volatility_window",
        },
    )
    return SignalConfig(
        entry_z=_to_float(raw.get("entry_z", 2.0), section="signal.entry_z"),
        exit_z=_to_float(raw.get("exit_z", 0.5), section="signal.exit_z"),
        stop_z=_to_float(raw.get("stop_z", 3.0), section="signal.stop_z"),
        max_hold_days=_to_int(
            raw.get("max_hold_days", 10), section="signal.max_hold_days"
        ),
        cooldown_days=_to_int(
            raw.get("cooldown_days", 0), section="signal.cooldown_days"
        ),
        volatility_window=_to_int(
            raw.get("volatility_window", 30), section="signal.volatility_window"
        ),
    )


def _parse_markov_filter(raw: Mapping[str, Any]) -> MarkovFilterConfig:
    _expect_keys(
        "markov_filter",
        raw,
        {
            "enabled",
            "horizon_days",
            "min_revert_prob",
            "min_train_observations",
            "min_state_observations",
            "transition_smoothing",
            "neutral_z",
            "entry_z",
        },
    )
    return MarkovFilterConfig(
        enabled=_to_bool(raw.get("enabled", False), section="markov_filter.enabled"),
        horizon_days=_to_int(
            raw.get("horizon_days", 30), section="markov_filter.horizon_days"
        ),
        min_revert_prob=_to_float(
            raw.get("min_revert_prob", 0.5),
            section="markov_filter.min_revert_prob",
        ),
        min_train_observations=_to_int(
            raw.get("min_train_observations", 30),
            section="markov_filter.min_train_observations",
        ),
        min_state_observations=_to_int(
            raw.get("min_state_observations", 5),
            section="markov_filter.min_state_observations",
        ),
        transition_smoothing=_to_float(
            raw.get("transition_smoothing", 0.0),
            section="markov_filter.transition_smoothing",
        ),
        neutral_z=_to_float_opt(
            raw.get("neutral_z"), section="markov_filter.neutral_z"
        ),
        entry_z=_to_float_opt(raw.get("entry_z"), section="markov_filter.entry_z"),
    )


def _parse_half_life(raw: Mapping[str, Any]) -> HalfLifeConfig:
    _expect_keys(
        "pair_prefilter.half_life",
        raw,
        {"min_days", "max_days", "max_hold_multiple", "min_derived_days"},
    )
    return HalfLifeConfig(
        min_days=_to_float(
            raw.get("min_days", 5.0), section="pair_prefilter.half_life.min_days"
        ),
        max_days=_to_float(
            raw.get("max_days", 120.0), section="pair_prefilter.half_life.max_days"
        ),
        max_hold_multiple=_to_float(
            raw.get("max_hold_multiple", 2.0),
            section="pair_prefilter.half_life.max_hold_multiple",
        ),
        min_derived_days=_to_int(
            raw.get("min_derived_days", 5),
            section="pair_prefilter.half_life.min_derived_days",
        ),
    )


def _parse_pair_prefilter(raw: Mapping[str, Any]) -> PairPrefilterConfig:
    _expect_keys(
        "pair_prefilter", raw, {"prefilter_active", "coint_alpha", "min_obs", "half_life"}
    )
    half_life = _parse_half_life(
        _as_dict(raw.get("half_life"), section="pair_prefilter.half_life")
    )
    return PairPrefilterConfig(
        prefilter_active=_to_bool(
            raw.get("prefilter_active", True),
            section="pair_prefilter.prefilter_active",
        ),
        coint_alpha=_to_float(
            raw.get("coint_alpha", 0.05), section="pair_prefilter.coint_alpha"
        ),
        min_obs=_to_int(raw.get("min_obs", 30), section="pair_prefilter.min_obs"),
        half_life=half_life,
    )


def _parse_spread_zscore(raw: Mapping[str, Any]) -> SpreadZscoreConfig:
    _expect_keys("spread_zscore", raw, {"z_window", "z_min_periods", "freeze_stats"})
    return SpreadZscoreConfig(
        z_window=_to_int(raw.get("z_window", 30), section="spread_zscore.z_window"),
        z_min_periods=_to_int_opt(
            raw.get("z_min_periods"), section="spread_zscore.z_min_periods"
        ),
        freeze_stats=_to_bool(
            raw.get("freeze_stats", False), section="spread_zscore.freeze_stats"
        ),
    )


def _parse_borrow_rates(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("borrow.per_asset_rate_annual must be a mapping")
    out: dict[str, float] = {}
    for symbol, rate in raw.items():
        out[_normalize_symbol(symbol, section="borrow.per_asset_rate_annual.symbol")] = _to_float(
            rate, section="borrow.per_asset_rate_annual.rate_annual"
        )
    return out


def _parse_borrow_rate_points(raw: Any) -> tuple[BorrowRatePoint, ...]:
    if raw is None:
        return ()
    rows: list[BorrowRatePoint] = []
    if isinstance(raw, Mapping):
        for symbol, value in raw.items():
            sym = _normalize_symbol(symbol, section="borrow.rates.symbol")
            if isinstance(value, Mapping):
                for date, rate in value.items():
                    rows.append(
                        BorrowRatePoint(
                            date=_to_str(date, section="borrow.rates.date"),
                            symbol=sym,
                            rate_annual=_to_float(
                                rate, section="borrow.rates.rate_annual"
                            ),
                        )
                    )
                continue
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    item_raw = _as_dict(item, section=f"borrow.rates[{sym}][{idx}]")
                    _expect_keys(
                        f"borrow.rates[{sym}][{idx}]",
                        item_raw,
                        {"date", "rate_annual"},
                    )
                    rows.append(
                        BorrowRatePoint(
                            date=_to_str(
                                item_raw.get("date"),
                                section=f"borrow.rates[{sym}][{idx}].date",
                            ),
                            symbol=sym,
                            rate_annual=_to_float(
                                item_raw.get("rate_annual"),
                                section=f"borrow.rates[{sym}][{idx}].rate_annual",
                            ),
                        )
                    )
                continue
            raise TypeError("borrow.rates mapping entries must be date maps or row lists")
        return tuple(rows)
    if not isinstance(raw, list):
        raise TypeError("borrow.rates must be a list or mapping")
    for idx, item in enumerate(raw):
        item_raw = _as_dict(item, section=f"borrow.rates[{idx}]")
        _expect_keys(f"borrow.rates[{idx}]", item_raw, {"date", "symbol", "rate_annual"})
        rows.append(
            BorrowRatePoint(
                date=_to_str(item_raw.get("date"), section=f"borrow.rates[{idx}].date"),
                symbol=_normalize_symbol(
                    item_raw.get("symbol"), section=f"borrow.rates[{idx}].symbol"
                ),
                rate_annual=_to_float(
                    item_raw.get("rate_annual"),
                    section=f"borrow.rates[{idx}].rate_annual",
                ),
            )
        )
    return tuple(rows)


def _parse_borrow_availability(raw: Any) -> tuple[BorrowAvailabilityPoint, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError("borrow.availability must be a list")
    rows: list[BorrowAvailabilityPoint] = []
    for idx, item in enumerate(raw):
        item_raw = _as_dict(item, section=f"borrow.availability[{idx}]")
        _expect_keys(
            f"borrow.availability[{idx}]",
            item_raw,
            {"date", "symbol", "available"},
        )
        rows.append(
            BorrowAvailabilityPoint(
                date=_to_str(
                    item_raw.get("date"),
                    section=f"borrow.availability[{idx}].date",
                ),
                symbol=_normalize_symbol(
                    item_raw.get("symbol"),
                    section=f"borrow.availability[{idx}].symbol",
                ),
                available=_to_bool(
                    item_raw.get("available"),
                    section=f"borrow.availability[{idx}].available",
                ),
            )
        )
    return tuple(rows)


def _parse_borrow_enforcement(raw: Mapping[str, Any]) -> BorrowEnforcementConfig:
    _expect_keys(
        "borrow.enforcement",
        raw,
        {"enabled", "mode", "recall_grace_days", "buyin_penalty_bps"},
    )
    return BorrowEnforcementConfig(
        enabled=_to_bool(raw.get("enabled", False), section="borrow.enforcement.enabled"),
        mode=_to_literal(
            raw.get("mode", "penalty_only"),
            section="borrow.enforcement.mode",
            allowed=("penalty_only", "clip_exit"),
        ),
        recall_grace_days=_to_int(
            raw.get("recall_grace_days", 2),
            section="borrow.enforcement.recall_grace_days",
        ),
        buyin_penalty_bps=_to_float(
            raw.get("buyin_penalty_bps", 0.0),
            section="borrow.enforcement.buyin_penalty_bps",
        ),
    )


def _parse_borrow(raw: Mapping[str, Any]) -> BorrowConfig:
    _expect_keys(
        "borrow",
        raw,
        {
            "enabled",
            "accrual_mode",
            "day_count",
            "include_exit_day",
            "min_days",
            "day_basis",
            "default_rate_annual",
            "per_asset_rate_annual",
            "rates",
            "availability",
            "enforcement",
            "synthetic_jitter_sigma",
            "ftd_block_threshold",
        },
    )
    enforcement = _parse_borrow_enforcement(
        _as_dict(raw.get("enforcement"), section="borrow.enforcement")
    )
    return BorrowConfig(
        enabled=_to_bool(raw.get("enabled", False), section="borrow.enabled"),
        accrual_mode=_to_literal(
            raw.get("accrual_mode", "entry_notional"),
            section="borrow.accrual_mode",
            allowed=("entry_notional", "mtm_daily"),
        ),
        day_count=_to_literal(
            raw.get("day_count", "busdays"),
            section="borrow.day_count",
            allowed=("busdays", "calendar_days", "sessions"),
        ),
        include_exit_day=_to_bool(
            raw.get("include_exit_day", False),
            section="borrow.include_exit_day",
        ),
        min_days=_to_int(raw.get("min_days", 1), section="borrow.min_days"),
        day_basis=_to_int(raw.get("day_basis", 252), section="borrow.day_basis"),
        default_rate_annual=_to_float(
            raw.get("default_rate_annual", 0.0),
            section="borrow.default_rate_annual",
        ),
        per_asset_rate_annual=_parse_borrow_rates(raw.get("per_asset_rate_annual")),
        rates=_parse_borrow_rate_points(raw.get("rates")),
        availability=_parse_borrow_availability(raw.get("availability")),
        enforcement=enforcement,
        synthetic_jitter_sigma=_to_float(
            raw.get("synthetic_jitter_sigma", 0.0),
            section="borrow.synthetic_jitter_sigma",
        ),
        ftd_block_threshold=_to_float(
            raw.get("ftd_block_threshold", 0.0),
            section="borrow.ftd_block_threshold",
        ),
    )


def _parse_execution_fees(section: str, raw: Mapping[str, Any]) -> ExecutionFeesConfig:
    _expect_keys(section, raw, {"per_trade", "bps", "per_share", "min_fee", "max_fee"})
    return ExecutionFeesConfig(
        per_trade=_to_float(raw.get("per_trade", 0.0), section=f"{section}.per_trade"),
        bps=_to_float(raw.get("bps", 0.0), section=f"{section}.bps"),
        per_share=_to_float(raw.get("per_share", 0.0), section=f"{section}.per_share"),
        min_fee=_to_float(raw.get("min_fee", 0.0), section=f"{section}.min_fee"),
        max_fee=_to_float(raw.get("max_fee", 0.0), section=f"{section}.max_fee"),
    )


def _parse_lob_post_costs(raw: Mapping[str, Any]) -> LOBPostCostConfig:
    _expect_keys(
        "execution.lob.post_costs",
        raw,
        {
            "per_trade",
            "maker_bps",
            "taker_bps",
            "maker_per_share",
            "taker_per_share",
            "min_fee",
            "max_fee",
        },
    )
    return LOBPostCostConfig(
        per_trade=_to_float(
            raw.get("per_trade", 0.0), section="execution.lob.post_costs.per_trade"
        ),
        maker_bps=_to_float(
            raw.get("maker_bps", 0.0), section="execution.lob.post_costs.maker_bps"
        ),
        taker_bps=_to_float(
            raw.get("taker_bps", 0.0), section="execution.lob.post_costs.taker_bps"
        ),
        maker_per_share=_to_float(
            raw.get("maker_per_share", 0.0),
            section="execution.lob.post_costs.maker_per_share",
        ),
        taker_per_share=_to_float(
            raw.get("taker_per_share", 0.0),
            section="execution.lob.post_costs.taker_per_share",
        ),
        min_fee=_to_float(
            raw.get("min_fee", 0.0), section="execution.lob.post_costs.min_fee"
        ),
        max_fee=_to_float(
            raw.get("max_fee", 0.0), section="execution.lob.post_costs.max_fee"
        ),
    )


def _parse_lob_liq_model(raw: Mapping[str, Any]) -> LOBLiquidityConfig:
    _expect_keys(
        "execution.lob.liq_model",
        raw,
        {
            "enabled",
            "asof_shift",
            "vol_window",
            "adv_window",
            "min_periods_frac",
            "spread_floor_bps",
            "spread_sigma_mult",
            "spread_adv_mult",
            "adv_ref_usd",
            "depth_frac_of_adv_shares",
            "depth_gamma",
            "min_depth_shares",
            "max_depth_shares",
            "min_level_shares",
            "lam_adv_power",
            "lam_min",
            "lam_max",
            "cancel_base",
            "cancel_sigma_mult",
            "cancel_min",
            "cancel_max",
            "max_add_frac_of_top",
            "max_cancel_frac_of_top",
            "max_add_min",
            "max_cancel_min",
            "max_add_max",
            "max_cancel_max",
            "tick_subpenny",
            "tick_penny",
            "tick_switch_price",
        },
    )
    return LOBLiquidityConfig(
        enabled=_to_bool(raw.get("enabled", False), section="execution.lob.liq_model.enabled"),
        asof_shift=_to_int(raw.get("asof_shift", 1), section="execution.lob.liq_model.asof_shift"),
        vol_window=_to_int(raw.get("vol_window", 30), section="execution.lob.liq_model.vol_window"),
        adv_window=_to_int(raw.get("adv_window", 60), section="execution.lob.liq_model.adv_window"),
        min_periods_frac=_to_float(raw.get("min_periods_frac", 0.5), section="execution.lob.liq_model.min_periods_frac"),
        spread_floor_bps=_to_float(raw.get("spread_floor_bps", 1.0), section="execution.lob.liq_model.spread_floor_bps"),
        spread_sigma_mult=_to_float(raw.get("spread_sigma_mult", 0.005), section="execution.lob.liq_model.spread_sigma_mult"),
        spread_adv_mult=_to_float(raw.get("spread_adv_mult", 15.0), section="execution.lob.liq_model.spread_adv_mult"),
        adv_ref_usd=_to_float(raw.get("adv_ref_usd", 1_000_000.0), section="execution.lob.liq_model.adv_ref_usd"),
        depth_frac_of_adv_shares=_to_float(raw.get("depth_frac_of_adv_shares", 0.001), section="execution.lob.liq_model.depth_frac_of_adv_shares"),
        depth_gamma=_to_float(raw.get("depth_gamma", 0.7), section="execution.lob.liq_model.depth_gamma"),
        min_depth_shares=_to_int(raw.get("min_depth_shares", 25), section="execution.lob.liq_model.min_depth_shares"),
        max_depth_shares=_to_int(raw.get("max_depth_shares", 250_000), section="execution.lob.liq_model.max_depth_shares"),
        min_level_shares=_to_int(raw.get("min_level_shares", 1), section="execution.lob.liq_model.min_level_shares"),
        lam_adv_power=_to_float(raw.get("lam_adv_power", 0.25), section="execution.lob.liq_model.lam_adv_power"),
        lam_min=_to_float(raw.get("lam_min", 0.25), section="execution.lob.liq_model.lam_min"),
        lam_max=_to_float(raw.get("lam_max", 25.0), section="execution.lob.liq_model.lam_max"),
        cancel_base=_to_float(raw.get("cancel_base", 0.15), section="execution.lob.liq_model.cancel_base"),
        cancel_sigma_mult=_to_float(raw.get("cancel_sigma_mult", 1.5), section="execution.lob.liq_model.cancel_sigma_mult"),
        cancel_min=_to_float(raw.get("cancel_min", 0.01), section="execution.lob.liq_model.cancel_min"),
        cancel_max=_to_float(raw.get("cancel_max", 0.9), section="execution.lob.liq_model.cancel_max"),
        max_add_frac_of_top=_to_float(raw.get("max_add_frac_of_top", 0.5), section="execution.lob.liq_model.max_add_frac_of_top"),
        max_cancel_frac_of_top=_to_float(raw.get("max_cancel_frac_of_top", 0.25), section="execution.lob.liq_model.max_cancel_frac_of_top"),
        max_add_min=_to_int(raw.get("max_add_min", 50), section="execution.lob.liq_model.max_add_min"),
        max_cancel_min=_to_int(raw.get("max_cancel_min", 25), section="execution.lob.liq_model.max_cancel_min"),
        max_add_max=_to_int(raw.get("max_add_max", 50_000), section="execution.lob.liq_model.max_add_max"),
        max_cancel_max=_to_int(raw.get("max_cancel_max", 50_000), section="execution.lob.liq_model.max_cancel_max"),
        tick_subpenny=_to_float(raw.get("tick_subpenny", 0.0001), section="execution.lob.liq_model.tick_subpenny"),
        tick_penny=_to_float(raw.get("tick_penny", 0.01), section="execution.lob.liq_model.tick_penny"),
        tick_switch_price=_to_float(raw.get("tick_switch_price", 1.0), section="execution.lob.liq_model.tick_switch_price"),
    )


def _parse_lob_fill_model(raw: Mapping[str, Any]) -> LOBFillModelConfig:
    _expect_keys(
        "execution.lob.fill_model",
        raw,
        {
            "enabled",
            "base_fill",
            "safe_depth_share",
            "depth_share_50",
            "depth_shape",
            "safe_participation",
            "participation_50",
            "participation_shape",
            "sigma_mult",
            "beta_kappa_base",
            "beta_kappa_adv_power",
            "beta_kappa_min",
            "beta_kappa_max",
            "allow_reject",
            "reject_below",
            "min_fill_if_filled",
        },
    )
    return LOBFillModelConfig(
        enabled=_to_bool(raw.get("enabled", False), section="execution.lob.fill_model.enabled"),
        base_fill=_to_float(raw.get("base_fill", 1.0), section="execution.lob.fill_model.base_fill"),
        safe_depth_share=_to_float(raw.get("safe_depth_share", 0.25), section="execution.lob.fill_model.safe_depth_share"),
        depth_share_50=_to_float(raw.get("depth_share_50", 0.50), section="execution.lob.fill_model.depth_share_50"),
        depth_shape=_to_float(raw.get("depth_shape", 1.5), section="execution.lob.fill_model.depth_shape"),
        safe_participation=_to_float(raw.get("safe_participation", 0.002), section="execution.lob.fill_model.safe_participation"),
        participation_50=_to_float(raw.get("participation_50", 0.008), section="execution.lob.fill_model.participation_50"),
        participation_shape=_to_float(raw.get("participation_shape", 1.2), section="execution.lob.fill_model.participation_shape"),
        sigma_mult=_to_float(raw.get("sigma_mult", 1.5), section="execution.lob.fill_model.sigma_mult"),
        beta_kappa_base=_to_float(raw.get("beta_kappa_base", 75.0), section="execution.lob.fill_model.beta_kappa_base"),
        beta_kappa_adv_power=_to_float(raw.get("beta_kappa_adv_power", 0.2), section="execution.lob.fill_model.beta_kappa_adv_power"),
        beta_kappa_min=_to_float(raw.get("beta_kappa_min", 10.0), section="execution.lob.fill_model.beta_kappa_min"),
        beta_kappa_max=_to_float(raw.get("beta_kappa_max", 500.0), section="execution.lob.fill_model.beta_kappa_max"),
        allow_reject=_to_bool(raw.get("allow_reject", True), section="execution.lob.fill_model.allow_reject"),
        reject_below=_to_float(raw.get("reject_below", 0.002), section="execution.lob.fill_model.reject_below"),
        min_fill_if_filled=_to_float(raw.get("min_fill_if_filled", 0.01), section="execution.lob.fill_model.min_fill_if_filled"),
    )


def _parse_lob_order_flow_side(
    section: str,
    raw: Mapping[str, Any],
) -> LOBOrderFlowSideConfig:
    _expect_keys(
        section,
        raw,
        {
            "mode",
            "maker_price",
            "maker_prob",
            "maker_max_top_frac",
            "maker_touch_prob",
            "fallback_to_taker",
        },
    )
    return LOBOrderFlowSideConfig(
        mode=_to_literal(
            raw.get("mode", "taker"),
            section=f"{section}.mode",
            allowed=("taker", "maker", "mixed"),
        ),
        maker_price=_to_literal(
            raw.get("maker_price", "best"),
            section=f"{section}.maker_price",
            allowed=("best", "mid"),
        ),
        maker_prob=_to_float(raw.get("maker_prob", 0.5), section=f"{section}.maker_prob"),
        maker_max_top_frac=_to_float(
            raw.get("maker_max_top_frac", 0.25),
            section=f"{section}.maker_max_top_frac",
        ),
        maker_touch_prob=_to_float(
            raw.get("maker_touch_prob", 1.0),
            section=f"{section}.maker_touch_prob",
        ),
        fallback_to_taker=_to_bool(
            raw.get("fallback_to_taker", True),
            section=f"{section}.fallback_to_taker",
        ),
    )


def _parse_lob_order_flow(raw: Mapping[str, Any]) -> LOBOrderFlowConfig:
    _expect_keys(
        "execution.lob.order_flow",
        raw,
        {
            "mode",
            "maker_price",
            "maker_prob",
            "maker_max_top_frac",
            "maker_touch_prob",
            "fallback_to_taker",
            "entry",
            "exit",
        },
    )
    base_raw = {
        key: value for key, value in raw.items() if key not in {"entry", "exit"}
    }
    base = _parse_lob_order_flow_side("execution.lob.order_flow", base_raw)
    entry = _parse_lob_order_flow_side(
        "execution.lob.order_flow.entry",
        {**asdict(base), **_as_dict(raw.get("entry"), section="execution.lob.order_flow.entry")},
    )
    exit_cfg = _parse_lob_order_flow_side(
        "execution.lob.order_flow.exit",
        {**asdict(base), **_as_dict(raw.get("exit"), section="execution.lob.order_flow.exit")},
    )
    return LOBOrderFlowConfig(
        mode=base.mode,
        maker_price=base.maker_price,
        maker_prob=base.maker_prob,
        maker_max_top_frac=base.maker_max_top_frac,
        maker_touch_prob=base.maker_touch_prob,
        fallback_to_taker=base.fallback_to_taker,
        entry=entry,
        exit=exit_cfg,
    )


def _parse_lob_stress(raw: Mapping[str, Any]) -> LOBStressModelConfig:
    _expect_keys(
        "execution.lob.stress_model",
        raw,
        {"enabled", "intensity", "max_entry_delay_days", "max_exit_grace_days", "panic_cross_bps"},
    )
    return LOBStressModelConfig(
        enabled=_to_bool(raw.get("enabled", True), section="execution.lob.stress_model.enabled"),
        intensity=_to_float(raw.get("intensity", 1.0), section="execution.lob.stress_model.intensity"),
        max_entry_delay_days=_to_int(raw.get("max_entry_delay_days", 1), section="execution.lob.stress_model.max_entry_delay_days"),
        max_exit_grace_days=_to_int(raw.get("max_exit_grace_days", 2), section="execution.lob.stress_model.max_exit_grace_days"),
        panic_cross_bps=_to_float(raw.get("panic_cross_bps", 50.0), section="execution.lob.stress_model.panic_cross_bps"),
    )


def _parse_light(raw: Mapping[str, Any]) -> LightExecutionConfig:
    _expect_keys("execution.light", raw, {"enabled", "reject_on_missing_price", "fees"})
    return LightExecutionConfig(
        enabled=_to_bool(raw.get("enabled", True), section="execution.light.enabled"),
        reject_on_missing_price=_to_bool(
            raw.get("reject_on_missing_price", True),
            section="execution.light.reject_on_missing_price",
        ),
        fees=_parse_execution_fees(
            "execution.light.fees",
            _as_dict(raw.get("fees"), section="execution.light.fees"),
        ),
    )


def _parse_lob(raw: Mapping[str, Any]) -> LOBExecutionConfig:
    _expect_keys(
        "execution.lob",
        raw,
        {
            "enabled",
            "tick",
            "levels",
            "size_per_level",
            "min_spread_ticks",
            "steps_per_day",
            "lam",
            "max_add",
            "bias_top",
            "cancel_prob",
            "max_cancel",
            "liq_model",
            "fill_model",
            "post_costs",
            "order_flow",
            "stress_model",
        },
    )
    return LOBExecutionConfig(
        enabled=_to_bool(raw.get("enabled", True), section="execution.lob.enabled"),
        tick=_to_float(raw.get("tick", 0.01), section="execution.lob.tick"),
        levels=_to_int(raw.get("levels", 5), section="execution.lob.levels"),
        size_per_level=_to_int(
            raw.get("size_per_level", 1_000), section="execution.lob.size_per_level"
        ),
        min_spread_ticks=_to_int(
            raw.get("min_spread_ticks", 1),
            section="execution.lob.min_spread_ticks",
        ),
        steps_per_day=_to_int(
            raw.get("steps_per_day", 4), section="execution.lob.steps_per_day"
        ),
        lam=_to_float(raw.get("lam", 2.0), section="execution.lob.lam"),
        max_add=_to_int(raw.get("max_add", 500), section="execution.lob.max_add"),
        bias_top=_to_float(raw.get("bias_top", 0.7), section="execution.lob.bias_top"),
        cancel_prob=_to_float(
            raw.get("cancel_prob", 0.15), section="execution.lob.cancel_prob"
        ),
        max_cancel=_to_int(
            raw.get("max_cancel", 200), section="execution.lob.max_cancel"
        ),
        liq_model=_parse_lob_liq_model(
            _as_dict(raw.get("liq_model"), section="execution.lob.liq_model")
        ),
        fill_model=_parse_lob_fill_model(
            _as_dict(raw.get("fill_model"), section="execution.lob.fill_model")
        ),
        post_costs=_parse_lob_post_costs(
            _as_dict(raw.get("post_costs"), section="execution.lob.post_costs")
        ),
        order_flow=_parse_lob_order_flow(
            _as_dict(raw.get("order_flow"), section="execution.lob.order_flow")
        ),
        stress_model=_parse_lob_stress(
            _as_dict(raw.get("stress_model"), section="execution.lob.stress_model")
        ),
    )


def _parse_execution(raw: Mapping[str, Any]) -> ExecutionConfig:
    _expect_keys("execution", raw, {"mode", "max_participation", "light", "lob"})
    mode = _to_literal(
        raw.get("mode"), section="execution.mode", allowed=("lob", "light")
    )
    has_light = "light" in raw
    has_lob = "lob" in raw
    if mode == "light" and (not has_light or has_lob):
        raise ValueError(
            "execution.mode='light' requires execution.light and forbids execution.lob"
        )
    if mode == "lob" and (not has_lob or has_light):
        raise ValueError(
            "execution.mode='lob' requires execution.lob and forbids execution.light"
        )
    light_cfg = (
        _parse_light(_as_dict(raw.get("light"), section="execution.light"))
        if has_light
        else LightExecutionConfig()
    )
    lob_cfg = (
        _parse_lob(_as_dict(raw.get("lob"), section="execution.lob"))
        if has_lob
        else LOBExecutionConfig()
    )
    return ExecutionConfig(
        mode=mode,
        max_participation=_to_float(
            raw.get("max_participation", 0.10),
            section="execution.max_participation",
        ),
        light=light_cfg,
        lob=lob_cfg,
    )


def _parse_reporting(raw: Mapping[str, Any]) -> ReportingConfig:
    _expect_keys("reporting", raw, {"mode", "train_visuals", "test_tearsheet"})
    visuals_raw = raw.get("train_visuals", ("cv_scores", "equity"))
    if not isinstance(visuals_raw, (list, tuple)):
        raise TypeError("reporting.train_visuals must be a list")
    visuals = tuple(_to_str(v, section="reporting.train_visuals") for v in visuals_raw)
    tearsheet_raw = _as_dict(raw.get("test_tearsheet"), section="reporting.test_tearsheet")
    _expect_keys("reporting.test_tearsheet", tearsheet_raw, {"enabled", "dpi"})
    tearsheet = ReportingTearsheetConfig(
        enabled=_to_bool(
            tearsheet_raw.get("enabled", True),
            section="reporting.test_tearsheet.enabled",
        ),
        dpi=_to_int(
            tearsheet_raw.get("dpi", 150), section="reporting.test_tearsheet.dpi"
        ),
    )
    return ReportingConfig(
        mode=_to_literal(
            raw.get("mode", "core"),
            section="reporting.mode",
            allowed=("core", "debug"),
        ),
        train_visuals=visuals,
        test_tearsheet=tearsheet,
    )


def _parse_bo_cv(section: str, raw: Mapping[str, Any]) -> BOCVConfig:
    _expect_keys(
        section,
        raw,
        {
            "enabled",
            "scheme",
            "n_blocks",
            "k_test_blocks",
            "purge",
            "embargo",
            "max_folds",
            "aggregate",
            "trim_pct",
            "shuffle",
        },
    )
    return BOCVConfig(
        enabled=_to_bool(raw.get("enabled", False), section=f"{section}.enabled"),
        scheme=_to_literal(
            raw.get("scheme", "blocked"),
            section=f"{section}.scheme",
            allowed=("blocked", "cpcv"),
        ),
        n_blocks=_to_int(raw.get("n_blocks", 5), section=f"{section}.n_blocks"),
        k_test_blocks=_to_int(
            raw.get("k_test_blocks", 2), section=f"{section}.k_test_blocks"
        ),
        purge=_to_int(raw.get("purge", 0), section=f"{section}.purge"),
        embargo=_to_float(raw.get("embargo", 0.0), section=f"{section}.embargo"),
        max_folds=_to_int_opt(raw.get("max_folds"), section=f"{section}.max_folds"),
        aggregate=_to_literal(
            raw.get("aggregate", "median"),
            section=f"{section}.aggregate",
            allowed=("median", "mean", "trimmed_mean"),
        ),
        trim_pct=_to_float(raw.get("trim_pct", 0.10), section=f"{section}.trim_pct"),
        shuffle=_to_bool(raw.get("shuffle", False), section=f"{section}.shuffle"),
    )


def _parse_bo(raw: Mapping[str, Any]) -> BOConfig:
    _expect_keys(
        "bo",
        raw,
        {
            "enabled",
            "mode",
            "out_dir",
            "fast",
            "realistic",
            "entry_z_range",
            "exit_z_range",
            "stop_z_range",
            "init_points",
            "n_iter",
            "patience",
            "min_revert_prob_range",
            "horizon_days_range",
            "markov_init_points",
            "markov_n_iter",
            "markov_patience",
        },
    )
    fast_raw = _as_dict(raw.get("fast"), section="bo.fast")
    _expect_keys("bo.fast", fast_raw, {"cv"})
    realistic_raw = _as_dict(raw.get("realistic"), section="bo.realistic")
    _expect_keys("bo.realistic", realistic_raw, {"metric", "cv"})
    fast = BOFastConfig(
        cv=_parse_bo_cv("bo.fast.cv", _as_dict(fast_raw.get("cv"), section="bo.fast.cv"))
    )
    realistic = BORealisticConfig(
        metric=_to_literal(
            realistic_raw.get("metric", "sharpe"),
            section="bo.realistic.metric",
            allowed=("sharpe", "cagr", "calmar"),
        ),
        cv=_parse_bo_cv(
            "bo.realistic.cv",
            _as_dict(realistic_raw.get("cv"), section="bo.realistic.cv"),
        ),
    )
    return BOConfig(
        enabled=_to_bool(raw.get("enabled", False), section="bo.enabled"),
        mode=_to_literal(
            raw.get("mode", "fast"), section="bo.mode", allowed=("fast", "realistic")
        ),
        out_dir=_to_str(raw.get("out_dir", "runs/results/bo"), section="bo.out_dir"),
        fast=fast,
        realistic=realistic,
        entry_z_range=cast(
            tuple[float, float] | None,
            _to_range(raw.get("entry_z_range"), section="bo.entry_z_range", cast_item=float),
        ),
        exit_z_range=cast(
            tuple[float, float] | None,
            _to_range(raw.get("exit_z_range"), section="bo.exit_z_range", cast_item=float),
        ),
        stop_z_range=cast(
            tuple[float, float] | None,
            _to_range(raw.get("stop_z_range"), section="bo.stop_z_range", cast_item=float),
        ),
        init_points=_to_int(raw.get("init_points", 8), section="bo.init_points"),
        n_iter=_to_int(raw.get("n_iter", 24), section="bo.n_iter"),
        patience=_to_int(raw.get("patience", 0), section="bo.patience"),
        min_revert_prob_range=cast(
            tuple[float, float] | None,
            _to_range(
                raw.get("min_revert_prob_range"),
                section="bo.min_revert_prob_range",
                cast_item=float,
            ),
        ),
        horizon_days_range=cast(
            tuple[int, int] | None,
            _to_range(
                raw.get("horizon_days_range"),
                section="bo.horizon_days_range",
                cast_item=int,
            ),
        ),
        markov_init_points=_to_int_opt(
            raw.get("markov_init_points"), section="bo.markov_init_points"
        ),
        markov_n_iter=_to_int_opt(
            raw.get("markov_n_iter"), section="bo.markov_n_iter"
        ),
        markov_patience=_to_int_opt(
            raw.get("markov_patience"), section="bo.markov_patience"
        ),
    )


def parse_config(mapping: Mapping[str, Any]) -> AppConfig:
    if not isinstance(mapping, Mapping):
        raise TypeError("config root must be a mapping")
    root = dict(mapping)
    _expect_keys("root", root, _ROOT_KEYS)
    _require_sections(
        root,
        ("data", "backtest", "strategy", "signal", "spread_zscore", "execution"),
    )
    return AppConfig(
        runtime=_parse_runtime(_as_dict(root.get("runtime"), section="runtime")),
        data=_parse_data(_as_dict(root.get("data"), section="data")),
        backtest=_parse_backtest(_as_dict(root.get("backtest"), section="backtest")),
        risk=_parse_risk(_as_dict(root.get("risk"), section="risk")),
        strategy=_parse_strategy(_as_dict(root.get("strategy"), section="strategy")),
        signal=_parse_signal(_as_dict(root.get("signal"), section="signal")),
        spread_zscore=_parse_spread_zscore(
            _as_dict(root.get("spread_zscore"), section="spread_zscore")
        ),
        markov_filter=_parse_markov_filter(
            _as_dict(root.get("markov_filter"), section="markov_filter")
        ),
        pair_prefilter=_parse_pair_prefilter(
            _as_dict(root.get("pair_prefilter"), section="pair_prefilter")
        ),
        borrow=_parse_borrow(_as_dict(root.get("borrow"), section="borrow")),
        execution=_parse_execution(
            _as_dict(root.get("execution"), section="execution")
        ),
        reporting=_parse_reporting(
            _as_dict(root.get("reporting"), section="reporting")
        ),
        bo=_parse_bo(_as_dict(root.get("bo"), section="bo")),
        cv=_parse_bo_cv("cv", _as_dict(root.get("cv"), section="cv")),
    )


def load_config(path: Path) -> AppConfig:
    return parse_config(load_yaml_dict(path))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    out = cast(dict[str, Any], _to_jsonable(asdict(cfg)))
    execution = out.get("execution")
    if isinstance(execution, dict):
        mode = str(execution.get("mode", "")).strip().lower()
        if mode == "light":
            execution.pop("lob", None)
        elif mode == "lob":
            execution.pop("light", None)
    return out
