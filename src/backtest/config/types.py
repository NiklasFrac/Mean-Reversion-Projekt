"""
Typed backtest configuration models and small stable execution helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

import pandas as pd

__all__ = [
    "AppConfig",
    "BOConfig",
    "BOCVConfig",
    "BOFastConfig",
    "BORealisticConfig",
    "BacktestSettings",
    "BorrowAvailabilityPoint",
    "BorrowConfig",
    "BorrowCtx",
    "BorrowEnforcementConfig",
    "BorrowRatePoint",
    "DataConfig",
    "ExecutionConfig",
    "ExecutionFeesConfig",
    "Fill",
    "HalfLifeConfig",
    "LOBExecutionConfig",
    "LOBFillModelConfig",
    "LOBLiquidityConfig",
    "LOBOrderFlowConfig",
    "LOBOrderFlowSideConfig",
    "LOBPostCostConfig",
    "LOBStressModelConfig",
    "LightExecutionConfig",
    "MarkovFilterConfig",
    "PairPrefilterConfig",
    "PricingCfg",
    "ReportingConfig",
    "ReportingTearsheetConfig",
    "RiskConfig",
    "RuntimeConfig",
    "ShortAvailabilityHeuristicConfig",
    "Side",
    "SignalConfig",
    "SplitWindowConfig",
    "SpreadZscoreConfig",
    "StrategyConfig",
    "WalkforwardConfig",
    "WindowRangeConfig",
]


class Fill(TypedDict, total=False):
    qty: int
    price: float
    ts: pd.Timestamp
    liquidity: Literal["M", "T"]
    order_id: str


Side = Literal["buy", "sell"]


class BorrowCtx(Protocol):
    day_basis: int


@dataclass(frozen=True)
class PricingCfg:
    reference: str = "mid_on_submit"


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    require_execution_hooks: bool = False


@dataclass(frozen=True)
class DataConfig:
    input_mode: Literal["explicit", "analysis_meta"] = "explicit"
    analysis_meta_path: str | None = None
    prices_path: str = ""
    pairs_path: str = ""
    adv_map_path: str | None = None
    calendar_name: str = "XNYS"
    prefer_col: str = "close"


@dataclass(frozen=True)
class SplitWindowConfig:
    start: str
    end: str
    entry_end: str | None = None
    exit_end: str | None = None


@dataclass(frozen=True)
class WindowRangeConfig:
    start: str | None = None
    end: str | None = None
    analysis_cfg_path: str | None = None


@dataclass(frozen=True)
class WalkforwardConfig:
    enabled: bool = False
    train_mode: Literal["expanding", "rolling"] = "rolling"
    initial_train_months: int = 18
    test_months: int = 3
    step_months: int = 3


@dataclass(frozen=True)
class BacktestSettings:
    initial_capital: float = 1_000_000.0
    risk_per_trade: float = 0.01
    execution_lag_bars: int = 1
    calendar_mapping: Literal["prior", "strict", "next"] = "prior"
    strict_calendar_only: bool = False
    annualization_factor: int | None = None
    settlement_lag_bars: int = 0
    range: WindowRangeConfig = field(default_factory=WindowRangeConfig)
    walkforward: WalkforwardConfig = field(default_factory=WalkforwardConfig)
    splits: dict[str, SplitWindowConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class ShortAvailabilityHeuristicConfig:
    enabled: bool = False
    min_price: float = 0.0
    min_adv_usd: float = 0.0
    block_on_missing: bool = True


@dataclass(frozen=True)
class RiskConfig:
    enabled: bool = False
    max_open_positions: int | None = None
    max_trade_pct: float = 0.10
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 1.0
    max_per_name_pct: float | None = None
    max_positions_per_symbol: int | None = None
    require_shortable_flag: bool = True
    cap_by_availability: bool = True
    strict: bool = False
    short_availability_heuristic: ShortAvailabilityHeuristicConfig = field(
        default_factory=ShortAvailabilityHeuristicConfig
    )


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "baseline"
    pair_z_window_as_volatility_window: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.0
    max_hold_days: int = 10
    cooldown_days: int = 0
    volatility_window: int = 30


@dataclass(frozen=True)
class MarkovFilterConfig:
    enabled: bool = False
    horizon_days: int = 30
    min_revert_prob: float = 0.5
    min_train_observations: int = 30
    min_state_observations: int = 5
    transition_smoothing: float = 0.0
    neutral_z: float | None = None
    entry_z: float | None = None


@dataclass(frozen=True)
class HalfLifeConfig:
    min_days: float = 5.0
    max_days: float = 120.0
    max_hold_multiple: float = 2.0
    min_derived_days: int = 5


@dataclass(frozen=True)
class PairPrefilterConfig:
    prefilter_active: bool = True
    coint_alpha: float = 0.05
    min_obs: int = 30
    half_life: HalfLifeConfig = field(default_factory=HalfLifeConfig)


@dataclass(frozen=True)
class SpreadZscoreConfig:
    z_window: int = 30
    z_min_periods: int | None = None
    freeze_stats: bool = False


@dataclass(frozen=True)
class BorrowRatePoint:
    date: str
    symbol: str
    rate_annual: float


@dataclass(frozen=True)
class BorrowAvailabilityPoint:
    date: str
    symbol: str
    available: bool


@dataclass(frozen=True)
class BorrowEnforcementConfig:
    enabled: bool = False
    mode: Literal["penalty_only", "clip_exit"] = "penalty_only"
    recall_grace_days: int = 2
    buyin_penalty_bps: float = 0.0


@dataclass(frozen=True)
class BorrowConfig:
    enabled: bool = False
    accrual_mode: Literal["entry_notional", "mtm_daily"] = "entry_notional"
    day_count: Literal["busdays", "calendar_days", "sessions"] = "busdays"
    include_exit_day: bool = False
    min_days: int = 1
    day_basis: int = 252
    default_rate_annual: float = 0.0
    per_asset_rate_annual: dict[str, float] = field(default_factory=dict)
    rates: tuple[BorrowRatePoint, ...] = ()
    availability: tuple[BorrowAvailabilityPoint, ...] = ()
    enforcement: BorrowEnforcementConfig = field(default_factory=BorrowEnforcementConfig)
    synthetic_jitter_sigma: float = 0.0
    ftd_block_threshold: float = 0.0


@dataclass(frozen=True)
class ExecutionFeesConfig:
    per_trade: float = 0.0
    bps: float = 0.0
    per_share: float = 0.0
    min_fee: float = 0.0
    max_fee: float = 0.0


@dataclass(frozen=True)
class LightExecutionConfig:
    enabled: bool = True
    reject_on_missing_price: bool = True
    fees: ExecutionFeesConfig = field(default_factory=ExecutionFeesConfig)


@dataclass(frozen=True)
class LOBLiquidityConfig:
    enabled: bool = False
    asof_shift: int = 1
    vol_window: int = 30
    adv_window: int = 60
    min_periods_frac: float = 0.5
    spread_floor_bps: float = 1.0
    spread_sigma_mult: float = 0.005
    spread_adv_mult: float = 15.0
    adv_ref_usd: float = 1_000_000.0
    depth_frac_of_adv_shares: float = 0.001
    depth_gamma: float = 0.7
    min_depth_shares: int = 25
    max_depth_shares: int = 250_000
    min_level_shares: int = 1
    lam_adv_power: float = 0.25
    lam_min: float = 0.25
    lam_max: float = 25.0
    cancel_base: float = 0.15
    cancel_sigma_mult: float = 1.5
    cancel_min: float = 0.01
    cancel_max: float = 0.9
    max_add_frac_of_top: float = 0.5
    max_cancel_frac_of_top: float = 0.25
    max_add_min: int = 50
    max_cancel_min: int = 25
    max_add_max: int = 50_000
    max_cancel_max: int = 50_000
    tick_subpenny: float = 0.0001
    tick_penny: float = 0.01
    tick_switch_price: float = 1.0


@dataclass(frozen=True)
class LOBFillModelConfig:
    enabled: bool = False
    base_fill: float = 1.0
    safe_depth_share: float = 0.25
    depth_share_50: float = 0.50
    depth_shape: float = 1.5
    safe_participation: float = 0.002
    participation_50: float = 0.008
    participation_shape: float = 1.2
    sigma_mult: float = 1.5
    beta_kappa_base: float = 75.0
    beta_kappa_adv_power: float = 0.2
    beta_kappa_min: float = 10.0
    beta_kappa_max: float = 500.0
    allow_reject: bool = True
    reject_below: float = 0.002
    min_fill_if_filled: float = 0.01


@dataclass(frozen=True)
class LOBPostCostConfig:
    per_trade: float = 0.0
    maker_bps: float = 0.0
    taker_bps: float = 0.0
    maker_per_share: float = 0.0
    taker_per_share: float = 0.0
    min_fee: float = 0.0
    max_fee: float = 0.0


@dataclass(frozen=True)
class LOBOrderFlowSideConfig:
    mode: Literal["taker", "maker", "mixed"] = "taker"
    maker_price: Literal["best", "mid"] = "best"
    maker_prob: float = 0.5
    maker_max_top_frac: float = 0.25
    maker_touch_prob: float = 1.0
    fallback_to_taker: bool = True


@dataclass(frozen=True)
class LOBOrderFlowConfig:
    mode: Literal["taker", "maker", "mixed"] = "taker"
    maker_price: Literal["best", "mid"] = "best"
    maker_prob: float = 0.5
    maker_max_top_frac: float = 0.25
    maker_touch_prob: float = 1.0
    fallback_to_taker: bool = True
    entry: LOBOrderFlowSideConfig = field(default_factory=LOBOrderFlowSideConfig)
    exit: LOBOrderFlowSideConfig = field(default_factory=LOBOrderFlowSideConfig)


@dataclass(frozen=True)
class LOBStressModelConfig:
    enabled: bool = True
    intensity: float = 1.0
    max_entry_delay_days: int = 1
    max_exit_grace_days: int = 2
    panic_cross_bps: float = 50.0


@dataclass(frozen=True)
class LOBExecutionConfig:
    enabled: bool = True
    tick: float = 0.01
    levels: int = 5
    size_per_level: int = 1_000
    min_spread_ticks: int = 1
    steps_per_day: int = 4
    lam: float = 2.0
    max_add: int = 500
    bias_top: float = 0.7
    cancel_prob: float = 0.15
    max_cancel: int = 200
    liq_model: LOBLiquidityConfig = field(default_factory=LOBLiquidityConfig)
    fill_model: LOBFillModelConfig = field(default_factory=LOBFillModelConfig)
    post_costs: LOBPostCostConfig = field(default_factory=LOBPostCostConfig)
    order_flow: LOBOrderFlowConfig = field(default_factory=LOBOrderFlowConfig)
    stress_model: LOBStressModelConfig = field(default_factory=LOBStressModelConfig)


@dataclass(frozen=True)
class ExecutionConfig:
    mode: Literal["lob", "light"] = "lob"
    max_participation: float = 0.10
    light: LightExecutionConfig = field(default_factory=LightExecutionConfig)
    lob: LOBExecutionConfig = field(default_factory=LOBExecutionConfig)


@dataclass(frozen=True)
class ReportingTearsheetConfig:
    enabled: bool = True
    dpi: int = 150


@dataclass(frozen=True)
class ReportingConfig:
    mode: Literal["core", "debug"] = "core"
    train_visuals: tuple[str, ...] = ("cv_scores", "equity")
    test_tearsheet: ReportingTearsheetConfig = field(
        default_factory=ReportingTearsheetConfig
    )

    @property
    def debug_enabled(self) -> bool:
        return self.mode == "debug"


@dataclass(frozen=True)
class BOCVConfig:
    enabled: bool = False
    scheme: Literal["blocked", "cpcv"] = "blocked"
    n_blocks: int = 5
    k_test_blocks: int = 2
    purge: int = 0
    embargo: float = 0.0
    max_folds: int | None = None
    aggregate: Literal["median", "mean", "trimmed_mean"] = "median"
    trim_pct: float = 0.10
    shuffle: bool = False


@dataclass(frozen=True)
class BOFastConfig:
    cv: BOCVConfig = field(default_factory=BOCVConfig)


@dataclass(frozen=True)
class BORealisticConfig:
    metric: Literal["sharpe", "cagr", "calmar"] = "sharpe"
    cv: BOCVConfig = field(default_factory=BOCVConfig)


@dataclass(frozen=True)
class BOConfig:
    enabled: bool = False
    mode: Literal["fast", "realistic"] = "fast"
    out_dir: str = "runs/results/bo"
    fast: BOFastConfig = field(default_factory=BOFastConfig)
    realistic: BORealisticConfig = field(default_factory=BORealisticConfig)
    entry_z_range: tuple[float, float] | None = None
    exit_z_range: tuple[float, float] | None = None
    stop_z_range: tuple[float, float] | None = None
    init_points: int = 8
    n_iter: int = 24
    patience: int = 0
    min_revert_prob_range: tuple[float, float] | None = None
    horizon_days_range: tuple[int, int] | None = None
    markov_init_points: int | None = None
    markov_n_iter: int | None = None
    markov_patience: int | None = None


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    spread_zscore: SpreadZscoreConfig = field(default_factory=SpreadZscoreConfig)
    markov_filter: MarkovFilterConfig = field(default_factory=MarkovFilterConfig)
    pair_prefilter: PairPrefilterConfig = field(default_factory=PairPrefilterConfig)
    borrow: BorrowConfig = field(default_factory=BorrowConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    bo: BOConfig = field(default_factory=BOConfig)
    cv: BOCVConfig = field(default_factory=BOCVConfig)
