from __future__ import annotations

from typing import Any, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backtest.utils.tz import NY_TZ


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _PassThroughFrozenModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class CalendarCfg(_StrictFrozenModel):
    exchange: Literal["XNYS", "XNAS", "XETR"] = "XNYS"
    mapping: Literal["strict", "prior"] = "strict"
    entry: Literal["next_open", "prev_close"] = "next_open"
    exit: Literal["next_close", "same_close"] = "next_close"
    lag_bars: int = 1
    tz: str = NY_TZ

    @field_validator("lag_bars")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("calendar.lag_bars must be >= 0")
        return v


class ExecBase(_StrictFrozenModel):
    mode: Literal["light", "lob"]


class LightFees(_StrictFrozenModel):
    per_trade: float = 0.0
    bps: float = 0.0
    per_share: float = 0.0
    min_fee: float = 0.0
    max_fee: float = 0.0

    @field_validator("per_trade", "bps", "per_share", "min_fee", "max_fee")
    @classmethod
    def _nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("execution.light.fees.* must be >= 0")
        return v


class ExecLightCfg(_StrictFrozenModel):
    enabled: bool = True
    reject_on_missing_price: bool = True
    fees: LightFees = Field(default_factory=LightFees)


class ExecLight(ExecBase):
    mode: Literal["light"]
    light: ExecLightCfg = Field(default_factory=ExecLightCfg)


class LobPostCosts(_StrictFrozenModel):
    per_trade: float = 0.0
    maker_bps: float = 0.0
    taker_bps: float = 0.0

    @field_validator("per_trade", "taker_bps")
    @classmethod
    def _nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("execution.lob.post_costs values must be >= 0")
        return v


class ExecLOB(ExecBase):
    mode: Literal["lob"]
    tick: float
    levels: int = 5
    size_per_level: int
    min_spread_ticks: int = 1
    lam: float = 0.0
    max_add: int = 0
    bias_top: float = 0.0
    cancel_prob: float = 0.0
    max_cancel: int = 0
    steps_per_day: int = 78
    post_costs: LobPostCosts = Field(default_factory=LobPostCosts)

    @field_validator("tick")
    @classmethod
    def _tick_pos(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("execution.lob.tick must be > 0")
        return v

    @field_validator("levels", "size_per_level", "min_spread_ticks", "steps_per_day")
    @classmethod
    def _int_pos(cls, v: int) -> int:
        if v < 1:
            raise ValueError("execution.lob integer params must be >= 1")
        return v

    @field_validator("lam", "bias_top", "cancel_prob")
    @classmethod
    def _float_bounds(cls, v: float) -> float:
        if v < 0:
            raise ValueError("execution.lob lam/bias_top/cancel_prob must be >= 0")
        return v


ExecCfg = Annotated[ExecLight | ExecLOB, Field(discriminator="mode")]


class BorrowCfg(_StrictFrozenModel):
    enabled: bool = False
    day_basis: int = 252
    csv_path: str | None = None

    @field_validator("day_basis")
    @classmethod
    def _dbasis(cls, v: int) -> int:
        if v not in (252, 360, 365):
            raise ValueError("borrow.day_basis should be one of {252,360,365}")
        return v

    @field_validator("csv_path")
    @classmethod
    def _gate(cls, v: str | None, info: Any) -> str | None:
        enabled = info.data.get("enabled", False)
        if enabled and not v:
            raise ValueError("borrow.enabled=True requires borrow.csv_path")
        return v


class RiskCaps(_StrictFrozenModel):
    max_gross: float = 1.0
    max_net: float = 1.0
    per_trade: float = 0.1
    per_name: float = 0.2
    clusters: dict[str, float] | None = None

    @field_validator("max_gross", "max_net", "per_trade", "per_name")
    @classmethod
    def _in_0_10(cls, v: float) -> float:
        if not (0 <= v <= 10):
            raise ValueError("risk caps out of plausible bounds (0..10)")
        return v


class RiskCfg(_StrictFrozenModel):
    caps: RiskCaps = Field(default_factory=RiskCaps)


class BOCfg(_PassThroughFrozenModel):
    enabled: bool = False


class RegimeCfg(_PassThroughFrozenModel):
    enabled: bool = False


class MonitoringCfg(_PassThroughFrozenModel):
    enabled: bool = False


class PathsCfg(_PassThroughFrozenModel):
    base_dir: str | None = None


class MLflowCfg(_PassThroughFrozenModel):
    enabled: bool = False


class BacktestConfig(_StrictFrozenModel):
    run_mode: Literal["aktuell", "konservativ"] = "konservativ"
    version: str | None = None
    global_seed: int | None = None

    calendar: CalendarCfg
    execution: ExecCfg
    borrow: BorrowCfg = Field(default_factory=BorrowCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    bo: BOCfg = Field(default_factory=BOCfg)
    regimes: RegimeCfg = Field(default_factory=RegimeCfg)
    monitoring: MonitoringCfg = Field(default_factory=MonitoringCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    mlflow: MLflowCfg = Field(default_factory=MLflowCfg)

    @field_validator("execution")
    @classmethod
    def _cross_checks(cls, v: ExecCfg) -> ExecCfg:
        return v
