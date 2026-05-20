from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    prices_path: Path


@dataclass(frozen=True)
class PairSelectionConfig:
    min_obs: int = 120
    min_corr: float = 0.80
    max_eg_pvalue: float = 0.05
    min_half_life: float = 2.0
    max_half_life: float = 60.0
    max_hurst: float = 0.45
    max_pairs: int = 100


@dataclass(frozen=True)
class WalkforwardConfig:
    enabled: bool = True
    train_mode: str = "rolling"
    train_months: int = 18
    test_months: int = 6
    step_months: int = 6


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    start: str | None = None
    end: str | None = None
    walkforward: WalkforwardConfig = field(default_factory=WalkforwardConfig)


@dataclass(frozen=True)
class StrategyConfig:
    entry_z: float = 1.0
    exit_z: float = 0.0
    stop_z: float = 3.0
    z_window: int = 45
    z_min_periods: int = 5
    max_hold_days: int = 150
    cooldown_days: int = 1


@dataclass(frozen=True)
class MarkovConfig:
    enabled: bool = True
    horizon_days: int = 60
    min_revert_prob: float = 0.70
    min_train_observations: int = 30
    min_state_observations: int = 5
    transition_smoothing: float = 0.0
    neutral_z: float | None = None
    entry_z: float | None = None


@dataclass(frozen=True)
class RiskConfig:
    max_open_pairs: int = 20
    max_pair_weight: float = 0.05
    max_drawdown: float = 0.25


@dataclass(frozen=True)
class CostsConfig:
    fee_bps: float = 1.0
    slippage_bps: float = 0.0


@dataclass(frozen=True)
class BOCVConfig:
    n_blocks: int = 5
    k_test_blocks: int = 2
    purge: int = 0
    embargo: float = 0.0


@dataclass(frozen=True)
class BOConfig:
    enabled: bool = False
    init_points: int = 5
    n_iter: int = 15
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    cv: BOCVConfig = field(default_factory=BOCVConfig)


@dataclass(frozen=True)
class OutputConfig:
    dir: Path = Path("runs/results/backtest")


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class Config:
    data: DataConfig
    seed: int = 1508
    pair_selection: PairSelectionConfig = field(default_factory=PairSelectionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    markov: MarkovConfig = field(default_factory=MarkovConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    bo: BOConfig = field(default_factory=BOConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data = raw.get("data") or {}
    bt = raw.get("backtest") or {}
    bo = raw.get("bo") or {}
    return Config(
        seed=int(raw.get("seed", 1508)),
        data=DataConfig(prices_path=Path(data["prices_path"])),
        pair_selection=PairSelectionConfig(**(raw.get("pair_selection") or {})),
        backtest=BacktestConfig(
            initial_capital=float(bt.get("initial_capital", 100_000.0)),
            start=bt.get("start"),
            end=bt.get("end"),
            walkforward=WalkforwardConfig(**(bt.get("walkforward") or {})),
        ),
        strategy=StrategyConfig(**(raw.get("strategy") or {})),
        markov=MarkovConfig(**(raw.get("markov") or {})),
        risk=RiskConfig(**(raw.get("risk") or {})),
        costs=CostsConfig(**(raw.get("costs") or {})),
        bo=BOConfig(
            enabled=bool(bo.get("enabled", False)),
            init_points=int(bo.get("init_points", 5)),
            n_iter=int(bo.get("n_iter", 15)),
            ranges={k: tuple(v) for k, v in (bo.get("ranges") or {}).items()},
            cv=BOCVConfig(**(bo.get("cv") or {})),
        ),
        output=OutputConfig(
            dir=Path((raw.get("output") or {}).get("dir", "runs/results/backtest"))
        ),
        logging=LoggingConfig(**(raw.get("logging") or {})),
    )


def config_dict(cfg: Config) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    return clean(asdict(cfg))
