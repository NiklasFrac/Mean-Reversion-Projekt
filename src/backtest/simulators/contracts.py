from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PairState:
    pair_key: str
    y_symbol: str
    x_symbol: str
    raw_state: Mapping[str, Any]


@dataclass(frozen=True)
class EntryIntent:
    intent_id: str
    pair_key: str
    signal_date: pd.Timestamp
    signal: int
    state: PairState
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedTrade:
    intent_id: str
    pair_key: str
    signal_date: pd.Timestamp
    planned_entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp
    signal: int
    size: int
    beta_entry: float
    y_symbol: str
    x_symbol: str
    y_units: int
    x_units: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionFill:
    leg: str
    event: str
    symbol: str
    ts: pd.Timestamp | None
    price: float | None
    units: float
    side: str
    liquidity: str

    @property
    def notional(self) -> float:
        if self.price is None:
            return 0.0
        return abs(float(self.units) * float(self.price))


@dataclass(frozen=True)
class EntryResult:
    status: str
    executed_at: pd.Timestamp | None
    delay_days: int = 0
    reject_reason: str = ""
    fills: tuple[ExecutionFill, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    row_data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitResult:
    status: str
    executed_at: pd.Timestamp | None
    delay_days: int = 0
    forced_exit: bool = False
    reject_reason: str = ""
    fills: tuple[ExecutionFill, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    row_data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalTrade:
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class ExecutionContext:
    cfg_obj: Any
    price_data: Mapping[str, pd.Series]
    market_data_panel: pd.DataFrame | None
    adv_map: Mapping[str, float] | None
    calendar: pd.DatetimeIndex


@dataclass(frozen=True)
class IntentSimulationResult:
    trades: pd.DataFrame
    entry_intents: pd.DataFrame
    state_transitions: pd.DataFrame
    debug_artifacts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestResult:
    stats: pd.DataFrame
    trades: pd.DataFrame
    entry_intents: pd.DataFrame = field(default_factory=pd.DataFrame)
    state_transitions: pd.DataFrame = field(default_factory=pd.DataFrame)
    debug_artifacts: Mapping[str, Any] = field(default_factory=dict)
