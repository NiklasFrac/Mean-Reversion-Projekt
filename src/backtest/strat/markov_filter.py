from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

_STATE_LABELS = (
    "extreme_negative",
    "inner_negative",
    "neutral",
    "inner_positive",
    "extreme_positive",
)
_N_STATES = len(_STATE_LABELS)
_NEUTRAL_STATE = 2


@dataclass(frozen=True)
class MarkovFilterOutput:
    entry_gate: pd.Series
    hit_prob: pd.Series
    states: pd.Series
    diagnostics: dict[str, Any]


def _identity_row(size: int, idx: int) -> np.ndarray:
    row = np.zeros(size, dtype=float)
    row[int(idx)] = 1.0
    return row


def _to_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _to_int(value: Any, default: int, *, minimum: int) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(int(minimum), int(out))


def _default_output(
    z: pd.Series,
    *,
    enabled: bool,
    reason: str,
    diagnostics_extra: Mapping[str, Any] | None = None,
) -> MarkovFilterOutput:
    idx = pd.DatetimeIndex(z.index)
    entry_gate = pd.Series(True, index=idx, dtype=bool, name="markov_entry_gate")
    hit_prob = pd.Series(np.nan, index=idx, dtype=float, name="markov_hit_prob")
    states = pd.Series(pd.NA, index=idx, dtype="Int64", name="markov_state")
    diagnostics: dict[str, Any] = {
        "enabled": bool(enabled),
        "active": False,
        "reason": str(reason),
        "state_labels": list(_STATE_LABELS),
    }
    if diagnostics_extra:
        diagnostics.update(dict(diagnostics_extra))
    return MarkovFilterOutput(
        entry_gate=entry_gate,
        hit_prob=hit_prob,
        states=states,
        diagnostics=diagnostics,
    )


def _resolve_cfg(
    cfg: Mapping[str, Any],
    *,
    entry_z: float,
    exit_z: float,
) -> dict[str, Any]:
    raw = (
        cfg.get("markov_filter", {})
        if isinstance(cfg.get("markov_filter"), Mapping)
        else {}
    )
    enabled = bool(raw.get("enabled", False))
    horizon_days = _to_int(
        raw.get("horizon_days", raw.get("horizon", 10)), 10, minimum=1
    )
    min_revert_prob = _to_float(
        raw.get("min_revert_prob", raw.get("threshold", 0.5)), 0.5
    )
    min_revert_prob = float(min(max(min_revert_prob, 0.0), 1.0))
    min_train_observations = _to_int(
        raw.get("min_train_observations", 30), 30, minimum=2
    )
    min_state_observations = _to_int(raw.get("min_state_observations", 5), 5, minimum=1)
    transition_smoothing = max(
        0.0, _to_float(raw.get("transition_smoothing", 0.0), 0.0)
    )
    neutral_z = abs(_to_float(raw.get("neutral_z", exit_z), exit_z))
    extreme_z = abs(_to_float(raw.get("entry_z", entry_z), entry_z))
    return {
        "enabled": enabled,
        "horizon_days": int(horizon_days),
        "min_revert_prob": float(min_revert_prob),
        "min_train_observations": int(min_train_observations),
        "min_state_observations": int(min_state_observations),
        "transition_smoothing": float(transition_smoothing),
        "neutral_z": float(neutral_z),
        "entry_z": float(extreme_z),
    }


def discretize_z_to_states(
    z: pd.Series,
    *,
    neutral_z: float,
    entry_z: float,
) -> pd.Series:
    idx = pd.DatetimeIndex(z.index)
    values = pd.to_numeric(z, errors="coerce").to_numpy(dtype=float, copy=False)
    out = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    out[finite & (values <= -float(entry_z))] = 0.0
    out[finite & (values > -float(entry_z)) & (values < -float(neutral_z))] = 1.0
    out[finite & (np.abs(values) <= float(neutral_z))] = 2.0
    out[finite & (values > float(neutral_z)) & (values < float(entry_z))] = 3.0
    out[finite & (values >= float(entry_z))] = 4.0
    return pd.Series(out, index=idx, dtype="Float64", name="markov_state").astype(
        "Int64"
    )


def _transition_counts(states: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((_N_STATES, _N_STATES), dtype=float)
    visits = np.zeros(_N_STATES, dtype=int)
    vals = states.to_numpy(dtype=object, copy=False)
    for raw in vals:
        if pd.isna(raw):
            continue
        visits[int(raw)] += 1
    for prev_raw, curr_raw in zip(vals[:-1], vals[1:]):
        if pd.isna(prev_raw) or pd.isna(curr_raw):
            continue
        counts[int(prev_raw), int(curr_raw)] += 1.0
    return counts, visits


def _transition_matrix(
    counts: np.ndarray, *, smoothing: float
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.full_like(counts, np.nan, dtype=float)
    outgoing = counts.sum(axis=1).astype(int)
    for row_idx in range(counts.shape[0]):
        row = counts[row_idx]
        raw_total = float(row.sum())
        if raw_total <= 0.0:
            continue
        if smoothing > 0.0:
            row = row + float(smoothing)
            raw_total = float(row.sum())
        probs[row_idx] = row / raw_total
    return probs, outgoing


def _finite_horizon_hit_probability(
    probs: np.ndarray,
    *,
    horizon_days: int,
) -> np.ndarray:
    mat = np.array(probs, dtype=float, copy=True)
    for row_idx in range(mat.shape[0]):
        if row_idx == _NEUTRAL_STATE:
            mat[row_idx] = _identity_row(mat.shape[1], _NEUTRAL_STATE)
            continue
        row = mat[row_idx]
        if not np.all(np.isfinite(row)) or float(row.sum()) <= 0.0:
            mat[row_idx] = _identity_row(mat.shape[1], row_idx)
    stepped = np.linalg.matrix_power(mat, int(horizon_days))
    return stepped[:, _NEUTRAL_STATE].astype(float, copy=False)


def build_markov_entry_filter(
    cfg: Mapping[str, Any],
    *,
    z: pd.Series,
    train_index: pd.DatetimeIndex,
    eval_index: pd.DatetimeIndex,
    entry_z: float,
    exit_z: float,
) -> MarkovFilterOutput:
    params = _resolve_cfg(cfg, entry_z=entry_z, exit_z=exit_z)
    enabled = bool(params["enabled"])
    diagnostics_base = {
        "horizon_days": int(params["horizon_days"]),
        "min_revert_prob": float(params["min_revert_prob"]),
        "min_train_observations": int(params["min_train_observations"]),
        "min_state_observations": int(params["min_state_observations"]),
        "transition_smoothing": float(params["transition_smoothing"]),
        "neutral_z": float(params["neutral_z"]),
        "entry_z": float(params["entry_z"]),
    }
    if not enabled:
        return _default_output(
            z,
            enabled=False,
            reason="disabled",
            diagnostics_extra=diagnostics_base,
        )

    neutral_z = float(params["neutral_z"])
    extreme_z = float(params["entry_z"])
    if (
        not np.isfinite(neutral_z)
        or not np.isfinite(extreme_z)
        or neutral_z < 0.0
        or extreme_z <= neutral_z
    ):
        return _default_output(
            z,
            enabled=True,
            reason="invalid_state_thresholds",
            diagnostics_extra=diagnostics_base,
        )

    states = discretize_z_to_states(z, neutral_z=neutral_z, entry_z=extreme_z)
    train_states = states.reindex(pd.DatetimeIndex(train_index))
    valid_train = train_states.dropna()
    if int(len(valid_train)) < int(params["min_train_observations"]):
        return _default_output(
            z,
            enabled=True,
            reason="insufficient_train_observations",
            diagnostics_extra={
                **diagnostics_base,
                "n_train_observations": int(len(valid_train)),
            },
        )

    counts, visits = _transition_counts(train_states)
    probs, outgoing = _transition_matrix(
        counts, smoothing=float(params["transition_smoothing"])
    )
    hit_prob_by_state = _finite_horizon_hit_probability(
        probs, horizon_days=int(params["horizon_days"])
    )

    idx = pd.DatetimeIndex(z.index)
    entry_gate = pd.Series(True, index=idx, dtype=bool, name="markov_entry_gate")
    hit_prob = pd.Series(np.nan, index=idx, dtype=float, name="markov_hit_prob")
    eval_idx = pd.DatetimeIndex(eval_index)
    eval_states = states.reindex(eval_idx)
    min_state_observations = int(params["min_state_observations"])
    for ts, raw_state in eval_states.items():
        if pd.isna(raw_state):
            continue
        state = int(raw_state)
        state_prob = float(hit_prob_by_state[state])
        if np.isfinite(state_prob):
            hit_prob.at[ts] = state_prob
        enough_support = (
            int(visits[state]) >= min_state_observations and int(outgoing[state]) > 0
        )
        if enough_support and np.isfinite(state_prob):
            entry_gate.at[ts] = bool(state_prob >= float(params["min_revert_prob"]))

    blocked_eval_days = int((~entry_gate.reindex(eval_idx).fillna(True)).sum())
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "active": True,
        "reason": "ok",
        "state_labels": list(_STATE_LABELS),
        "horizon_days": int(params["horizon_days"]),
        "min_revert_prob": float(params["min_revert_prob"]),
        "min_train_observations": int(params["min_train_observations"]),
        "min_state_observations": int(params["min_state_observations"]),
        "transition_smoothing": float(params["transition_smoothing"]),
        "neutral_z": float(neutral_z),
        "entry_z": float(extreme_z),
        "n_train_observations": int(len(valid_train)),
        "blocked_eval_days": int(blocked_eval_days),
        "state_visit_counts": {
            label: int(visits[i]) for i, label in enumerate(_STATE_LABELS)
        },
        "state_outgoing_counts": {
            label: int(outgoing[i]) for i, label in enumerate(_STATE_LABELS)
        },
        "hit_prob_by_state": {
            label: (
                float(hit_prob_by_state[i])
                if np.isfinite(hit_prob_by_state[i])
                else None
            )
            for i, label in enumerate(_STATE_LABELS)
        },
    }
    return MarkovFilterOutput(
        entry_gate=entry_gate,
        hit_prob=hit_prob,
        states=states,
        diagnostics=diagnostics,
    )
