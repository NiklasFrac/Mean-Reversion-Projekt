from __future__ import annotations

import numpy as np
import pandas as pd

from processing.filtering_stage import FinalizedInputs, run_filtering_stages


def _build_inputs() -> tuple[FinalizedInputs, pd.DataFrame]:
    idx = pd.date_range("2024-01-02", periods=90, freq="B", tz="America/New_York")
    adj = pd.Series(10.0 + np.linspace(0.0, 1.0, len(idx)), index=idx)
    ratio = pd.Series(1.0, index=idx)
    ratio.iloc[45:] = 0.2
    unadj = adj * ratio

    close = pd.DataFrame({"SPLT": adj.astype(float)}, index=idx)
    panel = {
        "close": close.copy(),
        "open": close * 1.00,
        "high": close * 1.01,
        "low": close * 0.99,
    }
    volume = pd.DataFrame({"SPLT": np.full(len(idx), 1000.0)}, index=idx)
    masks = {"SPLT": pd.Series(True, index=idx)}

    finalized = FinalizedInputs(
        close=close,
        panel_fields=panel,
        volume=volume,
        pre_q_exec={"checks": {}},
        ref_index=idx,
        tradable_masks=masks,
    )
    close_raw_unadj = pd.DataFrame({"SPLT": unadj.astype(float)}, index=idx)
    return finalized, close_raw_unadj


def test_reverse_split_drop_on_score_only_switches_drop_behavior() -> None:
    finalized, close_raw_unadj = _build_inputs()
    stage1_cfg = {
        "ohlc_mask": {
            "enabled": True,
            "eps_abs": 1.0e-12,
            "eps_rel": 1.0e-8,
            "mask_nonpositive": True,
        },
        "zero_volume_with_price": {
            "enabled": True,
            "min_price_for_zero_volume_rule": 0.0,
        },
    }
    stage2_cfg = {
        "keep_pct_threshold": 0.80,
        "max_start_na": 3,
        "max_end_na": 3,
        "max_gap_bars": 3,
        "hard_drop": True,
        "min_tradable_rows_for_ohl_gate": 40,
        "ohl_missing_pct_max": 0.25,
    }
    reverse_base = {
        "enabled": True,
        "max_split_jump_gap": 3,
        "jump_threshold": 0.90,
        "level_window": 63,
        "level_mad_k": 6.0,
        "post_persist_bars": 5,
        "factor_tolerance": 0.20,
        "weights": {
            "jump": 0.35,
            "level": 0.25,
            "persistence": 0.25,
            "factor": 0.15,
        },
        "score_flag_threshold": 0.55,
        "score_drop_threshold": 0.70,
    }

    keep_result = run_filtering_stages(
        finalized=finalized,
        stage1_cfg=stage1_cfg,
        stage2_cfg=stage2_cfg,
        reverse_split_cfg={**reverse_base, "drop_on_score_only": False},
        caps_cfg={"enabled": False},
        outlier_cfg={"enabled": False},
        staleness_cfg={"enabled": False},
        close_raw_unadj=close_raw_unadj,
        strict_inputs=True,
    )
    assert "SPLT" in keep_result.close.columns
    assert keep_result.stages["reverse_split"]["n_dropped"] == 0
    assert keep_result.stages["reverse_split"]["n_flagged"] == 1
    assert (
        keep_result.stages["reverse_split"]["thresholds"]["drop_on_score_only"] is False
    )

    drop_result = run_filtering_stages(
        finalized=finalized,
        stage1_cfg=stage1_cfg,
        stage2_cfg=stage2_cfg,
        reverse_split_cfg={**reverse_base, "drop_on_score_only": True},
        caps_cfg={"enabled": False},
        outlier_cfg={"enabled": False},
        staleness_cfg={"enabled": False},
        close_raw_unadj=close_raw_unadj,
        strict_inputs=True,
    )
    assert "SPLT" not in drop_result.close.columns
    assert drop_result.stages["reverse_split"]["n_dropped"] == 1
    assert drop_result.stages["reverse_split"]["n_flagged"] == 0
    assert drop_result.drop_reasons["SPLT"] == "reverse_split_corporate_action_artifact"
    assert (
        drop_result.stages["reverse_split"]["thresholds"]["drop_on_score_only"] is True
    )


def test_reverse_split_drop_on_flag_threshold_bypasses_quality_gate() -> None:
    finalized, close_raw_unadj = _build_inputs()
    stage1_cfg = {
        "ohlc_mask": {
            "enabled": True,
            "eps_abs": 1.0e-12,
            "eps_rel": 1.0e-8,
            "mask_nonpositive": True,
        },
        "zero_volume_with_price": {
            "enabled": True,
            "min_price_for_zero_volume_rule": 0.0,
        },
    }
    stage2_cfg = {
        "keep_pct_threshold": 0.80,
        "max_start_na": 3,
        "max_end_na": 3,
        "max_gap_bars": 3,
        "hard_drop": True,
        "min_tradable_rows_for_ohl_gate": 40,
        "ohl_missing_pct_max": 0.25,
    }
    reverse_cfg = {
        "enabled": True,
        "max_split_jump_gap": 3,
        "jump_threshold": 0.90,
        "level_window": 63,
        "level_mad_k": 6.0,
        "post_persist_bars": 5,
        "factor_tolerance": 0.20,
        "weights": {
            "jump": 0.35,
            "level": 0.25,
            "persistence": 0.25,
            "factor": 0.15,
        },
        "score_flag_threshold": 0.55,
        "score_drop_threshold": 0.95,
        "drop_on_score_only": False,
        "drop_on_flag_threshold": True,
    }

    result = run_filtering_stages(
        finalized=finalized,
        stage1_cfg=stage1_cfg,
        stage2_cfg=stage2_cfg,
        reverse_split_cfg=reverse_cfg,
        caps_cfg={"enabled": False},
        outlier_cfg={"enabled": False},
        staleness_cfg={"enabled": False},
        close_raw_unadj=close_raw_unadj,
        strict_inputs=True,
    )

    assert "SPLT" not in result.close.columns
    assert result.stages["reverse_split"]["n_dropped"] == 1
    assert result.stages["reverse_split"]["n_flagged"] == 0
    assert result.drop_reasons["SPLT"] == "reverse_split_corporate_action_artifact"
    assert (
        result.stages["reverse_split"]["thresholds"]["drop_on_flag_threshold"] is True
    )
    reverse_drop_evt = result.events[
        (result.events["stage"] == "reverse_split")
        & (result.events["action"] == "drop_symbol")
    ]
    assert not reverse_drop_evt.empty
    assert float(reverse_drop_evt.iloc[0]["threshold"]) == float(
        reverse_cfg["score_flag_threshold"]
    )


def test_stage2_large_gap_soft_drop_emits_warning_and_keeps_symbol() -> None:
    idx = pd.date_range("2024-01-02", periods=12, freq="B", tz="America/New_York")
    close = pd.DataFrame(
        {
            "GAP": [
                10.0,
                10.5,
                11.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                11.5,
                12.0,
                12.5,
                13.0,
                13.5,
            ]
        },
        index=idx,
    )
    panel = {
        "close": close.copy(),
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
    }
    volume = pd.DataFrame({"GAP": np.full(len(idx), 1000.0)}, index=idx)
    masks = {"GAP": pd.Series(True, index=idx)}
    finalized = FinalizedInputs(
        close=close,
        panel_fields=panel,
        volume=volume,
        pre_q_exec={"checks": {}},
        ref_index=idx,
        tradable_masks=masks,
    )

    result = run_filtering_stages(
        finalized=finalized,
        stage1_cfg={
            "ohlc_mask": {"enabled": False},
            "zero_volume_with_price": {"enabled": False},
        },
        stage2_cfg={
            "keep_pct_threshold": 0.0,
            "max_start_na": 99,
            "max_end_na": 99,
            "max_gap_bars": 2,
            "hard_drop": False,
            "min_tradable_rows_for_ohl_gate": 40,
            "ohl_missing_pct_max": 1.0,
        },
        reverse_split_cfg={"enabled": False},
        caps_cfg={"enabled": False},
        outlier_cfg={"enabled": False},
        staleness_cfg={"enabled": False},
        close_raw_unadj=None,
        strict_inputs=False,
    )

    assert "GAP" in result.close.columns
    assert "GAP" not in result.removed_symbols
    assert "GAP" not in result.drop_reasons
    large_gap_events = result.events[result.events["rule"] == "large_gap"]
    assert not large_gap_events.empty
    evt = large_gap_events.iloc[0]
    assert evt["stage"] == "stage2"
    assert evt["severity"] == "warning"
    assert evt["action"] == "keep_with_gap"
