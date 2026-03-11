from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.reporting import artifacts, perf_timer, tearsheet


def test_perf_timer_measures_runtime() -> None:
    res = perf_timer.measure_runtime(lambda: 123)
    assert res.value == 123
    assert res.runtime_sec >= 0.0


def test_tearsheet_write_outputs(tmp_path: Path) -> None:
    idx = pd.bdate_range("2024-01-02", periods=20)
    eq = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx, name="equity")
    stats_df = pd.DataFrame({"equity": eq, "returns": eq.pct_change().fillna(0.0)})

    out_dir = tmp_path / "ts"
    tearsheet.write_tearsheet(
        eq,
        stats_df=stats_df,
        out_dir=out_dir,
        rolling_window=5,
        use_log_scale=False,
        dpi=80,
        include_pdf=False,
        save_svg=False,
    )
    pngs = list(out_dir.glob("*.png"))
    assert pngs


def test_tearsheet_with_pdf(tmp_path: Path) -> None:
    idx = pd.bdate_range("2024-01-02", periods=10)
    eq = pd.Series(np.linspace(100.0, 105.0, len(idx)), index=idx, name="equity")
    stats_df = pd.DataFrame(
        {
            "equity": eq,
            "returns": eq.pct_change().fillna(0.0),
            "phase": ["test"] * len(eq),
            "start": [idx[0]] * len(eq),
            "end": [idx[-1]] * len(eq),
        },
        index=idx,
    )
    out_dir = tmp_path / "ts_pdf"
    tearsheet.write_tearsheet(
        eq,
        stats_df=stats_df,
        out_dir=out_dir,
        rolling_window=3,
        include_pdf=True,
        save_svg=True,
        phase_filter="test",
    )
    assert (out_dir / "stats_table.png").exists()
    summary = tearsheet.summarize_stats(eq)
    assert not summary.empty


def test_artifacts_cache_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    store = artifacts.ArtifactStore.default()
    cfg = {"a": 1}

    def builder():
        return {"ok": True}

    ser = artifacts.JSONSerializer()
    out = artifacts.load_or_build(
        stage=artifacts.Stage.REPORTS,
        config=cfg,
        builder=builder,
        serializer=ser,
        store=store,
    )
    assert out["ok"] is True

    out2 = artifacts.load_or_build(
        stage=artifacts.Stage.REPORTS,
        config=cfg,
        builder=builder,
        serializer=ser,
        store=store,
    )
    assert out2["ok"] is True

    h = artifacts.hash_json_sha256({"x": 1})
    assert isinstance(h, str) and len(h) > 0
