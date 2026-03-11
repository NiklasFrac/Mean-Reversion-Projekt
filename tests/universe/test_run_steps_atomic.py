from __future__ import annotations

import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from universe import run_steps


def test_persist_unadjusted_panels_uses_atomic_writer_and_renames_collision(tmp_path):
    warmup_path = tmp_path / "raw_unadj.pkl"
    vol_path = tmp_path / "vol_unadj.pkl"

    calls: list[str] = []

    def _atomic(obj: object, path):
        calls.append(str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    run_steps._persist_unadjusted_panels(
        data_cfg={
            "raw_prices_unadj_warmup_cache": str(warmup_path),
            "raw_prices_unadj_cache": str(warmup_path),
            "raw_volume_unadj_cache": str(vol_path),
        },
        prices_unadj=pd.DataFrame(
            {"AAA_close": [1.0]}, index=pd.date_range("2024-01-01", periods=1)
        ),
        volumes_unadj=pd.DataFrame(
            {"AAA": [100.0]}, index=pd.date_range("2024-01-01", periods=1)
        ),
        artifacts={},
        atomic_write_pickle_fn=_atomic,
    )

    assert len(calls) == 2
    assert any(path.endswith("raw_unadj_filtered.pkl") for path in calls)
    assert any(path.endswith("vol_unadj.pkl") for path in calls)


def test_persist_adjusted_panels_uses_atomic_writer(tmp_path):
    prices = pd.DataFrame(
        {"AAA_close": [1.0]}, index=pd.date_range("2024-01-01", periods=1)
    )
    vols = pd.DataFrame({"AAA": [100.0]}, index=pd.date_range("2024-01-01", periods=1))

    writes: list[str] = []

    def _atomic(obj: object, path):
        writes.append(str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _artifact_targets(**_kwargs):
        return (tmp_path / "prices.pkl", tmp_path / "vols.pkl", None, None)

    deps = SimpleNamespace(
        artifact_targets=_artifact_targets, atomic_write_pickle=_atomic
    )

    artifacts: dict[str, str] = {}
    run_steps._persist_adjusted_panels(
        deps=deps,
        data_cfg={
            "raw_prices_cache": str(tmp_path / "prices.pkl"),
            "volume_path": str(tmp_path / "vols.pkl"),
        },
        runtime_cfg={"use_hashed_artifacts": False},
        prices_adj=prices,
        volumes_adj=vols,
        artifacts=artifacts,
    )

    assert len(writes) == 2
    assert str(tmp_path / "prices.pkl") in writes
    assert str(tmp_path / "vols.pkl") in writes
    assert artifacts["prices"].endswith("prices.pkl")
    assert artifacts["volumes"].endswith("vols.pkl")


def test_persist_adjusted_panels_skips_copy_when_mirror_equals_source(
    tmp_path, monkeypatch
):
    prices = pd.DataFrame(
        {"AAA_close": [1.0]}, index=pd.date_range("2024-01-01", periods=1)
    )
    vols = pd.DataFrame({"AAA": [100.0]}, index=pd.date_range("2024-01-01", periods=1))

    def _atomic(obj: object, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    prices_path = tmp_path / "prices.pkl"
    vols_path = tmp_path / "vols.pkl"

    def _artifact_targets(**_kwargs):
        return (prices_path, vols_path, prices_path, vols_path)

    copy_calls: list[tuple[str, str]] = []

    def _copy2(src: str, dst: str):
        copy_calls.append((src, dst))

    monkeypatch.setattr(run_steps.shutil, "copy2", _copy2)
    deps = SimpleNamespace(
        artifact_targets=_artifact_targets, atomic_write_pickle=_atomic
    )

    artifacts: dict[str, str] = {}
    run_steps._persist_adjusted_panels(
        deps=deps,
        data_cfg={
            "raw_prices_cache": str(prices_path),
            "volume_path": str(vols_path),
        },
        runtime_cfg={"use_hashed_artifacts": False},
        prices_adj=prices,
        volumes_adj=vols,
        artifacts=artifacts,
    )

    assert copy_calls == []
    assert artifacts["prices_canonical"] == str(prices_path)
    assert artifacts["volumes_canonical"] == str(vols_path)


def test_build_download_settings_reads_request_timeout():
    settings = run_steps._build_download_settings(
        data_cfg={},
        runtime_cfg={"progress_bar": False, "request_timeout": 17},
        tickers_final=["AAA"],
        derive_date_range_fn=lambda _cfg: ("2024-01-01", "2024-01-31"),
    )
    assert settings.request_timeout == 17.0


def test_build_download_settings_parses_bool_and_timeout_strings():
    settings = run_steps._build_download_settings(
        data_cfg={
            "allow_incomplete_history": "true",
            "adjust_dividends": "false",
            "strict_unadjusted_validation": "1",
        },
        runtime_cfg={"progress_bar": "false", "request_timeout": "bad"},
        tickers_final=["AAA"],
        derive_date_range_fn=lambda _cfg: ("2024-01-01", "2024-01-31"),
    )
    assert settings.allow_incomplete_history is True
    assert settings.adjust_dividends is False
    assert settings.strict_unadjusted_validation is True
    assert settings.progress_bar is False
    assert settings.request_timeout is None


def test_build_download_settings_sanitizes_invalid_download_knobs():
    settings = run_steps._build_download_settings(
        data_cfg={
            "download_batch": -3,
            "download_pause": -1.0,
            "download_retries": -7,
            "backoff_factor": 0.0,
        },
        runtime_cfg={"progress_bar": False},
        tickers_final=["AAA"],
        derive_date_range_fn=lambda _cfg: ("2024-01-01", "2024-01-31"),
    )
    assert settings.batch_size == 1
    assert settings.pause == 0.0
    assert settings.retries == 0
    assert settings.backoff == 2.0


def test_download_and_persist_history_sets_policy_text_for_filter_asof(tmp_path):
    deps = run_steps.RunnerDeps(
        fetch_price_volume_data=lambda *a, **k: (pd.DataFrame(), pd.DataFrame()),
        retry_missing_history=lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), []),
        normalize_panel=lambda df, _interval: df,
        fetch_unadjusted_panels=lambda *a, **k: (
            pd.DataFrame(),
            pd.DataFrame(),
            [],
            {},
        ),
        validate_unadjusted_vs_adjusted=lambda **kwargs: None,
        build_data_quality_report=lambda **kwargs: {},
        write_universe_csv=lambda *_a, **_k: None,
        write_universe_ext_csv=lambda *_a, **_k: None,
        artifact_targets=lambda **_k: (
            tmp_path / "prices.pkl",
            tmp_path / "vols.pkl",
            None,
            None,
        ),
        atomic_write_pickle=lambda *_a, **_k: None,
        norm_symbol=lambda s: str(s).upper(),
    )

    extra_out: dict[str, object] = {}
    df_uni = pd.DataFrame(index=pd.Index([], name="ticker"))
    df_funda = pd.DataFrame(index=pd.Index([], name="ticker"))
    _, _, _, _, _ = run_steps.download_and_persist_history(
        deps=deps,
        df_universe=df_uni,
        df_fundamentals=df_funda,
        tickers_final=[],
        data_cfg={
            "group_download": "false",
            "adjust_dividends": False,
            "adv_path": str(tmp_path / "adv_map.csv"),
            "raw_prices_cache": str(tmp_path / "raw_prices.pkl"),
            "volume_path": str(tmp_path / "raw_volume.pkl"),
            "raw_prices_unadj_warmup_cache": str(
                tmp_path / "raw_prices_unadj_warmup.pkl"
            ),
            "raw_prices_unadj_cache": str(tmp_path / "raw_prices_unadj.pkl"),
            "raw_volume_unadj_cache": str(tmp_path / "raw_volume_unadj.pkl"),
        },
        universe_cfg={},
        runtime_cfg={},
        stop_event=None,
        extra_out=extra_out,
        derive_date_range_fn=lambda _cfg: ("2024-01-01", "2024-01-31"),
    )

    policy = extra_out.get("data_policy")
    assert isinstance(policy, dict)
    filters_asof = policy.get("filters_asof")
    assert isinstance(filters_asof, dict)
    assert filters_asof.get("price") == "warmup_median_no_row_fallback"
    assert (
        filters_asof.get("dollar_adv")
        == "warmup_hist_if_column_present_else_snapshot_no_row_fallback"
    )
    assert policy.get("unadjusted_coverage_definition") == "price_and_volume"
    assert policy.get("group_download") is False


def test_download_and_persist_history_persists_panels_aligned_to_final_tickers(
    tmp_path,
):
    idx = pd.date_range("2024-01-01", periods=2)
    prices_adj = pd.DataFrame(
        {"AAA_close": [1.0, 1.1], "BBB_close": [2.0, 2.1]},
        index=idx,
    )
    volumes_adj = pd.DataFrame({"AAA": [100.0, 110.0]}, index=idx)
    prices_unadj = prices_adj.copy()
    volumes_unadj = volumes_adj.copy()

    def _atomic(obj: object, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    deps = run_steps.RunnerDeps(
        fetch_price_volume_data=lambda *a, **k: (prices_adj, volumes_adj),
        retry_missing_history=lambda *a, **k: (prices_adj, volumes_adj, ["BBB"]),
        normalize_panel=lambda df, _interval: df,
        fetch_unadjusted_panels=lambda *a, **k: (
            prices_unadj,
            volumes_unadj,
            ["BBB"],
            {},
        ),
        validate_unadjusted_vs_adjusted=lambda **kwargs: None,
        build_data_quality_report=lambda **kwargs: {},
        write_universe_csv=lambda *_a, **_k: None,
        write_universe_ext_csv=lambda *_a, **_k: None,
        artifact_targets=lambda **_k: (
            tmp_path / "prices.pkl",
            tmp_path / "vols.pkl",
            None,
            None,
        ),
        atomic_write_pickle=_atomic,
        norm_symbol=lambda s: str(s).upper(),
    )

    df_uni = pd.DataFrame(index=pd.Index(["AAA", "BBB"], name="ticker"))
    df_funda = pd.DataFrame(index=pd.Index(["AAA", "BBB"], name="ticker"))
    extra_out: dict[str, object] = {}

    _, _, tickers_final, _, _ = run_steps.download_and_persist_history(
        deps=deps,
        df_universe=df_uni,
        df_fundamentals=df_funda,
        tickers_final=["AAA", "BBB"],
        data_cfg={
            "allow_incomplete_history": True,
            "adjust_dividends": True,
            "raw_prices_unadj_warmup_cache": str(tmp_path / "warmup_unadj.pkl"),
            "raw_prices_unadj_cache": str(tmp_path / "raw_unadj.pkl"),
            "raw_volume_unadj_cache": str(tmp_path / "raw_vol_unadj.pkl"),
        },
        universe_cfg={"adv_cache": str(tmp_path / "adv.pkl")},
        runtime_cfg={"use_hashed_artifacts": False, "progress_bar": False},
        stop_event=None,
        extra_out=extra_out,
        derive_date_range_fn=lambda _cfg: ("2024-01-01", "2024-01-31"),
    )

    assert tickers_final == ["AAA"]

    adj_prices_saved = pickle.loads((tmp_path / "prices.pkl").read_bytes())
    adj_vols_saved = pickle.loads((tmp_path / "vols.pkl").read_bytes())
    unadj_prices_saved = pickle.loads((tmp_path / "raw_unadj.pkl").read_bytes())
    unadj_vols_saved = pickle.loads((tmp_path / "raw_vol_unadj.pkl").read_bytes())

    assert list(adj_prices_saved.columns) == ["AAA_close"]
    assert list(adj_vols_saved.columns) == ["AAA"]
    assert list(unadj_prices_saved.columns) == ["AAA_close"]
    assert list(unadj_vols_saved.columns) == ["AAA"]


def test_download_and_persist_history_requires_close_and_volume_for_adjusted_coverage(
    tmp_path,
):
    idx = pd.date_range("2024-01-01", periods=2)
    prices_adj = pd.DataFrame({"AAA_close": [1.0, 1.1]}, index=idx)
    volumes_adj = pd.DataFrame(index=idx)

    deps = run_steps.RunnerDeps(
        fetch_price_volume_data=lambda *a, **k: (prices_adj, volumes_adj),
        retry_missing_history=lambda *a, **k: (prices_adj, volumes_adj, []),
        normalize_panel=lambda df, _interval: df,
        fetch_unadjusted_panels=lambda *a, **k: (
            pd.DataFrame(),
            pd.DataFrame(),
            [],
            {},
        ),
        validate_unadjusted_vs_adjusted=lambda **kwargs: None,
        build_data_quality_report=lambda **kwargs: {},
        write_universe_csv=lambda *_a, **_k: None,
        write_universe_ext_csv=lambda *_a, **_k: None,
        artifact_targets=lambda **_k: (
            tmp_path / "prices.pkl",
            tmp_path / "vols.pkl",
            None,
            None,
        ),
        atomic_write_pickle=lambda *_a, **_k: None,
        norm_symbol=lambda s: str(s).upper(),
    )

    df_uni = pd.DataFrame(index=pd.Index(["AAA"], name="ticker"))
    df_funda = pd.DataFrame(index=pd.Index(["AAA"], name="ticker"))

    with pytest.raises(RuntimeError, match="Missing history"):
        run_steps.download_and_persist_history(
            deps=deps,
            df_universe=df_uni,
            df_fundamentals=df_funda,
            tickers_final=["AAA"],
            data_cfg={"allow_incomplete_history": False, "adjust_dividends": False},
            universe_cfg={},
            runtime_cfg={"use_hashed_artifacts": False, "progress_bar": False},
            stop_event=None,
            extra_out={},
            derive_date_range_fn=lambda _cfg: ("2024-01-01", "2024-01-31"),
        )
