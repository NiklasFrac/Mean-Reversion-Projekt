import pytest

from backtest.simulators import fill_model


def test_fill_model_cfg_from_exec_lob_invalid_values() -> None:
    cfg = fill_model.FillModelCfg.from_exec_lob(
        {"fill_model": {"enabled": True, "base_fill": "bad", "beta_kappa_min": "nan"}}
    )
    assert cfg.enabled is True
    assert cfg.base_fill == 1.0
    assert cfg.beta_kappa_min == 10.0


def test_fill_model_disabled_returns_full_fill() -> None:
    cfg = fill_model.FillModelCfg(enabled=False)
    frac, diag = fill_model.sample_package_fill_fraction(
        cfg=cfg,
        seed=1,
        shard_id=0,
        depth_total_shares_pair=1000.0,
        qty_pair_shares=100.0,
        adv_usd_pair=1_000_000.0,
        adv_ref_usd=1_000_000.0,
        participation_usd=1000.0,
        sigma_pair=0.02,
    )
    assert frac == 1.0
    assert diag["expected"] == 1.0


def test_fill_model_reject_and_min_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_reject = fill_model.FillModelCfg(
        enabled=True, allow_reject=True, reject_below=1.1
    )
    frac, _diag = fill_model.sample_package_fill_fraction(
        cfg=cfg_reject,
        seed=2,
        shard_id=0,
        depth_total_shares_pair=1000.0,
        qty_pair_shares=10.0,
        adv_usd_pair=1_000_000.0,
        adv_ref_usd=1_000_000.0,
        participation_usd=100.0,
        sigma_pair=0.01,
    )
    assert frac == 0.0

    class DummyRng:
        def beta(self, *_args, **_kwargs) -> float:
            return 0.1

    monkeypatch.setattr(
        fill_model.np.random, "Generator", lambda *_args, **_kwargs: DummyRng()
    )
    cfg_min = fill_model.FillModelCfg(
        enabled=True, allow_reject=False, min_fill_if_filled=0.9
    )
    frac2, _diag2 = fill_model.sample_package_fill_fraction(
        cfg=cfg_min,
        seed=3,
        shard_id=1,
        depth_total_shares_pair=1000.0,
        qty_pair_shares=900.0,
        adv_usd_pair=1_000_000.0,
        adv_ref_usd=1_000_000.0,
        participation_usd=50_000.0,
        sigma_pair=0.05,
    )
    assert frac2 == pytest.approx(0.9)
