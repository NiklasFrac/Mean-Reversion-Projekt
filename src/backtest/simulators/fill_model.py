"""
Fill uncertainty model (heuristic, free-data compatible).

Goal
----
Model the empirical reality that even for marketable orders, *executed size* can be
less than intended size due to (unobserved) microstructure constraints.

With free data (daily OHLCV + ADV proxies) we cannot calibrate to true tape-based
fill probabilities. Therefore this module implements a *bounded, monotone* heuristic:

  expected_fill = f(order_size_vs_depth, participation_vs_ADV, volatility)

and optional stochasticity via a Beta distribution (deterministic under seeds).

This is designed to be used as a **package fill fraction** for pair trades so that
the hedge ratio is preserved (both legs scaled by the same fill fraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = ["FillModelCfg", "sample_package_fill_fraction"]


def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


@dataclass(frozen=True)
class FillModelCfg:
    enabled: bool = False

    # Expected-fill components
    base_fill: float = 1.0
    safe_depth_share: float = 0.15
    depth_share_50: float = 0.35
    depth_shape: float = 1.5
    safe_participation: float = 0.001  # 0.1% of $ADV
    participation_50: float = 0.005  # 0.5% of $ADV
    participation_shape: float = 1.2
    sigma_mult: float = 1.5

    # Stochasticity (Beta distribution around expected_fill)
    beta_kappa_base: float = 75.0
    beta_kappa_adv_power: float = 0.2
    beta_kappa_min: float = 10.0
    beta_kappa_max: float = 500.0

    # Rejections (no fill) — optional, conservative.
    allow_reject: bool = True
    reject_below: float = 0.01  # if sampled fill < reject_below -> reject

    # Minimum fill if not rejected (avoid rounding to zero for tiny fills)
    min_fill_if_filled: float = 0.02

    @staticmethod
    def from_exec_lob(exec_lob: Mapping[str, Any] | None) -> "FillModelCfg":
        d = dict(exec_lob or {})
        fm = d.get("fill_model") if isinstance(d.get("fill_model"), Mapping) else {}
        fm = dict(fm or {})
        return FillModelCfg(
            enabled=bool(fm.get("enabled", False)),
            base_fill=_safe_float(fm.get("base_fill", 1.0), 1.0),
            safe_depth_share=_safe_float(fm.get("safe_depth_share", 0.15), 0.15),
            depth_share_50=_safe_float(fm.get("depth_share_50", 0.35), 0.35),
            depth_shape=_safe_float(fm.get("depth_shape", 1.5), 1.5),
            safe_participation=_safe_float(fm.get("safe_participation", 0.001), 0.001),
            participation_50=_safe_float(fm.get("participation_50", 0.005), 0.005),
            participation_shape=_safe_float(fm.get("participation_shape", 1.2), 1.2),
            sigma_mult=_safe_float(fm.get("sigma_mult", 1.5), 1.5),
            beta_kappa_base=_safe_float(fm.get("beta_kappa_base", 75.0), 75.0),
            beta_kappa_adv_power=_safe_float(fm.get("beta_kappa_adv_power", 0.2), 0.2),
            beta_kappa_min=_safe_float(fm.get("beta_kappa_min", 10.0), 10.0),
            beta_kappa_max=_safe_float(fm.get("beta_kappa_max", 500.0), 500.0),
            allow_reject=bool(fm.get("allow_reject", True)),
            reject_below=_safe_float(fm.get("reject_below", 0.01), 0.01),
            min_fill_if_filled=_safe_float(fm.get("min_fill_if_filled", 0.02), 0.02),
        )


def _expected_fill(
    *,
    depth_share: float,
    participation: float,
    sigma: float,
    cfg: FillModelCfg,
) -> float:
    # monotone decreasing in depth_share and participation; weakly decreasing in sigma
    ds50 = max(1e-12, float(cfg.depth_share_50))
    ps50 = max(1e-12, float(cfg.participation_50))
    a = max(0.1, float(cfg.depth_shape))
    b = max(0.1, float(cfg.participation_shape))

    ds = max(0.0, float(depth_share) - max(0.0, float(cfg.safe_depth_share)))
    ps = max(0.0, float(participation) - max(0.0, float(cfg.safe_participation)))

    mu_depth = 1.0 / (1.0 + float((ds / ds50) ** a)) if ds > 0 else 1.0
    mu_part = 1.0 / (1.0 + float((ps / ps50) ** b)) if ps > 0 else 1.0
    mu_vol = float(np.exp(-float(cfg.sigma_mult) * max(0.0, float(sigma))))
    mu = float(cfg.base_fill) * mu_depth * mu_part * mu_vol
    return _clip(mu, 0.0, 1.0)


def _beta_kappa(*, adv_usd_pair: float, adv_ref_usd: float, cfg: FillModelCfg) -> float:
    adv_scale = max(1e-12, float(adv_usd_pair) / max(1e-12, float(adv_ref_usd)))
    k = float(cfg.beta_kappa_base) * float(adv_scale ** float(cfg.beta_kappa_adv_power))
    return _clip(k, float(cfg.beta_kappa_min), float(cfg.beta_kappa_max))


def sample_package_fill_fraction(
    *,
    cfg: FillModelCfg,
    seed: int | None,
    shard_id: int,
    depth_total_shares_pair: float,
    qty_pair_shares: float,
    adv_usd_pair: float,
    adv_ref_usd: float,
    participation_usd: float,
    sigma_pair: float,
) -> tuple[float, dict[str, float]]:
    """
    Return (fill_frac, diagnostics).

    The returned fill_frac is in [0,1]. If cfg.allow_reject is True, it can be 0.
    """
    if not cfg.enabled:
        return 1.0, {"expected": 1.0, "sampled": 1.0}

    depth_total = max(1.0, float(depth_total_shares_pair))
    qty = max(0.0, float(qty_pair_shares))
    depth_share = qty / depth_total
    part = max(0.0, float(participation_usd))
    sigma = max(0.0, float(sigma_pair))

    mu = _expected_fill(
        depth_share=depth_share, participation=part, sigma=sigma, cfg=cfg
    )

    # Deterministic RNG if seed is given; otherwise still deterministic per-process via default_rng.
    if seed is None or seed < 0:
        rng = np.random.default_rng()
    else:
        ss = np.random.SeedSequence([int(seed), int(shard_id), 91173])
        rng = np.random.Generator(np.random.PCG64(ss))

    kappa = _beta_kappa(adv_usd_pair=adv_usd_pair, adv_ref_usd=adv_ref_usd, cfg=cfg)
    # Beta parameters; handle edge cases cleanly
    mu_c = _clip(mu, 1e-6, 1.0 - 1e-6)
    a = mu_c * kappa
    b = (1.0 - mu_c) * kappa
    sampled = float(rng.beta(a, b))

    if cfg.allow_reject and sampled < float(cfg.reject_below):
        sampled = 0.0
    elif sampled > 0.0 and sampled < float(cfg.min_fill_if_filled):
        sampled = float(cfg.min_fill_if_filled)

    sampled = _clip(sampled, 0.0, 1.0)
    return sampled, {
        "expected": float(mu),
        "sampled": float(sampled),
        "depth_share": float(depth_share),
        "participation_usd": float(part),
        "sigma": float(sigma),
        "kappa": float(kappa),
    }
