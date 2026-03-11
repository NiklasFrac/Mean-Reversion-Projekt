"""
External plausibility targets for execution costs under free-data constraints.

These are intentionally *broad* to avoid overfitting. They encode stylized facts:
- Most-liquid names: small costs (few bps)
- Least-liquid names: larger but still bounded costs
- Impact increases with participation

No target uses simulator outputs as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostTargets:
    # Total execution cost in bps (median) per ADV decile (1=least liquid, 10=most liquid).
    # Broad ranges to accommodate daily-data limitations.
    total_bps_median_lo: list[float]
    total_bps_median_hi: list[float]

    # Max outlier share allowed (fraction of trades) for extreme costs.
    outlier_bps_threshold: float
    outlier_frac_hi: float

    # Expected max median spread ticks in top ADV decile (proxy).
    top_decile_spread_ticks_hi: float

    # Participation monotonicity check: impact_bps median should be increasing with participation bins.
    require_participation_monotone: bool = True


DEFAULT_TARGETS = CostTargets(
    # Decile 1 (least liquid) -> decile 10 (most liquid)
    total_bps_median_lo=[6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
    total_bps_median_hi=[90.0, 70.0, 55.0, 45.0, 35.0, 28.0, 22.0, 16.0, 12.0, 8.0],
    outlier_bps_threshold=200.0,
    outlier_frac_hi=0.01,  # 1%
    top_decile_spread_ticks_hi=2.5,
    require_participation_monotone=True,
)
