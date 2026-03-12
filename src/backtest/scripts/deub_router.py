# tools/debug_router.py
import math
import sys

import yaml


def ceil_to_lot(x, lot):
    return int(math.ceil(x / lot) * lot)


def floor_to_lot(x, lot):
    return int(math.floor(x / lot) * lot)


def parse_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_parent_shares(parent_notional, px, lot):
    if px <= 0:
        return 0
    sh = int(parent_notional // (px * lot)) * lot
    return max(sh, 0)


def explain(cfg, parent_notional, px, lot):
    r = cfg.get("backtest", {}).get("router", {})
    rules = r.get("rules", {})
    slice_cfg = r.get("slice", {})
    rounding = (slice_cfg.get("rounding", {}) or {}).get("mode", "floor").lower()
    min_child_notional = slice_cfg.get("min_child_notional", 0) or 0
    max_child_notional = slice_cfg.get("max_child_notional", float("inf"))
    min_clip = rules.get("min_clip", 1)  # can be shares OR a fraction
    max_clip = rules.get("max_clip", 0.25)
    venue_over = r.get("venue_overrides", {}) or {}

    parent_sh = compute_parent_shares(parent_notional, px, lot)
    print(
        f"\nParent: notional=${parent_notional:,.0f}  px=${px:.4f}  lot={lot}  -> parent_shares={parent_sh}"
    )

    # Two interpretations: (A) min_clip as shares (as in your `1`) and (B) as a fraction of the parent
    candA_sh = min_clip if min_clip >= 1 else math.ceil(parent_sh * min_clip)
    candB_sh = math.ceil(parent_sh * max(min_clip, 0.0)) if min_clip < 1 else min_clip
    # We use A as the "shares interpretation" and B as the "fraction interpretation"
    candidates = []
    for label, sh in [
        ("min_clip_as_shares", candA_sh),
        ("min_clip_as_fraction", candB_sh),
        ("max_clip_fraction", math.ceil(parent_sh * max_clip)),
    ]:
        if rounding == "ceil":
            sh = ceil_to_lot(sh, lot)
        else:
            sh = max(1 if sh == 0 else sh, floor_to_lot(sh, lot))
        notional = sh * px
        candidates.append((label, sh, notional))

    print("\nSlice candidates (before venue checks):")
    for label, sh, nto in candidates:
        gate = []
        if nto < min_child_notional:
            gate.append(f"min_child_notional({min_child_notional})")
        if nto > max_child_notional:
            gate.append(f"max_child_notional({max_child_notional})")
        gates = " | ".join(gate) if gate else "OK"
        print(f"  - {label:22s}: shares={sh:>6}  notional=${nto:,.2f}  => {gates}")

    print("\nPer-Venue-Check (min_notional etc.):")
    for venue, ov in venue_over.items():
        active = ov.get("active", True)
        if not active:
            print(f"  {venue}: INACTIVE")
            continue
        v_min_notional = ov.get("min_notional", 0) or 0
        lot_size = ov.get("lot_size", lot) or lot
        tick = ov.get("tick_size", 0.01)
        rel = ov.get("reliability", 1.0)
        mp = ov.get("max_participation", None)

        # check each candidate against venue min_notional
        msgs = []
        for label, sh, nto in candidates:
            ok = nto >= v_min_notional
            msgs.append(
                f"{label}: {'OK' if ok else f'BLOCK (min_notional {v_min_notional})'}"
            )
        print(
            f"  {venue}: lot={lot_size} tick={tick} reliability={rel} max_part={mp} | "
            + " | ".join(msgs)
        )

    print("\nNote:")
    print("  - If ALL candidates fail min_child_notional OR min_notional,")
    print(
        "    the result is an empty child-order list -> 'Routing produced no child orders'."
    )
    print("  - 'rounding: ceil' prevents zero-share orders for very small clips.")
    print("  - Set venue min_notional = 0 if you want to test small tickets.")


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python tools/debug_router.py <config_backtest.yaml> <parent_notional_$> <price_$> [lot_size]"
        )
        sys.exit(2)
    cfg = parse_cfg(sys.argv[1])
    parent_notional = float(sys.argv[2])
    px = float(sys.argv[3])
    lot = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    explain(cfg, parent_notional, px, lot)


if __name__ == "__main__":
    main()
