# src/backtest/orderbook_sim.py
"""
Orderbook LOB Simulator (Tier-1)
- Deterministic tick handling (ROUND_HALF_UP), stable ordering, reproducible RNG
- NEW: deterministic RNG per shard (global_seed + shard_id)  -> set via ctor/set_seed/set_shard
- Market & limit orders: IOC/FOK/GTC; Post-Only with 'reject' or 'slide'
- Price/Time priority per level with FIFO queue for own resting orders
- Rich fill reports (VWAP, signed slippage in price & ticks, role)
- step(): Poisson refills + cancellations + (optional) depth maintenance + aggressive IOC flow
- Maker realism: step() simulates external taker flow that can consume own resting orders (queue FIFO)
- Fill-attribution: internal per-OID fill log; see collect_fills_for_oid(oid)

NEW HOOKS (all optional, no-ops if unset):
- Latency Hook (for timestamps/latency metrics):
    latency_fn(ctx: dict) -> dict  (return merged into report under "latency")
- Auction Hook (delegate to external auction logic when active):
    auction_fn(order: dict, ctx: dict) -> {"handled": True, "report": dict} | None

NEW: Event-Loop pre-allocation
- Internal reusable numpy buffers to reduce per-step allocations (adds/cancels/aggressors)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np

logger = logging.getLogger("orderbook")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# high precision for rounding to tick
getcontext().prec = 28


def _round_to_tick(p: float, tick: float) -> float:
    """
    ROUND_HALF_UP to nearest tick (no banker's rounding).
    Works for positive prices; negatives are guarded (not expected for equities).
    """
    p = float(p)
    t = float(tick)
    if t <= 0:
        raise ValueError("tick must be > 0")
    # Use Decimal(str(...)) to avoid binary FP artifacts at half-ticks.
    q = Decimal(str(p)) / Decimal(str(t))
    r = q.quantize(Decimal("1"), rounding=ROUND_HALF_UP)  # integer multiple
    return float(r * Decimal(str(t)))


# --- Queue-Primitives --------------------------------------------------------


@dataclass
class RestingOrder:
    oid: int
    qty: int  # remaining
    owner: Optional[str] = None  # optional label


class LimitLevel:
    """
    A price level with:
      - public: anonymous aggregated liquidity (background book)
      - queue : FIFO of our own resting orders (price/time priority)
    total_size = public + sum(queue.qty)
    """

    def __init__(self, price: float, public: int = 0) -> None:
        self.price: float = float(price)
        self.public: int = int(max(0, public))
        self.queue: deque[RestingOrder] = deque()

    # --- sizes / operations ---
    @property
    def total_size(self) -> int:
        return int(self.public + sum(o.qty for o in self.queue))

    def add_public(self, dq: int) -> None:
        self.public = max(0, int(self.public) + int(dq))

    def cancel_public(self, dq: int) -> int:
        """Reduce public depth only; return canceled qty."""
        want = int(max(0, dq))
        take = min(self.public, want)
        self.public -= take
        return int(take)

    def add_queue(self, ro: RestingOrder) -> None:
        if ro.qty > 0:
            self.queue.append(ro)

    def take(
        self, want: int, *, event_log: Optional[List[Tuple[int, float, int]]] = None
    ) -> List[Tuple[float, int]]:
        """
        Consume liquidity: public first, then FIFO queue.
        Returns fills [(price, qty), ...] at level price.
        If event_log is provided, queue-fills are appended as (oid, price, qty) for attribution.
        """
        fills: List[Tuple[float, int]] = []
        remaining = int(max(0, want))
        if remaining <= 0:
            return fills

        # public first
        if self.public > 0 and remaining > 0:
            take_pub = min(self.public, remaining)
            self.public -= take_pub
            remaining -= take_pub
            if take_pub > 0:
                fills.append((self.price, take_pub))

        # then FIFO queue
        while remaining > 0 and self.queue:
            head = self.queue[0]
            take_q = min(head.qty, remaining)
            if take_q <= 0:
                break
            head.qty -= take_q
            remaining -= take_q
            fills.append((self.price, take_q))
            if event_log is not None:
                # record attributed fill for this OID
                event_log.append((head.oid, self.price, take_q))
            if head.qty <= 0:
                self.queue.popleft()

        return fills

    def cancel_from_queue_tail(self, dq: int) -> int:
        """
        Stochastic cancels on our queue (other participants): remove from queue tail.
        Return: actually reduced qty.
        """
        want = int(max(0, dq))
        done = 0
        while want > 0 and self.queue:
            tail = self.queue[-1]
            take = min(tail.qty, want)
            tail.qty -= take
            want -= take
            done += take
            if tail.qty <= 0:
                self.queue.pop()
        return done

    def cancel_any(self, dq: int, prefer_public: bool = True) -> int:
        """Reduce sizes; prefer public or queue tail depending on flag."""
        want = int(max(0, dq))
        done = 0
        if want == 0:
            return 0
        if prefer_public and self.public > 0:
            take = min(self.public, want)
            self.public -= take
            want -= take
            done += take
        if want > 0:
            done += self.cancel_from_queue_tail(want)
        return done

    def snapshot_tuple(self) -> Tuple[float, int]:
        return (self.price, self.total_size)


# ---- RNG utils --------------------------------------------------------------


def _make_rng(
    global_seed: Optional[int], shard_id: Optional[int]
) -> np.random.Generator:
    """
    Deterministic RNG per shard: RNG = f(global_seed, shard_id).
    If global_seed is None -> non-deterministic default_rng().
    """
    if global_seed is None:
        return np.random.default_rng()
    sid = 0 if shard_id is None else int(shard_id)
    ss = np.random.SeedSequence([int(global_seed), sid])
    return np.random.Generator(np.random.PCG64(ss))


# ---- OrderBook --------------------------------------------------------------


class OrderBook:
    """
    Aggregated LOB with price/time priority, queue emulation and Post-Only.
      - asks: ascending prices (best ask first)
      - bids: descending prices (best bid first)

    Optional hooks (set via set_hooks or ctor):
      * latency_fn(ctx: dict) -> dict                                  # merged under report["latency"]
      * auction_fn(order: dict, ctx: dict) -> {"handled": True, "report": dict} | None
    """

    # ---- Construction -------------------------------------------------------
    def __init__(
        self,
        mid_price: float = 100.0,
        levels: int = 5,
        size_per_level: int = 1_000,
        tick: float = 0.01,
        seed: Optional[int] = None,
        *,
        min_spread_ticks: int = 1,
        level_sizes: Optional[List[int]] = None,
        shard_id: Optional[int] = None,
        check_invariants: bool = False,
        latency_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        auction_fn: Optional[
            Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]
        ] = None,
    ) -> None:
        if tick <= 0:
            raise ValueError("tick must be > 0")
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.tick: float = float(tick)
        self.levels_target: int = int(levels)
        self.default_size: int = int(max(size_per_level, 0))
        self._min_spread_ticks: int = int(max(1, min_spread_ticks))
        self._check_invariants: bool = bool(check_invariants)

        # deterministic RNG per shard
        self._global_seed: Optional[int] = int(seed) if seed is not None else None
        self._shard_id: Optional[int] = int(shard_id) if shard_id is not None else None
        self.rng: np.random.Generator = _make_rng(self._global_seed, self._shard_id)

        # Order-ID sequence & index
        self._next_oid: int = 1
        # oid -> (side, level_ref)
        self._oid_index: Dict[int, Tuple[str, LimitLevel]] = {}

        # attributed fill log: list of (oid, price, qty)
        self._fill_events: List[Tuple[int, float, int]] = []

        # Hooks
        self._latency_fn = latency_fn
        self._auction_fn = auction_fn

        # Pre-alloc buffers for step()
        self._buf_cap: int = max(256, levels * 16)
        self._buf_side = np.empty(
            self._buf_cap, dtype=np.int8
        )  # 0=ask/ sell, 1=bid/buy
        self._buf_size = np.empty(self._buf_cap, dtype=np.int32)
        self._buf_idx = np.empty(self._buf_cap, dtype=np.int32)

        # Build symmetric book around mid (no crossing)
        m = float(mid_price)
        self.asks: deque[LimitLevel] = deque()
        self.bids: deque[LimitLevel] = deque()
        sizes: Optional[List[int]] = None
        if level_sizes:
            try:
                sizes = [int(max(0, int(x))) for x in list(level_sizes)]
            except Exception:
                sizes = None
        if sizes is not None and len(sizes) < self.levels_target:
            sizes = sizes + [int(sizes[-1] if sizes else self.default_size)] * (
                self.levels_target - len(sizes)
            )
        for i in range(1, self.levels_target + 1):
            sz = int(sizes[i - 1]) if sizes is not None else self.default_size
            self.asks.append(
                LimitLevel(_round_to_tick(m + i * self.tick, self.tick), sz)
            )
            self.bids.append(
                LimitLevel(_round_to_tick(m - i * self.tick, self.tick), sz)
            )
        self._sanitize()

    def set_min_spread_ticks(self, min_spread_ticks: int) -> None:
        """Update the spread constraint and re-sanitize the book (no crossing, enforce min spread)."""
        self._min_spread_ticks = int(max(1, int(min_spread_ticks)))
        self._sanitize()

    def set_public_level_sizes(self, level_sizes: List[int]) -> None:
        """
        Overwrite *public* depth at each level (queue/resting orders are preserved).

        `level_sizes[i]` corresponds to i=0..levels-1 away from mid (best level first).
        """
        if not level_sizes:
            return
        try:
            sizes = [int(max(0, int(x))) for x in list(level_sizes)]
        except Exception:
            return
        if len(sizes) < self.levels_target:
            sizes = sizes + [int(sizes[-1] if sizes else self.default_size)] * (
                self.levels_target - len(sizes)
            )
        sizes = sizes[: self.levels_target]

        for i, lvl in enumerate(self.asks):
            if i >= self.levels_target:
                break
            lvl.public = int(sizes[i])
        for i, lvl in enumerate(self.bids):
            if i >= self.levels_target:
                break
            lvl.public = int(sizes[i])

        self._sanitize()

    # ---- Hook setter --------------------------------------------------------
    def set_hooks(
        self,
        *,
        latency_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        auction_fn: Optional[
            Callable[[Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]
        ] = None,
    ) -> None:
        if latency_fn is not None:
            self._latency_fn = latency_fn
        if auction_fn is not None:
            self._auction_fn = auction_fn

    # ---- Top-of-book / diagnostics -----------------------------------------
    def best_ask(self) -> Optional[float]:
        return float(self.asks[0].price) if self.asks else None

    def best_bid(self) -> Optional[float]:
        return float(self.bids[0].price) if self.bids else None

    def spread(self) -> Optional[float]:
        if not self.asks or not self.bids:
            return None
        return float(self.best_ask() - self.best_bid())  # type: ignore

    def spread_ticks(self) -> Optional[float]:
        s = self.spread()
        return None if s is None else (s / self.tick)

    def mid(self) -> Optional[float]:
        if not self.asks or not self.bids:
            return None
        return (self.best_bid() + self.best_ask()) / 2.0  # type: ignore

    def depth(
        self, side: str = "both", n_levels: Optional[int] = None
    ) -> Dict[str, List[Tuple[float, int]]]:
        """Return depth ladders (price, size) up to n_levels (None = full)."""
        side = side.lower()
        ladders: Dict[str, List[Tuple[float, int]]] = {"bids": [], "asks": []}
        if side in {"both", "bid", "bids"}:
            take = list(self.bids)[: (n_levels or len(self.bids))]
            ladders["bids"] = [lv.snapshot_tuple() for lv in take]
        if side in {"both", "ask", "asks"}:
            take = list(self.asks)[: (n_levels or len(self.asks))]
            ladders["asks"] = [lv.snapshot_tuple() for lv in take]
        return ladders

    def total_liquidity(self, side: str) -> int:
        side = side.lower()
        if side.startswith("b"):
            return int(sum(lv.total_size for lv in self.bids))
        if side.startswith("a"):
            return int(sum(lv.total_size for lv in self.asks))
        raise ValueError("side must be 'bid(s)' or 'ask(s)'")

    def imbalance(self) -> float:
        """Orderbook imbalance in [−1, +1]: (B − A) / (B + A)."""
        b = self.total_liquidity("bids")
        a = self.total_liquidity("asks")
        denom = max(1, a + b)
        return float((b - a) / denom)

    def snapshot(self) -> Dict[str, object]:
        return {
            "tick": self.tick,
            "min_spread_ticks": self._min_spread_ticks,
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "spread": self.spread(),
            "levels_target": self.levels_target,
            "asks": [lv.snapshot_tuple() for lv in self.asks],
            "bids": [lv.snapshot_tuple() for lv in self.bids],
        }

    def clone(self, copy_rng_state: bool = False) -> "OrderBook":
        ob = OrderBook(
            mid_price=self.mid() or 0.0,
            levels=self.levels_target,
            size_per_level=self.default_size,
            tick=self.tick,
            seed=self._global_seed,
            shard_id=self._shard_id,
            min_spread_ticks=self._min_spread_ticks,
            check_invariants=self._check_invariants,
            latency_fn=self._latency_fn,
            auction_fn=self._auction_fn,
        )
        # Deep copy levels incl. queue
        ob.asks = deque()
        for lv in self.asks:
            nv = LimitLevel(lv.price, lv.public)
            nv.queue = deque(RestingOrder(o.oid, o.qty, o.owner) for o in lv.queue)
            ob.asks.append(nv)
        ob.bids = deque()
        for lv in self.bids:
            nv = LimitLevel(lv.price, lv.public)
            nv.queue = deque(RestingOrder(o.oid, o.qty, o.owner) for o in lv.queue)
            ob.bids.append(nv)
        # OID index copy (structure only)
        ob._next_oid = self._next_oid
        for dq, side in ((ob.asks, "ask"), (ob.bids, "bid")):
            for lv in dq:
                for o in lv.queue:
                    ob._oid_index[o.oid] = (side, lv)
        if copy_rng_state:
            # deterministic fork from current RNG state
            ob.rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
        return ob

    # ---- RNG / Shard mgmt ---------------------------------------------------
    def set_seed(self, seed: Optional[int]) -> None:
        self._global_seed = int(seed) if seed is not None else None
        self.rng = _make_rng(self._global_seed, self._shard_id)

    def set_shard(self, shard_id: Optional[int]) -> None:
        self._shard_id = int(shard_id) if shard_id is not None else None
        self.rng = _make_rng(self._global_seed, self._shard_id)

    # ---- Trading API --------------------------------------------------------
    def _new_oid(self) -> int:
        oid = self._next_oid
        self._next_oid += 1
        return oid

    def _apply_latency(self, report: Dict[str, Any], ctx: Dict[str, Any]) -> None:
        if self._latency_fn is None:
            return
        try:
            lat = self._latency_fn(ctx) or {}
            if isinstance(lat, dict):
                report["latency"] = {**(report.get("latency", {}) or {}), **lat}
        except Exception as e:
            logger.debug("latency hook failed: %s", e)

    def _apply_auction(
        self,
        side: str,
        order_type: str,
        size: int,
        price: Optional[float],
        extra: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self._auction_fn is None:
            return None
        order = {
            "side": side,
            "order_type": order_type,
            "qty": int(abs(size)),
            "limit_price": price,
            **extra,
        }
        try:
            ctx = {
                "bbo": {
                    "bid": self.best_bid(),
                    "ask": self.best_ask(),
                    "spread": self.spread(),
                },
                "tick": self.tick,
            }
            out = self._auction_fn(order, ctx)
            if isinstance(out, dict) and out.get("handled", False):
                rep = dict(out.get("report", {}))
                rep.setdefault("role", "auction")
                # ensure standard keys present
                rep.setdefault("filled_size", int(rep.get("filled_size", 0)))
                rep.setdefault("avg_price", rep.get("avg_price"))
                rep.setdefault("vwap", rep.get("vwap", rep.get("avg_price")))
                rep.setdefault(
                    "signed_slippage", float(rep.get("signed_slippage", 0.0))
                )
                rep.setdefault("slippage_ticks", float(rep.get("slippage_ticks", 0.0)))
                rep.setdefault("unfilled_size", int(rep.get("unfilled_size", 0)))
                rep.setdefault("fills", rep.get("fills", []))
                rep.setdefault("pre_best_bid", self.best_bid())
                rep.setdefault("pre_best_ask", self.best_ask())
                rep.setdefault("post_best_bid", self.best_bid())
                rep.setdefault("post_best_ask", self.best_ask())
                return rep
        except Exception as e:
            logger.debug("auction hook failed: %s", e)
        return None

    def process_market_order(
        self,
        size: int,
        side: str = "buy",
        *,
        limit_price: Optional[float] = None,
        tif: str = "ioc",  # ioc | fok
        **kwargs: Any,  # extra fields (e.g. symbol) forwarded to hooks
    ) -> Dict[str, object]:
        """
        Consume liquidity until size filled, respecting limit_price.
        - tif='fok': if not enough size under limit → no fills & no state change.
        Returns a standard report (see below).
        """
        tif = str(tif).lower()
        if tif not in {"ioc", "fok"}:
            raise ValueError("tif must be 'ioc' or 'fok'")
        if size == 0:
            return {
                "filled_size": 0,
                "avg_price": None,
                "vwap": None,
                "signed_slippage": 0.0,
                "slippage_ticks": 0.0,
                "unfilled_size": 0,
                "fills": [],
                "pre_best_bid": self.best_bid(),
                "pre_best_ask": self.best_ask(),
                "post_best_bid": self.best_bid(),
                "post_best_ask": self.best_ask(),
                "role": "taker",
            }

        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        # Auction hook (can fully handle the order)
        auction_rep = self._apply_auction(side, "market", size, limit_price, kwargs)
        if auction_rep is not None:
            self._apply_latency(
                auction_rep,
                {"side": side, "order_type": "market", "size": size, **kwargs},
            )
            return auction_rep

        pre_bb, pre_ba = self.best_bid(), self.best_ask()
        pre_mid = self.mid()
        if pre_mid is None:
            # empty book – nothing fills
            rep: Dict[str, object] = {
                "filled_size": 0,
                "avg_price": None,
                "vwap": None,
                "signed_slippage": 0.0,
                "slippage_ticks": 0.0,
                "unfilled_size": abs(int(size)),
                "fills": [],
                "pre_best_bid": pre_bb,
                "pre_best_ask": pre_ba,
                "post_best_bid": self.best_bid(),
                "post_best_ask": self.best_ask(),
                "role": "taker",
            }
            self._apply_latency(
                rep, {"side": side, "order_type": "market", "size": size, **kwargs}
            )
            return rep

        remaining = abs(int(size))

        # Pre-check for FOK: enough under limit?
        if tif == "fok":
            can = self._capacity_under_limit(remaining, side, limit_price)
            if can < remaining:
                rep = {
                    "filled_size": 0,
                    "avg_price": None,
                    "vwap": None,
                    "signed_slippage": 0.0,
                    "slippage_ticks": 0.0,
                    "unfilled_size": remaining,
                    "fills": [],
                    "pre_best_bid": pre_bb,
                    "pre_best_ask": pre_ba,
                    "post_best_bid": self.best_bid(),
                    "post_best_ask": self.best_ask(),
                    "role": "taker",
                    "fok_rejected": True,
                }
                self._apply_latency(
                    rep, {"side": side, "order_type": "market", "size": size, **kwargs}
                )
                return rep

        # Execute
        filled = 0
        notional = 0.0
        fills: List[Tuple[float, int]] = []

        if side == "buy":
            while remaining > 0 and self.asks:
                level = self.asks[0]
                if limit_price is not None and level.price > float(limit_price):
                    break
                take = min(remaining, level.total_size)
                if take <= 0:
                    self.asks.popleft()
                    continue
                for px, qty in level.take(take, event_log=self._fill_events):
                    notional += qty * px
                    fills.append((px, qty))
                    filled += qty
                    remaining -= qty
                if level.total_size == 0:
                    self.asks.popleft()
        else:  # sell
            while remaining > 0 and self.bids:
                level = self.bids[0]
                if limit_price is not None and level.price < float(limit_price):
                    break
                take = min(remaining, level.total_size)
                if take <= 0:
                    self.bids.popleft()
                    continue
                for px, qty in level.take(take, event_log=self._fill_events):
                    notional += qty * px
                    fills.append((px, qty))
                    filled += qty
                    remaining -= qty
                if level.total_size == 0:
                    self.bids.popleft()

        avg_price = (notional / filled) if filled > 0 else None
        vwap = avg_price
        signed_slippage = 0.0
        slippage_ticks = 0.0
        if avg_price is not None and pre_mid is not None:
            if side == "buy":
                signed_slippage = float(avg_price - pre_mid)
            else:
                signed_slippage = float(pre_mid - avg_price)
            slippage_ticks = signed_slippage / self.tick

        self._sanitize()
        post_bb, post_ba = self.best_bid(), self.best_ask()

        rep = {
            "filled_size": int(filled),
            "avg_price": avg_price,
            "vwap": vwap,
            "signed_slippage": signed_slippage,
            "slippage_ticks": slippage_ticks,
            "unfilled_size": int(remaining),
            "fills": fills,
            "pre_best_bid": pre_bb,
            "pre_best_ask": pre_ba,
            "post_best_bid": post_bb,
            "post_best_ask": post_ba,
            "role": "taker",
        }
        self._apply_latency(
            rep, {"side": side, "order_type": "market", "size": size, **kwargs}
        )
        return rep

    def process_limit_order(
        self,
        side: str,
        price: float,
        size: int,
        *,
        tif: str = "gtc",  # gtc|ioc|fok
        post_only: bool = False,
        po_action: str = "reject",  # 'reject'| 'slide'
        owner: Optional[str] = None,
        **kwargs: Any,  # extra fields (e.g. symbol) forwarded to hooks
    ) -> Dict[str, object]:
        """
        Place a limit order; if marketable, it executes first (unless post_only).
        - tif='ioc' executes marketable part, discards rest
        - tif='fok' requires full immediate execution else zero & no state change
        - tif='gtc' rests any unfilled size at the given limit level
        - post_only=True: if marketable → 'reject' (no change) or 'slide' (to passive BBO)
        Returns additionally:
          'role': 'maker_posted' (if posted), 'taker' (if crossed), or 'po_rejected' (if rejected)
          'resting_oid' (if resting)
        """
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        tif = str(tif).lower()
        if tif not in {"gtc", "ioc", "fok"}:
            raise ValueError("tif must be 'gtc', 'ioc' or 'fok'")
        po_action = str(po_action).lower()
        if po_action not in {"reject", "slide"}:
            raise ValueError("po_action must be 'reject' or 'slide'")

        price = _round_to_tick(float(price), self.tick)

        # Auction hook (can fully handle the order)
        auction_rep = self._apply_auction(side, "limit", size, price, kwargs)
        if auction_rep is not None:
            self._apply_latency(
                auction_rep,
                {
                    "side": side,
                    "order_type": "limit",
                    "size": size,
                    "price": price,
                    **kwargs,
                },
            )
            return auction_rep

        # marketability test
        if side == "buy":
            mkt_limit = self.best_ask()
            is_marketable = mkt_limit is not None and price >= mkt_limit
        else:
            mkt_limit = self.best_bid()
            is_marketable = mkt_limit is not None and price <= mkt_limit

        if post_only and is_marketable:
            # Post-Only handling
            if po_action == "reject":
                rep: Dict[str, object] = {
                    "filled_size": 0,
                    "avg_price": None,
                    "vwap": None,
                    "signed_slippage": 0.0,
                    "slippage_ticks": 0.0,
                    "unfilled_size": int(size),
                    "fills": [],
                    "pre_best_bid": None,
                    "pre_best_ask": None,
                    "post_best_bid": self.best_bid(),
                    "post_best_ask": self.best_ask(),
                    "role": "po_rejected",
                    "resting_oid": None,
                    "post_only_rejected": True,
                }
                self._apply_latency(
                    rep,
                    {
                        "side": side,
                        "order_type": "limit",
                        "size": size,
                        "price": price,
                        **kwargs,
                    },
                )
                return rep
            # 'slide': move to passive BBO
            if side == "buy":
                safe = self.best_bid()
                if safe is None:
                    safe = price
                # prevent crossing
                while self.best_ask() is not None and safe >= self.best_ask():  # type: ignore
                    safe = _round_to_tick(safe - self.tick, self.tick)
            else:
                safe = self.best_ask()
                if safe is None:
                    safe = price
                while self.best_bid() is not None and safe <= self.best_bid():  # type: ignore
                    safe = _round_to_tick(safe + self.tick, self.tick)
            rep = self._rest_order(side, safe, int(size), owner=owner)
            self._apply_latency(
                rep,
                {
                    "side": side,
                    "order_type": "limit",
                    "size": size,
                    "price": safe,
                    **kwargs,
                },
            )
            return rep

        if not is_marketable:
            # pure resting (if GTC), else no-op report
            if tif == "gtc" and size > 0:
                rep = self._rest_order(side, price, int(size), owner=owner)
                self._apply_latency(
                    rep,
                    {
                        "side": side,
                        "order_type": "limit",
                        "size": size,
                        "price": price,
                        **kwargs,
                    },
                )
                return rep
            rep = {
                "filled_size": 0,
                "avg_price": None,
                "vwap": None,
                "signed_slippage": 0.0,
                "slippage_ticks": 0.0,
                "unfilled_size": int(size),
                "fills": [],
                "pre_best_bid": None,
                "pre_best_ask": None,
                "post_best_bid": self.best_bid(),
                "post_best_ask": self.best_ask(),
                "role": "none",
                "resting_oid": None,
            }
            self._apply_latency(
                rep,
                {
                    "side": side,
                    "order_type": "limit",
                    "size": size,
                    "price": price,
                    **kwargs,
                },
            )
            return rep

        # marketable: execute under limit
        report: Dict[str, object] = self.process_market_order(
            size=int(size),
            side=side,
            limit_price=price,
            tif="fok" if tif == "fok" else "ioc",
            **kwargs,
        )
        # Rest according to TIF possibly rest
        unfilled = int(cast(int | float, report.get("unfilled_size", 0)))
        if tif == "gtc" and unfilled > 0:
            rest_rep = self._rest_order(side, price, unfilled, owner=owner)
            report["unfilled_size"] = 0
            report["role"] = "taker"  # immediate part is taker; rest lies as maker
            report["resting_oid"] = rest_rep.get("resting_oid")
        else:
            report["role"] = "taker"
            report["resting_oid"] = None
        self._apply_latency(
            report,
            {
                "side": side,
                "order_type": "limit",
                "size": size,
                "price": price,
                **kwargs,
            },
        )
        return report

    def cancel_order(self, oid: int, qty: Optional[int] = None) -> int:
        """
        Cancel own resting order partially/completely.
        Returns: actually canceled qty.
        """
        info = self._oid_index.get(int(oid))
        if not info:
            return 0
        side, level = info
        want = int(10**12) if qty is None else int(max(0, qty))
        done = 0
        new_q: deque[RestingOrder] = deque()
        while level.queue:
            o = level.queue.popleft()
            if o.oid == oid and want > 0:
                take = min(o.qty, want)
                o.qty -= take
                want -= take
                done += take
            if o.qty > 0:
                new_q.append(o)
        level.queue = new_q
        if oid in self._oid_index and done > 0 and (qty is None or want == 0):
            if not any(o.oid == oid for o in level.queue):
                self._oid_index.pop(oid, None)
        self._sanitize()
        return done

    def order_position(self, oid: int) -> Optional[Dict[str, int]]:
        """
        Queue position for own order:
          - qty_ahead: public + sum of orders ahead
          - qty_behind: sum of orders behind
          - level_total: total size at the level
        """
        info = self._oid_index.get(int(oid))
        if not info:
            return None
        side, lv = info
        # Defensive: if the OID is no longer in the level queue, drop the stale index.
        if not any(o.oid == oid for o in lv.queue):
            self._oid_index.pop(oid, None)
            return None
        ahead = lv.public
        behind = 0
        seen = False
        for o in lv.queue:
            if o.oid == oid:
                seen = True
            elif not seen:
                ahead += o.qty
            else:
                behind += o.qty
        return {
            "qty_ahead": int(ahead),
            "qty_behind": int(behind),
            "level_total": int(lv.total_size),
        }

    # ---- Fill attribution ---------------------------------------------------
    def collect_fills_for_oid(self, oid: int) -> List[Tuple[float, int]]:
        """
        Return and remove attributed fills for a given OID accumulated since last call.
        Each entry is (price, qty).
        """
        out: List[Tuple[float, int]] = []
        keep: List[Tuple[int, float, int]] = []
        for ev in self._fill_events:
            e_oid, px, q = ev
            if e_oid == oid:
                out.append((float(px), int(q)))
            else:
                keep.append(ev)
        self._fill_events = keep
        return out

    # ---- Simulation step ----------------------------------------------------
    def _ensure_buf(self, need: int) -> None:
        if need <= self._buf_cap:
            return
        # grow by powers of two
        cap = self._buf_cap
        while cap < need:
            cap *= 2
        self._buf_cap = cap
        self._buf_side = np.empty(cap, dtype=np.int8)
        self._buf_size = np.empty(cap, dtype=np.int32)
        self._buf_idx = np.empty(cap, dtype=np.int32)

    def step(
        self,
        lam: float = 2.0,
        max_add: int = 500,
        bias_top: float = 0.7,
        maintain_levels: bool = True,
        *,
        cancel_prob: float = 0.15,
        max_cancel: Optional[int] = None,
        # aggressive market order flow (external takers)
        aggr_prob: float = 0.0,
        aggr_max: int = 0,
    ) -> None:
        """
        Random refill + (optional) cancellations + optional depth maintenance + aggressive IOC flow.
        - Adds ~ Poisson(lam):
            · side∈{bid,ask} uniformly
            · level index geometric bias to top (parameter bias_top)
            · add to PUBLIC liquidity (keeps own queue positions stable)
        - With probability cancel_prob: perform a cancellation event (size∈[1,max_cancel])
          · random side + uniform level; reduces PUBLIC depth only
        - With probability aggr_prob: perform an aggressive IOC market order (size in [1, aggr_max])
          · This consumes top-of-book and can fill our resting orders (maker realism)
        - maintain_levels: ensure at least `levels_target` visible price levels on each side
        """
        if lam < 0:
            lam = 0.0
        adds = int(self.rng.poisson(lam))
        max_add = int(max(2, int(max_add)))
        if not (0.0 <= cancel_prob <= 1.0):
            cancel_prob = 0.0
        max_cancel = int(max_cancel or max_add)
        if not (0.0 <= aggr_prob <= 1.0):
            aggr_prob = 0.0
        aggr_max = int(max(0, aggr_max))

        self._ensure_buf(max(adds, 1))

        def pick_level(side_deque: deque[LimitLevel]) -> int:
            if not side_deque:
                return 0
            n = len(side_deque)
            geom_p = float(np.clip(bias_top, 1e-3, 0.999))
            k = int(self.rng.geometric(geom_p) - 1)  # 0,1,2,...
            return int(min(k, n - 1))

        def pick_level_uniform(side_deque: deque[LimitLevel]) -> int:
            if not side_deque:
                return 0
            return int(self.rng.integers(0, len(side_deque)))

        # Refill events (PUBLIC) - vectorized sampling into reusable buffers
        if adds > 0:
            self._buf_side[:adds] = (self.rng.random(adds) < 0.5).astype(
                np.int8
            )  # 1=bid, 0=ask
            self._buf_size[:adds] = self.rng.integers(
                1, max_add + 1, size=adds, dtype=np.int32
            )
            for i in range(adds):
                side_flag = self._buf_side[i]
                size_i = int(self._buf_size[i])
                if side_flag:  # bid
                    if not self.bids:
                        ref = self.best_ask()
                        price = _round_to_tick((ref or 0.0) - self.tick, self.tick)
                        self.bids.appendleft(LimitLevel(price, size_i))
                    else:
                        idx = pick_level(self.bids)
                        self.bids[idx].add_public(size_i)
                else:  # ask
                    if not self.asks:
                        ref = self.best_bid()
                        price = _round_to_tick((ref or 0.0) + self.tick, self.tick)
                        self.asks.appendleft(LimitLevel(price, size_i))
                    else:
                        idx = pick_level(self.asks)
                        self.asks[idx].add_public(size_i)

        # Cancellation event
        if self.rng.random() < cancel_prob:
            side_flag = int(self.rng.random() < 0.5)  # 1=bid, 0=ask
            dq = int(self.rng.integers(1, max_cancel + 1))
            if side_flag and self.bids:
                idx = pick_level_uniform(self.bids)
                self.bids[idx].cancel_public(dq)
            elif (not side_flag) and self.asks:
                idx = pick_level_uniform(self.asks)
                self.asks[idx].cancel_public(dq)

        # Aggressive IOC market order (external taker)
        if aggr_max > 0 and (self.rng.random() < aggr_prob):
            side = "buy" if self.rng.random() < 0.5 else "sell"
            size = int(self.rng.integers(1, aggr_max + 1))
            # IOC – consume up to aggr_max at top of book (attribution will log OID fills)
            self.process_market_order(size=size, side=side, tif="ioc")

        if maintain_levels:
            self._ensure_depth()

        self._sanitize()

    # ---- External alignment utilities --------------------------------------
    def recenter(self, new_mid: float, preserve_spread: bool = True) -> None:
        """
        Recenter book around a new midpoint (e.g., after external price move).
        If preserve_spread=True, keep current spread; else enforce min_spread_ticks.
        """
        m_old = self.mid()
        if m_old is None:
            return
        delta = float(new_mid) - m_old
        for dq in (self.asks, self.bids):
            for lv in dq:
                lv.price = _round_to_tick(lv.price + delta, self.tick)
        self._sanitize()
        if not preserve_spread:
            self._ensure_min_spread()

    # ---- Internals ----------------------------------------------------------
    def _rest_order(
        self, side: str, price: float, size: int, *, owner: Optional[str] = None
    ) -> Dict[str, object]:
        """Helper: book a GTC order into the level queue."""
        price = _round_to_tick(price, self.tick)
        dq = self.bids if side.startswith("b") else self.asks
        # merge/insert level
        level: Optional[LimitLevel] = None
        for lv in dq:
            if math.isclose(lv.price, price, rel_tol=0, abs_tol=self.tick * 1e-9):
                level = lv
                break
        if level is None:
            level = LimitLevel(price, 0)
            dq.append(level)
            # sort by price
            if side.startswith("b"):
                dq = deque(sorted(dq, key=lambda lv: -lv.price))
                self.bids = dq
            else:
                dq = deque(sorted(dq, key=lambda lv: lv.price))
                self.asks = dq
        # enqueue
        oid = self._new_oid()
        ro = RestingOrder(oid=oid, qty=int(size), owner=owner)
        level.add_queue(ro)
        self._oid_index[oid] = ("bid" if side.startswith("b") else "ask", level)
        self._sanitize()
        return {
            "filled_size": 0,
            "avg_price": None,
            "vwap": None,
            "signed_slippage": 0.0,
            "slippage_ticks": 0.0,
            "unfilled_size": 0,
            "fills": [],
            "pre_best_bid": None,
            "pre_best_ask": None,
            "post_best_bid": self.best_bid(),
            "post_best_ask": self.best_ask(),
            "role": "maker_posted",
            "resting_oid": oid,
            "rest_price": price,
        }

    def _capacity_under_limit(
        self, want: int, side: str, limit_price: Optional[float]
    ) -> int:
        """How much can be filled under (for buys) / over (for sells) the limit."""
        cap = 0
        if side == "buy":
            for lv in self.asks:
                if limit_price is not None and lv.price > float(limit_price):
                    break
                cap += lv.total_size
                if cap >= want:
                    break
        else:
            for lv in self.bids:
                if limit_price is not None and lv.price < float(limit_price):
                    break
                cap += lv.total_size
                if cap >= want:
                    break
        return int(cap)

    def _ensure_depth(self) -> None:
        """Ensure at least `levels_target` price levels on each side; extend ladder if depleted."""
        # extend asks upward
        while len(self.asks) < self.levels_target:
            last = (
                self.asks[-1].price
                if self.asks
                else (self.best_bid() or 0.0) + self.tick
            )
            new_p = _round_to_tick(last + self.tick, self.tick)
            self.asks.append(LimitLevel(new_p, self.default_size))
        # extend bids downward
        while len(self.bids) < self.levels_target:
            last = (
                self.bids[-1].price
                if self.bids
                else (self.best_ask() or 0.0) - self.tick
            )
            new_p = _round_to_tick(last - self.tick, self.tick)
            self.bids.append(LimitLevel(new_p, self.default_size))

    def _ensure_min_spread(self) -> None:
        """Guarantee at least min_spread_ticks between best bid/ask."""
        if not self.asks or not self.bids:
            return
        ba = self.asks[0]
        bb = self.bids[0]
        min_spread = self._min_spread_ticks * self.tick
        spread = float(ba.price - bb.price)
        if spread < float(min_spread) - 1e-12:
            mid = (bb.price + ba.price) / 2.0
            bid_new = _round_to_tick(mid - min_spread / 2.0, self.tick)
            ask_new = _round_to_tick(mid + min_spread / 2.0, self.tick)
            if float(ask_new - bid_new) < float(min_spread) - 1e-12:
                ask_new = _round_to_tick(float(bid_new) + float(min_spread), self.tick)
            bb.price = bid_new
            ba.price = ask_new

    def _sanitize(self) -> None:
        """Remove empty levels; ensure monotone prices; enforce min spread; rebuild indices."""
        # drop zeros at top
        while self.asks and self.asks[0].total_size <= 0:
            self.asks.popleft()
        while self.bids and self.bids[0].total_size <= 0:
            self.bids.popleft()

        # remove empty inside levels
        self.asks = deque([lv for lv in self.asks if lv.total_size > 0])
        self.bids = deque([lv for lv in self.bids if lv.total_size > 0])

        # sort invariants (first pass)
        self.asks = deque(sorted(self.asks, key=lambda lv: lv.price))  # ascending
        self.bids = deque(sorted(self.bids, key=lambda lv: -lv.price))  # descending

        # ensure positive spread
        self._ensure_min_spread()

        # Re-sort after top-of-book price adjustments to preserve order for min_spread_ticks > 1.
        self.asks = deque(sorted(self.asks, key=lambda lv: lv.price))
        self.bids = deque(sorted(self.bids, key=lambda lv: -lv.price))

        # Optionally merge duplicate price levels (coalesce consecutive equal prices).
        def _merge_duplicates(
            dq: deque[LimitLevel], *, ascending: bool
        ) -> deque[LimitLevel]:
            out: deque[LimitLevel] = deque()
            last: Optional[LimitLevel] = None
            for lv in dq:
                if last is not None and math.isclose(
                    last.price, lv.price, rel_tol=0, abs_tol=self.tick * 1e-12
                ):
                    last.public += lv.public
                    last.queue.extend(lv.queue)
                else:
                    out.append(lv)
                    last = out[-1]
            return out

        self.asks = _merge_duplicates(self.asks, ascending=True)
        self.bids = _merge_duplicates(self.bids, ascending=False)

        # Rebuild OID index to avoid stale mappings after fills/cancels outside cancel_order().
        self._oid_index.clear()
        for lv in self.asks:
            for o in lv.queue:
                self._oid_index[o.oid] = ("ask", lv)
        for lv in self.bids:
            for o in lv.queue:
                self._oid_index[o.oid] = ("bid", lv)

        if self._check_invariants:
            self.assert_invariants()

    def assert_invariants(self) -> None:
        """Hard invariant checks (used in tests / diagnostics)."""
        # Size invariants
        for side, dq in (("ask", self.asks), ("bid", self.bids)):
            prev_price = None
            for lv in dq:
                if lv.public < 0:
                    raise AssertionError(f"{side}: negative public size at {lv.price}")
                for o in lv.queue:
                    if o.qty <= 0:
                        raise AssertionError(
                            f"{side}: non-positive queued qty at {lv.price}"
                        )
                if prev_price is not None:
                    if side == "ask" and lv.price < prev_price - (self.tick * 1e-9):
                        raise AssertionError("asks not sorted ascending")
                    if side == "bid" and lv.price > prev_price + (self.tick * 1e-9):
                        raise AssertionError("bids not sorted descending")
                prev_price = lv.price

        # Spread invariants
        if self.asks and self.bids:
            best_ask = float(self.asks[0].price)
            best_bid = float(self.bids[0].price)
            if best_bid >= best_ask - (self.tick * 1e-12):
                raise AssertionError(f"crossed book: bid={best_bid} ask={best_ask}")
            min_spread = float(self._min_spread_ticks) * float(self.tick)
            if (best_ask - best_bid) + (self.tick * 1e-12) < min_spread:
                raise AssertionError("min spread violated")

    # niceties
    def __repr__(self) -> str:
        mid = self.mid()
        return (
            f"OrderBook(mid={mid if mid is not None else float('nan'):.4f} tick={self.tick:g} "
            f"spread={(self.spread() or 0.0):.4f} levels={self.levels_target} "
            f"A0={self.best_ask()} B0={self.best_bid()})"
        )
