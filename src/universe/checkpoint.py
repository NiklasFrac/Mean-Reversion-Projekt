from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from universe.fs_atomic import attempt_atomic_replace

__all__ = ["norm_symbol", "Checkpointer"]

logger = logging.getLogger("runner_universe")

EXCHANGE_SUFFIXES_KEEP_DOT = {
    ".AS",
    ".AT",
    ".BR",
    ".BRU",
    ".CO",
    ".DE",
    ".EU",
    ".F",
    ".HE",
    ".HK",
    ".IR",
    ".L",
    ".LS",
    ".MC",
    ".MI",
    ".NL",
    ".OL",
    ".PA",
    ".PL",
    ".SW",
    ".SWX",
    ".T",
    ".TO",
    ".TSX",
    ".V",
    ".VI",
    ".VX",
}


def norm_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    if "/" in s:
        # yfinance expects share-class separators as "-" (e.g., BRK-B). Normalize early
        # so caches/checkpoints and downstream fetches use the vendor-friendly form.
        s = s.replace("/", "-")
    if "." in s and s.count(".") == 1:
        base, suffix = s.split(".", 1)
        if base:
            suffix_token = f".{suffix}"
            suffix_upper = suffix_token.upper()
            if suffix_upper not in EXCHANGE_SUFFIXES_KEEP_DOT:
                if len(suffix) == 1:
                    s = f"{base}-{suffix}"
                else:
                    s = f"{base}-{suffix}"
            else:
                s = f"{base}.{suffix}"
    elif "." in s:
        s = s.replace(".", "-")
    return s


class Checkpointer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {"entries": {}, "symbol_seeds": {}}

    def load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    if "entries" in raw:
                        self._state = {
                            "entries": raw.get("entries", {}),
                            "symbol_seeds": raw.get("symbol_seeds", {}),
                        }
                    elif "done" in raw and isinstance(raw["done"], list):
                        self._state = {
                            "entries": {
                                norm_symbol(sym): {"ts": 0.0}
                                for sym in raw["done"]
                                if isinstance(sym, str)
                            },
                            "symbol_seeds": {},
                        }
                    else:
                        self._state = {"entries": {}, "symbol_seeds": {}}
                elif isinstance(raw, list):
                    self._state = {
                        "entries": {
                            norm_symbol(sym): {"ts": 0.0}
                            for sym in raw
                            if isinstance(sym, str)
                        },
                        "symbol_seeds": {},
                    }
                else:
                    self._state = {"entries": {}, "symbol_seeds": {}}
                self._entries()
                self._symbol_seeds()
        except Exception:
            pass

    def _entries(self) -> dict[str, dict[str, Any]]:
        entries = self._state.get("entries")
        clean: dict[str, dict[str, Any]] = {}
        if isinstance(entries, dict):
            for sym, meta in entries.items():
                if not isinstance(sym, str):
                    continue
                norm = norm_symbol(sym)
                data = meta if isinstance(meta, dict) else {}
                ts = data.get("ts")
                cfg = data.get("cfg_hash")
                meta_clean: dict[str, Any] = {}
                if isinstance(ts, (int, float)):
                    meta_clean["ts"] = float(ts)
                if isinstance(cfg, str) and cfg:
                    meta_clean["cfg_hash"] = cfg
                status = str(data.get("status") or "").strip().lower()
                if status in {"ok", "failed"}:
                    meta_clean["status"] = status
                if not meta_clean:
                    meta_clean = {}
                clean[norm] = meta_clean
        self._state["entries"] = clean
        return clean

    def _symbol_seeds(self) -> dict[str, dict[str, Any]]:
        seeds = self._state.get("symbol_seeds")
        clean: dict[str, dict[str, Any]] = {}
        if isinstance(seeds, dict):
            for key, meta in seeds.items():
                if not isinstance(key, str):
                    continue
                data = meta if isinstance(meta, dict) else {}
                symbols = data.get("symbols")
                if isinstance(symbols, list):
                    collected = []
                    for sym in symbols:
                        norm = norm_symbol(sym)
                        if norm:
                            collected.append(norm)
                else:
                    collected = []
                if not collected:
                    continue
                entry: dict[str, Any] = {"symbols": collected}
                ts = data.get("ts")
                if isinstance(ts, (int, float)):
                    entry["ts"] = float(ts)
                prov = data.get("provenance")
                if isinstance(prov, dict) and prov:
                    entry["provenance"] = prov.copy()
                clean[key] = entry
        self._state["symbol_seeds"] = clean
        return clean

    def done_symbols(self) -> set[str]:
        return set(self._entries().keys())

    def entry(self, sym: str) -> dict[str, Any] | None:
        return self._entries().get(norm_symbol(sym))

    def entries(self) -> dict[str, dict[str, Any]]:
        return {sym: meta.copy() for sym, meta in self._entries().items()}

    @staticmethod
    def _is_entry_done(
        entry: dict[str, Any] | None,
        *,
        cfg_hash: str | None = None,
        max_age: float | None = None,
        now_ts: float | None = None,
    ) -> bool:
        if entry is None:
            return False
        if cfg_hash and entry.get("cfg_hash") not in (None, cfg_hash):
            return False
        if max_age is not None and max_age >= 0:
            ts = float(entry.get("ts") or 0.0)
            now_val = float(now_ts if now_ts is not None else time.time())
            if ts <= 0.0 or now_val - ts > max_age:
                return False
        return True

    def is_done(
        self,
        sym: str,
        *,
        cfg_hash: str | None = None,
        max_age: float | None = None,
    ) -> bool:
        entry = self._entries().get(norm_symbol(sym))
        return self._is_entry_done(entry, cfg_hash=cfg_hash, max_age=max_age)

    def valid_symbols(
        self, *, cfg_hash: str | None = None, max_age: float | None = None
    ) -> set[str]:
        return {
            sym
            for sym in self._entries()
            if self.is_done(sym, cfg_hash=cfg_hash, max_age=max_age)
        }

    def failed_symbols(
        self, *, cfg_hash: str | None = None, max_age: float | None = None
    ) -> set[str]:
        out: set[str] = set()
        now_ts = time.time()
        for sym, meta in self._entries().items():
            if str(meta.get("status") or "").strip().lower() != "failed":
                continue
            if self._is_entry_done(
                meta, cfg_hash=cfg_hash, max_age=max_age, now_ts=now_ts
            ):
                out.add(sym)
        return out

    def purge_invalid(
        self, *, cfg_hash: str | None = None, max_age: float | None = None
    ) -> int:
        with self._lock:
            entries = self._entries()
            now_ts = time.time()
            drop = [
                sym
                for sym, meta in entries.items()
                if not self._is_entry_done(
                    meta, cfg_hash=cfg_hash, max_age=max_age, now_ts=now_ts
                )
            ]
            for sym in drop:
                entries.pop(sym, None)
            if drop:
                self._persist()
            return len(drop)

    def drop_many(self, symbols: Iterable[str]) -> int:
        with self._lock:
            entries = self._entries()
            removed = 0
            for sym in symbols:
                if entries.pop(norm_symbol(sym), None) is not None:
                    removed += 1
            if removed:
                self._persist()
            return removed

    def retain_only(self, symbols: Iterable[str]) -> int:
        keep = {norm_symbol(sym) for sym in symbols}
        with self._lock:
            entries = self._entries()
            drop = [sym for sym in entries if sym not in keep]
            for sym in drop:
                entries.pop(sym, None)
            if drop:
                self._persist()
            return len(drop)

    def mark_done(
        self,
        sym: str,
        *,
        cfg_hash: str | None = None,
        timestamp: float | None = None,
        persist: bool = True,
    ) -> None:
        with self._lock:
            entries = self._entries()
            norm = norm_symbol(sym)
            meta = entries.get(norm, {})
            meta["ts"] = float(timestamp if timestamp is not None else time.time())
            meta["status"] = "ok"
            if cfg_hash:
                meta["cfg_hash"] = cfg_hash
            entries[norm] = meta
            if persist:
                self._persist()

    def mark_done_many(
        self,
        symbols: Iterable[str],
        *,
        cfg_hash: str | None = None,
        timestamps: Mapping[str, float] | None = None,
        default_timestamp: float | None = None,
    ) -> int:
        ts_map: dict[str, float] = {}
        if isinstance(timestamps, Mapping):
            for sym, ts in timestamps.items():
                try:
                    ts_map[norm_symbol(sym)] = float(ts)
                except Exception:
                    continue
        now_ts = float(
            default_timestamp if default_timestamp is not None else time.time()
        )
        updated = 0
        with self._lock:
            entries = self._entries()
            for sym in symbols:
                norm = norm_symbol(sym)
                if not norm:
                    continue
                meta = entries.get(norm, {})
                meta["ts"] = float(ts_map.get(norm, now_ts))
                meta["status"] = "ok"
                if cfg_hash:
                    meta["cfg_hash"] = cfg_hash
                entries[norm] = meta
                updated += 1
            if updated:
                self._persist()
        return updated

    def mark_failed_many(
        self,
        symbols: Iterable[str],
        *,
        cfg_hash: str | None = None,
        default_timestamp: float | None = None,
    ) -> int:
        now_ts = float(
            default_timestamp if default_timestamp is not None else time.time()
        )
        updated = 0
        with self._lock:
            entries = self._entries()
            for sym in symbols:
                norm = norm_symbol(sym)
                if not norm:
                    continue
                meta = entries.get(norm, {})
                meta["ts"] = now_ts
                meta["status"] = "failed"
                if cfg_hash:
                    meta["cfg_hash"] = cfg_hash
                entries[norm] = meta
                updated += 1
            if updated:
                self._persist()
        return updated

    def symbol_seed(
        self,
        *,
        cfg_hash: str | None = None,
        max_age: float | None = None,
    ) -> list[str] | None:
        entry = self.symbol_seed_entry(cfg_hash=cfg_hash, max_age=max_age)
        if not entry:
            return None
        return list(entry.get("symbols", []))

    def symbol_seed_entry(
        self,
        *,
        cfg_hash: str | None = None,
        max_age: float | None = None,
    ) -> dict[str, Any] | None:
        seeds = self._symbol_seeds()
        key = cfg_hash or "__default__"
        entry = seeds.get(key) or seeds.get("__default__")
        if not entry:
            return None
        if max_age is not None and max_age >= 0:
            ts = float(entry.get("ts") or 0.0)
            if ts <= 0.0 or time.time() - ts > max_age:
                return None
        return dict(entry)

    def store_symbol_seed(
        self,
        symbols: Iterable[str],
        *,
        cfg_hash: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        clean = []
        for sym in symbols:
            norm = norm_symbol(sym)
            if norm:
                clean.append(norm)
        if not clean:
            return
        prov = provenance if isinstance(provenance, dict) and provenance else None
        with self._lock:
            seeds = self._symbol_seeds()
            key = cfg_hash or "__default__"
            entry: dict[str, Any] = {"symbols": clean, "ts": float(time.time())}
            if prov is not None:
                entry["provenance"] = dict(prov)
            seeds[key] = entry
            self._state["symbol_seeds"] = seeds
            self._persist()

    def reset(self) -> None:
        with self._lock:
            self._state = {"entries": {}, "symbol_seeds": {}}
            self._persist()

    def _persist(self) -> None:
        entries = self._entries()
        seeds = self._symbol_seeds()
        payload: dict[str, Any] = {
            "entries": {sym: entries[sym] for sym in sorted(entries)}
        }
        if seeds:
            payload["symbol_seeds"] = {
                key: {
                    "symbols": meta.get("symbols", []),
                    **({"ts": float(meta["ts"])} if "ts" in meta else {}),
                    **(
                        {"provenance": meta.get("provenance")}
                        if isinstance(meta.get("provenance"), dict)
                        and meta.get("provenance")
                        else {}
                    ),
                }
                for key, meta in seeds.items()
            }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        import os as _os

        last_exc = attempt_atomic_replace(
            tmp,
            self.path,
            attempts=8,
            replace_fn=_os.replace,
            sleep_fn=time.sleep,
        )
        if last_exc is not None:
            try:
                alt = self.path.with_suffix(
                    self.path.suffix + f".bak_{int(time.time())}"
                )
                _os.replace(str(tmp), str(alt))
                logger.warning(
                    "Checkpoint persist fallback used; target may be locked (%s). "
                    "Saved backup checkpoint to %s%s",
                    self.path,
                    alt,
                    f" (reason: {last_exc})" if last_exc is not None else "",
                )
            except Exception:
                logger.error(
                    "Checkpoint persist failed: could not replace %s and could not write backup.",
                    self.path,
                )
