"""
Universe Runner

Orchestrates a full Universe build:
- Load/validate config
- Build universe (ticker seed + fundamentals + filters + warmup ADV) via `universe.builder`
- Download price/volume history (adjusted + optional unadjusted panels)
- Write artifacts (CSV/manifest) and auditability outputs (optional run-scoped copies)

Design guarantees (by default):
- Selection/methodology is owned by `universe.builder`/`universe.adv`/`universe.filters`.
- This runner is orchestration only: it should not change selection logic.
- All additional metadata is additive (manifest/report) and should not change the universe result.
"""

from __future__ import annotations

import argparse
import _thread
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from universe import downloads as _downloads
from universe.artifact_keys import KEY_TICKERS_EXT_CSV, set_artifact_path
from universe.builder import build_universe
from universe.checkpoint import norm_symbol
from universe.coercion import cfg_int
from universe.config import load_cfg_or_default, validate_cfg
from universe.downloads import derive_date_range as _derive_date_range
from universe.downloads import (
    normalize_panel_for_universe as _normalize_panel_for_universe,
)
from universe.downloads import (
    validate_unadjusted_vs_adjusted as _validate_unadjusted_vs_adjusted,
)
from universe.monitoring import logger, setup_logging, setup_prometheus, stage_timer
from universe.outputs import (
    persist_universe_run_artifacts,
    write_manifest,
    write_universe_csv,
    write_universe_ext_csv,
)
from universe.run_steps import RunnerDeps, download_and_persist_history
from universe.storage import artifact_targets, resolve_artifact_paths
from universe.utils import _atomic_write_pickle, _generate_run_id, _sha1
from universe.vendor import _retry_missing_history, fetch_price_volume_data

SCHEMA_VERSION = "1.6.3"

__all__ = [
    "SCHEMA_VERSION",
    "_build_data_quality_report",
    "_fetch_unadjusted_panels",
    "_normalize_panel_for_universe",
    "_run_main_with_force",
    "_retry_missing_history",
    "artifact_targets",
    "build_universe",
    "fetch_price_volume_data",
    "main",
    "norm_symbol",
]

_INTERRUPT_INFO: dict[str, Any] = {
    "signal_name": None,
    "signal_num": None,
    "received_at": None,
    "console_event_name": None,
    "console_event_num": None,
    "console_received_at": None,
}


def _process_context() -> str:
    try:
        import psutil  # type: ignore

        p = psutil.Process(os.getpid())
        chain: list[str] = []
        cur = p
        for _ in range(6):
            chain.append(f"{cur.pid}:{cur.name()}")
            parent = cur.parent()
            if parent is None:
                break
            cur = parent
        return " <- ".join(chain)
    except Exception:
        return f"pid={os.getpid()} ppid={os.getppid()}"


def _process_context_verbose(*, depth: int = 8) -> str:
    try:
        import psutil  # type: ignore

        p = psutil.Process(os.getpid())
        chain: list[str] = []
        cur = p
        for _ in range(max(1, int(depth))):
            try:
                cmd = " ".join(cur.cmdline())
            except Exception:
                cmd = ""
            cmd_compact = " ".join(cmd.split())
            if len(cmd_compact) > 200:
                cmd_compact = cmd_compact[:197] + "..."
            chain.append(
                f"{cur.pid}:{cur.name()} ppid={cur.ppid()} cmd={cmd_compact!r}"
            )
            parent = cur.parent()
            if parent is None:
                break
            cur = parent
        return " <- ".join(chain)
    except Exception:
        return _process_context()


def _thread_dump(*, stack_limit: int = 12) -> str:
    try:
        frames = sys._current_frames()
    except Exception:
        return "<thread dump unavailable>"
    parts: list[str] = []
    for t in threading.enumerate():
        ident = t.ident
        if ident is None:
            continue
        frame = frames.get(ident)
        if frame is None:
            continue
        try:
            stack = "".join(
                traceback.format_stack(frame, limit=max(1, int(stack_limit)))
            )
        except Exception:
            stack = "<failed to format stack>"
        parts.append(
            f"thread={t.name!r} ident={ident} daemon={bool(t.daemon)}\n{stack}"
        )
    return "\n---\n".join(parts) if parts else "<no thread frames>"


def _install_windows_console_ctrl_tracing() -> dict[str, Any]:
    previous: dict[str, Any] = {}
    if os.name != "nt":
        return previous
    try:
        import ctypes
    except Exception:
        return previous

    ctrl_map = {
        0: "CTRL_C_EVENT",
        1: "CTRL_BREAK_EVENT",
        2: "CTRL_CLOSE_EVENT",
        5: "CTRL_LOGOFF_EVENT",
        6: "CTRL_SHUTDOWN_EVENT",
    }

    handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    def _ctrl_handler(ctrl_type: int) -> bool:
        now_ts = float(time.time())
        evt_num = int(ctrl_type)
        evt_name = ctrl_map.get(evt_num, f"UNKNOWN_CTRL_EVENT_{evt_num}")
        _INTERRUPT_INFO["console_event_name"] = evt_name
        _INTERRUPT_INFO["console_event_num"] = evt_num
        _INTERRUPT_INFO["console_received_at"] = now_ts
        try:
            logger.warning(
                "Windows console control event received: %s (%d). Process context: %s",
                evt_name,
                evt_num,
                _process_context_verbose(),
            )
        except Exception:
            pass
        # Return False so Python/default handlers still receive the event.
        return False

    cb = handler_type(_ctrl_handler)
    try:
        ok = bool(kernel32.SetConsoleCtrlHandler(cb, True))
    except Exception:
        ok = False
    if ok:
        previous["kernel32"] = kernel32
        previous["callback"] = cb
    return previous


def _restore_windows_console_ctrl_tracing(previous: dict[str, Any]) -> None:
    kernel32 = previous.get("kernel32")
    cb = previous.get("callback")
    if kernel32 is None or cb is None:
        return
    try:
        kernel32.SetConsoleCtrlHandler(cb, False)
    except Exception:
        pass


def _log_signal_debug_context_once() -> None:
    env_keys = [
        "VSCODE_PID",
        "VSCODE_CLI",
        "WT_SESSION",
        "TERM_PROGRAM",
        "PROMPT",
        "PYTHONUNBUFFERED",
    ]
    env_excerpt = {k: os.environ.get(k) for k in env_keys if os.environ.get(k)}
    logger.info(
        "Signal debug context: pid=%d ppid=%d exe=%s env=%s chain=%s",
        os.getpid(),
        os.getppid(),
        sys.executable,
        env_excerpt,
        _process_context_verbose(),
    )


def _install_interrupt_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def _on_signal(signum: int, _frame: Any) -> None:
        now_ts = float(time.time())
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = str(signum)
        _INTERRUPT_INFO["signal_name"] = sig_name
        _INTERRUPT_INFO["signal_num"] = int(signum)
        _INTERRUPT_INFO["received_at"] = now_ts
        try:
            stack = "".join(traceback.format_stack(limit=20))
            console_evt = _INTERRUPT_INFO.get("console_event_name")
            console_evt_num = _INTERRUPT_INFO.get("console_event_num")
            logger.warning("Signal context: %s", _process_context())
            logger.warning("Signal context verbose: %s", _process_context_verbose())
            if console_evt:
                logger.warning(
                    "Related console event before signal: %s (%s).",
                    console_evt,
                    console_evt_num,
                )
            logger.warning("Signal handler stack at %s:\n%s", sig_name, stack)
            logger.warning("Thread dump at %s:\n%s", sig_name, _thread_dump())
        except Exception:
            pass
        raise KeyboardInterrupt(f"Received signal {sig_name} ({int(signum)}).")

    for sig_attr in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_attr, None)
        if sig is None:
            continue
        try:
            previous[int(sig)] = signal.getsignal(sig)
            signal.signal(sig, _on_signal)
        except Exception:
            continue
    return previous


def _restore_interrupt_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except Exception:
            continue


def _install_signal_origin_tracing() -> dict[str, Any]:
    previous: dict[str, Any] = {}
    sigint_val = int(getattr(signal, "SIGINT", 2))

    def _log_origin(source: str) -> None:
        stack = "".join(traceback.format_stack(limit=16))
        logger.warning("%s triggered; stack trace follows:\n%s", source, stack)

    try:
        previous["os.kill"] = os.kill

        def _wrapped_os_kill(pid: int, sig: int) -> Any:
            try:
                pid_int = int(pid)
            except Exception:
                pid_int = -1
            try:
                sig_int = int(sig)
            except Exception:
                sig_int = -1
            if sig_int == sigint_val and pid_int in (0, os.getpid()):
                _log_origin("os.kill(..., SIGINT)")
            return previous["os.kill"](pid, sig)

        os.kill = _wrapped_os_kill  # type: ignore[assignment]
    except Exception:
        pass

    if hasattr(signal, "raise_signal"):
        try:
            previous["signal.raise_signal"] = signal.raise_signal

            def _wrapped_raise_signal(sig: int) -> Any:
                try:
                    sig_int = int(sig)
                except Exception:
                    sig_int = -1
                if sig_int == sigint_val:
                    _log_origin("signal.raise_signal(SIGINT)")
                return previous["signal.raise_signal"](sig)

            signal.raise_signal = _wrapped_raise_signal  # type: ignore[assignment]
        except Exception:
            pass

    if hasattr(_thread, "interrupt_main"):
        try:
            previous["_thread.interrupt_main"] = _thread.interrupt_main

            def _wrapped_interrupt_main(signum: int = sigint_val) -> Any:
                try:
                    sig_int = int(signum)
                except Exception:
                    sig_int = sigint_val
                if sig_int == sigint_val:
                    _log_origin("_thread.interrupt_main(SIGINT)")
                try:
                    return previous["_thread.interrupt_main"](signum)
                except TypeError:
                    return previous["_thread.interrupt_main"]()

            _thread.interrupt_main = _wrapped_interrupt_main  # type: ignore[assignment]
        except Exception:
            pass

    return previous


def _restore_signal_origin_tracing(previous: dict[str, Any]) -> None:
    for key, value in previous.items():
        try:
            if key == "os.kill":
                os.kill = value  # type: ignore[assignment]
            elif key == "signal.raise_signal":
                signal.raise_signal = value  # type: ignore[assignment]
            elif key == "_thread.interrupt_main":
                _thread.interrupt_main = value  # type: ignore[assignment]
        except Exception:
            continue


def _build_data_quality_report(
    *,
    prices_unadjusted: pd.DataFrame,
    volumes_reported: pd.DataFrame | None = None,
    max_examples: int = 10,
) -> dict[str, Any]:
    if prices_unadjusted is None or prices_unadjusted.empty:
        base = {"ohlc": {"violations": 0, "examples": []}}
        if volumes_reported is None or volumes_reported.empty:
            return base
        vol_num = volumes_reported.apply(pd.to_numeric, errors="coerce")
        neg = int((vol_num < 0).sum().sum())
        zero = int((vol_num == 0).sum().sum())
        vol_examples_empty_ohlc: list[dict[str, Any]] = []
        if neg > 0:
            neg_locs = (vol_num < 0).stack()
            neg_locs = neg_locs[neg_locs]
            for ts, sym in list(neg_locs.index)[: int(max_examples)]:
                vol_examples_empty_ohlc.append(
                    {
                        "ticker": str(sym),
                        "ts": str(ts),
                        "volume": float(vol_num.loc[ts, sym]),
                    }
                )
        base["volume"] = {
            "negative_values": neg,
            "zero_values": zero,
            "examples": vol_examples_empty_ohlc,
        }
        return base

    tickers: set[str] = set()
    for col in prices_unadjusted.columns:
        name = str(col)
        for suf in ("_open", "_high", "_low", "_close"):
            if name.endswith(suf):
                tickers.add(name[: -len(suf)])
                break

    violations = 0
    examples: list[dict[str, Any]] = []
    for sym in sorted(tickers):
        cols = {k: f"{sym}_{k}" for k in ("open", "high", "low", "close")}
        if any(c not in prices_unadjusted.columns for c in cols.values()):
            continue
        o = pd.to_numeric(prices_unadjusted[cols["open"]], errors="coerce")
        h = pd.to_numeric(prices_unadjusted[cols["high"]], errors="coerce")
        low = pd.to_numeric(prices_unadjusted[cols["low"]], errors="coerce")
        c = pd.to_numeric(prices_unadjusted[cols["close"]], errors="coerce")
        frame = pd.concat({"open": o, "high": h, "low": low, "close": c}, axis=1)
        frame = frame.dropna(how="all")
        if frame.empty:
            continue
        for ts, row in frame.iterrows():
            try:
                oo, hh, ll, cc = (row["open"], row["high"], row["low"], row["close"])
                if pd.isna(oo) or pd.isna(hh) or pd.isna(ll) or pd.isna(cc):
                    continue
                ok = (hh >= max(oo, cc, ll)) and (ll <= min(oo, cc, hh)) and (ll <= hh)
                if not ok:
                    violations += 1
                    if len(examples) < int(max_examples):
                        examples.append(
                            {
                                "ticker": sym,
                                "ts": str(ts),
                                "open": float(oo),
                                "high": float(hh),
                                "low": float(ll),
                                "close": float(cc),
                            }
                        )
            except Exception:
                continue

    report: dict[str, Any] = {
        "ohlc": {"violations": int(violations), "examples": examples}
    }
    if volumes_reported is not None and not volumes_reported.empty:
        vol_num = volumes_reported.apply(pd.to_numeric, errors="coerce")
        neg = int((vol_num < 0).sum().sum())
        zero = int((vol_num == 0).sum().sum())
        vol_examples: list[dict[str, Any]] = []
        if neg > 0:
            neg_locs = (vol_num < 0).stack()
            neg_locs = neg_locs[neg_locs]
            for ts, sym in list(neg_locs.index)[: int(max_examples)]:
                vol_examples.append(
                    {
                        "ticker": str(sym),
                        "ts": str(ts),
                        "volume": float(vol_num.loc[ts, sym]),
                    }
                )
        report["volume"] = {
            "negative_values": neg,
            "zero_values": zero,
            "examples": vol_examples,
        }
    return report


def _fetch_unadjusted_panels(
    tickers: list[str],
    *,
    download_start_date: Any,
    download_end_date: Any,
    download_interval: str,
    batch_size: int,
    pause: float,
    max_retries: int,
    backoff_factor: float,
    use_threads: bool,
    max_download_workers: int | None,
    stop_event: threading.Event | None,
    progress_bar: bool,
    request_timeout: float | None,
    junk_filter: Callable[[str], bool] | None,
    coverage_threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    return _downloads.fetch_unadjusted_panels(
        tickers,
        download_start_date=download_start_date,
        download_end_date=download_end_date,
        download_interval=download_interval,
        batch_size=batch_size,
        pause=pause,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        use_threads=use_threads,
        max_download_workers=max_download_workers,
        stop_event=stop_event,
        progress_bar=progress_bar,
        request_timeout=request_timeout,
        junk_filter=junk_filter,
        coverage_threshold=coverage_threshold,
        fetch_price_volume_data_fn=fetch_price_volume_data,
        retry_missing_history_fn=_retry_missing_history,
        normalize_panel_fn=_normalize_panel_for_universe,
    )


def main(cfg_path: Path | str = Path("runs/configs/config_universe.yaml")) -> None:
    cfg_path = Path(cfg_path)
    cfg_raw = load_cfg_or_default(cfg_path)

    cfg = validate_cfg(cfg_raw)
    setup_logging(cfg.raw)
    setup_prometheus(cfg.raw)

    universe_cfg = cfg.universe
    runtime_cfg = cfg.runtime
    data_cfg = cfg.data
    artifact_paths = resolve_artifact_paths(
        universe_cfg=universe_cfg,
        data_cfg=data_cfg,
        runtime_cfg=runtime_cfg,
    )

    cfg_hash = _sha1(cfg_path) if cfg_path.exists() else None
    run_id = str(
        runtime_cfg.get("run_id_override") or os.getenv("UNIVERSE_RUN_ID") or ""
    ).strip()
    if not run_id:
        run_id = _generate_run_id(cfg_hash)
    logger.info("=== Universe Runner gestartet ===")
    logger.info("Config: %s", str(cfg_path))

    stop_event = threading.Event()

    with stage_timer("build_universe"):
        df_universe, df_fundamentals, monitoring, extra = build_universe(
            cfg.raw, cfg_path, run_id, stop_event=stop_event
        )

    extra_out: dict[str, Any] = dict(extra or {})
    extra_out["run_id"] = run_id
    extra_out.setdefault("cfg_path", str(cfg_path))
    extra_out.setdefault("artifacts", {})

    out_tickers = artifact_paths.output_tickers_csv
    out_tickers_ext = artifact_paths.output_tickers_ext_csv
    out_manifest = artifact_paths.manifest

    adv_window = cfg_int(
        data_cfg, "adv_window", 30, min_value=1, logger=logger, section_name="data"
    )
    tickers_final = [norm_symbol(t) for t in df_universe.index.tolist()]

    deps = RunnerDeps(
        fetch_price_volume_data=fetch_price_volume_data,
        retry_missing_history=_retry_missing_history,
        normalize_panel=_normalize_panel_for_universe,
        fetch_unadjusted_panels=_fetch_unadjusted_panels,
        validate_unadjusted_vs_adjusted=_validate_unadjusted_vs_adjusted,
        build_data_quality_report=_build_data_quality_report,
        write_universe_csv=write_universe_csv,
        write_universe_ext_csv=write_universe_ext_csv,
        artifact_targets=artifact_targets,
        atomic_write_pickle=_atomic_write_pickle,
        norm_symbol=norm_symbol,
    )
    df_universe, df_fundamentals, tickers_final, adv_csv_path, adv_csv_filtered_path = (
        download_and_persist_history(
            deps=deps,
            df_universe=df_universe,
            df_fundamentals=df_fundamentals,
            tickers_final=tickers_final,
            data_cfg=data_cfg,
            universe_cfg=universe_cfg,
            runtime_cfg=runtime_cfg,
            stop_event=stop_event,
            extra_out=extra_out,
            derive_date_range_fn=_derive_date_range,
        )
    )

    write_universe_csv(df_universe, out_tickers)
    if out_tickers_ext is not None:
        write_universe_ext_csv(
            df_universe, df_fundamentals, out_tickers_ext, adv_window=adv_window
        )
        set_artifact_path(
            extra_out["artifacts"], key=KEY_TICKERS_EXT_CSV, path=out_tickers_ext
        )

    n_initial = int(
        extra_out.get("n_tickers_total")
        or len(extra_out.get("tickers_all", []) or [])
        or int(df_universe.shape[0])
    )
    write_manifest(
        manifest_path=out_manifest,
        cfg_path=cfg_path,
        cfg_hash=cfg_hash,
        run_id=run_id,
        n_initial=n_initial,
        n_final=int(df_universe.shape[0]),
        monitoring=monitoring,
        extra=extra_out,
        schema_version=SCHEMA_VERSION,
    )

    persist_universe_run_artifacts(
        cfg_path=cfg_path,
        cfg_hash=cfg_hash,
        run_id=run_id,
        universe_cfg=universe_cfg,
        runtime_cfg=runtime_cfg,
        data_cfg=data_cfg,
        out_tickers=out_tickers,
        out_tickers_ext=out_tickers_ext,
        out_manifest=out_manifest,
        adv_csv=adv_csv_path,
        adv_csv_filtered=adv_csv_filtered_path,
        tickers_final=tickers_final,
        df_fundamentals=df_fundamentals,
        df_universe=df_universe,
        monitoring=monitoring,
        stats=extra_out,
        artifact_paths=artifact_paths,
    )

    logger.info(
        "=== Universe Runner abgeschlossen (Tickers: %d) ===", int(df_universe.shape[0])
    )


def _run_main_with_force(cfg_path: Path | str, *, force: bool) -> None:
    """
    Execute `main` while optionally forcing a rebuild for this process only.

    We use the existing `UNIVERSE_FORCE` switch consumed inside `build_universe`
    so the force flag follows the exact same full pipeline path as a normal run.
    """
    if not force:
        main(cfg_path)
        return

    prev = os.environ.get("UNIVERSE_FORCE")
    os.environ["UNIVERSE_FORCE"] = "1"
    try:
        main(cfg_path)
    finally:
        if prev is None:
            os.environ.pop("UNIVERSE_FORCE", None)
        else:
            os.environ["UNIVERSE_FORCE"] = prev


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Universe Runner")
    ap.add_argument(
        "--cfg", type=Path, default=Path("runs/configs/config_universe.yaml")
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Set runtime.force_rebuild=true for this run.",
    )
    args = ap.parse_args()
    signal_origin_hooks = _install_signal_origin_tracing()
    console_ctrl_hooks = _install_windows_console_ctrl_tracing()
    previous_handlers = _install_interrupt_signal_handlers()
    _log_signal_debug_context_once()

    try:
        _run_main_with_force(args.cfg, force=bool(args.force))
    except KeyboardInterrupt as exc:
        sig_name = _INTERRUPT_INFO.get("signal_name")
        sig_num = _INTERRUPT_INFO.get("signal_num")
        console_evt = _INTERRUPT_INFO.get("console_event_name")
        console_evt_num = _INTERRUPT_INFO.get("console_event_num")
        if sig_name:
            logger.warning(
                "Universe run interrupted by signal %s (%s).", sig_name, sig_num
            )
        else:
            logger.warning("Universe run interrupted.")
        if console_evt:
            logger.warning(
                "Last Windows console event: %s (%s).",
                console_evt,
                console_evt_num,
            )
        detail = str(exc).strip()
        if detail and not detail.startswith("Received signal"):
            logger.warning("Interrupt detail: %s", detail)
        sys.exit(130)
    except Exception as e:
        logger.error("Universe run failed: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        _restore_interrupt_signal_handlers(previous_handlers)
        _restore_signal_origin_tracing(signal_origin_hooks)
        _restore_windows_console_ctrl_tracing(console_ctrl_hooks)
