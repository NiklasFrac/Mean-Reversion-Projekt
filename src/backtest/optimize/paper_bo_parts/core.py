from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from backtest.utils.tz import utc_now

BayesianOptimization: Any
try:  # pragma: no cover
    from bayes_opt import BayesianOptimization as _BayesianOptimization

    BayesianOptimization = _BayesianOptimization
    _BAYES_OK = True
except Exception:  # pragma: no cover
    BayesianOptimization = None
    _BAYES_OK = False

logger = logging.getLogger("backtest.paper_bo")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    )
    logger.addHandler(h)
logger.setLevel(logging.INFO)

BAD_SCORE: float = -1e6

_TRIALS_FILE = "bo_trials.csv"
_TRIAL_COLUMNS = [
    "timestamp",
    "component",
    "model_id",
    "params_json",
    "metric",
    "score",
    "sharpe",
    "fold",
    "fold_id",
    "is_score",
    "oos_score",
]

_LOCK_TIMEOUT_SEC = 30.0
_LOCK_POLL_SEC = 0.05


def _lock_path_for(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.lock")


@contextmanager
def _file_lock(
    path: Path, *, timeout: float = _LOCK_TIMEOUT_SEC, poll: float = _LOCK_POLL_SEC
) -> Iterator[None]:
    lock_path = _lock_path_for(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")
    acquired = False
    start = time.monotonic()
    try:
        while True:
            try:
                lock_file.seek(0)
                if os.name == "nt":  # Windows
                    import msvcrt  # type: ignore

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl  # type: ignore

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
                acquired = True
                break
            except (OSError, BlockingIOError):
                if timeout is not None and (time.monotonic() - start) >= float(timeout):
                    logger.warning("BO file lock timeout: %s", lock_path)
                    raise TimeoutError(f"BO file lock timeout: {lock_path}") from None
                time.sleep(float(poll))
        yield
    finally:
        try:
            if acquired:
                lock_file.seek(0)
                if os.name == "nt":
                    import msvcrt  # type: ignore

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl  # type: ignore

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        finally:
            lock_file.close()


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _candidate_id(params: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(params), sort_keys=True, default=str).encode("utf-8")
    import hashlib

    return hashlib.md5(payload).hexdigest()[:16]


def _append_trial_row(path: Path, row: Mapping[str, Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(path):
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_TRIAL_COLUMNS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in _TRIAL_COLUMNS})


def _log_trial(
    out_dir: Path,
    *,
    component: str,
    params: Mapping[str, Any],
    score: float | None = None,
    fold_id: int | None = None,
    is_score: float | None = None,
    oos_score: float | None = None,
) -> None:
    try:
        row = {
            "timestamp": utc_now().isoformat(timespec="seconds"),
            "component": component,
            "model_id": _candidate_id(params),
            "params_json": json.dumps(
                dict(params), sort_keys=True, ensure_ascii=False, default=str
            ),
            "metric": "sharpe",
            "score": float(score) if score is not None else None,
            "sharpe": float(score) if score is not None else None,
            "fold": int(fold_id) if fold_id is not None else None,
            "fold_id": int(fold_id) if fold_id is not None else None,
            "is_score": float(is_score) if is_score is not None else None,
            "oos_score": float(oos_score) if oos_score is not None else None,
        }
        _append_trial_row(out_dir / _TRIALS_FILE, row)
    except Exception as exc:
        logger.warning("BO trial log failed: %s", exc, exc_info=True)


def _persist_trials_json(out_dir: Path, stage: str, opt: Any) -> None:
    try:
        data = {
            "res": [
                {"params": r.get("params", {}), "target": r.get("target")}
                for r in getattr(opt, "res", [])
            ]
        }
        path = out_dir / f"bo_{stage}_trials.json"
        payload = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        with _file_lock(path):
            _atomic_write_text(path, payload)
    except Exception as exc:
        logger.warning("BO trials JSON persist failed: %s", exc, exc_info=True)


def _load_trials_json(out_dir: Path, stage: str) -> list[dict[str, Any]]:
    p = out_dir / f"bo_{stage}_trials.json"
    if not p.exists():
        return []
    try:
        with _file_lock(p):
            raw = p.read_text(encoding="utf-8")
        try:
            obj = json.loads(raw)
        except Exception as exc:
            logger.warning("BO trials JSON parse failed: %s (%s)", p, exc)
            return []
        res = obj.get("res", [])
        if not isinstance(res, list):
            logger.warning("BO trials JSON has unexpected shape: %s", p)
            return []
        return res
    except Exception as exc:
        logger.warning("BO trials JSON read failed: %s (%s)", p, exc)
        return []


def _register_previous_trials(
    opt: Any,
    trials: Iterable[Mapping[str, Any]],
    pbounds: Mapping[str, tuple[float, float]],
) -> int:
    added = 0
    for t in trials:
        params_raw = dict(t.get("params", {}))
        target = t.get("target", None)
        try:
            params = {k: float(v) for k, v in params_raw.items()}
        except Exception:
            continue

        out_of_bounds = False
        for k, v in params.items():
            if k not in pbounds:
                out_of_bounds = True
                break
            lo, hi = pbounds[k]
            if not (float(lo) <= float(v) <= float(hi)):
                out_of_bounds = True
                break
        if out_of_bounds:
            continue

        try:
            opt.register(
                params=params, target=float(target) if target is not None else None
            )
            added += 1
        except Exception:
            continue
    return added


def _bayes_optimize(
    *,
    out_dir: Path,
    stage: str,
    pbounds: Mapping[str, tuple[float, float]],
    objective,
    seed: int,
    init_points: int,
    n_iter: int,
    patience: int = 0,
) -> tuple[dict[str, float], float]:
    if not _BAYES_OK or BayesianOptimization is None:
        raise RuntimeError("bayes_opt is not available")
    opt = BayesianOptimization(
        f=objective,
        pbounds=dict(pbounds),
        random_state=int(seed),
        verbose=0,
        allow_duplicate_points=False,
    )

    prev = _load_trials_json(out_dir, stage)
    n_registered = 0
    if prev:
        n_registered = int(_register_previous_trials(opt, prev, pbounds=pbounds))

    budget_total = int(max(0, init_points)) + int(max(0, n_iter))
    remaining = int(max(0, budget_total - n_registered))
    init_to_run = int(min(int(max(0, init_points)), remaining))
    iter_to_run = int(max(0, remaining - init_to_run))

    best_seen = -np.inf
    no_imp = 0
    try:
        if init_to_run > 0:
            opt.maximize(init_points=init_to_run, n_iter=0)
        max_state = opt.max if isinstance(getattr(opt, "max", None), Mapping) else {}
        best_seen = float(max_state.get("target", -np.inf))
        for _ in range(iter_to_run):
            opt.maximize(init_points=0, n_iter=1)
            max_state = (
                opt.max if isinstance(getattr(opt, "max", None), Mapping) else {}
            )
            cur = float(max_state.get("target", -np.inf))
            if patience > 0:
                if cur > best_seen + 1e-12:
                    best_seen = cur
                    no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= patience:
                        break
    except Exception as e:
        logger.warning("BO stage %s failed: %s", stage, e)

    _persist_trials_json(out_dir, stage, opt)
    max_state = opt.max if isinstance(getattr(opt, "max", None), Mapping) else {}
    best_params = dict(max_state.get("params", {}))
    best_score = float(max_state.get("target", BAD_SCORE))
    return best_params, best_score
