from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable


def attempt_atomic_replace(
    tmp: Path,
    dst: Path,
    *,
    attempts: int = 8,
    replace_fn: Callable[[str, str], None] = os.replace,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Exception | None:
    last_exc: Exception | None = None
    for i in range(max(1, int(attempts))):
        try:
            replace_fn(str(tmp), str(dst))
            return None
        except PermissionError as exc:
            last_exc = exc
            sleep_fn(0.05 * (2**i))
        except Exception as exc:
            last_exc = exc
            break
    return last_exc


def atomic_replace_or_raise(
    tmp: Path,
    dst: Path,
    *,
    attempts: int = 8,
    replace_fn: Callable[[str, str], None] = os.replace,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    last_exc = attempt_atomic_replace(
        tmp,
        dst,
        attempts=attempts,
        replace_fn=replace_fn,
        sleep_fn=sleep_fn,
    )
    if last_exc is not None:
        raise RuntimeError(
            f"Atomic replace failed for {tmp} -> {dst}: {last_exc}"
        ) from last_exc
