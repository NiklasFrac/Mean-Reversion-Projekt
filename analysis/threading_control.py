"""Optional thread limiting to avoid BLAS/NumExpr oversubscription."""

from __future__ import annotations

import atexit
import os
from typing import Any, Optional

try:  # pragma: no cover
    from threadpoolctl import ThreadpoolController  # type: ignore[import-untyped]

    try:
        import numexpr as _ne  # type: ignore[import-untyped]  # optional
    except Exception:
        _ne = None
    _TP_CONTROLLER: Any = ThreadpoolController()
    _TP_LIMIT_CTX = None  # persistent entered context
except Exception:  # pragma: no cover
    _TP_CONTROLLER = None
    _ne = None
    _TP_LIMIT_CTX = None


def set_thread_limits(
    blas_threads: Optional[int] = None, numexpr_threads: Optional[int] = None
) -> None:
    """Enforce BLAS/NumExpr thread limits to avoid oversubscription (idempotent)."""
    try:
        if _TP_CONTROLLER and blas_threads is not None and int(blas_threads) > 0:
            global _TP_LIMIT_CTX
            if _TP_LIMIT_CTX is None:
                _TP_LIMIT_CTX = _TP_CONTROLLER.limit(limits=int(blas_threads))
                _TP_LIMIT_CTX.__enter__()
                atexit.register(
                    lambda: _TP_LIMIT_CTX and _TP_LIMIT_CTX.__exit__(None, None, None)
                )
            os.environ["MKL_NUM_THREADS"] = str(int(blas_threads))
            os.environ["OPENBLAS_NUM_THREADS"] = str(int(blas_threads))
            os.environ["OMP_NUM_THREADS"] = str(int(blas_threads))
        else:
            if blas_threads is not None and int(blas_threads) > 0:
                os.environ["MKL_NUM_THREADS"] = str(int(blas_threads))
                os.environ["OPENBLAS_NUM_THREADS"] = str(int(blas_threads))
                os.environ["OMP_NUM_THREADS"] = str(int(blas_threads))
    except Exception:
        if blas_threads is not None and int(blas_threads) > 0:
            os.environ["MKL_NUM_THREADS"] = str(int(blas_threads))
            os.environ["OPENBLAS_NUM_THREADS"] = str(int(blas_threads))
            os.environ["OMP_NUM_THREADS"] = str(int(blas_threads))
    if numexpr_threads is not None and int(numexpr_threads) > 0:
        os.environ["NUMEXPR_MAX_THREADS"] = str(int(numexpr_threads))
        if _ne is None:
            return
        try:
            _ne.set_num_threads(int(numexpr_threads))
        except Exception:
            pass


__all__ = ["set_thread_limits"]
