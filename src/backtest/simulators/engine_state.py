from __future__ import annotations

from ..utils.tz import NY_TZ, get_naive_is_utc

# --- Timezone policy (single source of truth) ----------------------------------
# Backtest consumes processing-normalized prices. Default exchange TZ is NY_TZ,
# with WF_NAIVE_IS_UTC only steering legacy naive timestamp interpretation.
_EX_TZ: str = NY_TZ
_NAIVE_IS_UTC: bool = get_naive_is_utc()
