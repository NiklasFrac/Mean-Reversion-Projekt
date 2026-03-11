from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from universe.adv_outputs import emit_adv_cache_and_filtered_csv


def test_emit_adv_outputs_writes_filtered_csv_and_cache(tmp_path):
    adv_csv = tmp_path / "adv.csv"
    adv_filtered = tmp_path / "adv_filtered.csv"
    adv_cache = tmp_path / "adv_cache.pkl"
    pd.DataFrame(
        {
            "ticker": ["aaa", "BBB"],
            "dollar_adv_hist": [100.0, 200.0],
        }
    ).to_csv(adv_csv, index=False)

    writes: list[Path] = []

    def _atomic(obj: object, path: Path) -> None:
        writes.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    artifacts: dict[str, str] = {}
    out_csv, out_filtered = emit_adv_cache_and_filtered_csv(
        data_cfg={},
        universe_cfg={},
        tickers_final=["AAA"],
        artifacts=artifacts,
        norm_symbol_fn=lambda s: str(s).upper(),
        atomic_write_pickle_fn=_atomic,
        artifact_paths=type(
            "_Paths",
            (),
            {
                "adv_csv": adv_csv,
                "adv_csv_filtered": adv_filtered,
                "adv_cache": adv_cache,
            },
        )(),
    )

    assert out_csv == adv_csv
    assert out_filtered == adv_filtered
    assert adv_filtered.exists()
    assert "adv_csv_filtered" in artifacts
    assert "adv_cache" in artifacts
    cached = pickle.loads(adv_cache.read_bytes())
    assert cached == {"AAA": 100.0}
    assert writes == [adv_cache]


def test_emit_adv_outputs_logs_when_filtered_csv_write_fails(
    tmp_path, monkeypatch, caplog
):
    adv_csv = tmp_path / "adv.csv"
    adv_filtered = tmp_path / "adv_filtered.csv"
    adv_cache = tmp_path / "adv_cache.pkl"
    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "dollar_adv_hist": [123.0],
        }
    ).to_csv(adv_csv, index=False)

    real_to_csv = pd.DataFrame.to_csv

    def _boom(self, path_or_buf=None, *args, **kwargs):  # type: ignore[no-untyped-def]
        if path_or_buf is not None and str(path_or_buf) == str(adv_filtered):
            raise OSError("disk full")
        return real_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", _boom)

    def _atomic(obj: object, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    artifacts: dict[str, str] = {}
    with caplog.at_level("WARNING", logger="runner_universe"):
        out_csv, out_filtered = emit_adv_cache_and_filtered_csv(
            data_cfg={},
            universe_cfg={},
            tickers_final=["AAA"],
            artifacts=artifacts,
            norm_symbol_fn=lambda s: str(s).upper(),
            atomic_write_pickle_fn=_atomic,
            artifact_paths=type(
                "_Paths",
                (),
                {
                    "adv_csv": adv_csv,
                    "adv_csv_filtered": adv_filtered,
                    "adv_cache": adv_cache,
                },
            )(),
        )

    assert out_csv == adv_csv
    assert out_filtered is None
    assert "adv_csv_filtered" not in artifacts
    assert "adv_cache" in artifacts
    assert any(
        "Failed to write filtered ADV CSV" in rec.message for rec in caplog.records
    )
