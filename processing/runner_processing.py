"""
CLI wrapper to run data processing pipeline (OFFLINE).
Usage:
  python -m processing.runner_processing --cfg runs/configs/config_processing.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("runner_processing")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Runner for data processing (offline)")
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("runs/configs/config_processing.yaml"),
        help="Path to config",
    )
    args = parser.parse_args()
    cfg_path = args.cfg
    if not cfg_path.is_file():
        logger.error("Config file not found: %s", cfg_path)
        sys.exit(2)

    try:
        from processing.pipeline import main as pipeline_main
    except Exception as e:  # pragma: no cover - defensive guard
        logger.error("Failed to import pipeline.main: %s", e)
        sys.exit(3)

    logger.info("Starting data processing using config: %s", cfg_path)
    pipeline_main(cfg_path)


if __name__ == "__main__":
    cli()
