# Mean-Reversion-Projekt

This repository contains a reproducible quantitative research pipeline with
four operational stages:

- `universe`: builds the investable universe and upstream artefacts with provenance
- `processing`: turns upstream artefacts into execution-ready panels, diagnostics, and ADV outputs
- `analysis`: runs pair-selection research, rolling metrics, and bootstrap/FDR evaluation
- `backtest`: evaluates strategy behavior, execution assumptions, and walk-forward performance from analysis outputs

## Prerequisites

- Python `3.12` (`>=3.12,<3.14`)
- `uv` installed, for example via `python -m pip install --upgrade uv`
- Run commands from the repository root

## Project Structure

- `src/universe/` - universe build, vendor downloads, manifests, and raw outputs
- `src/processing/` - input resolution, cleaning/filling, diagnostics, and ADV generation
- `src/analysis/` - pair-selection analysis, rolling metrics, bootstrap/FDR, and provenance
- `src/backtest/` - downstream evaluation and strategy research
- `src/tests/universe/` - universe test suite
- `src/tests/processing/` - processing test suite
- `src/tests/analysis/` - analysis test suite
- `src/tests/backtest/` - backtest test suite
- `runs/configs/` - runtime configurations
- `runs/data/` - generated mutable "latest" outputs and run-scoped artefacts

## Stage Handoff

- `universe` writes raw prices, raw volume, manifests, and optional immutable per-run outputs under `runs/data/by_run/...`
- `processing` consumes universe artefacts and emits processed execution outputs, diagnostics, and ADV artefacts
- `analysis` consumes processed outputs and writes pair candidates, rolling diagnostics, bootstrap/FDR statistics, and provenance metadata
- `backtest` consumes processed prices plus analysis outputs and writes run-scoped reports under `runs/results/performance/BT-*` and BO diagnostics under `runs/results/bo/*` when enabled
- For audit, paper, or reproducibility work, prefer immutable run-scoped artefacts under `runs/data/by_run/...` when available

## Installation

Base environment:

```bash
uv sync
```

Analysis runtime dependencies:

```bash
uv sync --extra analysis
```

Backtest runtime dependencies:

```bash
uv sync --extra backtest
```

Optional processing features (for example calendars and MLflow):

```bash
uv sync --extra processing
```

Full local environment for the full four-stage research stack:

```bash
uv sync --extra analysis --extra backtest --extra processing
```

## Run Pipelines

Recommended order: `universe -> processing -> analysis -> backtest`.
The default entry-point configs live under `runs/configs/`.

Universe:

```bash
uv run python -m universe.runner_universe --cfg runs/configs/config_universe.yaml
```

Processing:

```bash
uv run python -m processing.runner_processing --cfg runs/configs/config_processing.yaml
```

Analysis:

```bash
uv run --extra analysis python -m analysis.runner_analysis --cfg runs/configs/config_analysis.yaml
```

Backtest:

```bash
uv run --extra backtest python -m backtest.runner_backtest --cfg runs/configs/config_backtest.yaml
```

## Quality Gates (Local)

The `mypy` profiles are stage-specific and target the production source
packages; stage tests are validated through `pytest`.

Universe:

```bash
uv run ruff check src/universe src/tests/universe
uv run ruff format --check src/universe src/tests/universe
uv run mypy --config-file mypy_universe.ini
uv run pytest -q src/tests/universe
```

Processing:

```bash
uv run ruff check src/processing src/tests/processing
uv run ruff format --check src/processing src/tests/processing
uv run mypy --config-file mypy_processing.ini
uv run pytest -q src/tests/processing
```

Analysis:

```bash
uv run ruff check src/analysis src/tests/analysis
uv run ruff format --check src/analysis src/tests/analysis
uv run --extra analysis mypy --config-file mypy_analysis.ini
uv run --extra analysis pytest -q src/tests/analysis
```

Backtest:

```bash
uv run ruff check src/backtest src/tests/backtest
uv run ruff format --check src/backtest src/tests/backtest
uv run --extra backtest mypy --config-file mypy_backtest.ini
uv run --extra backtest pytest -q src/tests/backtest
```

Pre-commit / pre-push:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pre-commit run --all-files --hook-stage pre-push
```

## CI Overview

- `universe-ci`: Ruff, format check, Mypy, and Universe tests
- `processing-ci`: `core` / `optional` matrix, optional-processing import coverage, Ruff, Mypy, and Processing tests
- `analysis-ci`: frozen `uv` install with `analysis` extra, dependency import smoke check, Ruff, Mypy, and Analysis tests
- `backtest-ci`: frozen `uv` install with `backtest` extra, dependency import smoke check, Ruff, Mypy, and Backtest tests

## Reproducibility Files

- `pyproject.toml`: Python version, package discovery, extras, and repo-wide `pytest`/`ruff`/`uv` defaults.
- `uv.lock`: frozen dependency set for deterministic local and CI installs via `uv sync --frozen`.
- `.pre-commit-config.yaml`: local `ruff`/`mypy`/`pytest` hooks for all four stages.
- `mypy_*.ini`: stage-specific type-check profiles used locally and in CI.
- `runs/configs/*.yaml`: canonical stage entry-point configs, including `runs/configs/config_backtest.yaml`.
- `.github/workflows/*.yml`: CI source of truth mirroring the local quality gates.

## Further Documentation

- Analysis: `src/analysis/README.md`
- Universe: `src/universe/README.md`
- Processing: `src/processing/README.md`
- Backtest: `src/backtest/README.md`
