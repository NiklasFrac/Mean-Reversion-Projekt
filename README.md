# PythonProjekt

This repository contains a reproducible quantitative research pipeline with
three operational stages and one downstream consumer:

- `universe`: builds the investable universe and upstream artefacts with provenance
- `processing`: turns upstream artefacts into execution-ready panels, diagnostics, and ADV outputs
- `analysis`: runs pair-selection research, rolling metrics, and bootstrap/FDR evaluation
- `backtest`: consumes analysis outputs downstream and is documented separately from the core runbook

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
- `runs/configs/` - runtime configurations
- `runs/data/` - generated mutable "latest" outputs and run-scoped artefacts

## Stage Handoff

- `universe` writes raw prices, raw volume, manifests, and optional immutable per-run outputs under `runs/data/by_run/...`
- `processing` consumes universe artefacts and emits processed execution outputs, diagnostics, and ADV artefacts
- `analysis` consumes processed outputs and writes pair candidates, rolling diagnostics, bootstrap/FDR statistics, and provenance metadata
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

Optional processing features (for example calendars and MLflow):

```bash
uv sync --extra processing
```

Full local environment for the core research stack:

```bash
uv sync --extra analysis --extra processing
```

## Run Pipelines

Recommended order: `universe -> processing -> analysis`.
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

Pre-commit / pre-push:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pre-commit run --all-files --hook-stage pre-push
```

## CI Overview

- `universe-ci`: Ruff, format check, Mypy, and Universe tests
- `processing-ci`: `core` / `optional` matrix, optional-processing import coverage, Ruff, Mypy, and Processing tests
- `analysis-ci`: frozen `uv` install with `analysis` extra, dependency import smoke check, Ruff, Mypy, and Analysis tests

## Further Documentation

- Analysis: `src/analysis/README.md`
- Universe: `src/universe/README.md`
- Processing: `src/processing/README.md`
- Backtest: `src/backtest/README.md`
