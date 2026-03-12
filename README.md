# Mean-Reversion-Projekt

A reproducible Python research pipeline for **US equity pairs trading**, built around **causal preprocessing**, **statistically disciplined pair selection**, and **walk-forward evaluation**.

The repository is designed as a **quantitative research and engineering project**, not as a production trading system. Its purpose is to support transparent empirical investigation under **public-data constraints**, while keeping universe construction, preprocessing, statistical analysis, and downstream evaluation clearly separated.

## Highlights

- **Four-stage research pipeline:** `universe -> processing -> analysis -> backtest`
- **Causal preprocessing** of public equity data into execution-ready panels
- **Statistically disciplined pair selection** with rolling diagnostics, bootstrap-based evaluation, and FDR-controlled testing
- **Walk-forward backtesting** under explicit execution, borrow, and portfolio-level assumptions
- **Reproducibility** through configuration-driven runs, manifests, pinned dependencies, and stage handoff
- **Engineering quality controls** via tests, linting, formatting, CI, and stage-specific type checking

## Research Scope

This repository studies a constrained **US equity statistical-arbitrage** setting using **public daily data**. The focus is not on maximizing headline backtest metrics, but on building a **transparent, modular, and auditable research stack** in which upstream data construction, downstream evaluation, and implementation assumptions remain explicit.

## Pipeline Overview

```text
Universe -> Processing -> Analysis -> Backtest
```

- **Universe** builds the candidate equity universe and upstream artefacts with provenance.
- **Processing** transforms upstream artefacts into cleaned, execution-ready panels, diagnostics, and ADV outputs.
- **Analysis** runs pair-selection research, rolling diagnostics, and bootstrap/FDR-based statistical evaluation.
- **Backtest** evaluates strategy behaviour, execution assumptions, and walk-forward performance using upstream artefacts.

## Methodological Safeguards

- train-only estimation and split-consistent downstream evaluation
- explicit separation between upstream artefact generation and downstream consumption
- configuration-driven entry points for all operational stages
- reproducibility through manifests, pinned dependencies, and canonical runtime configs
- stage-specific quality gates covering linting, formatting, typing, and tests
- explicit treatment of execution-related assumptions in downstream evaluation

## Repository Structure

- `src/universe/` - universe build, vendor downloads, manifests, and raw upstream outputs
- `src/processing/` - input resolution, cleaning/filling, diagnostics, and ADV generation
- `src/analysis/` - pair-selection analysis, rolling metrics, bootstrap/FDR, and provenance
- `src/backtest/` - downstream evaluation and strategy research
- `src/tests/` - stage-specific test suites
- `runs/configs/` - canonical runtime configurations

## Further Documentation

- **Universe:** `src/universe/README.md`
- **Processing:** `src/processing/README.md`
- **Analysis:** `src/analysis/README.md`
- **Backtest:** `src/backtest/README.md`

## Prerequisites

- Python `3.12` (`>=3.12,<3.14`)
- `uv` installed, for example via:

```bash
python -m pip install --upgrade uv
```

- Run commands from the repository root

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

Optional processing features:

```bash
uv sync --extra processing
```

Full local environment for the complete four-stage research stack:

```bash
uv sync --extra analysis --extra backtest --extra processing
```

## Quick Start

Recommended stage order:

```text
universe -> processing -> analysis -> backtest
```

### Universe

```bash
uv run python -m universe.runner_universe --cfg runs/configs/config_universe.yaml
```

### Processing

```bash
uv run python -m processing.runner_processing --cfg runs/configs/config_processing.yaml
```

### Analysis

```bash
uv run --extra analysis python -m analysis.runner_analysis --cfg runs/configs/config_analysis.yaml
```

### Backtest

```bash
uv run --extra backtest python -m backtest.runner_backtest --cfg runs/configs/config_backtest.yaml
```

## Stage Handoff

The pipeline is designed so that each stage consumes **explicit upstream artefacts** rather than recomputing the full stack implicitly.

- **Universe** produces upstream market artefacts, manifests, and symbol-level inputs.
- **Processing** consumes universe artefacts and emits cleaned execution-side panels, diagnostics, and ADV artefacts.
- **Analysis** consumes processed outputs and writes pair candidates, rolling diagnostics, statistical evaluation, and provenance metadata.
- **Backtest** consumes processed prices plus analysis outputs and writes downstream performance and research outputs.

For audit or reproducibility work, prefer **immutable run-scoped artefacts** when available.

## Quality Gates (Local)

The `mypy` profiles are stage-specific and target the production source packages; stage tests are validated through `pytest`.

### Universe

```bash
uv run ruff check src/universe src/tests/universe
uv run ruff format --check src/universe src/tests/universe
uv run mypy --config-file mypy_universe.ini
uv run pytest -q src/tests/universe
```

### Processing

```bash
uv run ruff check src/processing src/tests/processing
uv run ruff format --check src/processing src/tests/processing
uv run mypy --config-file mypy_processing.ini
uv run pytest -q src/tests/processing
```

### Analysis

```bash
uv run ruff check src/analysis src/tests/analysis
uv run ruff format --check src/analysis src/tests/analysis
uv run --extra analysis mypy --config-file mypy_analysis.ini
uv run --extra analysis pytest -q src/tests/analysis
```

### Backtest

```bash
uv run ruff check src/backtest src/tests/backtest
uv run ruff format --check src/backtest src/tests/backtest
uv run --extra backtest mypy --config-file mypy_backtest.ini
uv run --extra backtest pytest -q src/tests/backtest
```

### Pre-commit / pre-push

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pre-commit run --all-files --hook-stage pre-push
```

## CI Overview

- **universe-ci**: Ruff, format check, Mypy, and Universe tests
- **processing-ci**: core / optional matrix, optional-processing import coverage, Ruff, Mypy, and Processing tests
- **analysis-ci**: frozen `uv` install with analysis extra, dependency import smoke check, Ruff, Mypy, and Analysis tests
- **backtest-ci**: frozen `uv` install with backtest extra, dependency import smoke check, Ruff, Mypy, and Backtest tests

## Reproducibility

- `pyproject.toml` defines Python version bounds, package discovery, extras, and repo-wide tooling defaults
- `uv.lock` pins the dependency set for deterministic local and CI installs
- `.pre-commit-config.yaml` defines local quality hooks
- `mypy_*.ini` provides stage-specific typing profiles
- `runs/configs/*.yaml` contains canonical stage entry-point configurations
- `.github/workflows/*.yml` mirrors the local quality gates in CI

## Limitations

- the repository uses public daily data and therefore operates under data-vendor and frequency constraints
- it is a research pipeline, not a live execution system
- downstream conclusions depend on upstream data quality, filtering choices, and implementation assumptions
- performance outputs should be interpreted in the context of the stated research design

## License

The code in this repository is licensed under the **MIT License**. Data files and other third-party artefacts may be subject to separate upstream terms and are not covered by the MIT License unless explicitly stated.