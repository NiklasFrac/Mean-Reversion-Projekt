# Analysis Pipeline README

## Purpose
- Research-stage pipeline that consumes processed price outputs and produces pair candidates, rolling diagnostics, bootstrap/FDR statistics, and provenance metadata.
- Expected upstream input is the processing stage output, preferably from immutable run-scoped artefacts under `runs/data/by_run/...`.

## Installation
- Base tooling: `uv sync`
- Analysis runtime dependencies: `uv sync --extra analysis`
- If you also need processing optional features locally: `uv sync --extra analysis --extra processing`

## Run Instructions
- CLI: `uv run --extra analysis python -m analysis.runner_analysis --cfg runs/configs/config_analysis.yaml`
- Runner supports overrides such as `--override-prices`, `--override-out`, `--n-jobs`, `--n-resamples`, and `--dry-run`.

## Quality Gates
- Lint: `uv run ruff check src/analysis src/tests/analysis`
- Format check: `uv run ruff format --check src/analysis src/tests/analysis`
- Format write: `uv run ruff format src/analysis src/tests/analysis`
- Type-check: `uv run --extra analysis mypy --config-file mypy_analysis.ini`
- Tests: `uv run --extra analysis pytest -q src/tests/analysis`
- Git hooks: `uv run pre-commit install --hook-type pre-commit --hook-type pre-push`

## Runtime Notes
- Analysis depends on `scipy`, `scikit-learn`, and thread-limiting helpers provided through the `analysis` extra.
- Thread controls are best-effort; if optional acceleration libraries are unavailable, the stage still falls back to environment-level thread caps.
- Test markers `unit` and `integration` are registered in `pyproject.toml` so local pytest runs stay warning-clean.

## Outputs
- Pair candidate files and summary metadata.
- Rolling/bootstrap-derived statistics and diagnostics payloads.
- Provenance information including config hash and library versions used during the run.
