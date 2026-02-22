# Test Suite

This folder contains the automated `pytest` suite for `pickinsta`.

## Run Tests

From the project root:

```bash
.venv/bin/pytest -q
```

Run a single file:

```bash
.venv/bin/pytest -q tests/test_scoring_and_reports.py
```

## What Is Covered

- `test_main.py`
  - CLI argument parsing and `run_pipeline(...)` invocation
  - Package version sanity check

- `test_crop.py`
  - Smart crop output dimensions
  - Debug artifact generation
  - Fallback crop behavior when subject detection fails

- `test_cropping_regression.py`
  - Regression guard for crop behavior on real fixtures in `tests/cropping`
  - Fails if generated crops drift too far from saved expected outputs

- `test_full_integration.py`
  - Pipeline orchestration (resize, dedupe, scoring, crop, reports)
  - Output/report file creation
  - Missing input handling

- `test_env_and_cache.py`
  - `.env` parsing
  - Anthropic/HF token resolution behavior
  - Claude model resolution and fallback candidate generation
  - Claude per-image cache read/write validation

- `test_scoring_and_reports.py`
  - Markdown report content/escaping
  - Technical batch ranking behavior
  - CLIP scoring/fallback behavior
  - Claude path validation, including Anthropic API key wiring

## Notes

- Tests are designed to be deterministic and fast.
- External services/models (Anthropic, CLIP downloads, YOLO downloads) are mocked in tests.
- `tests/benchmarks/benchmark_ollama_yolo.py` is a manual benchmark script and is not part of the standard `pytest` runbook.
- `tests/benchmarks/benchmark_ollama_models.py` is a manual benchmark script for cross-model speed comparisons and is not part of the standard `pytest` runbook.
- Manual debug scripts and debug output artifacts are in:
  - `/Users/renatobo/development/pickinsta/debug`
