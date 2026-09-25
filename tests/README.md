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

Measure branch coverage locally:

```bash
.venv/bin/pytest --cov=pickinsta --cov-report=term-missing
```

CI reports coverage once on the newest supported Python version while running the
standard test suite on every supported version. Coverage is currently a measured
baseline rather than a percentage gate; this avoids rewarding superficial tests
before the large pipeline module is split into testable boundaries.

The initial branch-coverage baseline is 57% (58 tests on Python 3.14). Treat this
as an observation, not a target: new and changed code should be meaningfully tested,
and a failure threshold can be introduced after module boundaries improve.

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
  - `run_manifest.json` complete/degraded status and issue counters
  - Transactional managed-artifact publication, rollback, and user-file preservation
  - Missing input handling

- `test_telemetry.py`
  - Cumulative stage timings and invocation counts, including failed blocks
  - Technical/vision cache hit, miss, and ratio aggregation
  - Runtime/package metadata and cross-platform peak-memory normalization
  - Recursive secret-key redaction and strict JSON-safe manifest serialization

- `test_env_and_cache.py`
  - `.env` parsing
  - Anthropic/HF token resolution behavior
  - Claude model resolution and fallback candidate generation
  - Claude schema-v2 cache identity validation across source, scorer, model, prompt,
    and scoring options
  - Technical cache schema/fingerprint invalidation and atomic cache writes

- `test_scoring_and_reports.py`
  - Markdown report content/escaping
  - Technical batch ranking behavior
  - CLIP scoring/fallback behavior
  - Claude path validation, including Anthropic API key wiring
  - Ollama response parsing, retry/circuit-breaker behavior, and env wiring

## Notes

- Tests are designed to be deterministic and fast.
- External services/models (Anthropic, CLIP downloads, YOLO downloads) are mocked in tests.
- `tests/benchmarks/benchmark_ollama_yolo.py` is a manual benchmark script and is not part of the standard `pytest` runbook.
- `tests/benchmarks/benchmark_ollama_models.py` is a manual benchmark script for cross-model speed comparisons and is not part of the standard `pytest` runbook.
- `tests/benchmarks/dedup_scalability_benchmark.py` is a deterministic manual
  benchmark of production perceptual-hash grouping against the retained linear
  reference. It reports exact before/after comparison counts and median timing
  without a brittle wall-clock threshold; see `tests/benchmarks/DEDUP_SCALABILITY.md`.
- `tests/benchmarks/worker_scaling_benchmark.py` compares bounded Python worker
  pools with OpenCV native-thread counts using deterministic generated images and
  no models; see `tests/benchmarks/WORKER_TUNING.md` for the production configuration
  contract and acceptance tests.
- `tests/benchmarks/offline_stage_benchmark.py` records cold, warm, and cached
  resize/technical-scoring medians and p95 without models or network access; see
  `tests/benchmarks/OFFLINE_BENCHMARK.md`. Its cache checks validate reuse, while its
  timings should only be compared on similar, otherwise-idle hardware.
- Manual debug scripts and debug output artifacts are in:
  - `/Users/renatobo/development/pickinsta/debug`

## Opt-in real integrations

`test_real_integrations.py` contains small compatibility probes for Anthropic,
Ollama, CLIP, and YOLO. They are deliberately excluded from normal behavior by
two opt-ins: the global `PICKINSTA_RUN_REAL_INTEGRATIONS=1` switch and a
component-specific switch. Without both, the probe skips before making a network
request, loading a model, or reading credentials.

Run only these probes with:

```bash
.venv/bin/pytest -m real_integration tests/test_real_integrations.py
```

Component setup:

- Anthropic: set `PICKINSTA_RUN_ANTHROPIC_INTEGRATION=1` and
  `ANTHROPIC_API_KEY`. This makes one paid API request. `ANTHROPIC_MODEL` is
  optional.
- Ollama: set `PICKINSTA_RUN_OLLAMA_INTEGRATION=1` and name an already-installed
  vision model with `PICKINSTA_OLLAMA_MODEL`. The default endpoint is local;
  `PICKINSTA_OLLAMA_BASE_URL` can override it.
- CLIP: set `PICKINSTA_RUN_CLIP_INTEGRATION=1` and
  `PICKINSTA_INTEGRATION_ALLOW_MODEL_DOWNLOADS=1`. The second flag explicitly
  acknowledges that Transformers may download the model on first use.
- YOLO: set `PICKINSTA_RUN_YOLO_INTEGRATION=1` and point
  `PICKINSTA_YOLO_MODEL` to an existing local model file. The probe never uses
  Pickinsta's automatic model-download fallback.

For example, a local YOLO-only check is:

```bash
PICKINSTA_RUN_REAL_INTEGRATIONS=1 \
PICKINSTA_RUN_YOLO_INTEGRATION=1 \
PICKINSTA_YOLO_MODEL=/absolute/path/to/model.pt \
.venv/bin/pytest -m yolo_integration tests/test_real_integrations.py
```

Do not enable these probes in the standard CI matrix. A dedicated manual job
should inject credentials from its secret store and select only the intended
marker; never pass secrets as command-line arguments or print the environment.
