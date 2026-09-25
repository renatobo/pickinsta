# Architecture, Reliability, and Performance Action Plan

## Purpose

This plan turns the July 2026 repository assessment into an incremental delivery program.
It prioritizes correctness and reproducibility before modularization or performance tuning.
The intent is to preserve existing pipeline behavior while making changes easier to verify,
operate, and optimize.

## Initial Assessment

`pickinsta` has mature domain logic, deterministic crop fixtures, useful integration tests,
and resilient external-scorer fallbacks. Its main constraint is structural: configuration,
caching, preprocessing, scoring, model integration, cropping, reporting, gallery generation,
recursive execution, and CLI parsing are concentrated in a single module of more than 5,600
lines.

The highest immediate risks are:

1. The supported Python versions disagree between package metadata, CI, and documentation.
2. Technical-score caches are invalidated by image modification time only, so algorithm or
   dependency changes can reuse obsolete scores.
3. Claude cache reads ignore the selected model, so scores can cross model boundaries.
4. Output publication is not transactional and can leave partial or stale artifacts.
5. Broad exception handling can silently degrade a run without a complete failure record.
6. Existing benchmarks do not consistently separate cold, warm, and cached execution.

As of 2026-07-18, the implementation program below is complete. The legacy integration point is a
sub-300-line compatibility facade, the largest production module is an 808-line dependency-free
HTML template, cache/output contracts are versioned and transactional, and the final local suite
passes with 86% branch-aware coverage. The initial risks above are retained as the audit baseline.

## Delivery Principles

- Preserve behavior before reorganizing code.
- Make cache and output correctness explicit before optimizing throughput.
- Move one tested boundary at a time; do not rewrite the pipeline wholesale.
- Measure performance on a fixed corpus before accepting an optimization.
- Keep compatibility imports while modules are extracted.
- Require an acceptance check for every phase.

## Phase 0: Restore Trustworthy Contracts

**Priority:** P0
**Target effort:** 1-2 days

### Actions

- Align supported Python versions across `pyproject.toml`, CI, README, and troubleshooting.
- Test Python 3.11 through 3.14 in CI.
- Align the pre-commit Ruff revision with the declared development dependency.
- Add dependency consistency and package build/install smoke checks to CI.
- Decide and document a reproducible dependency strategy for development and CI.
- Expand Ruff rules incrementally beyond undefined-name checks.

### Acceptance criteria

- Every advertised Python version installs the project and passes lint and tests in CI.
- CI builds an installable package artifact and verifies its installation.
- Pre-commit and project tooling use compatible Ruff versions.
- The dependency reproducibility policy is documented.

## Phase 1: Correctness and Failure Safety

**Priority:** P0-P1
**Target effort:** 2-4 days

### Actions

- Version every cache payload.
- Add an algorithm fingerprint to technical-score cache keys.
- Include scorer, model, prompt, preprocessing settings, and relevant model options in vision
  cache identity.
- Make cross-model cache reuse an explicit opt-in behavior, if retained.
- Write caches atomically through a temporary file and rename.
- Replace silent data-loss paths with structured stage warnings and failure records.
- Distinguish complete, degraded, and failed runs.
- Publish output through a temporary run directory and atomic promotion.

### Acceptance criteria

- Algorithm, model, scorer, or relevant configuration changes cannot reuse incompatible cache
  entries.
- Corrupt cache files safely miss without damaging the next write.
- Interrupted runs cannot appear as complete output directories.
- Reports include processed, skipped, degraded, and failed counts with reasons.

## Phase 2: Modularize Without Behavior Changes

**Priority:** P1
**Target effort:** 1-2 weeks

### Target boundaries

```text
pickinsta/
  cli.py
  config.py
  models.py
  pipeline/
    orchestration.py
    resize.py
    deduplication.py
    technical_scoring.py
    cropping.py
  vision/
    base.py
    clip.py
    claude.py
    ollama.py
  infrastructure/
    cache.py
    model_store.py
    filesystem.py
  reporting/
    markdown.py
    gallery.py
```

### Actions

1. Extract configuration and data models.
2. Extract cache and filesystem services.
3. Put each vision scorer behind a common protocol.
4. Extract deduplication, technical scoring, and cropping.
5. Extract report and gallery generation.
6. Reduce `run_pipeline` to stage coordination.
7. Preserve existing imports until downstream callers are migrated.

### Acceptance criteria

- Existing public behavior and regression fixtures remain unchanged.
- Orchestration modules contain no scoring or rendering algorithms.
- New modules have one clear domain responsibility.
- No production module remains a multi-thousand-line integration point.

## Phase 3: Establish Performance Baselines

**Priority:** P1
**Target effort:** 3-5 days

### Actions

- Create a fixed, versioned representative benchmark corpus.
- Measure cold, warm, and cached runs separately.
- Run at least three repetitions and report median and p95.
- Record CPU, peak memory, package versions, model, image count, cache hits, and failures.
- Record wall time per pipeline stage.
- Add lightweight CI performance guards for non-model stages.

### Acceptance criteria

- Benchmark results are reproducible on the same host and corpus.
- Reports distinguish preprocessing, scoring, cropping, and cache effects.
- Regressions can be attributed to a specific stage.

## Phase 4: Profile-Guided Optimization

**Priority:** P2
**Target effort:** 1-2 weeks after baselines exist

### Candidate improvements

- Compute deduplication features in one image-read pass and one persistent process pool.
- Restrict burst comparisons using timestamp buckets before histogram and ORB matching.
- Batch CLIP preprocessing and inference.
- Persist YOLO detections as pipeline artifacts and reuse them across scoring and cropping.
- Add explicit worker limits and coordinate them with OpenCV native threading.
- Avoid repeatedly hashing the same source during one run.
- Include stage timings and cache-hit ratios in reports.

### Acceptance criteria

- Each optimization demonstrates improvement on the fixed corpus.
- Ranking, crop regression, and failure behavior do not degrade.
- Memory use stays within a documented bound.

## Phase 5: Operational Hardening

**Priority:** P2
**Target effort:** 3-5 days

### Actions

- Replace ad hoc console output with structured logging and a human-friendly renderer.
- Generate a run manifest containing configuration, versions, timings, cache hits, warnings,
  and failures.
- Add cancellation and interruption tests.
- Add disk-full, read-only input, corrupt-cache, and partial-output tests.
- Add an opt-in real-integration suite for model and API compatibility.
- Measure test coverage before defining a coverage threshold.

### Acceptance criteria

- Every run produces enough structured evidence to explain degraded results.
- Expected filesystem and interruption failures have regression coverage.
- External integration drift can be detected without making the normal test suite network-bound.

## Recommended Sequence

1. Fix the delivery contract and cache correctness.
2. Make output publication and failure reporting reliable.
3. Extract stable architectural boundaries.
4. Establish credible performance baselines.
5. Optimize only from measured profiles.
6. Complete operational hardening.

## Progress Log

Update this section as work lands. Link changes to tests or verification evidence.

- [x] Phase 0: supported-runtime contract, packaging checks, and tooling alignment
- [x] Phase 0: incremental Ruff rule expansion
- [x] Phase 1: technical cache identity, Claude model isolation, and atomic cache writes
- [x] Phase 1: complete vision cache identity and schema versioning
- [x] Phase 1: structured failures and transactional output
- [x] Phase 2: domain model, filesystem, and vision-cache boundaries
- [x] Phase 2: configuration, pipeline stages, scorers, reporting, CLI, and compatibility facade
- [x] Phase 3: reproducible offline benchmarks and CI invariants
- [x] Phase 4: indexed perceptual-hash grouping with exact-equivalence benchmarks
- [x] Phase 4: bounded worker/OpenCV policy and evidence-based optimization decisions
- [x] Phase 5: coverage baseline and opt-in real-integration contract
- [x] Phase 5: structured events, manifests, interruption, disk-failure, read-only-input, corrupt-cache, and rollback hardening

## Completion Evidence

- Final local gate: 392 passed, 4 opt-in integrations skipped; 86% branch-aware package coverage.
- Ruff, compileall, architecture boundary guards, and `git diff --check` pass.
- A freshly built wheel installs in an isolated environment, reports version `1.3.0`, and exposes
  `pickinsta = pickinsta.cli:main`.
- Offline cold/warm/cached, dedup scalability, and worker-scaling benchmarks pass their portable
  correctness guards. See [performance-decisions.md](performance-decisions.md) for measured results
  and candidates intentionally deferred until a representative model-enabled corpus exists.
