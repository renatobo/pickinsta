# Performance Decisions

This record separates measured production changes from ideas that still need a representative
model-enabled corpus. Timing results are host-specific; deterministic equivalence and operation
counts are the portable acceptance signals.

## Accepted changes

### Indexed perceptual-hash grouping

The production grouper now uses an exact BK-tree while preserving the first-created-group rule.
The retained linear implementation is the reference oracle. On the deterministic synthetic corpus,
the index reduced Hamming-distance comparisons by 74.2% at 128 images, 80.7% at 256, and 85.3% at
512. Output equivalence is checked on every benchmark repetition and by randomized threshold and
collision tests.

### Bounded Python and OpenCV workers

Resize, local feature extraction, technical scoring, and crop execution now use explicit worker
caps. OpenCV native threading can be coordinated with those caps to avoid nested oversubscription.
The worker benchmark verifies identical ordered score dictionaries for every tested configuration;
wall-clock measurements remain descriptive because they depend on the host scheduler and CPU.

### Cache and stage observability

Run manifests now record per-stage wall time, invocation counts, cache hits/misses and hit ratios,
runtime/package metadata, issue counts, warnings, and peak resident memory when the platform exposes
it. The offline benchmark separately measures cold, warm, and cached execution and CI checks that
cached technical scoring retains a material benefit without imposing a machine-specific absolute
deadline.

## Evaluated but not yet justified

The following ideas remain deliberately unimplemented:

- batching CLIP inference;
- persisting YOLO detections across scoring and cropping;
- merging all deduplication feature extraction into one long-lived process pool;
- timestamp-bucketing histogram and ORB burst comparisons.

The deterministic offline corpus excludes optional models and does not represent real camera burst
distributions, so it cannot demonstrate that these changes improve end-to-end behavior or memory
use. Implementing them now would add cache invalidation and lifecycle complexity without evidence.
Reconsider them only with a versioned, consented real-image corpus and model-enabled cold/warm
profiles. Require unchanged ranking/crop fixtures, lower median or p95 in the affected stage, and a
documented peak-memory bound before accepting any of them.

## Reproduction

Run the three documented benchmarks from the repository root:

```bash
.venv/bin/python tests/benchmarks/offline_stage_benchmark.py \
  --repetitions 5 --images 4 --output /tmp/pickinsta-benchmark.json
.venv/bin/python tests/benchmarks/dedup_scalability_benchmark.py \
  --sizes 128 256 512 1024 --repetitions 5 \
  --output /tmp/pickinsta-dedup-scalability.json
.venv/bin/python tests/benchmarks/worker_scaling_benchmark.py \
  --images 8 --repetitions 5 --output /tmp/pickinsta-worker-scaling.json
```
