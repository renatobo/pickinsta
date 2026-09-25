# Offline stage benchmark

`offline_stage_benchmark.py` measures the local resize and technical-scoring stages without calling CLIP, Ollama, Claude, YOLO downloads, or any network service. It generates the same small corpus on every run.

The production resize and technical-scoring workers are invoked sequentially so scheduling and OS semaphore limits do not distort or prevent measurements. Optional subject-model detection is disabled, preventing model downloads and inference. This benchmark therefore tracks deterministic local computation and cache reuse, not parallel coordinator or model throughput.

Run it from the repository root:

```bash
.venv/bin/python tests/benchmarks/offline_stage_benchmark.py \
  --repetitions 5 --images 4 --output /tmp/pickinsta-benchmark.json
```

Each repetition records three modes:

- `cold`: resize outputs and technical score caches are absent.
- `warm`: resized files are reused, while technical scores are recomputed.
- `cached`: resized files and technical score caches are reused.

The JSON report includes every timing sample, median and interpolated p95 per stage, failure counts, corpus configuration, and Python/platform/package metadata. Compare results only on similar hardware and report cold, warm, and cached measurements separately. Use at least three repetitions; five or more are preferable for performance investigations.

## CI guard

CI runs a small three-repetition version on Python 3.14 only:

```bash
.venv/bin/python tests/benchmarks/offline_stage_benchmark.py \
  --repetitions 3 --images 4 \
  --check --max-cached-to-warm-ratio 0.75 \
  --output /tmp/pickinsta-benchmark.json
```

`--check` verifies portable invariants: every mode completes without failures,
cached mode hits the technical cache for every image, and cold/warm modes do not
claim cache hits. The optional ratio compares cached and warm technical-scoring
medians from the same process. CI uses a conservative `0.75` maximum to catch a
material loss of cache benefit without imposing an absolute wall-clock limit on
different GitHub runners. Omit the ratio when only correctness invariants are
desired.
