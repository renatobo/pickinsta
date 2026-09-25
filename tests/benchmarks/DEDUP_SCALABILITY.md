# Deduplication scalability benchmark

`dedup_scalability_benchmark.py` exercises the production perceptual-hash
grouping implementation against a retained linear reference. It uses stable,
well-distributed synthetic 256-bit hashes, so it performs no image decoding,
process-pool work, model inference, filesystem timing, or network access.

Run it from the repository root:

```bash
.venv/bin/python tests/benchmarks/dedup_scalability_benchmark.py \
  --sizes 128 256 512 1024 --repetitions 5 \
  --output /tmp/pickinsta-dedup-scalability.json
```

The all-unique scenario is adversarial for the old linear scan: its exact
distance-comparison count is `n(n-1)/2`. The production implementation uses an
exact BK-tree metric index for ImageHash Hamming distance. Its result is checked
against the reference on every repetition before measurements are reported.

The JSON report records raw timing samples, median timings, exact before/after
distance-comparison counts, reduction ratios, group counts, configuration, and
runtime metadata. Use at least three repetitions. Compare wall-clock results
only on similar hardware; deterministic comparison counts are the primary
scalability signal.

Validation requires exact output equivalence and at least a 50% comparison
reduction for corpus sizes of 64 or more. Separate randomized tests cover hash
collisions, thresholds, and cases where multiple representatives match; the
first-created group must still win regardless of metric-tree traversal order.
