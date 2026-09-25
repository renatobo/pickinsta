#!/usr/bin/env python3
"""Deterministic benchmark of production perceptual-hash grouping."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pickinsta.pipeline.deduplication import HashGroupingStats, group_perceptual_hashes

SCHEMA_VERSION = 2
DEFAULT_SIZES = (128, 256, 512, 1024)
HASH_THRESHOLD = 5


@dataclass(frozen=True)
class SyntheticHash:
    """Deterministic 256-bit hash with ImageHash subtraction semantics."""

    value: int

    def __sub__(self, other: object) -> int:
        if not isinstance(other, SyntheticHash):
            return NotImplemented
        return (self.value ^ other.value).bit_count()


def percentile(values: list[float], rank: float) -> float:
    """Return a linearly interpolated percentile."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * rank / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return value ^ (value >> 31)


def generate_unique_hashes(size: int) -> list[SyntheticHash]:
    """Generate a stable corpus of well-distributed unique 256-bit hashes."""
    if size < 1:
        raise ValueError("size must be positive")
    return [
        SyntheticHash(sum(_splitmix64(index * 4 + part) << (part * 64) for part in range(4)))
        for index in range(size)
    ]


def group_reference(
    images: list[Path], hashes: dict[Path, SyntheticHash], threshold: int
) -> tuple[dict[SyntheticHash, list[Path]], int]:
    """Retained linear reference implementing the legacy first-match loop."""
    groups: dict[SyntheticHash, list[Path]] = {}
    comparisons = 0
    for image in images:
        image_hash = hashes[image]
        for representative, group in groups.items():
            comparisons += 1
            if abs(image_hash - representative) <= threshold:
                group.append(image)
                break
        else:
            groups[image_hash] = [image]
    return groups, comparisons


def _measure(size: int) -> tuple[float, float, int, int, int]:
    values = generate_unique_hashes(size)
    images = [Path(f"image-{index:06d}.jpg") for index in range(size)]
    hashes = dict(zip(images, values))

    reference_started = time.perf_counter()
    reference, reference_comparisons = group_reference(images, hashes, HASH_THRESHOLD)
    reference_elapsed = time.perf_counter() - reference_started

    stats = HashGroupingStats()
    indexed_started = time.perf_counter()
    indexed = group_perceptual_hashes(images, hashes, HASH_THRESHOLD, stats=stats)
    indexed_elapsed = time.perf_counter() - indexed_started
    if indexed != reference:
        raise AssertionError("indexed grouping differs from the linear reference")
    return (
        reference_elapsed,
        indexed_elapsed,
        reference_comparisons,
        stats.distance_comparisons,
        len(indexed),
    )


def run_benchmark(*, sizes: tuple[int, ...], repetitions: int = 5) -> dict[str, object]:
    """Compare production metric-index grouping with the retained reference."""
    if repetitions < 3:
        raise ValueError("repetitions must be at least 3")
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("sizes must contain positive integers")

    results = []
    for size in sizes:
        samples = [_measure(size) for _ in range(repetitions)]
        reference_counts = {sample[2] for sample in samples}
        indexed_counts = {sample[3] for sample in samples}
        group_counts = {sample[4] for sample in samples}
        reference_count = reference_counts.pop()
        indexed_count = indexed_counts.pop()
        results.append(
            {
                "image_count": size,
                "reference_samples_seconds": [sample[0] for sample in samples],
                "indexed_samples_seconds": [sample[1] for sample in samples],
                "reference_median_seconds": statistics.median(sample[0] for sample in samples),
                "indexed_median_seconds": statistics.median(sample[1] for sample in samples),
                "reference_comparison_count": reference_count,
                "indexed_comparison_count": indexed_count,
                "comparison_reduction_ratio": 1.0 - indexed_count / reference_count,
                "group_count": group_counts.pop(),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "dedup-perceptual-hash-grouping-scalability",
        "scenario": "all-unique-256-bit-perceptual-hashes",
        "configuration": {
            "sizes": list(sizes),
            "repetitions": repetitions,
            "threshold": HASH_THRESHOLD,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "results": results,
    }


def validate_report(report: dict[str, object]) -> list[str]:
    """Validate equivalence and a conservative operation-count improvement."""
    violations: list[str] = []
    for result in report["results"]:
        size = int(result["image_count"])
        expected = size * (size - 1) // 2
        if int(result["reference_comparison_count"]) != expected:
            violations.append(f"size {size}: reference count is not n(n-1)/2")
        if int(result["group_count"]) != size:
            violations.append(f"size {size}: all-unique corpus did not remain unique")
        if size >= 64 and float(result["comparison_reduction_ratio"]) < 0.5:
            violations.append(f"size {size}: indexed comparison reduction is below 50%")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(sizes=tuple(args.sizes), repetitions=args.repetitions)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    violations = validate_report(report)
    if violations:
        print("\n".join(violations))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
