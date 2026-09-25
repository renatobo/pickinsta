#!/usr/bin/env python3
"""Deterministic, offline benchmark for Pickinsta's local pipeline stages."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

import pickinsta.ig_image_selector as selector

SCHEMA_VERSION = 1
MODES = ("cold", "warm", "cached")


def percentile(values: list[float], percentile_rank: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentile_rank <= 100.0:
        raise ValueError("percentile rank must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_rank / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(durations: list[float]) -> dict[str, float | int]:
    """Summarize repeated timings using stable, JSON-friendly statistics."""
    if not durations:
        raise ValueError("at least one duration is required")
    return {
        "repetitions": len(durations),
        "median_seconds": statistics.median(durations),
        "p95_seconds": percentile(durations, 95.0),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
    }


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def environment_metadata() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": selector.os.cpu_count(),
        "packages": {
            name: _package_version(name)
            for name in ("pickinsta", "Pillow", "numpy", "opencv-python-headless", "ImageHash")
        },
    }


def generate_corpus(folder: Path, image_count: int) -> list[Path]:
    """Create a small deterministic image corpus without network or fixtures."""
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(image_count):
        y, x = np.mgrid[0:480, 0:640]
        pixels = np.stack(
            (
                35 + (x * (index + 1) / 25) % 90,
                55 + (y * (index + 2) / 30) % 100,
                75 + ((x + y) * (index + 1) / 40) % 110,
            ),
            axis=2,
        ).astype(np.uint8)
        image = Image.fromarray(pixels, "RGB")
        path = folder / f"generated_{index:02d}.jpg"
        image.save(path, "JPEG", quality=92)
        paths.append(path)
    return paths


def _timed(operation: Callable[[], object]) -> tuple[object, float]:
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def _remove_technical_caches(images: list[Path]) -> None:
    for image in images:
        selector._tech_cache_path(image).unlink(missing_ok=True)


def resize_stage(source: Path, work: Path) -> tuple[list[Path], dict[Path, Path]]:
    """Run the production resize worker sequentially with its normal reuse rule.

    Keeping orchestration sequential avoids platform semaphore availability from
    becoming a benchmark prerequisite and makes cross-run comparisons less noisy.
    """
    work.mkdir(parents=True, exist_ok=True)
    resized: list[Path] = []
    source_map: dict[Path, Path] = {}
    for image in sorted(source.iterdir()):
        if image.suffix.lower() not in selector.SUPPORTED_EXTENSIONS:
            continue
        destination = work / f"{image.stem}.jpg"
        if destination.exists() and destination.stat().st_mtime >= image.stat().st_mtime:
            result = (destination, image)
        else:
            result = selector._resize_one_image((image, destination))
        if result is not None:
            output, original = result
            resized.append(output)
            source_map[output] = original
    return resized, source_map


def technical_score_stage(
    images: list[Path],
) -> tuple[list[tuple[Path, dict[str, float]]], int]:
    """Score sequentially with cache reuse and optional model detection disabled."""
    results: list[tuple[Path, dict[str, float]]] = []
    cache_hits = 0
    original_detector = selector._detect_subject_mask
    selector._detect_subject_mask = lambda _image: None
    try:
        for image in images:
            scores = selector._load_tech_cache(image)
            if scores is None:
                scores = selector.score_technical(image)
                selector._save_tech_cache(image, scores)
            else:
                cache_hits += 1
            results.append((image, scores))
    finally:
        selector._detect_subject_mask = original_detector
    return results, cache_hits


def run_benchmark(
    *, workspace: Path, repetitions: int = 3, image_count: int = 4
) -> dict[str, object]:
    """Run cold, warm, and cached local-stage measurements."""
    if repetitions < 3:
        raise ValueError("repetitions must be at least 3")
    if image_count < 1:
        raise ValueError("image_count must be positive")

    source = workspace / "source"
    work = workspace / "work"
    generated = generate_corpus(source, image_count)
    # Compute this while all production functions are intact. The score stage
    # temporarily disables optional model detection, and cache writes use this
    # fingerprint to identify the underlying technical-scoring implementation.
    selector._technical_algorithm_fingerprint()
    samples: dict[str, list[dict[str, object]]] = {mode: [] for mode in MODES}

    for repetition in range(1, repetitions + 1):
        shutil.rmtree(work, ignore_errors=True)

        resized_result, resize_cold = _timed(lambda: resize_stage(source, work))
        resized, source_map = resized_result
        _remove_technical_caches(resized)
        scored_result, score_cold = _timed(lambda resized=resized: technical_score_stage(resized))
        scored, cache_hits = scored_result
        samples["cold"].append(
            _sample(
                repetition,
                resize_cold,
                score_cold,
                image_count,
                len(resized),
                len(scored),
                cache_hits,
            )
        )

        _remove_technical_caches(resized)
        resized_result, resize_warm = _timed(lambda: resize_stage(source, work))
        resized, source_map = resized_result
        scored_result, score_warm = _timed(lambda resized=resized: technical_score_stage(resized))
        scored, cache_hits = scored_result
        samples["warm"].append(
            _sample(
                repetition,
                resize_warm,
                score_warm,
                image_count,
                len(resized),
                len(scored),
                cache_hits,
            )
        )

        resized_result, resize_cached = _timed(lambda: resize_stage(source, work))
        resized, source_map = resized_result
        scored_result, score_cached = _timed(lambda resized=resized: technical_score_stage(resized))
        scored, cache_hits = scored_result
        samples["cached"].append(
            _sample(
                repetition,
                resize_cached,
                score_cached,
                image_count,
                len(resized),
                len(scored),
                cache_hits,
            )
        )

    modes = {}
    for mode, mode_samples in samples.items():
        modes[mode] = {
            "samples": mode_samples,
            "summary": {
                stage: summarize([float(sample[stage]) for sample in mode_samples])
                for stage in ("resize_seconds", "technical_score_seconds", "total_seconds")
            },
            "failure_count": sum(int(sample["failure_count"]) for sample in mode_samples),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "offline-local-stages",
        "notes": [
            "Resize and technical scoring workers are measured sequentially to avoid scheduling noise.",
            "Optional subject-model detection is disabled; no model is downloaded or invoked.",
        ],
        "configuration": {
            "repetitions": repetitions,
            "image_count": image_count,
            "modes": list(MODES),
        },
        "environment": environment_metadata(),
        "modes": modes,
        "corpus": {
            "kind": "deterministically-generated",
            "seed_base": 20_260_718,
            "files": [p.name for p in generated],
        },
    }


def _sample(
    repetition: int,
    resize_seconds: float,
    technical_score_seconds: float,
    expected: int,
    resized: int,
    scored: int,
    technical_cache_hits: int,
) -> dict[str, object]:
    return {
        "repetition": repetition,
        "resize_seconds": resize_seconds,
        "technical_score_seconds": technical_score_seconds,
        "total_seconds": resize_seconds + technical_score_seconds,
        "resized_count": resized,
        "scored_count": scored,
        "technical_cache_hits": technical_cache_hits,
        "failure_count": (expected - resized) + (resized - scored),
    }


def validate_report(
    report: dict[str, object],
    *,
    max_cached_to_warm_ratio: float | None = None,
) -> list[str]:
    """Return invariant violations suitable for a portable CI guard.

    Absolute duration limits are deliberately excluded because GitHub runner
    load and hardware vary. The optional ratio compares modes from the same
    process, which is useful for detecting broken cache reuse.
    """
    if max_cached_to_warm_ratio is not None and not 0 < max_cached_to_warm_ratio < 1:
        raise ValueError("max cached-to-warm ratio must be between 0 and 1")

    configuration = report["configuration"]
    modes = report["modes"]
    image_count = int(configuration["image_count"])
    violations: list[str] = []
    for mode in MODES:
        mode_data = modes[mode]
        if int(mode_data["failure_count"]) != 0:
            violations.append(f"{mode} mode reported failures")

    cached_samples = modes["cached"]["samples"]
    if any(int(sample["technical_cache_hits"]) != image_count for sample in cached_samples):
        violations.append("cached mode did not report a technical-cache hit for every image")

    for mode in ("cold", "warm"):
        if any(int(sample["technical_cache_hits"]) != 0 for sample in modes[mode]["samples"]):
            violations.append(f"{mode} mode unexpectedly reported technical-cache hits")

    if max_cached_to_warm_ratio is not None:
        cached_median = float(
            modes["cached"]["summary"]["technical_score_seconds"]["median_seconds"]
        )
        warm_median = float(modes["warm"]["summary"]["technical_score_seconds"]["median_seconds"])
        ratio = cached_median / warm_median if warm_median > 0 else float("inf")
        if ratio > max_cached_to_warm_ratio:
            violations.append(
                "cached technical scoring was not materially faster than warm scoring "
                f"(ratio {ratio:.3f}, maximum {max_cached_to_warm_ratio:.3f})"
            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("offline-stage-benchmark.json"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if correctness and cache-reuse invariants are violated.",
    )
    parser.add_argument(
        "--max-cached-to-warm-ratio",
        type=float,
        help=(
            "With --check, require cached technical scoring to be faster than warm "
            "scoring by this same-run ratio (for example, 0.75)."
        ),
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pickinsta-benchmark-") as temporary:
        report = run_benchmark(
            workspace=Path(temporary), repetitions=args.repetitions, image_count=args.images
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.check:
        violations = validate_report(report, max_cached_to_warm_ratio=args.max_cached_to_warm_ratio)
        if violations:
            for violation in violations:
                print(f"Benchmark guard failed: {violation}", file=sys.stderr)
            return 1
        print("Benchmark guard passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
