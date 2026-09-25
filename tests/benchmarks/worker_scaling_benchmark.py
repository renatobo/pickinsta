#!/usr/bin/env python3
"""Offline benchmark for Python-worker and OpenCV native-thread interaction."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from offline_stage_benchmark import generate_corpus

import pickinsta.ig_image_selector as selector


def _score(paths: list[Path], workers: int, cv_threads: int) -> list[dict[str, float]]:
    cv2.setNumThreads(cv_threads)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(selector.score_technical, paths))


def benchmark(
    workspace: Path,
    *,
    image_count: int = 8,
    repetitions: int = 3,
    worker_counts: tuple[int, ...] = (1, 2, 4),
    cv_thread_counts: tuple[int, ...] = (1, 2),
) -> dict[str, object]:
    """Measure bounded configurations and verify that scores do not change."""
    if image_count < 1 or repetitions < 1:
        raise ValueError("image_count and repetitions must be positive")
    paths = generate_corpus(workspace / "source", image_count)
    original_detector = selector._detect_subject_mask
    original_cv_threads = cv2.getNumThreads()
    selector._detect_subject_mask = lambda _image: None
    configurations: list[dict[str, object]] = []
    baseline: list[dict[str, float]] | None = None
    try:
        for workers in worker_counts:
            for cv_threads in cv_thread_counts:
                durations: list[float] = []
                latest: list[dict[str, float]] = []
                for _ in range(repetitions):
                    started = time.perf_counter()
                    latest = _score(paths, min(workers, image_count), cv_threads)
                    durations.append(time.perf_counter() - started)
                if baseline is None:
                    baseline = latest
                configurations.append(
                    {
                        "python_workers": workers,
                        "opencv_threads": cv_threads,
                        "median_seconds": statistics.median(durations),
                        "samples_seconds": durations,
                        "matches_baseline": latest == baseline,
                    }
                )
    finally:
        selector._detect_subject_mask = original_detector
        cv2.setNumThreads(original_cv_threads)
    return {
        "schema_version": 1,
        "benchmark": "offline-worker-scaling",
        "image_count": image_count,
        "repetitions": repetitions,
        "logical_cpu_count": selector.os.cpu_count(),
        "configurations": configurations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("worker-scaling.json"))
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pickinsta-worker-benchmark-") as temporary:
        report = benchmark(Path(temporary), image_count=args.images, repetitions=args.repetitions)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0 if all(c["matches_baseline"] for c in report["configurations"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
