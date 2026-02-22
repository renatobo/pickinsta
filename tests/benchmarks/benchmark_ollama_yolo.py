#!/usr/bin/env python3
"""Benchmark Ollama scoring with and without YOLO context and write a Markdown report."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pickinsta.ig_image_selector as selector
from pickinsta.ig_image_selector import ImageScore


@dataclass
class RunMetrics:
    yolo_enabled: bool
    run_index: int
    duration_sec: float
    images_count: int
    failed_count: int
    mean_final_score: float
    median_final_score: float
    top1_name: str
    top1_score: float

    @property
    def sec_per_image(self) -> float:
        if self.images_count == 0:
            return 0.0
        return self.duration_sec / self.images_count

    @property
    def images_per_min(self) -> float:
        if self.duration_sec <= 0:
            return 0.0
        return self.images_count * 60.0 / self.duration_sec


@contextmanager
def patched_env(key: str, value: str) -> Iterator[None]:
    prev = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _clone_candidates(candidates: list[ImageScore]) -> list[ImageScore]:
    return [
        ImageScore(
            path=item.path,
            source_path=item.source_path,
            technical=dict(item.technical),
        )
        for item in candidates
    ]


def _failed_item(item: ImageScore) -> bool:
    text = (item.one_line or "").lower()
    markers = [
        "vision scoring failed",
        "technical-only",
        "circuit breaker",
    ]
    return any(marker in text for marker in markers)


def _benchmark_mode(
    *,
    candidates: list[ImageScore],
    src: Path,
    runs: int,
    yolo_enabled: bool,
) -> list[RunMetrics]:
    metrics: list[RunMetrics] = []
    env_value = "true" if yolo_enabled else "false"
    with patched_env(selector.OLLAMA_USE_YOLO_ENV_VAR, env_value):
        for run_idx in range(1, runs + 1):
            run_candidates = _clone_candidates(candidates)
            t0 = time.perf_counter()
            ranked = selector.batch_vision_score(
                run_candidates,
                scorer="ollama",
                env_search_dir=src,
            )
            duration = time.perf_counter() - t0
            failed_count = sum(1 for item in ranked if _failed_item(item))
            scores = [item.final_score for item in ranked]
            top1 = ranked[0] if ranked else None
            metrics.append(
                RunMetrics(
                    yolo_enabled=yolo_enabled,
                    run_index=run_idx,
                    duration_sec=duration,
                    images_count=len(ranked),
                    failed_count=failed_count,
                    mean_final_score=statistics.fmean(scores) if scores else 0.0,
                    median_final_score=statistics.median(scores) if scores else 0.0,
                    top1_name=(top1.source_path or top1.path).name if top1 else "",
                    top1_score=top1.final_score if top1 else 0.0,
                )
            )
    return metrics


def _avg(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _summarize(metrics: list[RunMetrics]) -> dict[str, float]:
    return {
        "runs": float(len(metrics)),
        "avg_duration_sec": _avg([m.duration_sec for m in metrics]),
        "avg_sec_per_image": _avg([m.sec_per_image for m in metrics]),
        "avg_images_per_min": _avg([m.images_per_min for m in metrics]),
        "avg_failed_count": _avg([float(m.failed_count) for m in metrics]),
        "avg_mean_score": _avg([m.mean_final_score for m in metrics]),
        "avg_median_score": _avg([m.median_final_score for m in metrics]),
    }


def _write_report(
    *,
    report_path: Path,
    src: Path,
    candidates_count: int,
    runs: int,
    off_metrics: list[RunMetrics],
    on_metrics: list[RunMetrics],
) -> None:
    off = _summarize(off_metrics)
    on = _summarize(on_metrics)
    speedup = 0.0
    if on["avg_sec_per_image"] > 0:
        speedup = on["avg_sec_per_image"] / off["avg_sec_per_image"] if off["avg_sec_per_image"] > 0 else 0.0

    ts = datetime.now().isoformat(timespec="seconds")
    model = selector.resolve_ollama_model(search_dir=src)
    base_url = selector.resolve_ollama_base_url(search_dir=src)
    concurrency = selector.resolve_ollama_concurrency()
    retries = selector.resolve_ollama_max_retries()

    lines: list[str] = []
    lines.append("# Ollama YOLO Benchmark Report")
    lines.append("")
    lines.append(f"- Generated: `{ts}`")
    lines.append(f"- Input folder: `{src}`")
    lines.append(f"- Candidates scored per run: `{candidates_count}`")
    lines.append(f"- Runs per mode: `{runs}`")
    lines.append(f"- Ollama base URL: `{base_url}`")
    lines.append(f"- Ollama model: `{model}`")
    lines.append(f"- Concurrency: `{concurrency}`")
    lines.append(f"- Max retries: `{retries}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Mode | Avg sec/img | Avg imgs/min | Avg duration (s) | Avg failures/run | Avg mean final score | Avg median final score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| YOLO OFF | {off['avg_sec_per_image']:.2f} | {off['avg_images_per_min']:.2f} | {off['avg_duration_sec']:.2f} | {off['avg_failed_count']:.2f} | {off['avg_mean_score']:.4f} | {off['avg_median_score']:.4f} |"
    )
    lines.append(
        f"| YOLO ON | {on['avg_sec_per_image']:.2f} | {on['avg_images_per_min']:.2f} | {on['avg_duration_sec']:.2f} | {on['avg_failed_count']:.2f} | {on['avg_mean_score']:.4f} | {on['avg_median_score']:.4f} |"
    )
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if speedup > 0:
        lines.append(f"- YOLO OFF speed factor vs YOLO ON (sec/img): `{speedup:.2f}x`.")
    lines.append(
        "- Better throughput means lower `sec/img` and higher `imgs/min`."
    )
    lines.append(
        "- Score deltas (`mean/median final score`) show quality impact when enabling/disabling YOLO context."
    )
    lines.append("")
    lines.append("## Per-run Details")
    lines.append("")
    lines.append("| Mode | Run | Duration (s) | Sec/img | Imgs/min | Failures | Top1 | Top1 score |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for m in off_metrics + on_metrics:
        mode = "YOLO ON" if m.yolo_enabled else "YOLO OFF"
        lines.append(
            f"| {mode} | {m.run_index} | {m.duration_sec:.2f} | {m.sec_per_image:.2f} | {m.images_per_min:.2f} | {m.failed_count} | {m.top1_name} | {m.top1_score:.4f} |"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama batch scoring with and without YOLO context.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input folder containing original images.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of benchmark runs per mode (default: 2).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top-N baseline used for candidate cutoff when --all is not set (default: 10).",
    )
    parser.add_argument(
        "--vision-pct",
        type=float,
        default=0.5,
        help="Fraction of technically-scored images sent to vision scoring when --all is not set (default: 0.5).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Score all Stage 2 candidates.",
    )
    parser.add_argument(
        "--report",
        default="docs/ollama-yolo-benchmark-report.md",
        help="Output Markdown report path (default: docs/ollama-yolo-benchmark-report.md).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Input folder not found: {src}")
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    work = src.parent / f"{src.name}_work_bench"
    print(f"📐 Preparing shared Stage 0-2 candidates from: {src}")
    resized, source_map = selector.resize_for_processing(src, work)
    if not resized:
        raise SystemExit("No images found to benchmark.")
    unique = selector.deduplicate(resized)
    scored = selector.batch_technical_score(unique, source_map=source_map)

    if args.all:
        n_candidates = len(scored)
    else:
        n_candidates = max(args.top, int(len(scored) * args.vision_pct))
    candidates = scored[:n_candidates]
    if not candidates:
        raise SystemExit("No candidates selected for benchmark.")

    print(f"🧪 Benchmarking Ollama on {len(candidates)} candidates, {args.runs} run(s) per mode")
    print("➡️  Mode 1/2: YOLO OFF")
    off_metrics = _benchmark_mode(candidates=candidates, src=src, runs=args.runs, yolo_enabled=False)
    print("➡️  Mode 2/2: YOLO ON")
    on_metrics = _benchmark_mode(candidates=candidates, src=src, runs=args.runs, yolo_enabled=True)

    _write_report(
        report_path=report_path,
        src=src,
        candidates_count=len(candidates),
        runs=args.runs,
        off_metrics=off_metrics,
        on_metrics=on_metrics,
    )
    print(f"📝 Report written: {report_path}")


if __name__ == "__main__":
    main()
