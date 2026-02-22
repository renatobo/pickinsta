#!/usr/bin/env python3
"""Manual benchmark: compare Ollama model processing speed on the same image set."""

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


DEFAULT_MODELS = [
    "qwen3-vl:8b",
    "blaifa/InternVL3_5:8b",
    "blaifa/InternVL3_5:4B",
    "openbmb/minicpm-v4.5:8b",
]


@dataclass
class RunMetrics:
    model: str
    run_index: int
    duration_sec: float
    images_count: int
    failed_count: int

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
def patched_env(values: dict[str, str]) -> Iterator[None]:
    prev: dict[str, str | None] = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
        "ollama unavailable",
    ]
    return any(marker in text for marker in markers)


def _run_once(*, candidates: list[ImageScore], src: Path) -> tuple[float, int, int]:
    run_candidates = _clone_candidates(candidates)
    t0 = time.perf_counter()
    ranked = selector.batch_vision_score(
        run_candidates,
        scorer="ollama",
        env_search_dir=src,
    )
    duration = time.perf_counter() - t0
    failed_count = sum(1 for item in ranked if _failed_item(item))
    return duration, len(ranked), failed_count


def _benchmark_model(
    *,
    model: str,
    candidates: list[ImageScore],
    src: Path,
    runs: int,
    yolo_enabled: bool,
    warmup: bool,
) -> list[RunMetrics]:
    metrics: list[RunMetrics] = []
    env = {
        selector.PICKINSTA_OLLAMA_MODEL_ENV_VAR: model,
        selector.OLLAMA_USE_YOLO_ENV_VAR: "true" if yolo_enabled else "false",
    }
    with patched_env(env):
        if warmup:
            # Warm model cache with a single image only (avoid full-pass warmup cost).
            warmup_candidates = candidates[:1]
            if warmup_candidates:
                _run_once(candidates=warmup_candidates, src=src)
                time.sleep(10)
        for run_idx in range(1, runs + 1):
            duration, images_count, failed_count = _run_once(candidates=candidates, src=src)
            metrics.append(
                RunMetrics(
                    model=model,
                    run_index=run_idx,
                    duration_sec=duration,
                    images_count=images_count,
                    failed_count=failed_count,
                )
            )
    return metrics


def _avg(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _write_report(
    *,
    report_path: Path,
    src: Path,
    candidates_count: int,
    runs: int,
    yolo_enabled: bool,
    warmup: bool,
    all_metrics: list[RunMetrics],
) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    base_url = selector.resolve_ollama_base_url(search_dir=src)
    concurrency = selector.resolve_ollama_concurrency()
    retries = selector.resolve_ollama_max_retries()
    keep_alive = selector.resolve_ollama_keep_alive(search_dir=src)

    grouped: dict[str, list[RunMetrics]] = {}
    for metric in all_metrics:
        grouped.setdefault(metric.model, []).append(metric)

    summary_rows: list[tuple[str, float, float, float, float]] = []
    for model, metrics in grouped.items():
        summary_rows.append(
            (
                model,
                _avg([m.sec_per_image for m in metrics]),
                _avg([m.images_per_min for m in metrics]),
                _avg([m.duration_sec for m in metrics]),
                _avg([float(m.failed_count) for m in metrics]),
            )
        )
    summary_rows.sort(key=lambda row: row[1])

    fastest_sec_per_img = summary_rows[0][1] if summary_rows else 0.0

    lines: list[str] = []
    lines.append("# Ollama Model Speed Benchmark Report")
    lines.append("")
    lines.append(f"- Generated: `{ts}`")
    lines.append(f"- Input folder: `{src}`")
    lines.append(f"- Candidates scored per run: `{candidates_count}`")
    lines.append(f"- Runs per model: `{runs}`")
    lines.append(f"- Warmup run before timed runs: `{warmup}`")
    lines.append(f"- YOLO context enabled: `{yolo_enabled}`")
    lines.append(f"- Ollama base URL: `{base_url}`")
    lines.append(f"- Concurrency: `{concurrency}`")
    lines.append(f"- Max retries: `{retries}`")
    lines.append(f"- Keep alive: `{keep_alive}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Avg sec/img | Avg imgs/min | Avg duration (s) | Avg failures/run | Speed vs fastest |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model, sec_img, img_min, duration, failures in summary_rows:
        speed_factor = (sec_img / fastest_sec_per_img) if fastest_sec_per_img > 0 else 0.0
        lines.append(
            f"| {model} | {sec_img:.2f} | {img_min:.2f} | {duration:.2f} | {failures:.2f} | {speed_factor:.2f}x |"
        )

    lines.append("")
    lines.append("## Per-run Details")
    lines.append("")
    lines.append("| Model | Run | Duration (s) | Sec/img | Imgs/min | Failures |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model, metrics in grouped.items():
        for m in sorted(metrics, key=lambda x: x.run_index):
            lines.append(
                f"| {model} | {m.run_index} | {m.duration_sec:.2f} | {m.sec_per_image:.2f} | {m.images_per_min:.2f} | {m.failed_count} |"
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual benchmark to compare Ollama model processing speed on the same image set.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input folder containing original images.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs per model (default: 3).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Model tags to compare (space-separated). "
            "Default: qwen3-vl:8b blaifa/InternVL3_5:8b blaifa/InternVL3_5:4B openbmb/minicpm-v4.5:8b"
        ),
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
        "--yolo",
        action="store_true",
        help="Enable YOLO context during benchmark (default: off for pure model speed comparison).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable per-model warmup (default warmup scores 1 image, then waits 10s).",
    )
    parser.add_argument(
        "--report",
        default="docs/ollama-model-speed-benchmark-report.md",
        help="Output Markdown report path (default: docs/ollama-model-speed-benchmark-report.md).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    models = [m.strip() for m in args.models if m.strip()]
    if not src.exists():
        raise SystemExit(f"Input folder not found: {src}")
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if not models:
        raise SystemExit("No models provided.")

    warmup = not args.no_warmup

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

    print(
        f"🧪 Benchmarking {len(models)} model(s) on {len(candidates)} candidates, {args.runs} run(s) per model"
    )
    all_metrics: list[RunMetrics] = []
    for idx, model in enumerate(models, start=1):
        print(f"➡️  Model {idx}/{len(models)}: {model}")
        metrics = _benchmark_model(
            model=model,
            candidates=candidates,
            src=src,
            runs=args.runs,
            yolo_enabled=args.yolo,
            warmup=warmup,
        )
        all_metrics.extend(metrics)

    _write_report(
        report_path=report_path,
        src=src,
        candidates_count=len(candidates),
        runs=args.runs,
        yolo_enabled=args.yolo,
        warmup=warmup,
        all_metrics=all_metrics,
    )
    print(f"📝 Report written: {report_path}")


if __name__ == "__main__":
    main()
