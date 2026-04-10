#!/usr/bin/env python3
"""Manual benchmark: compare model speed and quality on the same image set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pickinsta.ig_image_selector as selector
from pickinsta.ig_image_selector import ImageScore


DEFAULT_OLLAMA_MODELS = [
    "qwen3-vl:8b",
    "blaifa/InternVL3_5:8b",
    "blaifa/InternVL3_5:4B",
    "openbmb/minicpm-v4.5:8b",
]
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"


@dataclass
class BenchmarkVariant:
    label: str
    scorer: str  # ollama | claude
    model: str
    yolo_enabled: bool
    prompt_variant: str = "default"


@dataclass
class RunMetrics:
    variant_label: str
    scorer: str
    model: str
    yolo_enabled: bool
    run_index: int
    duration_sec: float
    images_count: int
    failed_count: int
    ranked_rows: list[dict]

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


def _image_set_hash(candidates: list[ImageScore]) -> str:
    entries = sorted(
        (str(p := item.source_path or item.path), p.stat().st_size)
        for item in candidates
    )
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()[:16]


@dataclass
class BenchmarkStore:
    """Persistent JSON store of RunMetrics keyed on variant config + image set."""

    path: Path

    def _variant_key(self, variant: BenchmarkVariant) -> str:
        return f"{variant.scorer}|{variant.model}|{variant.yolo_enabled}|{variant.prompt_variant}"

    def _load_raw(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def get_cached_metrics(self, variant: BenchmarkVariant, image_set_hash: str) -> list[RunMetrics] | None:
        entry = self._load_raw().get(self._variant_key(variant))
        if entry and entry.get("image_set_hash") == image_set_hash:
            return [RunMetrics(**r) for r in entry["runs"]]
        return None

    def store_metrics(self, variant: BenchmarkVariant, metrics: list[RunMetrics], image_set_hash: str) -> None:
        data = self._load_raw()
        data[self._variant_key(variant)] = {
            "variant": asdict(variant),
            "runs": [asdict(m) for m in metrics],
            "image_set_hash": image_set_hash,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _failed_item(item: ImageScore) -> bool:
    text = (item.one_line or "").lower()
    markers = [
        "vision scoring failed",
        "technical-only",
        "circuit breaker",
        "ollama unavailable",
        "claude unavailable",
    ]
    return any(marker in text for marker in markers)


def _md_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


_VISION_KEYS = ("subject_clarity", "lighting", "color_pop", "emotion", "scroll_stop", "crop_4x5")


def _compute_sdi(ranked_rows: list[dict]) -> float:
    """Score Differentiation Index: mean stdev of 6 criteria across images."""
    stdevs: list[float] = []
    for row in ranked_rows:
        scores = [row.get(k) for k in _VISION_KEYS]
        nums = [float(s) for s in scores if s != "" and s is not None]
        if len(nums) >= 2:
            stdevs.append(statistics.stdev(nums))
    return statistics.fmean(stdevs) if stdevs else 0.0


def _count_unique_tuples(ranked_rows: list[dict]) -> int:
    """Count distinct (s1,s2,s3,s4,s5,s6) score combos across images."""
    tuples = set()
    for row in ranked_rows:
        scores = tuple(row.get(k, "") for k in _VISION_KEYS)
        if any(s != "" and s is not None for s in scores):
            tuples.add(scores)
    return len(tuples)


def _ranked_rows(ranked: list[ImageScore]) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(ranked, start=1):
        vision = item.vision if isinstance(item.vision, dict) else {}
        src_name = (item.source_path or item.path).name
        rows.append(
            {
                "rank": idx,
                "filename": src_name,
                "final_score": float(item.final_score),
                "technical": float(item.technical.get("composite", 0.0)),
                "subject_clarity": vision.get("subject_clarity", ""),
                "lighting": vision.get("lighting", ""),
                "color_pop": vision.get("color_pop", ""),
                "emotion": vision.get("emotion", ""),
                "scroll_stop": vision.get("scroll_stop", ""),
                "crop_4x5": vision.get("crop_4x5", ""),
                "vision_total": vision.get("total", ""),
                "one_line": item.one_line or "",
                "failed": _failed_item(item),
            }
        )
    return rows


def _run_once(*, candidates: list[ImageScore], src: Path, scorer: str) -> tuple[float, list[dict], int]:
    run_candidates = _clone_candidates(candidates)
    t0 = time.perf_counter()
    ranked = selector.batch_vision_score(
        run_candidates,
        scorer=scorer,
        env_search_dir=src,
    )
    duration = time.perf_counter() - t0
    failed_count = sum(1 for item in ranked if _failed_item(item))
    return duration, _ranked_rows(ranked), failed_count


def _benchmark_variant(
    *,
    variant: BenchmarkVariant,
    candidates: list[ImageScore],
    src: Path,
    runs: int,
    warmup: bool,
) -> list[RunMetrics]:
    metrics: list[RunMetrics] = []
    env = {}
    if variant.scorer == "ollama":
        env[selector.PICKINSTA_OLLAMA_MODEL_ENV_VAR] = variant.model
        env[selector.OLLAMA_USE_YOLO_ENV_VAR] = "true" if variant.yolo_enabled else "false"
        if variant.prompt_variant != "default":
            env[selector.OLLAMA_PROMPT_VARIANT_ENV_VAR] = variant.prompt_variant
    elif variant.scorer == "claude":
        env["ANTHROPIC_MODEL"] = variant.model

    with patched_env(env):
        if warmup:
            warmup_candidates = candidates[:1]
            if warmup_candidates:
                _run_once(candidates=warmup_candidates, src=src, scorer=variant.scorer)
                time.sleep(10)
        for run_idx in range(1, runs + 1):
            duration, ranked_rows, failed_count = _run_once(
                candidates=candidates,
                src=src,
                scorer=variant.scorer,
            )
            metrics.append(
                RunMetrics(
                    variant_label=variant.label,
                    scorer=variant.scorer,
                    model=variant.model,
                    yolo_enabled=variant.yolo_enabled,
                    run_index=run_idx,
                    duration_sec=duration,
                    images_count=len(ranked_rows),
                    failed_count=failed_count,
                    ranked_rows=ranked_rows,
                )
            )
    return metrics


def _avg(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _first_run_by_variant(metrics: list[RunMetrics]) -> dict[str, RunMetrics]:
    by_variant: dict[str, RunMetrics] = {}
    for m in metrics:
        if m.variant_label not in by_variant or m.run_index < by_variant[m.variant_label].run_index:
            by_variant[m.variant_label] = m
    return by_variant


def _write_report(
    *,
    report_path: Path,
    src: Path,
    candidates_count: int,
    runs: int,
    warmup: bool,
    variants: list[BenchmarkVariant],
    all_metrics: list[RunMetrics],
) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    base_url = selector.resolve_ollama_base_url(search_dir=src)
    concurrency = selector.resolve_ollama_concurrency()
    retries = selector.resolve_ollama_max_retries()
    keep_alive = selector.resolve_ollama_keep_alive(search_dir=src)

    grouped: dict[str, list[RunMetrics]] = {}
    for metric in all_metrics:
        grouped.setdefault(metric.variant_label, []).append(metric)

    first_runs = _first_run_by_variant(all_metrics)

    summary_rows: list[tuple[str, str, str, str, float, float, float, float, float, int]] = []
    for variant in variants:
        metrics = grouped.get(variant.label, [])
        if not metrics:
            continue
        avg_sec = _avg([m.sec_per_image for m in metrics])
        first_run = first_runs.get(variant.label)
        sdi = _compute_sdi(first_run.ranked_rows) if first_run else 0.0
        unique = _count_unique_tuples(first_run.ranked_rows) if first_run else 0
        summary_rows.append(
            (
                variant.label,
                variant.scorer,
                variant.model,
                "on" if variant.yolo_enabled else "off",
                avg_sec,
                _avg([m.images_per_min for m in metrics]),
                _avg([m.duration_sec for m in metrics]),
                _avg([float(m.failed_count) for m in metrics]),
                sdi,
                unique,
            )
        )
    summary_rows.sort(key=lambda row: row[4])
    fastest_sec_per_img = summary_rows[0][4] if summary_rows else 0.0

    lines: list[str] = []
    lines.append("# Model Benchmark Report (Speed + Quality)")
    lines.append("")
    lines.append(f"- Generated: `{ts}`")
    lines.append(f"- Input folder: `{src}`")
    lines.append(f"- Candidates scored per run: `{candidates_count}`")
    lines.append(f"- Runs per variant: `{runs}`")
    lines.append(f"- Warmup enabled: `{warmup}` (1 image + 10s wait)")
    lines.append(f"- Ollama base URL: `{base_url}`")
    lines.append(f"- Ollama concurrency: `{concurrency}`")
    lines.append(f"- Ollama max retries: `{retries}`")
    lines.append(f"- Ollama keep_alive: `{keep_alive}`")
    lines.append("")
    lines.append("## Speed Summary")
    lines.append("")
    lines.append("| Variant | Scorer | Model | YOLO | Avg sec/img | Avg imgs/min | Avg duration (s) | Avg failures/run | SDI | Unique tuples | Speed vs fastest |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, scorer_name, model, yolo, sec_img, img_min, duration, failures, sdi, unique in summary_rows:
        speed_factor = (sec_img / fastest_sec_per_img) if fastest_sec_per_img > 0 else 0.0
        lines.append(
            f"| {_md_escape(label)} | {scorer_name} | {_md_escape(model)} | {yolo} | "
            f"{sec_img:.2f} | {img_min:.2f} | {duration:.2f} | {failures:.2f} | {sdi:.2f} | {unique} | {speed_factor:.2f}x |"
        )

    lines.append("")
    lines.append("## Per-run Timing")
    lines.append("")
    lines.append("| Variant | Run | Duration (s) | Sec/img | Imgs/min | Failures |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for variant in variants:
        for m in sorted(grouped.get(variant.label, []), key=lambda x: x.run_index):
            lines.append(
                f"| {_md_escape(variant.label)} | {m.run_index} | {m.duration_sec:.2f} | "
                f"{m.sec_per_image:.2f} | {m.images_per_min:.2f} | {m.failed_count} |"
            )

    if first_runs:
        lines.append("")
        lines.append("## Image-by-Image Score Comparison (Run 1)")
        lines.append("")
        variant_labels = [v.label for v in variants if v.label in first_runs]
        images = sorted({row["filename"] for m in first_runs.values() for row in m.ranked_rows})
        header = "| Image |" + "".join([f" {_md_escape(v)} final | {_md_escape(v)} rank |" for v in variant_labels])
        sep = "|---|" + "".join(["---:|---:|" for _ in variant_labels])
        lines.append(header)
        lines.append(sep)
        for image in images:
            line = f"| {_md_escape(image)} |"
            for v in variant_labels:
                rows = first_runs[v].ranked_rows
                row = next((r for r in rows if r["filename"] == image), None)
                if row is None:
                    line += "  |  |"
                else:
                    line += f" {row['final_score']:.4f} | {row['rank']} |"
            lines.append(line)

    lines.append("")
    lines.append("## Ranked Quality Details (All Images)")
    lines.append("")
    lines.append("_Each section below is from timed run 1 for that variant._")
    for variant in variants:
        run = first_runs.get(variant.label)
        if run is None:
            continue
        lines.append("")
        lines.append(f"### {variant.label}")
        lines.append("")
        lines.append("| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for row in run.ranked_rows:
            lines.append(
                f"| {row['rank']} | {_md_escape(row['filename'])} | {row['final_score']:.4f} | "
                f"{row['technical']:.4f} | {row['vision_total']} | {row['subject_clarity']} | "
                f"{row['lighting']} | {row['color_pop']} | {row['emotion']} | {row['scroll_stop']} | "
                f"{row['crop_4x5']} | {'yes' if row['failed'] else 'no'} | {_md_escape(str(row['one_line']))} |"
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark models for speed + quality on the same candidate image set.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input folder containing original images.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of timed runs per variant (default: 1).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_OLLAMA_MODELS,
        help=(
            "Ollama model tags to compare (space-separated). "
            "Default: qwen3-vl:8b blaifa/InternVL3_5:8b blaifa/InternVL3_5:4B openbmb/minicpm-v4.5:8b"
        ),
    )
    parser.add_argument(
        "--variants",
        choices=["off", "on", "both"],
        default="off",
        help="YOLO variants to run for Ollama models (default: off).",
    )
    parser.add_argument(
        "--include-claude",
        action="store_true",
        help="Include Claude scorer as additional baseline variant.",
    )
    parser.add_argument(
        "--claude-model",
        default=DEFAULT_CLAUDE_MODEL,
        help=f"Claude model used when --include-claude is set (default: {DEFAULT_CLAUDE_MODEL}).",
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
        "--no-warmup",
        action="store_true",
        help="Disable per-variant warmup (default warmup scores 1 image, then waits 10s).",
    )
    parser.add_argument(
        "--prompt-variants",
        nargs="+",
        choices=["default", "claude+system"],
        default=["default"],
        metavar="VARIANT",
        help="One or more Gemma 4 prompt variants to benchmark (default: default).",
    )
    parser.add_argument(
        "--report",
        default="docs/ollama-model-benchmark-report.md",
        help="Output Markdown report path (default: docs/ollama-model-benchmark-report.md).",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help=(
            "Path to a JSON file for persisting benchmark results across runs. "
            "Cached variants are skipped and loaded from this file instead of recomputed. "
            "New results are merged back in."
        ),
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Ignore cached results in --results-file and recompute all variants.",
    )
    return parser.parse_args()


def _build_variants(args: argparse.Namespace, models: list[str]) -> list[BenchmarkVariant]:
    variants: list[BenchmarkVariant] = []
    prompt_variants = args.prompt_variants
    yolo_modes = [False]
    if args.variants == "on":
        yolo_modes = [True]
    elif args.variants == "both":
        yolo_modes = [False, True]

    for yolo_enabled in yolo_modes:
        yolo_label = "on" if yolo_enabled else "off"
        for model in models:
            for prompt_variant in prompt_variants:
                label = f"{model} | yolo={yolo_label}"
                if prompt_variant != "default":
                    label += f" | prompt={prompt_variant}"
                variants.append(
                    BenchmarkVariant(
                        label=label,
                        scorer="ollama",
                        model=model,
                        yolo_enabled=yolo_enabled,
                        prompt_variant=prompt_variant,
                    )
                )

    if args.include_claude:
        variants.append(
            BenchmarkVariant(
                label=f"{args.claude_model} | scorer=claude",
                scorer="claude",
                model=args.claude_model,
                yolo_enabled=False,
            )
        )
    return variants


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

    variants = _build_variants(args, models)
    if not variants:
        raise SystemExit("No benchmark variants selected.")

    warmup = not args.no_warmup

    work = src.parent / f"{src.name}_work_bench"
    print(f"📐 Preparing shared Stage 0-2 candidates from: {src}")
    resized, source_map = selector.resize_for_processing(src, work)
    if not resized:
        raise SystemExit("No images found to benchmark.")
    unique, _ = selector.deduplicate(resized)
    scored = selector.batch_technical_score(unique, source_map=source_map)

    if args.all:
        n_candidates = len(scored)
    else:
        n_candidates = max(args.top, int(len(scored) * args.vision_pct))
    candidates = scored[:n_candidates]
    if not candidates:
        raise SystemExit("No candidates selected for benchmark.")

    store = BenchmarkStore(Path(args.results_file).expanduser().resolve()) if args.results_file else None
    image_set_hash = _image_set_hash(candidates)

    print(
        f"🧪 Benchmarking {len(variants)} variant(s) on {len(candidates)} candidates, {args.runs} run(s) each"
    )
    if store:
        print(f"📦 Results store: {store.path}")
    all_metrics: list[RunMetrics] = []
    for idx, variant in enumerate(variants, start=1):
        if store and not args.rescore:
            cached = store.get_cached_metrics(variant, image_set_hash)
            if cached:
                print(f"⚡ Variant {idx}/{len(variants)}: {variant.label} — loaded from cache")
                all_metrics.extend(cached)
                continue
        print(f"➡️  Variant {idx}/{len(variants)}: {variant.label}")
        metrics = _benchmark_variant(
            variant=variant,
            candidates=candidates,
            src=src,
            runs=args.runs,
            warmup=warmup,
        )
        all_metrics.extend(metrics)
        if store:
            store.store_metrics(variant, metrics, image_set_hash)

    _write_report(
        report_path=report_path,
        src=src,
        candidates_count=len(candidates),
        runs=args.runs,
        warmup=warmup,
        variants=variants,
        all_metrics=all_metrics,
    )
    print(f"📝 Report written: {report_path}")


if __name__ == "__main__":
    main()
