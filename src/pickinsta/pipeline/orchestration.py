"""Application-level orchestration for Pickinsta pipeline use cases.

This module owns recursive discovery and dispatch.  The compatibility facade
injects collaborators from its live globals so downstream monkeypatches keep
working while the application flow remains independent of the legacy module.
"""

from __future__ import annotations

import os
import shutil  # noqa: F401 - compatibility export for downstream monkeypatches
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from pickinsta.config import WorkerSettings
from pickinsta.events import console_event
from pickinsta.pipeline.dedup_run import _run_dedup_only_bound
from pickinsta.pipeline.full_run import _run_pipeline_bound

print = partial(console_event, "recursive_pipeline")


@dataclass(frozen=True, slots=True)
class RunContext:
    """Resolved paths and execution policy shared by one pipeline run."""

    input_path: Path
    output_path: Path
    work_path: Path
    worker_settings: WorkerSettings

    @classmethod
    def resolve(
        cls,
        input_folder: str,
        output_folder: str,
        work_folder: str | None,
        worker_settings: WorkerSettings,
        *,
        recursive: bool = False,
    ) -> "RunContext":
        source = Path(input_folder)
        output = Path(output_folder)
        if work_folder:
            work = Path(work_folder)
        elif recursive:
            work = output.parent / f"{output.name}_work"
        else:
            work = source.parent / f"{source.name}_work"
        return cls(source, output, work, worker_settings)


@dataclass(frozen=True, slots=True)
class RecursiveCollaborators:
    """Side effects required by the recursive use case."""

    resolve_workers: Callable[[], WorkerSettings]
    configure_opencv: Callable[[WorkerSettings], None]
    run_pipeline: Callable[..., list[dict[str, Any]]]
    write_summary: Callable[..., tuple[Path, Path]]
    generate_gallery_index: Callable[[Path], list[Path]]
    supported_extensions: frozenset[str]


@dataclass(frozen=True, slots=True)
class FacadeCollaborators:
    """Explicit patchable dependencies required by full and dedup use cases."""

    RunTelemetry: Callable[[], Any]
    resolve_worker_settings: Callable[[], WorkerSettings]
    configure_opencv_threads: Callable[[WorkerSettings], None]
    resize_for_processing: Callable[..., Any]
    deduplicate: Callable[..., Any]
    batch_technical_score: Callable[..., Any]
    batch_vision_score: Callable[..., Any]
    score_technical_with_cache: Callable[..., Any]
    prepare_claude_crop_first_candidates: Callable[..., Any]
    crop_one_image_no_debug: Callable[..., Any]
    crop_one_image: Callable[..., Any]
    generate_dedup_gallery: Callable[..., Any]
    generate_gallery: Callable[..., Any]
    write_markdown_report: Callable[..., Any]
    atomic_write_json: Callable[..., Any]
    publish_managed_artifacts: Callable[..., Any]
    build_run_manifest: Callable[..., Any]
    bounded_worker_count: Callable[..., int]
    call_worker_stage: Callable[..., Any]
    call_with_supported_kwargs: Callable[..., Any]
    build_vision_prompt: Callable[..., str]
    resolve_account_context: Callable[..., str]
    claude_prompt_sha256: Callable[[str], str]
    resolve_claude_model: Callable[[], str]
    claude_cache_options: Callable[..., dict]
    load_claude_score_from_file_cache: Callable[..., Any]
    file_sha256: Callable[[Path], str]
    safe_float: Callable[..., float]
    Image: Any
    max_resize_px: int
    output_width: int
    output_height: int
    min_crop_score: float

    @classmethod
    def from_module(cls, module: ModuleType) -> "FacadeCollaborators":
        values = vars(module)
        return cls(
            RunTelemetry=values["RunTelemetry"],
            resolve_worker_settings=values["resolve_worker_settings"],
            configure_opencv_threads=values["configure_opencv_threads"],
            resize_for_processing=values["resize_for_processing"],
            deduplicate=values["deduplicate"],
            batch_technical_score=values["batch_technical_score"],
            batch_vision_score=values["batch_vision_score"],
            score_technical_with_cache=values["_score_technical_with_cache"],
            prepare_claude_crop_first_candidates=values["_prepare_claude_crop_first_candidates"],
            crop_one_image_no_debug=values["_crop_one_image_no_debug"],
            crop_one_image=values["_crop_one_image"],
            generate_dedup_gallery=values["_generate_dedup_gallery"],
            generate_gallery=values["generate_gallery"],
            write_markdown_report=values["write_markdown_report"],
            atomic_write_json=values["_atomic_write_json"],
            publish_managed_artifacts=values["_publish_managed_artifacts"],
            build_run_manifest=values["build_run_manifest"],
            bounded_worker_count=values["bounded_worker_count"],
            call_worker_stage=values["_call_worker_stage"],
            call_with_supported_kwargs=values["_call_with_supported_kwargs"],
            build_vision_prompt=values["build_vision_prompt"],
            resolve_account_context=values["resolve_account_context"],
            claude_prompt_sha256=values["claude_prompt_sha256"],
            resolve_claude_model=values["resolve_claude_model"],
            claude_cache_options=values["claude_cache_options"],
            load_claude_score_from_file_cache=values["load_claude_score_from_file_cache"],
            file_sha256=values["_file_sha256"],
            safe_float=values["_safe_float"],
            Image=values["Image"],
            max_resize_px=values["MAX_RESIZE_PX"],
            output_width=values["OUTPUT_WIDTH"],
            output_height=values["OUTPUT_HEIGHT"],
            min_crop_score=values["CLAUDE_MIN_CROP4X5_OUTPUT_SCORE"],
        )


def crop_one_image_no_debug(args: tuple, *, cropper: Callable[..., Path]) -> tuple[int, dict]:
    """Worker adapter that makes the crop dependency explicit."""
    from pickinsta.pipeline.cropping import _crop_one_image_no_debug

    return _crop_one_image_no_debug(args, cropper=cropper)


def crop_one_image(args: tuple, *, cropper: Callable[..., Path]) -> tuple[int, dict]:
    """Debug crop worker adapter with an injected crop implementation."""
    from pickinsta.pipeline.cropping import _crop_one_image

    return _crop_one_image(args, cropper=cropper)


def has_supported_images(folder: Path, supported_extensions: frozenset[str]) -> bool:
    """Return whether ``folder`` directly contains a supported image."""
    try:
        return any(
            path.is_file() and path.suffix.lower() in supported_extensions
            for path in folder.iterdir()
        )
    except OSError:
        return False


def discover_recursive_input_folders(
    root: Path, supported_extensions: frozenset[str]
) -> list[Path]:
    """Return leaf folders containing supported images in deterministic order."""
    if not root.exists():
        return []

    candidates: set[Path] = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(Path(name).suffix.lower() in supported_extensions for name in filenames):
            candidates.add(Path(dirpath))

    if not candidates and has_supported_images(root, supported_extensions):
        candidates.add(root)

    ordered = sorted(candidates, key=lambda path: (len(path.parts), str(path)))
    return [
        folder
        for folder in ordered
        if not any(other != folder and other.is_relative_to(folder) for other in ordered)
    ]


def _folder_summary(
    *, folder: Path, relative: Path, output: Path, output_root: Path, report: list[dict]
) -> dict[str, Any]:
    report_path = output / "selection_report.json"
    relative_text = "." if relative == Path(".") else str(relative)
    return {
        "relative_folder": relative_text,
        "output_relative": relative_text,
        "input_folder": str(folder),
        "output_folder": str(output),
        "report_path": str(report_path),
        "report_relative": str(report_path.relative_to(output_root)),
        "selected_count": len(report),
        "top_score": float(report[0]["final_score"]) if report else 0.0,
        "avg_final_score": (
            sum(float(item["final_score"]) for item in report) / len(report) if report else 0.0
        ),
        "uncertain_crops": sum(1 for item in report if item.get("uncertain_crop")),
        "report": report,
    }


def run_pipeline_recursive(
    input_folder: str,
    output_folder: str = "selected",
    work_folder: str | None = None,
    top_n: int = 10,
    scorer: str = "clip",
    vision_candidates_pct: float = 0.5,
    claude_model: str | None = None,
    score_all: bool = False,
    claude_crop_first: bool = False,
    rescore: bool = False,
    worker_settings: WorkerSettings | None = None,
    *,
    collaborators: RecursiveCollaborators,
) -> list[dict[str, Any]]:
    """Run the full pipeline for every leaf image folder under an input root."""
    settings = worker_settings or collaborators.resolve_workers()
    context = RunContext.resolve(input_folder, output_folder, work_folder, settings, recursive=True)
    if not context.input_path.exists():
        print(f"❌ Input folder not found: {context.input_path}")
        sys.exit(1)

    collaborators.configure_opencv(settings)
    targets = discover_recursive_input_folders(
        context.input_path, collaborators.supported_extensions
    )
    if not targets:
        print(f"❌ No supported images found under {context.input_path}")
        sys.exit(1)

    context.output_path.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("🏍️  Recursive Instagram Image Selection Pipeline")
    print(f"   Input root:  {context.input_path}")
    print(f"   Output root: {context.output_path}")
    print(f"   Work root:   {context.work_path}")
    print(f"   Scorer:      {scorer}")
    print(f"   Folders:     {len(targets)}")
    print(f"   Top N:       {top_n}")
    print("=" * 60)

    summaries: list[dict[str, Any]] = []
    for folder in targets:
        relative = folder.relative_to(context.input_path)
        output = context.output_path / relative
        work = context.work_path / relative
        print(f"\n📁 Recursive folder: {folder}")
        try:
            report = collaborators.run_pipeline(
                input_folder=str(folder),
                output_folder=str(output),
                work_folder=str(work),
                top_n=top_n,
                scorer=scorer,
                vision_candidates_pct=vision_candidates_pct,
                claude_model=claude_model,
                score_all=score_all,
                claude_crop_first=claude_crop_first,
                rescore=rescore,
                worker_settings=settings,
                _configure_opencv=False,
            )
        except Exception as exc:
            relative_text = "." if relative == Path(".") else str(relative)
            print(f"  ⚠ Folder failed; continuing: {folder}: {exc}")
            summaries.append(
                {
                    "relative_folder": relative_text,
                    "output_relative": relative_text,
                    "input_folder": str(folder),
                    "output_folder": str(output),
                    "report_path": None,
                    "report_relative": None,
                    "selected_count": 0,
                    "top_score": 0.0,
                    "avg_final_score": 0.0,
                    "uncertain_crops": 0,
                    "status": "failed",
                    "error": str(exc),
                    "report": [],
                }
            )
            continue
        summary = _folder_summary(
            folder=folder,
            relative=relative,
            output=output,
            output_root=context.output_path,
            report=report,
        )
        summary["status"] = "complete"
        summaries.append(summary)

    recursive_json, recursive_md = collaborators.write_summary(
        root_input=context.input_path,
        output_root=context.output_path,
        scorer=scorer,
        top_n=top_n,
        folder_summaries=summaries,
    )
    generated = collaborators.generate_gallery_index(context.output_path)
    print(f"\n{'=' * 60}")
    print(f"🏆 Recursive run complete: {len(summaries)} folders processed")
    print(f"🗂️  Work root retained: {context.work_path}")
    print(f"📋 Recursive JSON Report: {recursive_json}")
    print(f"📝 Recursive Markdown Report: {recursive_md}")
    if generated:
        print(f"🌐 Generated gallery/index pages: {len(generated)}")
    print("=" * 60)
    return summaries


def run_dedup_only(*args, collaborators: FacadeCollaborators, **kwargs):
    """Execute dedup-only orchestration with explicitly supplied side effects."""
    return _run_dedup_only_bound(*args, collaborators=collaborators, **kwargs)


def run_pipeline(*args, collaborators: FacadeCollaborators, **kwargs):
    """Execute full orchestration with explicitly supplied stage collaborators."""
    return _run_pipeline_bound(*args, collaborators=collaborators, **kwargs)
