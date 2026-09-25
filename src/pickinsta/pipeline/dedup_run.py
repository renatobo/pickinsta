"""Dedup-only pipeline use-case execution."""

from __future__ import annotations

import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Optional

from pickinsta.config import WorkerSettings
from pickinsta.events import console_event

print = partial(console_event, "dedup_pipeline")


def _run_dedup_only_bound(
    input_folder: str,
    output_folder: str = "selected",
    work_folder: Optional[str] = None,
    worker_settings: WorkerSettings | None = None,
    *,
    collaborators: Any,
):
    """
    Dedup-only mode: resize → deduplicate → output all unique images.
    Best shot per burst selected by sharpness, then technical re-evaluation.
    Outputs full/hd/cropped variants only — no scoring, no ranking, no debug.
    """
    RunTelemetry = collaborators.RunTelemetry
    resolve_worker_settings = collaborators.resolve_worker_settings
    configure_opencv_threads = collaborators.configure_opencv_threads
    resize_for_processing = collaborators.resize_for_processing
    deduplicate = collaborators.deduplicate
    _score_technical_with_cache = collaborators.score_technical_with_cache
    _crop_one_image_no_debug = collaborators.crop_one_image_no_debug
    _generate_dedup_gallery = collaborators.generate_dedup_gallery
    build_run_manifest = collaborators.build_run_manifest
    _atomic_write_json = collaborators.atomic_write_json
    _publish_managed_artifacts = collaborators.publish_managed_artifacts
    bounded_worker_count = collaborators.bounded_worker_count
    _call_worker_stage = collaborators.call_worker_stage
    Image = collaborators.Image
    MAX_RESIZE_PX = collaborators.max_resize_px

    src = Path(input_folder)
    out = Path(output_folder)
    work = Path(work_folder) if work_folder else src.parent / f"{src.name}_work"
    telemetry = RunTelemetry()

    if not src.exists():
        print(f"❌ Input folder not found: {src}")
        sys.exit(1)

    worker_settings = worker_settings or resolve_worker_settings()
    configure_opencv_threads(worker_settings)

    print("=" * 60)
    print("🏍️  Dedup-Only Mode")
    print(f"   Input:  {src}")
    print(f"   Output: {out}")
    print("=" * 60)

    # Stage 0: Resize
    print(f"\n📐 Stage 0: Resizing to max {MAX_RESIZE_PX}px...")
    with telemetry.stage("resize"):
        resized, source_map = _call_worker_stage(
            resize_for_processing, src, work, worker_settings=worker_settings
        )
    if not resized:
        print("❌ No valid images found.")
        sys.exit(1)

    # Stage 1: Deduplicate
    print("\n🔍 Stage 1: Deduplicating (burst detection)...")
    with telemetry.stage("deduplication"):
        unique, burst_map = _call_worker_stage(
            deduplicate, resized, worker_settings=worker_settings
        )

    # Stage 2b: Burst re-evaluation with technical scoring
    burst_timing = telemetry.stage("burst_reevaluation")
    burst_timing.__enter__()
    burst_items = [(path, burst_map[path]) for path in unique if path in burst_map]
    burst_swaps = 0
    if burst_items:
        from concurrent.futures import ThreadPoolExecutor as _TPEb

        alt_paths = []
        for rep, members in burst_items:
            for p in members:
                if p != rep:
                    alt_paths.append(p)
        alt_scores: dict[Path, dict] = {}
        if alt_paths:
            n_workers = bounded_worker_count("thread", len(alt_paths), worker_settings)
            with _TPEb(max_workers=n_workers) as pool:
                for _, res in zip(alt_paths, pool.map(_score_technical_with_cache, alt_paths)):
                    if res is not None:
                        alt_scores[res[0]] = res[1]
        # Also score the representatives
        rep_paths = [rep for rep, _ in burst_items]
        if rep_paths:
            n_workers = bounded_worker_count("thread", len(rep_paths), worker_settings)
            with _TPEb(max_workers=n_workers) as pool:
                for result in zip(rep_paths, pool.map(_score_technical_with_cache, rep_paths)):
                    path, res = result
                    if res is not None:
                        alt_scores[res[0]] = res[1]

        new_unique = []
        for path in unique:
            if path not in burst_map:
                new_unique.append(path)
                continue
            members = burst_map[path]
            best_path = path
            best_composite = alt_scores.get(path, {}).get("composite", 0)
            for m in members:
                mc = alt_scores.get(m, {}).get("composite", 0)
                if mc > best_composite:
                    best_composite = mc
                    best_path = m
            if best_path != path:
                burst_swaps += 1
            new_unique.append(best_path)
        unique = new_unique

    if burst_swaps:
        print(f"  🔄 Burst re-evaluation: swapped {burst_swaps} images for better technical scores")
    burst_timing.__exit__(None, None, None)

    # Build burst lookup for gallery (keyed on selected path)
    burst_info_map: dict[Path, list[Path]] = {}
    for orig_rep, members in burst_map.items():
        # Find which unique path corresponds to this burst
        for u in unique:
            if u == orig_rep or u in members:
                burst_info_map[u] = members
                break

    # Output
    out.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pickinsta-run-", dir=out))
    run_issues: list[dict[str, str]] = []
    print(f"\n✂️  Outputting {len(unique)} images (full/hd/cropped)...")

    # Naming: <stem>_full.<ext>, <stem>_hd.jpg, <stem>_cropped.jpg
    output_plan = []
    used_output_stems: set[str] = set()
    for work_path in unique:
        base_stem = (source_map.get(work_path) or work_path).stem
        output_stem = base_stem
        suffix = 2
        while output_stem.casefold() in used_output_stems:
            output_stem = f"{base_stem}_{suffix}"
            suffix += 1
        used_output_stems.add(output_stem.casefold())
        source_ext = (source_map.get(work_path) or work_path).suffix
        output_plan.append(
            {
                "work_path": work_path,
                "source_path": source_map.get(work_path),
                "dest_cropped": staging / f"{output_stem}_cropped.jpg",
                "dest_hd": staging / f"{output_stem}_hd.jpg",
                "dest_full": staging / f"{output_stem}_full{source_ext}",
                "display_name": (source_map.get(work_path) or work_path).name,
            }
        )

    output_timing = telemetry.stage("output_generation")
    output_timing.__enter__()
    crop_args = [(i, plan["work_path"], plan["dest_cropped"]) for i, plan in enumerate(output_plan)]
    crop_results: dict[int, dict] = {}
    n_workers = bounded_worker_count("thread", len(crop_args), worker_settings)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for idx, meta in pool.map(_crop_one_image_no_debug, crop_args):
            crop_results[idx] = meta

    gallery_items = []
    written = 0
    for i, plan in enumerate(output_plan):
        crop_meta = crop_results.get(i, {})
        if crop_meta.get("_failed"):
            print(f"  ⚠ Could not crop {plan['work_path'].name}")
            run_issues.append(
                {
                    "stage": "crop",
                    "artifact": plan["display_name"],
                    "reason": "crop failed; image omitted from published output",
                }
            )
            continue

        # HD
        try:
            shutil.copy2(str(plan["work_path"]), str(plan["dest_hd"]))
        except Exception as exc:
            run_issues.append(
                {"stage": "copy_hd", "artifact": plan["dest_hd"].name, "reason": str(exc)}
            )

        # Full
        source = plan["source_path"] or plan["work_path"]
        try:
            shutil.copy2(str(source), str(plan["dest_full"]))
        except Exception as exc:
            run_issues.append(
                {
                    "stage": "copy_full",
                    "artifact": plan["dest_full"].name,
                    "reason": str(exc),
                }
            )

        burst_members = burst_info_map.get(unique[i])

        # EXIF
        exif_info = {}
        source_file = plan["source_path"] or plan["work_path"]
        try:
            from PIL.ExifTags import TAGS as _EXIF_TAGS

            with Image.open(source_file) as _eimg:
                raw_exif = _eimg._getexif() or {}
            tagged = {_EXIF_TAGS.get(k, k): v for k, v in raw_exif.items()}
            if tagged.get("Make"):
                make = tagged["Make"].strip()
                model = tagged.get("Model", "").strip()
                exif_info["camera"] = model if model.startswith(make) else f"{make} {model}".strip()
            if tagged.get("LensModel"):
                exif_info["lens"] = str(tagged["LensModel"]).strip()
            if tagged.get("FocalLength"):
                exif_info["focal"] = f"{float(tagged['FocalLength']):.0f}mm"
            if tagged.get("FNumber"):
                exif_info["aperture"] = f"f/{float(tagged['FNumber']):.1f}"
            if tagged.get("ExposureTime"):
                et = float(tagged["ExposureTime"])
                exif_info["shutter"] = f"{et:.1f}s" if et >= 1 else f"1/{int(round(1 / et))}s"
            if tagged.get("ISOSpeedRatings"):
                exif_info["iso"] = f"ISO {tagged['ISOSpeedRatings']}"
            if tagged.get("DateTimeOriginal"):
                exif_info["date"] = str(tagged["DateTimeOriginal"])
        except Exception:
            pass

        gallery_items.append(
            {
                "filename": plan["display_name"],
                "cropped": plan["dest_cropped"].name,
                "hd": plan["dest_hd"].name,
                "full": plan["dest_full"].name,
                "burst_count": len(burst_members) if burst_members else 0,
                "exif": exif_info,
            }
        )
        written += 1
    output_timing.__exit__(None, None, None)

    # Simplified gallery
    with telemetry.stage("reporting"):
        _generate_dedup_gallery(staging, gallery_items)
    run_summary = {
        "status": "degraded" if run_issues else "complete",
        "processed": written,
        "skipped": sum(issue["stage"] == "crop" for issue in run_issues),
        "failed": sum(issue["stage"] != "crop" for issue in run_issues),
        "issues": run_issues,
    }
    manifest = build_run_manifest(
        {
            "schema_version": 1,
            **run_summary,
            "input_folder": str(src),
            "output_folder": str(out),
            "scorer": "dedup-only",
            "artifacts": sorted(path.name for path in staging.iterdir() if path.is_file()),
        },
        telemetry,
        configuration={
            "input_folder": src,
            "output_folder": out,
            "work_folder": work,
            "mode": "dedup-only",
            "workers": {
                "processes": worker_settings.process_workers,
                "threads": worker_settings.thread_workers,
                "opencv_threads": worker_settings.opencv_threads,
            },
        },
        warnings=run_issues,
    )
    _atomic_write_json(staging / "run_manifest.json", manifest)
    artifact_names = sorted(path.name for path in staging.iterdir() if path.is_file())
    try:
        _publish_managed_artifacts(
            staging,
            out,
            artifact_names,
            completion_artifact="run_manifest.json",
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"🏆 Done! {written} unique images saved to {out}/")
    print(f"🗂️  Work folder retained: {work}")
    print(f"🌐 Gallery: {out / 'index.html'}")
    print(f"{'=' * 60}")
