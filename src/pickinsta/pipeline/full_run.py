"""Full selection pipeline use-case execution."""

from __future__ import annotations

import shutil
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Optional

from pickinsta.config import WorkerSettings
from pickinsta.events import console_event

print = partial(console_event, "full_pipeline")


def _run_pipeline_bound(
    input_folder: str,
    output_folder: str = "selected",
    work_folder: Optional[str] = None,
    top_n: int = 10,
    scorer: str = "clip",
    vision_candidates_pct: float = 0.5,
    claude_model: Optional[str] = None,
    score_all: bool = False,
    claude_crop_first: bool = False,
    rescore: bool = False,
    worker_settings: WorkerSettings | None = None,
    _configure_opencv: bool = True,
    *,
    collaborators: Any,
):
    """
    Full pipeline: resize → deduplicate → technical score → vision score → crop → output.

    Args:
        input_folder: Path to folder with raw event photos
        output_folder: Path for final 1080x1440 output images
        work_folder: Path for intermediate work files (default: <input>_work next to input)
        top_n: Number of top images to output
        scorer: "clip" (free/local), "claude" (best quality, API costs), or "ollama" (self-hosted)
        vision_candidates_pct: Send top N% of technically-scored images to vision scoring
        score_all: If True, send all technically-scored images to vision scoring
        claude_crop_first: If True with Claude scorer, pre-crop candidates before scoring
    """
    RunTelemetry = collaborators.RunTelemetry
    resolve_worker_settings = collaborators.resolve_worker_settings
    configure_opencv_threads = collaborators.configure_opencv_threads
    resize_for_processing = collaborators.resize_for_processing
    deduplicate = collaborators.deduplicate
    batch_technical_score = collaborators.batch_technical_score
    batch_vision_score = collaborators.batch_vision_score
    _score_technical_with_cache = collaborators.score_technical_with_cache
    _prepare_claude_crop_first_candidates = collaborators.prepare_claude_crop_first_candidates
    _crop_one_image = collaborators.crop_one_image
    generate_gallery = collaborators.generate_gallery
    write_markdown_report = collaborators.write_markdown_report
    _atomic_write_json = collaborators.atomic_write_json
    _publish_managed_artifacts = collaborators.publish_managed_artifacts
    build_run_manifest = collaborators.build_run_manifest
    bounded_worker_count = collaborators.bounded_worker_count
    _call_worker_stage = collaborators.call_worker_stage
    _call_with_supported_kwargs = collaborators.call_with_supported_kwargs
    build_vision_prompt = collaborators.build_vision_prompt
    resolve_account_context = collaborators.resolve_account_context
    claude_prompt_sha256 = collaborators.claude_prompt_sha256
    resolve_claude_model = collaborators.resolve_claude_model
    claude_cache_options = collaborators.claude_cache_options
    load_claude_score_from_file_cache = collaborators.load_claude_score_from_file_cache
    _file_sha256 = collaborators.file_sha256
    _safe_float = collaborators.safe_float
    MAX_RESIZE_PX = collaborators.max_resize_px
    OUTPUT_WIDTH = collaborators.output_width
    OUTPUT_HEIGHT = collaborators.output_height
    CLAUDE_MIN_CROP4X5_OUTPUT_SCORE = collaborators.min_crop_score

    src = Path(input_folder)
    out = Path(output_folder)
    work = Path(work_folder) if work_folder else src.parent / f"{src.name}_work"
    telemetry = RunTelemetry()
    run_issues: list[dict[str, str]] = []

    def record_issue(path: Path, stage: str, message: str) -> None:
        run_issues.append({"stage": stage, "image": path.name, "message": str(message)[:500]})

    if not src.exists():
        print(f"❌ Input folder not found: {src}")
        sys.exit(1)

    worker_settings = worker_settings or resolve_worker_settings()
    if _configure_opencv:
        configure_opencv_threads(worker_settings)

    print("=" * 60)
    print("🏍️  Instagram Image Selection Pipeline")
    print(f"   Input:  {src}")
    print(f"   Output: {out}")
    print(f"   Scorer: {scorer}")
    if scorer == "claude":
        print("   Claude cache: per file in input folder (<original_filename>.pickinsta.json)")
    print(f"   Top N:  {top_n}")
    print("=" * 60)

    # --- Stage 0: Resize ---
    print(f"\n📐 Stage 0: Resizing to max {MAX_RESIZE_PX}px...")
    with telemetry.stage("resize"):
        resized, source_map = _call_worker_stage(
            resize_for_processing, src, work, worker_settings=worker_settings
        )
    if not resized:
        print("❌ No valid images found.")
        sys.exit(1)

    # --- Stage 1: Deduplicate ---
    print("\n🔍 Stage 1: Deduplicating...")
    with telemetry.stage("deduplication"):
        unique, burst_map = _call_worker_stage(
            deduplicate, resized, worker_settings=worker_settings
        )

    # --- Stage 2: Technical scoring ---
    print("\n📊 Stage 2: Technical quality scoring...")
    with telemetry.stage("technical_scoring"):
        scored = _call_with_supported_kwargs(
            batch_technical_score,
            unique,
            source_map=source_map,
            worker_settings=worker_settings,
            cache_observer=lambda hit: telemetry.record_cache("technical", hit=hit),
            failure_observer=lambda path, stage, message: record_issue(path, stage, message),
        )

    # Tag burst groups on scored items
    for item in scored:
        if item.path in burst_map:
            item.burst_group = burst_map[item.path]
            item.burst_selected_by = "sharpness"

    # --- Stage 2b: Re-evaluate bursts for top candidates ---
    # For images selected from a burst, score all burst members technically
    # in parallel and swap in the best one if different from the sharpness pick.
    burst_timing = telemetry.stage("burst_reevaluation")
    burst_timing.__enter__()
    n_reeval = max(top_n * 2, int(len(scored) * 0.5))
    burst_items = [
        item for item in scored[:n_reeval] if item.burst_group and len(item.burst_group) >= 2
    ]
    burst_swaps = 0
    if burst_items:
        from concurrent.futures import ThreadPoolExecutor as _TPE2b

        # Collect all alt paths that need scoring (exclude already-scored representative)
        alt_paths: list[Path] = []
        for item in burst_items:
            for p in item.burst_group:
                if p != item.path:
                    alt_paths.append(p)
        # Score them all in parallel (ThreadPool — avoids YOLO fork deadlocks)
        alt_scores: dict[Path, dict] = {}
        if alt_paths:
            n_workers = bounded_worker_count("thread", len(alt_paths), worker_settings)
            with _TPE2b(max_workers=n_workers) as pool:
                for _, res in zip(alt_paths, pool.map(_score_technical_with_cache, alt_paths)):
                    if res is not None:
                        alt_scores[res[0]] = res[1]
        # Pick the best from each burst group
        for item in burst_items:
            best_path = item.path
            best_composite = item.technical.get("composite", 0)
            for alt_path in item.burst_group:
                if alt_path == item.path:
                    continue
                alt_tech = alt_scores.get(alt_path)
                if alt_tech and alt_tech.get("composite", 0) > best_composite:
                    best_composite = alt_tech["composite"]
                    best_path = alt_path
            if best_path != item.path:
                item.path = best_path
                item.source_path = source_map.get(best_path, item.source_path)
                item.technical = alt_scores.get(best_path, item.technical)
                item.burst_selected_by = "technical"
                burst_swaps += 1
    if burst_swaps:
        print(f"  🔄 Burst re-evaluation: swapped {burst_swaps} images for better technical scores")
        scored.sort(key=lambda x: x.technical.get("composite", 0), reverse=True)
    burst_timing.__exit__(None, None, None)

    # --- Stage 3: Vision scoring ---
    if score_all:
        n_candidates = len(scored)
    else:
        n_candidates = max(top_n, int(len(scored) * vision_candidates_pct))
    candidates = scored[:n_candidates]
    if scorer == "claude" and claude_crop_first:
        print(
            f"  ✂️  Claude crop-first mode: pre-cropping {len(candidates)} candidates to "
            f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT} before vision scoring..."
        )
        candidates = _prepare_claude_crop_first_candidates(candidates, work_folder=work)
    scope = "all" if score_all else f"top {len(candidates)}"
    if scorer == "claude":
        cost_per_image = 0.005
        preloaded_vision_cache: Optional[dict[Path, Optional[dict]]] = None
        if rescore:
            uncached = len(candidates)
        else:
            preloaded_vision_cache = {}
            claude_est_prompt = build_vision_prompt(resolve_account_context(search_dir=src))
            claude_est_hash = claude_prompt_sha256(claude_est_prompt)
            claude_est_model = claude_model or resolve_claude_model()
            uncached = 0
            for item in candidates:
                sp = item.source_path or item.path
                cache_options = claude_cache_options(score_path=item.path, source_path=sp)
                cached = load_claude_score_from_file_cache(
                    source_path=sp,
                    source_sha256=_file_sha256(sp),
                    model=claude_est_model,
                    prompt_sha256=claude_est_hash,
                    scorer="claude",
                    scoring_options=cache_options,
                    strict_model=True,
                )
                preloaded_vision_cache[sp] = cached
                if cached is None:
                    uncached += 1
        est_cost = uncached * cost_per_image
        print(
            f"\n💰 Claude estimate: {uncached}/{len(candidates)} images to score, "
            f"~${est_cost:.2f} ({len(candidates) - uncached} cached)"
        )
    print(f"\n🧠 Stage 3: Vision scoring {scope} candidates ({scorer})...")
    with telemetry.stage("vision_scoring"):
        ranked = _call_with_supported_kwargs(
            batch_vision_score,
            candidates,
            scorer=scorer,
            env_search_dir=src,
            claude_model=claude_model,
            rescore=rescore,
            cache_observer=lambda hit: telemetry.record_cache("vision", hit=hit),
            failure_observer=lambda path, stage, message: record_issue(
                path, f"{stage}_scoring", message
            ),
            preloaded_vision_cache=(preloaded_vision_cache if scorer == "claude" else None),
        )

    # --- Stage 4: Output top N cropped to 1080x1440 ---
    print(f"\n✂️  Stage 4: Cropping top {top_n} to {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}...")
    out.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pickinsta-run-", dir=out))

    # Also save a JSON report
    report = []

    if scorer in {"claude", "ollama"}:
        crop_safe = [
            item
            for item in ranked
            if _safe_float(item.vision.get("crop_4x5"), default=0.0)
            >= CLAUDE_MIN_CROP4X5_OUTPUT_SCORE
        ]
        if len(crop_safe) >= top_n:
            top = crop_safe[:top_n]
            print(
                f"  ✅ Claude crop gate: selecting images with crop_4x5 >= "
                f"{CLAUDE_MIN_CROP4X5_OUTPUT_SCORE:g}"
            )
        else:
            print(
                f"  ⚠ Claude crop gate: only {len(crop_safe)} images meet crop_4x5 >= "
                f"{CLAUDE_MIN_CROP4X5_OUTPUT_SCORE:g}; filling remaining slots by score."
            )
            crop_safe_ids = {id(item) for item in crop_safe}
            fallback = [item for item in ranked if id(item) not in crop_safe_ids]
            top = (crop_safe + fallback)[:top_n]
    else:
        top = ranked[:top_n]

    from concurrent.futures import ThreadPoolExecutor as _TPE4

    # Prepare output paths for each ranked image
    output_timing = telemetry.stage("output_generation")
    output_timing.__enter__()
    output_plan: list[dict] = []
    for i, item in enumerate(top):
        rank = i + 1
        output_stem = (item.source_path or item.path).stem
        output_plan.append(
            {
                "rank": rank,
                "item": item,
                "dest_cropped": staging / f"{rank:02d}_cropped_{output_stem}.jpg",
                "dest_hd": staging / f"{rank:02d}_hd_{output_stem}.jpg",
                "dest_full": staging
                / f"{rank:02d}_full_{output_stem}{(item.source_path or item.path).suffix}",
                "display_name": (item.source_path or item.path).name,
            }
        )

    crop_args = [(i, plan["item"].path, plan["dest_cropped"]) for i, plan in enumerate(output_plan)]
    crop_results: dict[int, dict] = {}
    n_crop_workers = bounded_worker_count("thread", len(crop_args), worker_settings)
    with _TPE4(max_workers=n_crop_workers) as pool:
        for idx, meta in pool.map(_crop_one_image, crop_args):
            crop_results[idx] = meta

    # Sequential: file copies + report assembly (fast I/O, needs ordering)
    for i, plan in enumerate(output_plan):
        item = plan["item"]
        rank = plan["rank"]
        dest_cropped = plan["dest_cropped"]
        dest_hd = plan["dest_hd"]
        dest_full = plan["dest_full"]
        display_name = plan["display_name"]
        crop_meta = crop_results.get(i, {})
        padded_written = False

        if crop_meta.get("_failed"):
            print(f"  ⚠ Could not process {item.path.name}")
            run_issues.append(
                {
                    "stage": "crop",
                    "artifact": display_name,
                    "reason": "crop failed; image omitted from published selection",
                }
            )
            continue

        # HD: 1920px longest edge (work copy)
        try:
            shutil.copy2(str(item.path), str(dest_hd))
        except Exception as e3:
            print(f"  ⚠ Could not copy HD version for {display_name}: {e3}")
            run_issues.append(
                {
                    "stage": "copy_hd",
                    "artifact": dest_hd.name,
                    "reason": str(e3),
                }
            )

        # Full: original source file
        try:
            source = item.source_path or item.path
            shutil.copy2(str(source), str(dest_full))
            padded_written = True
        except Exception as e3:
            print(f"  ⚠ Could not copy full version for {display_name}: {e3}")
            run_issues.append(
                {
                    "stage": "copy_full",
                    "artifact": dest_full.name,
                    "reason": str(e3),
                }
            )

        burst_info = None
        if item.burst_group and len(item.burst_group) > 1:
            burst_info = {
                "count": len(item.burst_group),
                "selected_by": item.burst_selected_by,
                "members": [p.name for p in item.burst_group],
            }

        report.append(
            {
                "rank": rank,
                "filename": display_name,
                "final_score": round(item.final_score, 4),
                "technical_composite": round(item.technical.get("composite", 0), 4),
                "vision_total": item.vision.get("total", 0),
                "one_line": item.one_line,
                "output_cropped": dest_cropped.name,
                "output_hd": dest_hd.name,
                "output_full": dest_full.name if padded_written else None,
                "uncertain_crop": bool(crop_meta.get("uncertain_crop", False)),
                "uncertain_crop_reasons": crop_meta.get("uncertain_crop_reasons", []),
                "burst": burst_info,
            }
        )

        print(f"  #{rank}: {display_name} → {dest_cropped.name}")
        print(
            f"       Score: {item.final_score:.3f} | Tech: {item.technical.get('composite', 0):.3f} | Vision: {item.vision.get('total', 0)}"
        )
        print(f"       {item.one_line}")
    output_timing.__exit__(None, None, None)

    # Save report
    report_json_path = staging / "selection_report.json"
    report_md_path = staging / "selection_report.md"
    run_summary = {
        "status": "degraded" if run_issues else "complete",
        "processed": len(report),
        "skipped": sum(issue["stage"] == "crop" for issue in run_issues),
        "failed": sum(issue["stage"] != "crop" for issue in run_issues),
        "issues": run_issues,
    }
    with telemetry.stage("reporting"):
        _atomic_write_json(report_json_path, report)
        write_markdown_report(
            report_md_path,
            input_folder=src,
            output_folder=out,
            scorer=scorer,
            top_n=top_n,
            selected_report=report,
            analyzed_items=ranked,
            run_summary=run_summary,
        )

        # Generate HTML gallery
        gallery_path = generate_gallery(staging, src)
    manifest_path = staging / "run_manifest.json"
    manifest = build_run_manifest(
        {
            "schema_version": 1,
            **run_summary,
            "input_folder": str(src),
            "output_folder": str(out),
            "scorer": scorer,
            "top_n_requested": top_n,
            "analyzed": len(ranked),
            "artifacts": sorted(path.name for path in staging.iterdir() if path.is_file()),
        },
        telemetry,
        configuration={
            "input_folder": src,
            "output_folder": out,
            "work_folder": work,
            "top_n": top_n,
            "scorer": scorer,
            "vision_candidates_pct": vision_candidates_pct,
            "score_all": score_all,
            "claude_crop_first": claude_crop_first,
            "rescore": rescore,
            "workers": {
                "processes": worker_settings.process_workers,
                "threads": worker_settings.thread_workers,
                "opencv_threads": worker_settings.opencv_threads,
            },
        },
        warnings=run_issues,
    )
    _atomic_write_json(manifest_path, manifest)
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

    report_json_path = out / report_json_path.name
    report_md_path = out / report_md_path.name
    gallery_path = out / gallery_path.name if gallery_path else None

    print(f"\n{'=' * 60}")
    print(f"🏆 Done ({run_summary['status']})! {len(report)} images saved to {out}/")
    print(f"🗂️  Work folder retained: {work}")
    print(f"📋 JSON Report: {report_json_path}")
    print(f"📝 Markdown Report: {report_md_path}")
    if gallery_path:
        print(f"🌐 Gallery: {gallery_path}")
    print(f"{'=' * 60}")

    return report
