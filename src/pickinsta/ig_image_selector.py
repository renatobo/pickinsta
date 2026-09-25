#!/usr/bin/env python3
"""Legacy compatibility facade for the extracted Pickinsta pipeline."""

import inspect
import os
import shutil
import sys
import time
from urllib.request import urlopen, urlretrieve

import cv2
import numpy as np
from PIL import Image

from pickinsta.clip_scorer import _clip_setup_hint, load_clip_model, score_with_clip
from pickinsta.config import (
    ACCOUNT_CONTEXT_ENV_VAR,
    DEFAULT_ACCOUNT_CONTEXT,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_BACKOFF_BASE_ENV_VAR,
    OLLAMA_CIRCUIT_BREAKER_ENV_VAR,
    OLLAMA_CONCURRENCY_ENV_VAR,
    OLLAMA_JPEG_QUALITY_ENV_VAR,
    OLLAMA_KEEP_ALIVE_ENV_VAR,
    OLLAMA_MAX_EDGE_ENV_VAR,
    OLLAMA_MAX_RETRIES_ENV_VAR,
    OLLAMA_TIMEOUT_ENV_VAR,
    OLLAMA_USE_YOLO_ENV_VAR,
    PICKINSTA_OLLAMA_BASE_URL_ENV_VAR,
    PICKINSTA_OLLAMA_MODEL_ENV_VAR,
    YOLO_MODEL_ENV_VAR,
    WorkerSettings,
    _read_env_file,
    resolve_account_context,
    resolve_anthropic_api_key,
    resolve_claude_model,
    resolve_ollama_base_url,
    resolve_ollama_circuit_breaker_errors,
    resolve_ollama_concurrency,
    resolve_ollama_jpeg_quality,
    resolve_ollama_keep_alive,
    resolve_ollama_max_image_edge,
    resolve_ollama_max_retries,
    resolve_ollama_model,
    resolve_ollama_retry_backoff_seconds,
    resolve_ollama_timeout_seconds,
    resolve_ollama_use_yolo_context,
    resolve_optional_hf_token,
    resolve_worker_settings,
)
from pickinsta.detection import yolo as yolo_detection
from pickinsta.infrastructure.filesystem import (
    atomic_write_json as _atomic_write_json,
)
from pickinsta.infrastructure.filesystem import (
    atomic_write_text as _atomic_write_text,
)
from pickinsta.infrastructure.filesystem import (
    publish_managed_artifacts as _publish_managed_artifacts,
)
from pickinsta.infrastructure.vision_cache import cache_file_for_source
from pickinsta.models import ImageScore
from pickinsta.pipeline import (
    crop_geometry,
    cropping,
    deduplication,
    orchestration,
    resize,
    technical_scoring,
    vision_scoring,
)
from pickinsta.pipeline.worker_tuning import bounded_worker_count, configure_opencv_threads
from pickinsta.reporting.gallery import (
    _generate_dedup_gallery,
    generate_gallery,
    generate_gallery_index,
)
from pickinsta.reporting.gallery_data import build_gallery_data as _gallery_build_data
from pickinsta.reporting.markdown import write_markdown_report
from pickinsta.reporting.recursive import (
    write_recursive_summary_report as _write_recursive_summary_report,
)
from pickinsta.telemetry import RunTelemetry, build_run_manifest
from pickinsta.vision import claude as claude_vision
from pickinsta.vision import facade_compat
from pickinsta.vision import ollama as ollama_vision
from pickinsta.vision import prompts as vision_prompts

__all__ = [
    "ACCOUNT_CONTEXT_ENV_VAR",
    "DEFAULT_ACCOUNT_CONTEXT",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "OLLAMA_BACKOFF_BASE_ENV_VAR",
    "OLLAMA_CIRCUIT_BREAKER_ENV_VAR",
    "OLLAMA_CONCURRENCY_ENV_VAR",
    "OLLAMA_JPEG_QUALITY_ENV_VAR",
    "OLLAMA_KEEP_ALIVE_ENV_VAR",
    "OLLAMA_MAX_EDGE_ENV_VAR",
    "OLLAMA_MAX_RETRIES_ENV_VAR",
    "OLLAMA_TIMEOUT_ENV_VAR",
    "OLLAMA_USE_YOLO_ENV_VAR",
    "PICKINSTA_OLLAMA_BASE_URL_ENV_VAR",
    "PICKINSTA_OLLAMA_MODEL_ENV_VAR",
    "YOLO_MODEL_ENV_VAR",
    "_gallery_build_data",
    "_read_env_file",
]
MAX_RESIZE_PX = resize.MAX_RESIZE_PX
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1080, 1440
SUPPORTED_EXTENSIONS, DEDUP_THRESHOLD = resize.SUPPORTED_EXTENSIONS, deduplication.DEDUP_THRESHOLD
TECHNICAL_CACHE_SCHEMA_VERSION = technical_scoring.TECHNICAL_CACHE_SCHEMA_VERSION
VISION_CACHE_SCHEMA_VERSION = facade_compat.VISION_CACHE_SCHEMA_VERSION
CLAUDE_SCORING_MAX_EDGE = claude_vision.CLAUDE_SCORING_MAX_EDGE
CLAUDE_SCORING_JPEG_QUALITY = claude_vision.CLAUDE_SCORING_JPEG_QUALITY
CLAUDE_MIN_CROP4X5_OUTPUT_SCORE = 6.0
OLLAMA_QWEN_NUM_PREDICT_LARGE_EDGE = ollama_vision.OLLAMA_QWEN_NUM_PREDICT_LARGE_EDGE
YOLO_MODEL_FILENAME, YOLO_MODEL_URL = (
    yolo_detection.YOLO_MODEL_FILENAME,
    yolo_detection.YOLO_MODEL_URL,
)


def _call_worker_stage(function, *args, worker_settings: WorkerSettings, **kwargs):
    if "worker_settings" in inspect.signature(function).parameters:
        kwargs["worker_settings"] = worker_settings
    return function(*args, **kwargs)


def _call_with_supported_kwargs(function, *args, **kwargs):
    parameters = inspect.signature(function).parameters.values()
    if not any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters):
        supported = {parameter.name for parameter in parameters}
        kwargs = {name: value for name, value in kwargs.items() if name in supported}
    return function(*args, **kwargs)


_is_model_not_found_error = facade_compat.is_model_not_found_error
_claude_model_candidates = facade_compat.claude_model_candidates
_file_sha256 = facade_compat.file_sha256
claude_cache_file_for_source = cache_file_for_source
claude_prompt_sha256 = facade_compat.claude_prompt_sha256
claude_cache_options = facade_compat.claude_cache_options
load_claude_score_from_file_cache = facade_compat.load_claude_score_from_file_cache
save_claude_score_to_file_cache = facade_compat.save_claude_score_to_file_cache


def _resize_one_image(args):
    return resize.resize_one_image(args)


def resize_for_processing(src_folder, work_folder, worker_settings=None):
    return resize.resize_for_processing(
        src_folder,
        work_folder,
        worker_settings,
        resize_worker=_resize_one_image,
    )


HIST_DEDUP_THRESHOLD = deduplication.HIST_DEDUP_THRESHOLD
HIST_DEDUP_TEMPORAL_THRESHOLD = deduplication.HIST_DEDUP_TEMPORAL_THRESHOLD
HIST_DEDUP_TEMPORAL_ORB_THRESHOLD = deduplication.HIST_DEDUP_TEMPORAL_ORB_THRESHOLD
HIST_DEDUP_THUMB_SIZE = deduplication.HIST_DEDUP_THUMB_SIZE
BURST_MAX_INTERVAL_SEC = deduplication.BURST_MAX_INTERVAL_SEC
ORB_MATCH_THRESHOLD = deduplication.ORB_MATCH_THRESHOLD
_image_histogram = deduplication.image_histogram
_exif_timestamp = deduplication.exif_timestamp
_quick_sharpness = deduplication.quick_sharpness
_compute_phash = deduplication.compute_phash
_compute_orb_descriptors = deduplication.compute_orb_descriptors
_orb_match_ratio = deduplication.orb_match_ratio


def _compute_dedup_features(img_path):
    return deduplication.compute_dedup_features(
        img_path,
        histogram_function=_image_histogram,
        timestamp_function=_exif_timestamp,
        sharpness_function=_quick_sharpness,
        orb_function=_compute_orb_descriptors,
    )


def deduplicate(images, threshold=DEDUP_THRESHOLD, worker_settings=None):
    return deduplication.deduplicate(
        images,
        threshold,
        worker_settings,
        phash_function=_compute_phash,
        sharpness_function=_quick_sharpness,
        features_function=_compute_dedup_features,
        orb_ratio_function=_orb_match_ratio,
    )


def _detect_subject_mask(img):
    return technical_scoring.detect_subject_mask(img, yolo_detect_subject)


_composition_score = technical_scoring.composition_score
_horizon_tilt_penalty = technical_scoring.horizon_tilt_penalty
_lead_room_score = technical_scoring.lead_room_score
_colorfulness_metric = technical_scoring.colorfulness_metric
_print_score_distribution = technical_scoring.print_score_distribution
_tech_cache_path = technical_scoring.tech_cache_path
_json_safe_cache_value = technical_scoring.json_safe_cache_value
_technical_algorithm_fingerprint = technical_scoring.technical_algorithm_fingerprint


def score_technical(image_path):
    return technical_scoring.score_technical(image_path, detect_subject_mask=_detect_subject_mask)


def _load_tech_cache(img_path):
    return technical_scoring.load_tech_cache(img_path, fingerprint=_technical_algorithm_fingerprint)


def _save_tech_cache(img_path, tech):
    return technical_scoring.save_tech_cache(
        img_path,
        tech,
        fingerprint=_technical_algorithm_fingerprint,
    )


def _score_technical_with_cache(img_path):
    return technical_scoring.score_technical_with_cache(
        img_path,
        scorer=score_technical,
        loader=_load_tech_cache,
        saver=_save_tech_cache,
    )


def batch_technical_score(
    images, source_map=None, worker_settings=None, *, cache_observer=None, failure_observer=None
):
    return technical_scoring.batch_technical_score(
        images,
        source_map,
        worker_settings,
        loader=_load_tech_cache,
        score_cached=_score_technical_with_cache,
        distribution=_print_score_distribution,
        cache_observer=cache_observer,
        failure_observer=failure_observer,
    )


VISION_PROMPT_TEMPLATE = vision_prompts.VISION_PROMPT_TEMPLATE
OLLAMA_COMPACT_JSON_PROMPT_TEMPLATE = vision_prompts.OLLAMA_COMPACT_JSON_PROMPT_TEMPLATE
OLLAMA_SYSTEM_PROMPT = vision_prompts.OLLAMA_SYSTEM_PROMPT
OLLAMA_STRICT_JSON_SCHEMA = vision_prompts.OLLAMA_STRICT_JSON_SCHEMA
VISION_SCORE_KEYS = vision_prompts.VISION_SCORE_KEYS
build_vision_prompt = vision_prompts.build_vision_prompt
build_ollama_compact_json_prompt = vision_prompts.build_ollama_compact_json_prompt
_extract_account_context_from_prompt = vision_prompts.extract_account_context_from_prompt
_is_qwen_ollama_model = ollama_vision._is_qwen_ollama_model
_is_gemma4_ollama_model = ollama_vision._is_gemma4_ollama_model
_resolve_ollama_num_predict = ollama_vision._resolve_ollama_num_predict
_sanitize_vision_one_line = ollama_vision._sanitize_vision_one_line
_normalize_ollama_vision_payload = ollama_vision._normalize_ollama_vision_payload
_extract_json_payload = ollama_vision._extract_json_payload
_parse_ollama_message_json_with_mode = ollama_vision._parse_ollama_message_json_with_mode
_parse_ollama_message_json = ollama_vision._parse_ollama_message_json
_parse_ollama_plaintext_scores = ollama_vision._parse_ollama_plaintext_scores
_ollama_neutral_fallback_from_text = ollama_vision._ollama_neutral_fallback_from_text
_encode_image_for_ollama = ollama_vision._encode_image_for_ollama
_is_retryable_ollama_error = ollama_vision._is_retryable_ollama_error
_ollama_setup_hint = ollama_vision._ollama_setup_hint
_safe_float = facade_compat.safe_float
_claude_crop_gate_multiplier = facade_compat.claude_crop_gate_multiplier
_claude_setup_hint = facade_compat.claude_setup_hint


def score_with_claude(*args, **kwargs):
    kwargs.setdefault("detect_subject", yolo_detect_subject)
    return claude_vision.score_with_claude(*args, **kwargs)


def score_with_ollama(*args, **kwargs):
    kwargs.setdefault("detect_subject", yolo_detect_subject)
    kwargs.setdefault("encode_image", _encode_image_for_ollama)
    kwargs.setdefault("open_url", urlopen)
    return ollama_vision.score_with_ollama(*args, **kwargs)


def batch_vision_score(
    candidates,
    scorer="clip",
    env_search_dir=None,
    claude_model=None,
    rescore=False,
    cache_observer=None,
    preloaded_vision_cache=None,
    failure_observer=None,
):
    return vision_scoring.batch_vision_score(
        candidates,
        scorer=scorer,
        env_search_dir=env_search_dir,
        claude_model=claude_model,
        rescore=rescore,
        cache_observer=cache_observer,
        preloaded_vision_cache=preloaded_vision_cache,
        failure_observer=failure_observer,
        collaborators=vision_scoring.FacadeCollaborators.from_module(sys.modules[__name__]),
    )


_bbox_iou_xywh = crop_geometry._bbox_iou_xywh
_bbox_center_distance_ratio = crop_geometry._bbox_center_distance_ratio
_bbox_union_xywh = crop_geometry._bbox_union_xywh
_classify_shot_type = crop_geometry._classify_shot_type
_guess_facing_direction = crop_geometry._guess_facing_direction
_ideal_subject_x = crop_geometry._ideal_subject_x
_ideal_subject_y = crop_geometry._ideal_subject_y
_expand_subject_bbox = crop_geometry._expand_subject_bbox
_horizontal_margin_bounds = crop_geometry._horizontal_margin_bounds
_subject_side_gap_ratios = crop_geometry._subject_side_gap_ratios
_crop_uncertainty_flags = crop_geometry._crop_uncertainty_flags
_score_crop_candidate = crop_geometry._score_crop_candidate
_combine_rider_motorcycle_box = yolo_detection._combine_rider_motorcycle_box


def resolve_yolo_model_path(debug=False):
    return yolo_detection.resolve_yolo_model_path(debug=debug, downloader=urlretrieve)


def _load_yolo_model(debug=False):
    return yolo_detection._load_yolo_model(debug=debug, path_resolver=resolve_yolo_model_path)


def yolo_detect_subject(img, debug=False):
    return yolo_detection.yolo_detect_subject(img, debug=debug, model_loader=_load_yolo_model)


write_padded_full_subject = cropping.write_padded_full_subject


def smart_crop(
    image_path,
    output_path,
    out_w=OUTPUT_WIDTH,
    out_h=OUTPUT_HEIGHT,
    debug=False,
    save_debug=False,
    use_yolo=True,
    meta_out=None,
):
    return cropping.smart_crop(
        image_path,
        output_path,
        out_w=out_w,
        out_h=out_h,
        debug=debug,
        save_debug=save_debug,
        use_yolo=use_yolo,
        meta_out=meta_out,
        detector=yolo_detect_subject,
        guess_facing=_guess_facing_direction,
        expand_bbox=_expand_subject_bbox,
    )


def _prepare_claude_crop_first_candidates(candidates, *, work_folder):
    return cropping._prepare_claude_crop_first_candidates(
        candidates,
        work_folder=work_folder,
        cropper=smart_crop,
    )


def _facade_collaborators():
    return orchestration.FacadeCollaborators.from_module(sys.modules[__name__])


def run_dedup_only(input_folder, output_folder="selected", work_folder=None, worker_settings=None):
    return orchestration.run_dedup_only(
        input_folder,
        output_folder,
        work_folder,
        worker_settings,
        collaborators=_facade_collaborators(),
    )


def _crop_one_image_no_debug(args):
    return orchestration.crop_one_image_no_debug(args, cropper=smart_crop)


def _crop_one_image(args):
    return orchestration.crop_one_image(args, cropper=smart_crop)


def run_pipeline(
    input_folder,
    output_folder="selected",
    work_folder=None,
    top_n=10,
    scorer="clip",
    vision_candidates_pct=0.5,
    claude_model=None,
    score_all=False,
    claude_crop_first=False,
    rescore=False,
    worker_settings=None,
    _configure_opencv=True,
):
    return orchestration.run_pipeline(
        input_folder,
        output_folder,
        work_folder,
        top_n,
        scorer,
        vision_candidates_pct,
        claude_model,
        score_all,
        claude_crop_first,
        rescore,
        worker_settings,
        _configure_opencv,
        collaborators=_facade_collaborators(),
    )


def run_pipeline_recursive(
    input_folder,
    output_folder="selected",
    work_folder=None,
    top_n=10,
    scorer="clip",
    vision_candidates_pct=0.5,
    claude_model=None,
    score_all=False,
    claude_crop_first=False,
    rescore=False,
    worker_settings=None,
):
    collaborators = orchestration.RecursiveCollaborators(
        resolve_workers=resolve_worker_settings,
        configure_opencv=configure_opencv_threads,
        run_pipeline=run_pipeline,
        write_summary=_write_recursive_summary_report,
        generate_gallery_index=generate_gallery_index,
        supported_extensions=frozenset(SUPPORTED_EXTENSIONS),
    )
    return orchestration.run_pipeline_recursive(
        input_folder,
        output_folder,
        work_folder,
        top_n,
        scorer,
        vision_candidates_pct,
        claude_model,
        score_all,
        claude_crop_first,
        rescore,
        worker_settings,
        collaborators=collaborators,
    )


def main():
    from pickinsta.cli import CliCollaborators
    from pickinsta.cli import main as cli_main

    return cli_main(collaborators=CliCollaborators.from_selector(sys.modules[__name__]))


if __name__ == "__main__":
    main()
_COMPATIBILITY_REFERENCES = (
    os,
    shutil,
    time,
    cv2,
    np,
    Image,
    ImageScore,
    _clip_setup_hint,
    load_clip_model,
    score_with_clip,
    resolve_account_context,
    resolve_anthropic_api_key,
    resolve_claude_model,
    resolve_ollama_base_url,
    resolve_ollama_circuit_breaker_errors,
    resolve_ollama_concurrency,
    resolve_ollama_jpeg_quality,
    resolve_ollama_keep_alive,
    resolve_ollama_max_image_edge,
    resolve_ollama_max_retries,
    resolve_ollama_model,
    resolve_ollama_retry_backoff_seconds,
    resolve_ollama_timeout_seconds,
    resolve_ollama_use_yolo_context,
    resolve_optional_hf_token,
    _atomic_write_json,
    _atomic_write_text,
    _publish_managed_artifacts,
    bounded_worker_count,
    _generate_dedup_gallery,
    generate_gallery,
    write_markdown_report,
    RunTelemetry,
    build_run_manifest,
)
