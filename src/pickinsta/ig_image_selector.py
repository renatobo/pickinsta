#!/usr/bin/env python3
"""
Instagram Image Selection Pipeline
=================================================

Processes an event photo dump into Instagram-ready cover candidates.

Pipeline:
  1. Resize all images to max 1920px (longest edge) — saves compute
  2. Deduplicate near-identical and similar images
  3. Score technical quality (sharpness, exposure, contrast, saturation, composition)
  4. Score aesthetic/Instagram potential via Vision LLM (Claude or CLIP)
  5. Rank and output top N candidates cropped to 1080x1440

Usage:
  pickinsta ./input --output ./selected --top 10
  pickinsta ./input --scorer clip      # free, local
  python -m pickinsta ./input --output ./selected --scorer claude    # best quality, ~$0.50/100 imgs

Requirements:
  pip install Pillow opencv-python-headless numpy imagehash

  For CLIP scoring (free, local):
    pip install transformers torch

  For Claude scoring (best quality):
    pip install anthropic
    export ANTHROPIC_API_KEY=your_key_here
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen, urlretrieve

import cv2
import numpy as np
from PIL import Image
from pickinsta.clip_scorer import _clip_setup_hint, load_clip_model, score_with_clip

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RESIZE_PX = 1920  # longest edge for processing
OUTPUT_WIDTH = 1080  # Instagram output width
OUTPUT_HEIGHT = 1440  # Instagram output height (3:4 ratio)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".bmp"}
DEDUP_THRESHOLD = 8  # perceptual hash distance; lower = stricter
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_OLLAMA_MODEL = "qwen2.5vl:7b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CLAUDE_MIN_CROP4X5_OUTPUT_SCORE = 6.0
DEFAULT_ACCOUNT_CONTEXT = "motorcycle enthusiast account"
ACCOUNT_CONTEXT_ENV_VAR = "PICKINSTA_ACCOUNT_CONTEXT"
PICKINSTA_OLLAMA_BASE_URL_ENV_VAR = "PICKINSTA_OLLAMA_BASE_URL"
PICKINSTA_OLLAMA_MODEL_ENV_VAR = "PICKINSTA_OLLAMA_MODEL"
OLLAMA_TIMEOUT_ENV_VAR = "PICKINSTA_OLLAMA_TIMEOUT_SEC"
OLLAMA_MAX_EDGE_ENV_VAR = "PICKINSTA_OLLAMA_MAX_IMAGE_EDGE"
OLLAMA_JPEG_QUALITY_ENV_VAR = "PICKINSTA_OLLAMA_JPEG_QUALITY"
OLLAMA_KEEP_ALIVE_ENV_VAR = "PICKINSTA_OLLAMA_KEEP_ALIVE"
OLLAMA_USE_YOLO_ENV_VAR = "PICKINSTA_OLLAMA_USE_YOLO_CONTEXT"
OLLAMA_CONCURRENCY_ENV_VAR = "PICKINSTA_OLLAMA_CONCURRENCY"
OLLAMA_MAX_RETRIES_ENV_VAR = "PICKINSTA_OLLAMA_MAX_RETRIES"
OLLAMA_BACKOFF_BASE_ENV_VAR = "PICKINSTA_OLLAMA_RETRY_BACKOFF_SEC"
OLLAMA_CIRCUIT_BREAKER_ENV_VAR = "PICKINSTA_OLLAMA_CIRCUIT_BREAKER_ERRORS"
YOLO_MODEL_FILENAME = "yolov8n.pt"
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/latest/download/yolov8n.pt"
YOLO_MODEL_ENV_VAR = "PICKINSTA_YOLO_MODEL"
MIN_FRONT_EDGE_GAP_RATIO = 0.03
YOLO_BBOX_PAD_RATIO = 0.06
YOLO_BBOX_PAD_MIN_PX = 12
EDGE_RISK_SUBJECT_FILL_RATIO = 0.60
EDGE_RISK_TIGHT_GAP_MULT = 1.15
EDGE_RISK_ALT_GAP_MULT = 2.0
EDGE_RISK_ALT_MAX_SCORE_DELTA = 0.12
MIN_TOP_EDGE_GAP_RATIO = 0.03
CROP_UNCERTAIN_EDGE_GAP_RATIO = 0.02
CROP_UNCERTAIN_EDGE_GAP_MIN_PX = 8

_YOLO_MODEL = None


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _read_env_file(env_path: Path) -> dict[str, str]:
    """Parse a simple .env file into key/value pairs."""
    values: dict[str, str] = {}
    try:
        content = env_path.read_text(encoding="utf-8")
    except OSError:
        return values

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value

    return values


def resolve_anthropic_api_key(search_dir: Optional[Path] = None) -> str:
    """
    Resolve ANTHROPIC_API_KEY from environment first, then .env files.
    Checks current working directory and, if provided, search_dir.
    """
    env_value = os.environ.get("ANTHROPIC_API_KEY")
    if env_value:
        return env_value

    candidates = [Path.cwd() / ".env"]
    if search_dir is not None:
        candidates.append(search_dir / ".env")

    seen: set[Path] = set()
    for env_file in candidates:
        resolved = env_file.resolve()
        if resolved in seen or not env_file.exists():
            continue
        seen.add(resolved)

        values = _read_env_file(env_file)
        key = values.get("ANTHROPIC_API_KEY")
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            print(f"  🔐 Loaded ANTHROPIC_API_KEY from {env_file}")
            return key

    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. Set it in the environment or add it to a .env file."
    )


def resolve_optional_hf_token(search_dir: Optional[Path] = None) -> Optional[str]:
    """
    Resolve optional HF auth token from environment first, then .env files.
    Never raises when missing.
    """
    existing = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if existing:
        os.environ.setdefault("HF_TOKEN", existing)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", existing)
        return existing

    candidates = [Path.cwd() / ".env"]
    if search_dir is not None:
        candidates.append(search_dir / ".env")

    seen: set[Path] = set()
    for env_file in candidates:
        resolved = env_file.resolve()
        if resolved in seen or not env_file.exists():
            continue
        seen.add(resolved)

        values = _read_env_file(env_file)
        token = values.get("HF_TOKEN") or values.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
            print(f"  🔐 Loaded HF_TOKEN from {env_file} (optional)")
            return token

    return None


def resolve_claude_model(cli_model: Optional[str] = None) -> str:
    """Resolve Claude model from CLI override, env, or default."""
    if cli_model:
        return cli_model
    return (
        os.environ.get("ANTHROPIC_MODEL") or os.environ.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL
    )


def _resolve_env_string(var_name: str, search_dir: Optional[Path] = None) -> Optional[str]:
    """Resolve a string env var from environment first, then .env files."""
    env_value = (os.environ.get(var_name) or "").strip()
    if env_value:
        return env_value

    candidates = [Path.cwd() / ".env"]
    if search_dir is not None:
        candidates.append(search_dir / ".env")

    seen: set[Path] = set()
    for env_file in candidates:
        resolved = env_file.resolve()
        if resolved in seen or not env_file.exists():
            continue
        seen.add(resolved)

        values = _read_env_file(env_file)
        value = (values.get(var_name) or "").strip()
        if value:
            os.environ.setdefault(var_name, value)
            print(f"  📝 Loaded {var_name} from {env_file}")
            return value
    return None


def resolve_ollama_base_url(search_dir: Optional[Path] = None) -> str:
    """Resolve Ollama base URL from env or fallback default."""
    return (
        _resolve_env_string(PICKINSTA_OLLAMA_BASE_URL_ENV_VAR, search_dir=search_dir)
        or DEFAULT_OLLAMA_BASE_URL
    )


def resolve_ollama_model(search_dir: Optional[Path] = None) -> str:
    """Resolve Ollama model from env or fallback default."""
    return _resolve_env_string(PICKINSTA_OLLAMA_MODEL_ENV_VAR, search_dir=search_dir) or DEFAULT_OLLAMA_MODEL


def _resolve_env_int(var_name: str, default: int) -> int:
    raw = (os.environ.get(var_name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_env_float(var_name: str, default: float) -> float:
    raw = (os.environ.get(var_name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _resolve_env_bool(var_name: str, default: bool) -> bool:
    raw = (os.environ.get(var_name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def resolve_ollama_timeout_seconds() -> int:
    """Resolve request timeout for Ollama API calls."""
    return max(30, _resolve_env_int(OLLAMA_TIMEOUT_ENV_VAR, 300))


def resolve_ollama_max_image_edge() -> int:
    """Resolve max edge for image payload sent to Ollama."""
    return max(256, _resolve_env_int(OLLAMA_MAX_EDGE_ENV_VAR, 1024))


def resolve_ollama_jpeg_quality() -> int:
    """Resolve JPEG quality used for Ollama payload encoding."""
    return min(95, max(30, _resolve_env_int(OLLAMA_JPEG_QUALITY_ENV_VAR, 80)))


def resolve_ollama_keep_alive(search_dir: Optional[Path] = None) -> str:
    """Resolve Ollama keep_alive duration to keep model warm between requests."""
    return _resolve_env_string(OLLAMA_KEEP_ALIVE_ENV_VAR, search_dir=search_dir) or "10m"


def resolve_ollama_use_yolo_context() -> bool:
    """Resolve whether to run YOLO context before Ollama scoring."""
    return _resolve_env_bool(OLLAMA_USE_YOLO_ENV_VAR, False)


def resolve_ollama_concurrency() -> int:
    """Resolve parallel request count for Ollama scoring."""
    return min(16, max(1, _resolve_env_int(OLLAMA_CONCURRENCY_ENV_VAR, 2)))


def resolve_ollama_max_retries() -> int:
    """Resolve max retry attempts per Ollama request."""
    return min(8, max(0, _resolve_env_int(OLLAMA_MAX_RETRIES_ENV_VAR, 2)))


def resolve_ollama_retry_backoff_seconds() -> float:
    """Resolve base exponential backoff between retries."""
    return min(10.0, max(0.05, _resolve_env_float(OLLAMA_BACKOFF_BASE_ENV_VAR, 0.75)))


def resolve_ollama_circuit_breaker_errors() -> int:
    """Resolve consecutive-error threshold before halting new submissions."""
    return min(50, max(1, _resolve_env_int(OLLAMA_CIRCUIT_BREAKER_ENV_VAR, 6)))


def resolve_account_context(search_dir: Optional[Path] = None) -> str:
    """
    Resolve account context from environment first, then .env files.
    Returns a sensible default when not set.
    """
    env_value = (os.environ.get(ACCOUNT_CONTEXT_ENV_VAR) or "").strip()
    if env_value:
        return env_value

    candidates = [Path.cwd() / ".env"]
    if search_dir is not None:
        candidates.append(search_dir / ".env")

    seen: set[Path] = set()
    for env_file in candidates:
        resolved = env_file.resolve()
        if resolved in seen or not env_file.exists():
            continue
        seen.add(resolved)

        values = _read_env_file(env_file)
        context = (values.get(ACCOUNT_CONTEXT_ENV_VAR) or "").strip()
        if context:
            os.environ.setdefault(ACCOUNT_CONTEXT_ENV_VAR, context)
            print(f"  📝 Loaded {ACCOUNT_CONTEXT_ENV_VAR} from {env_file}")
            return context

    return DEFAULT_ACCOUNT_CONTEXT


def _is_model_not_found_error(error: Exception) -> bool:
    text = str(error).lower()
    return "not_found_error" in text and "model" in text


def _claude_model_candidates(preferred: str) -> list[str]:
    """Build a small ordered set of Claude model candidates."""
    candidates = [preferred]

    # If using a dated snapshot (e.g. *-20250514), also try the non-dated alias.
    parts = preferred.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        candidates.append(parts[0])

    # Stable fallback aliases.
    candidates.extend([DEFAULT_CLAUDE_MODEL, "claude-3-5-sonnet-latest"])

    unique: list[str] = []
    seen: set[str] = set()
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            unique.append(model)
    return unique


def _file_sha256(path: Path) -> str:
    """Return SHA256 hex digest for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def claude_cache_file_for_source(source_path: Path) -> Path:
    """Return per-image Claude cache path next to the original image."""
    return Path(str(source_path) + ".pickinsta.json")


def claude_prompt_sha256(prompt: str) -> str:
    """Stable hash of the Claude scoring prompt in use."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_claude_score_from_file_cache(
    *,
    source_path: Path,
    source_sha256: str,
    model: str,
    prompt_sha256: str,
) -> Optional[dict]:
    """Load cached Claude score for a specific source file if still valid."""
    cache_file = claude_cache_file_for_source(source_path)
    if not cache_file.exists():
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("source_sha256") != source_sha256:
        return None
    if payload.get("model") != model:
        return None
    if payload.get("prompt_sha256") != prompt_sha256:
        return None

    vision = payload.get("vision")
    if isinstance(vision, dict):
        return vision
    return None


def save_claude_score_to_file_cache(
    *,
    source_path: Path,
    source_sha256: str,
    model: str,
    prompt_sha256: str,
    vision: dict,
) -> None:
    """Persist Claude score cache for one source image."""
    cache_file = claude_cache_file_for_source(source_path)
    payload = {
        "source_file": str(source_path),
        "source_sha256": source_sha256,
        "model": model,
        "prompt_sha256": prompt_sha256,
        "vision": vision,
    }
    cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ImageScore:
    path: Path
    source_path: Optional[Path] = None
    technical: dict = field(default_factory=dict)
    vision: dict = field(default_factory=dict)
    final_score: float = 0.0
    one_line: str = ""


def _md_escape(value: object) -> str:
    """Escape values for markdown table cells."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def write_markdown_report(
    report_path: Path,
    *,
    input_folder: Path,
    output_folder: Path,
    scorer: str,
    top_n: int,
    selected_report: list[dict],
    analyzed_items: list[ImageScore],
) -> None:
    """Write a markdown report with top selections and full analyzed results."""
    lines: list[str] = []
    lines.append("# pickinsta Selection Report")
    lines.append("")
    lines.append(f"- Input: `{input_folder}`")
    lines.append(f"- Output: `{output_folder}`")
    lines.append(f"- Scorer: `{scorer}`")
    lines.append(f"- Top N requested: `{top_n}`")
    lines.append(f"- Images analyzed in vision stage: `{len(analyzed_items)}`")
    lines.append("")
    lines.append("## Top Selected Outputs")
    lines.append("")
    lines.append("| Rank | Filename | Final | Tech | Vision | Output | Summary |")
    lines.append("|---:|---|---:|---:|---:|---|---|")
    for row in selected_report:
        lines.append(
            "| "
            f"{row.get('rank', '')} | "
            f"{_md_escape(row.get('filename', ''))} | "
            f"{row.get('final_score', '')} | "
            f"{row.get('technical_composite', '')} | "
            f"{row.get('vision_total', '')} | "
            f"{_md_escape(row.get('output', ''))} | "
            f"{_md_escape(row.get('one_line', ''))} |"
        )

    lines.append("")
    section_title = (
        "Claude Analysis (All Images Analyzed)"
        if scorer == "claude"
        else "Vision Analysis (All Images Analyzed)"
    )
    lines.append(f"## {section_title}")
    lines.append("")
    lines.append(
        "| Rank | Filename | Final | Tech | Vision | Subject | Lighting | Color | Emotion | Scroll | Crop | Summary |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for idx, item in enumerate(analyzed_items, start=1):
        vision = item.vision or {}
        display_name = item.source_path.name if item.source_path else item.path.name
        lines.append(
            "| "
            f"{idx} | "
            f"{_md_escape(display_name)} | "
            f"{item.final_score:.4f} | "
            f"{item.technical.get('composite', 0):.4f} | "
            f"{vision.get('total', 0)} | "
            f"{vision.get('subject_clarity', '')} | "
            f"{vision.get('lighting', '')} | "
            f"{vision.get('color_pop', '')} | "
            f"{vision.get('emotion', '')} | "
            f"{vision.get('scroll_stop', '')} | "
            f"{vision.get('crop_4x5', '')} | "
            f"{_md_escape(item.one_line)} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Stage 0: Resize to max 1920px
# ---------------------------------------------------------------------------


def resize_for_processing(
    src_folder: Path, work_folder: Path
) -> tuple[list[Path], dict[Path, Path]]:
    """
    Copy all images to work_folder, resized so the longest edge <= MAX_RESIZE_PX.
    Preserves EXIF orientation. Returns list of resized image paths.
    """
    work_folder.mkdir(parents=True, exist_ok=True)
    resized: list[Path] = []
    source_map: dict[Path, Path] = {}
    reused = 0

    src_images = [
        p for p in sorted(src_folder.iterdir()) if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    print(f"📁 Found {len(src_images)} images in {src_folder}")

    for img_path in src_images:
        try:
            dest = work_folder / f"{img_path.stem}.jpg"
            # Reuse prior output when it is up-to-date to avoid re-encoding.
            if dest.exists() and dest.stat().st_mtime >= img_path.stat().st_mtime:
                resized.append(dest)
                source_map[dest] = img_path
                reused += 1
                continue

            with Image.open(img_path) as img:
                # Handle EXIF rotation
                from PIL import ImageOps

                img = ImageOps.exif_transpose(img)

                # Convert to RGB if needed (handles RGBA, P mode, etc.)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                w, h = img.size
                longest = max(w, h)

                if longest > MAX_RESIZE_PX:
                    scale = MAX_RESIZE_PX / longest
                    new_size = (int(w * scale), int(h * scale))
                    img = img.resize(new_size, Image.LANCZOS)

                img.save(dest, "JPEG", quality=90)
                resized.append(dest)
                source_map[dest] = img_path
        except Exception as e:
            print(f"  ⚠ Skipping {img_path.name}: {e}")

    newly_resized = len(resized) - reused
    print(
        f"  ✅ Prepared {len(resized)} images (new: {newly_resized}, reused: {reused})"
        f" to max {MAX_RESIZE_PX}px → {work_folder}"
    )
    return resized, source_map


# ---------------------------------------------------------------------------
# Stage 1: Deduplication (perceptual hashing)
# ---------------------------------------------------------------------------


def deduplicate(images: list[Path], threshold: int = DEDUP_THRESHOLD) -> list[Path]:
    """
    Remove near-duplicate images using perceptual hashing.
    Returns one representative from each group of similar images.
    """
    import imagehash

    hash_groups: dict[imagehash.ImageHash, list[Path]] = {}

    for img_path in images:
        try:
            h = imagehash.phash(Image.open(img_path), hash_size=16)
            placed = False
            for existing_hash, group in hash_groups.items():
                if abs(h - existing_hash) <= threshold:
                    group.append(img_path)
                    placed = True
                    break
            if not placed:
                hash_groups[h] = [img_path]
        except Exception as e:
            print(f"  ⚠ Hash failed for {img_path.name}: {e}")

    # From each group, pick the largest file (usually highest quality)
    unique = []
    duplicates_removed = 0
    for group in hash_groups.values():
        best = max(group, key=lambda p: p.stat().st_size)
        unique.append(best)
        duplicates_removed += len(group) - 1

    print(
        f"  ✅ Dedup: {len(images)} → {len(unique)} unique ({duplicates_removed} duplicates removed)"
    )
    return unique


# ---------------------------------------------------------------------------
# Stage 2: Technical quality scoring
# ---------------------------------------------------------------------------


def _detect_subject_mask(img: np.ndarray) -> Optional[np.ndarray]:
    """Return a binary mask for the primary subject using YOLO, or None."""
    detection = yolo_detect_subject(img, debug=False)
    if detection is None:
        return None
    x, y, w, h, _cls, _conf = detection
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return mask


def _composition_score(
    center_x_norm: float,
    center_y_norm: float,
) -> float:
    """Score subject placement against Rule-of-Thirds and Phi Grid power points.

    Returns 0.0–1.0 using a smooth Gaussian falloff.  The best of the two
    grid systems is used (per composition rules: weight equally, take better).
    Dead-center placement (0.5, 0.5) is penalized unless very close to a
    power point.
    """
    thirds_points = [
        (0.333, 0.333),
        (0.667, 0.333),
        (0.333, 0.667),
        (0.667, 0.667),
    ]
    phi_points = [
        (0.382, 0.382),
        (0.618, 0.382),
        (0.382, 0.618),
        (0.618, 0.618),
    ]

    def _min_dist(pts):
        return min(np.hypot(center_x_norm - px, center_y_norm - py) for px, py in pts)

    best_dist = min(_min_dist(thirds_points), _min_dist(phi_points))
    # Normalize by image diagonal (sqrt(2) for a unit square) and apply
    # Gaussian-like decay; sigma chosen so that ±0.05 from a power point
    # scores ~0.95 and dead-center (~0.18 away) scores ~0.55.
    sigma = 0.12
    score = float(np.exp(-0.5 * (best_dist / sigma) ** 2))
    return score


def _horizon_tilt_penalty(gray: np.ndarray) -> float:
    """Estimate horizon tilt and return a 0.0–1.0 score (1.0 = level)."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=gray.shape[1] // 4, maxLineGap=20
    )
    if lines is None or len(lines) == 0:
        return 0.8  # no strong lines → assume acceptable

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # Only consider near-horizontal lines (within 15° of horizontal)
        if angle < 15 or angle > 165:
            angles.append(min(angle, 180 - angle))

    if not angles:
        return 0.8

    median_tilt = float(np.median(angles))
    # Penalize > 2° deviation (per composition rules)
    if median_tilt <= 2.0:
        return 1.0
    return max(0.0, 1.0 - (median_tilt - 2.0) / 10.0)


def _lead_room_score(
    img: np.ndarray,
    subject_mask: Optional[np.ndarray],
) -> float:
    """Evaluate lead room (facing space).

    Uses the horizontal gradient of the subject region to guess facing
    direction, then checks that 55–75% of horizontal space is ahead.
    Returns 0.0–1.0.
    """
    if subject_mask is None:
        return 0.5  # neutral when we can't tell

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Subject center-x
    cols = np.where(subject_mask.any(axis=0))[0]
    if len(cols) == 0:
        return 0.5
    cx = float(cols.mean()) / w

    # Rough facing direction from horizontal gradient within subject
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    masked = sobel_x.copy()
    masked[subject_mask == 0] = 0
    mean_grad = masked.mean()
    # positive gradient → features brighter on right → facing right
    facing_right = mean_grad >= 0

    if facing_right:
        space_ahead = 1.0 - cx
    else:
        space_ahead = cx

    # Ideal: 60–70% ahead (per composition rules); acceptable: 55–75%.
    if 0.55 <= space_ahead <= 0.75:
        return 1.0
    if 0.45 <= space_ahead <= 0.85:
        return 0.7
    return 0.3


def _colorfulness_metric(img: np.ndarray) -> float:
    """Hasler-Süsstrunk colorfulness metric, normalized to 0–1."""
    B, G, R = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    sigma = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mu = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    C = sigma + 0.3 * mu
    # Typical range 0–120; normalize so ~60 maps to 0.8
    return min(C / 80.0, 1.0)


def score_technical(image_path: Path) -> dict:
    """Score image on 7 metrics aligned with the composition-rules rubric.

    Metrics (all 0.0–1.0):
      sharpness          — Laplacian variance on subject region (weight 0.18)
      background_sep     — subject-to-bg sharpness ratio (weight 0.12)
      composition        — distance to nearest power point + horizon + lead room (weight 0.20)
      lighting           — histogram clipping + mean luminance (weight 0.18)
      color_harmony      — Hasler-Süsstrunk colorfulness + subject-bg contrast (weight 0.13)
      visual_clutter     — inverse edge density in background (weight 0.12)
      aesthetic          — proxy from contrast + saturation balance (weight 0.07)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    scores: dict[str, float] = {}

    # --- Detect subject for region-aware metrics ---
    subject_mask = _detect_subject_mask(img)
    bg_mask = None
    if subject_mask is not None:
        bg_mask = cv2.bitwise_not(subject_mask)

    # --- Metric 1: Sharpness (weight 0.18) ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    if subject_mask is not None:
        subj_lap = laplacian.copy()
        subj_lap[subject_mask == 0] = 0
        n_subj = max(1, int(subject_mask.sum() / 255))
        subj_var = float((subj_lap**2).sum() / n_subj)
    else:
        subj_var = float(laplacian.var())
    # Scale: 500+ on a 1080px-wide image is tack-sharp
    scale_factor = w / 1080.0
    scores["sharpness"] = min(subj_var / (500.0 * scale_factor), 1.0)

    # --- Metric 2: Background separation (weight 0.12) ---
    if subject_mask is not None and bg_mask is not None:
        bg_lap = laplacian.copy()
        bg_lap[bg_mask == 0] = 0
        n_bg = max(1, int(bg_mask.sum() / 255))
        bg_var = float((bg_lap**2).sum() / n_bg)
        ratio = subj_var / max(bg_var, 1e-6)
        # 3:1 = good (0.6), 5:1+ = excellent (1.0)
        scores["background_sep"] = min(ratio / 5.0, 1.0)
    else:
        scores["background_sep"] = 0.5  # neutral

    # --- Metric 3: Composition adherence (weight 0.20) ---
    # 3a) Subject placement on power points
    if subject_mask is not None:
        rows, cols = np.where(subject_mask > 0)
        if len(rows) > 0:
            cx_norm = float(cols.mean()) / w
            cy_norm = float(rows.mean()) / h
        else:
            cx_norm, cy_norm = 0.5, 0.5
    else:
        cx_norm, cy_norm = 0.5, 0.5

    placement = _composition_score(cx_norm, cy_norm)
    # 3b) Horizon tilt
    tilt = _horizon_tilt_penalty(gray)
    # 3c) Lead room
    lead = _lead_room_score(img, subject_mask)
    scores["composition"] = 0.50 * placement + 0.25 * tilt + 0.25 * lead

    # --- Metric 4: Lighting quality (weight 0.18) ---
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_px = hist.sum()
    clip_black = hist[:5].sum() / total_px  # pure black
    clip_white = hist[250:].sum() / total_px  # pure white
    clipping_penalty = min(clip_black + clip_white, 0.04) / 0.04  # < 2% each = good
    mean_lum = float(gray.mean())
    # Ideal mean luminance: 90–170 (per composition rules)
    if 90 <= mean_lum <= 170:
        lum_score = 1.0
    elif 60 <= mean_lum <= 200:
        lum_score = 0.7
    else:
        lum_score = 0.4
    scores["lighting"] = lum_score * (1.0 - clipping_penalty)

    # --- Metric 5: Color harmony (weight 0.13) ---
    colorfulness = _colorfulness_metric(img)
    # Subject-background color contrast
    if subject_mask is not None and bg_mask is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        subj_hue = hsv[:, :, 0][subject_mask > 0].mean() if (subject_mask > 0).any() else 0
        bg_hue = hsv[:, :, 0][bg_mask > 0].mean() if (bg_mask > 0).any() else 0
        hue_diff = abs(float(subj_hue) - float(bg_hue))
        if hue_diff > 90:
            hue_diff = 180 - hue_diff
        # Bigger hue difference = better contrast; ideal ~30-90
        hue_contrast = min(hue_diff / 60.0, 1.0)
    else:
        hue_contrast = 0.5
    scores["color_harmony"] = 0.6 * colorfulness + 0.4 * hue_contrast

    # --- Metric 6: Visual clutter (weight 0.12) ---
    edges = cv2.Canny(gray, 50, 150)
    if bg_mask is not None:
        bg_edges = edges.copy()
        bg_edges[bg_mask == 0] = 0
        n_bg_px = max(1, int(bg_mask.sum() / 255))
        bg_edge_density = float(bg_edges.sum() / 255) / n_bg_px
    else:
        bg_edge_density = float(edges.sum() / 255) / (h * w)
    # Lower edge density = cleaner background = higher score
    scores["visual_clutter"] = max(0.0, 1.0 - bg_edge_density * 10.0)

    # --- Metric 7: Overall aesthetic proxy (weight 0.07) ---
    contrast = min(gray.std() / 80.0, 1.0)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv_img[:, :, 1].mean() / 255.0
    # Moderate saturation (0.3-0.7) scores highest
    sat_balance = 1.0 - abs(sat - 0.5) * 2.0
    scores["aesthetic"] = 0.5 * contrast + 0.5 * sat_balance

    # --- Weighted composite (per composition rules) ---
    weights = {
        "sharpness": 0.18,
        "background_sep": 0.12,
        "composition": 0.20,
        "lighting": 0.18,
        "color_harmony": 0.13,
        "visual_clutter": 0.12,
        "aesthetic": 0.07,
    }
    scores["composite"] = sum(scores[k] * weights[k] for k in weights)

    return scores


def _print_score_distribution(results: list[ImageScore]) -> None:
    """Print a summary table and ASCII histogram of technical composite scores."""
    composites = [r.technical.get("composite", 0.0) for r in results]
    arr = np.array(composites)

    # Summary statistics
    print(f"  📈 Score distribution (n={len(arr)}):")
    print(
        f"     min={arr.min():.3f}  max={arr.max():.3f}  "
        f"mean={arr.mean():.3f}  median={float(np.median(arr)):.3f}  "
        f"std={arr.std():.3f}"
    )

    # Per-metric averages
    metric_names = [
        "sharpness",
        "background_sep",
        "composition",
        "lighting",
        "color_harmony",
        "visual_clutter",
        "aesthetic",
    ]
    avgs = {}
    for m in metric_names:
        vals = [r.technical.get(m, 0.0) for r in results]
        avgs[m] = np.mean(vals) if vals else 0.0
    parts = [f"{m}={avgs[m]:.2f}" for m in metric_names]
    print(f"     metric avgs: {', '.join(parts)}")

    # 10-bin histogram (0.0–1.0)
    counts, edges = np.histogram(arr, bins=10, range=(0.0, 1.0))
    max_count = int(counts.max()) if counts.max() > 0 else 1
    bar_width = 30  # max chars for the longest bar
    print("     ┌" + "─" * (bar_width + 18) + "┐")
    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"     │ {lo:.1f}–{hi:.1f} │ {bar:<{bar_width}} {count:>3} │")
    print("     └" + "─" * (bar_width + 18) + "┘")


def batch_technical_score(
    images: list[Path], source_map: Optional[dict[Path, Path]] = None
) -> list[ImageScore]:
    """Score all images technically and return sorted list."""
    results = []
    for img_path in images:
        try:
            tech = score_technical(img_path)
            results.append(
                ImageScore(
                    path=img_path,
                    source_path=source_map.get(img_path) if source_map else None,
                    technical=tech,
                )
            )
        except Exception as e:
            print(f"  ⚠ Tech score failed for {img_path.name}: {e}")

    results.sort(key=lambda x: x.technical.get("composite", 0), reverse=True)
    print(
        f"  ✅ Technical scoring complete. Top: {results[0].path.name} ({results[0].technical['composite']:.3f})"
    )
    _print_score_distribution(results)
    return results


# ---------------------------------------------------------------------------
# Stage 3a: Vision LLM scoring — Claude API
# ---------------------------------------------------------------------------

VISION_PROMPT_TEMPLATE = """Score this motorcycle photo for Instagram cover potential.

Context: {account_context}.

Rate each criterion 1-10 using these professional composition guidelines:

1. SUBJECT_CLARITY: Is the motorcycle/rider the clear focal point? Does it stand out
   from the background at thumbnail size? Subject-to-background sharpness ratio ≥3:1
   is good; busy backgrounds that compete for attention score low.

2. LIGHTING: Quality of light — golden hour or dramatic low-sun light is ideal.
   Penalize if chrome/metallic highlights are blown out, or if > 2% of pixels are
   clipped black/white. Mean luminance should feel balanced (not flat midday).

3. COLOR_POP: Evaluate color harmony (complementary/analogous/triadic schemes).
   Does the bike's color contrast with the background? Orange bike on blue sky = high;
   matching bike-and-background colors = low. Moderate, consistent saturation preferred.

4. EMOTION: Does the image convey motion, tension, power, or aspiration? Low camera
   angles (tank/axle level) make bikes look powerful; standing eye-height shots score
   lower. 3/4 view is the most flattering standard angle.

5. SCROLL_STOP: Would this image stop fast-scrolling on Instagram? Consider: dramatic
   composition, strong leading lines, clean negative space, and clear visual hierarchy.

6. CROP_4x5: Can this be cropped to 3:4 portrait (1080x1440) while maintaining good
   composition? Consider:
   - Is the subject placed near a rule-of-thirds power point (not dead center)?
   - Is there adequate lead room (60-70% of space ahead of the motorcycle's facing direction)?
   - Would cropping to portrait cut off wheels, handlebars, or exhaust?
   - Would the subject remain well-composed in Instagram's 3:4 grid thumbnail?

Return ONLY valid JSON, no markdown:
{{"subject_clarity": N, "lighting": N, "color_pop": N, "emotion": N, "scroll_stop": N, "crop_4x5": N, "total": N, "one_line": "why this works or doesn't"}}"""


def build_vision_prompt(account_context: str) -> str:
    """Build the Claude vision prompt with account-specific context."""
    context = account_context.strip() or DEFAULT_ACCOUNT_CONTEXT
    return VISION_PROMPT_TEMPLATE.format(account_context=context)


def score_with_claude(
    image_path: Path,
    api_key: Optional[str] = None,
    model: str = DEFAULT_CLAUDE_MODEL,
    client=None,
    use_yolo_context: bool = True,
    prompt: Optional[str] = None,
) -> dict:
    """
    Score a single image using Claude's vision API.

    Args:
        image_path: Path to image to score
        api_key: Claude API key
        model: Claude model to use
        client: Optional pre-initialized Claude client
        use_yolo_context: If True, detect subjects with YOLO and enhance prompt
    """
    import anthropic

    if client is None:
        client = anthropic.Anthropic(api_key=api_key)

    # Optional: Detect subjects with YOLO to enhance Claude's context
    yolo_context = ""
    if use_yolo_context:
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                detection = yolo_detect_subject(img, debug=False)
                if detection:
                    x, y, w, h, class_name, conf = detection
                    img_h, img_w = img.shape[:2]

                    # Calculate relative position
                    center_x = (x + w / 2) / img_w
                    center_y = (y + h / 2) / img_h
                    size_ratio = (w * h) / (img_w * img_h)

                    # Describe position
                    h_pos = "left" if center_x < 0.33 else "right" if center_x > 0.66 else "center"
                    v_pos = "top" if center_y < 0.33 else "bottom" if center_y > 0.66 else "middle"
                    position = (
                        f"{v_pos}-{h_pos}" if v_pos != "middle" or h_pos != "center" else "centered"
                    )

                    # Describe size
                    size_desc = (
                        "large" if size_ratio > 0.3 else "medium" if size_ratio > 0.1 else "small"
                    )

                    yolo_context = f"\n\n**Detected Subject**: {class_name} ({position}, {size_desc}, confidence: {conf:.0%})"
        except Exception:
            # Silently fail - YOLO context is optional
            pass

    # Read and encode — images are already resized to max 1920px
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    suffix = image_path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"

    # Enhance prompt with YOLO detection context if available
    enhanced_prompt = prompt or build_vision_prompt(DEFAULT_ACCOUNT_CONTEXT)
    if yolo_context:
        enhanced_prompt = enhanced_prompt + yolo_context

    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_data},
                    },
                    {"type": "text", "text": enhanced_prompt},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    return json.loads(_extract_json_payload(raw))


def _extract_json_payload(raw_text: str) -> str:
    """Extract JSON body from a model response that may include code fences."""
    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw


def _parse_ollama_message_json(body: str, parsed: dict) -> dict:
    """Parse JSON model output from Ollama chat `message.content` or `message.thinking`."""
    message = parsed.get("message") if isinstance(parsed, dict) else None
    if not isinstance(message, dict):
        raise RuntimeError(f"Ollama response did not include message object: {body[:300]}")

    content = str(message.get("content") or "").strip()
    thinking = str(message.get("thinking") or "").strip()
    for text in (content, thinking):
        if not text:
            continue
        try:
            return json.loads(_extract_json_payload(text))
        except Exception:
            continue

    for text in (content, thinking):
        fallback = _parse_ollama_plaintext_scores(text)
        if fallback is not None:
            return fallback

    for text in (content, thinking):
        fallback = _ollama_neutral_fallback_from_text(text)
        if fallback is not None:
            return fallback

    raise RuntimeError(
        "Ollama response did not include parseable JSON in message content/thinking: "
        f"{body[:300]}"
    )


def _parse_ollama_plaintext_scores(raw_text: str) -> Optional[dict]:
    """Fallback parser for rubric-like plain text when model ignores JSON format."""
    text = (raw_text or "").strip()
    if not text:
        return None

    key_patterns = {
        "subject_clarity": r"subject[\s_-]*clarity",
        "lighting": r"lighting",
        "color_pop": r"color[\s_-]*pop",
        "emotion": r"emotion",
        "scroll_stop": r"scroll[\s_-]*stop",
        "crop_4x5": r"crop[\s_-]*4x5",
    }

    def _extract_score(label_pattern: str) -> Optional[float]:
        patterns = [
            rf"(?is){label_pattern}\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
            rf"(?is){label_pattern}[^0-9]{{0,48}}([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
            rf"(?is){label_pattern}\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)",
            rf"(?is){label_pattern}[^0-9]{{0,48}}([0-9]+(?:\.[0-9]+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    continue
        return None

    scores: dict[str, int] = {}
    for key, label_pattern in key_patterns.items():
        score = _extract_score(label_pattern)
        if score is None:
            continue
        scores[key] = int(max(0, min(10, round(score))))

    if len(scores) < 3:
        return None

    present_values = list(scores.values())
    fill_value = int(round(sum(present_values) / len(present_values))) if present_values else 5
    fill_value = max(0, min(10, fill_value))
    for key in key_patterns:
        scores.setdefault(key, fill_value)

    total_match = re.search(r"(?is)\btotal\s*[:=-]?\s*([0-9]+(?:\.[0-9]+)?)", text)
    if total_match:
        total = int(max(0, min(60, round(float(total_match.group(1))))))
    else:
        total = sum(scores[k] for k in key_patterns)

    one_line = ""
    one_line_match = re.search(r"(?is)\bone[\s_-]*line\s*[:=-]\s*(.+)", text)
    if one_line_match:
        one_line = one_line_match.group(1).strip().splitlines()[0]
    if not one_line:
        first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        one_line = first_sentence[:180] if first_sentence else "Vision scoring summary"

    return {
        "subject_clarity": scores["subject_clarity"],
        "lighting": scores["lighting"],
        "color_pop": scores["color_pop"],
        "emotion": scores["emotion"],
        "scroll_stop": scores["scroll_stop"],
        "crop_4x5": scores["crop_4x5"],
        "total": total,
        "one_line": one_line,
    }


def _ollama_neutral_fallback_from_text(raw_text: str) -> Optional[dict]:
    """Last-resort fallback when model returns prose without structured numeric output."""
    text = (raw_text or "").strip()
    if not text:
        return None

    values: list[float] = []
    for match in re.finditer(r"(?is)\b([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b", text):
        try:
            values.append(float(match.group(1)))
        except Exception:
            continue
    if values:
        values.sort()
        mid = len(values) // 2
        base = values[mid] if len(values) % 2 == 1 else (values[mid - 1] + values[mid]) / 2.0
        score = int(max(0, min(10, round(base))))
    else:
        score = 5

    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    one_line = first_sentence[:180] if first_sentence else "Vision response prose; neutral fallback"
    total = int(max(0, min(60, score * 6)))

    return {
        "subject_clarity": score,
        "lighting": score,
        "color_pop": score,
        "emotion": score,
        "scroll_stop": score,
        "crop_4x5": score,
        "total": total,
        "one_line": one_line,
    }


def _encode_image_for_ollama(
    image_path: Path,
    *,
    max_edge: int,
    jpeg_quality: int,
) -> str:
    """Encode image as base64, downscaling/compressing to reduce Ollama payload size."""
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            longest = max(w, h)
            if longest > max_edge:
                scale = max_edge / float(longest)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")


def score_with_ollama(
    image_path: Path,
    base_url: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    use_yolo_context: bool = True,
    prompt: Optional[str] = None,
    timeout_seconds: int = 300,
    max_image_edge: int = 1024,
    jpeg_quality: int = 80,
    keep_alive: str = "10m",
) -> dict:
    """Score a single image using Ollama's vision API (/api/chat)."""
    yolo_context = ""
    if use_yolo_context:
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                detection = yolo_detect_subject(img, debug=False)
                if detection:
                    x, y, w, h, class_name, conf = detection
                    img_h, img_w = img.shape[:2]
                    center_x = (x + w / 2) / img_w
                    center_y = (y + h / 2) / img_h
                    size_ratio = (w * h) / (img_w * img_h)
                    h_pos = "left" if center_x < 0.33 else "right" if center_x > 0.66 else "center"
                    v_pos = "top" if center_y < 0.33 else "bottom" if center_y > 0.66 else "middle"
                    position = (
                        f"{v_pos}-{h_pos}" if v_pos != "middle" or h_pos != "center" else "centered"
                    )
                    size_desc = (
                        "large" if size_ratio > 0.3 else "medium" if size_ratio > 0.1 else "small"
                    )
                    yolo_context = (
                        f"\n\n**Detected Subject**: {class_name} "
                        f"({position}, {size_desc}, confidence: {conf:.0%})"
                    )
        except Exception:
            pass

    image_data = _encode_image_for_ollama(
        image_path,
        max_edge=max_image_edge,
        jpeg_quality=jpeg_quality,
    )

    enhanced_prompt = prompt or build_vision_prompt(DEFAULT_ACCOUNT_CONTEXT)
    if yolo_context:
        enhanced_prompt = enhanced_prompt + yolo_context

    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": keep_alive,
        "options": {"temperature": 0, "num_predict": 220},
        "messages": [{"role": "user", "content": enhanced_prompt, "images": [image_data]}],
    }

    endpoint = f"{base_url.rstrip('/')}/api/chat"
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as error:
        details = ""
        try:
            details = error.read().decode("utf-8", errors="ignore")
        except Exception:
            details = ""
        raise RuntimeError(f"Ollama request failed ({error.code}) at {endpoint}: {details[:300]}") from error
    except URLError as error:
        raise RuntimeError(f"Ollama connection failed for {endpoint}: {error}") from error

    parsed = json.loads(body)
    return _parse_ollama_message_json(body, parsed)


def _is_retryable_ollama_error(error: Exception) -> bool:
    text = str(error).lower()
    retryable_markers = [
        "timed out",
        "timeout",
        "connection failed",
        "connection reset",
        "temporarily unavailable",
        "429",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(marker in text for marker in retryable_markers)


def _claude_setup_hint(error: Exception) -> str:
    """Return a practical setup hint for common Claude initialization failures."""
    text = str(error).lower()
    if "no module named" in text or "import" in text:
        return (
            "Install Claude dependency in your active environment:\n"
            "  python -m pip install -e '.[claude]'"
        )
    if _is_model_not_found_error(error):
        return (
            "Claude model not found for your account.\n"
            f"Try: --claude-model {DEFAULT_CLAUDE_MODEL}\n"
            "or set ANTHROPIC_MODEL in your environment/.env."
        )
    return (
        "Claude initialization failed. Verify dependency and API key setup.\n"
        'Run: python -c "import anthropic; print(anthropic.__version__)".'
    )


def _ollama_setup_hint(error: Exception) -> str:
    """Return a practical setup hint for common Ollama initialization failures."""
    return (
        "Ollama setup failed. Verify PICKINSTA_OLLAMA_BASE_URL points to a running Ollama server,\n"
        "and that PICKINSTA_OLLAMA_MODEL is already pulled on that server (for example: qwen2.5vl:7b).\n"
        "You can increase client timeout with PICKINSTA_OLLAMA_TIMEOUT_SEC.\n"
        f"Original error: {error}"
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    """Best-effort float conversion with default fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _claude_crop_gate_multiplier(vision: dict) -> float:
    """Strongly gate ranking by Claude crop_4x5 confidence."""
    crop_score = _safe_float(vision.get("crop_4x5"), default=0.0)
    if crop_score <= 4.0:
        return 0.15
    if crop_score <= 5.0:
        return 0.35
    if crop_score <= 6.0:
        return 0.60
    if crop_score <= 7.0:
        return 0.80
    return 1.0


# ---------------------------------------------------------------------------
# Stage 3 dispatcher
# ---------------------------------------------------------------------------


def batch_vision_score(
    candidates: list[ImageScore],
    scorer: str = "clip",
    env_search_dir: Optional[Path] = None,
    claude_model: Optional[str] = None,
) -> list[ImageScore]:
    """Run vision scoring on candidate images."""

    clip_model, clip_processor = None, None
    claude_api_key = None
    claude_client = None
    claude_model_used = None
    ollama_model_used = None
    ollama_base_url = None
    ollama_timeout_seconds = resolve_ollama_timeout_seconds()
    ollama_max_image_edge = resolve_ollama_max_image_edge()
    ollama_jpeg_quality = resolve_ollama_jpeg_quality()
    ollama_keep_alive = "10m"
    ollama_use_yolo_context = resolve_ollama_use_yolo_context()
    ollama_concurrency = resolve_ollama_concurrency()
    ollama_max_retries = resolve_ollama_max_retries()
    ollama_retry_backoff = resolve_ollama_retry_backoff_seconds()
    ollama_circuit_breaker_errors = resolve_ollama_circuit_breaker_errors()
    claude_base_prompt = build_vision_prompt(DEFAULT_ACCOUNT_CONTEXT)
    claude_prompt_hash = claude_prompt_sha256(claude_base_prompt)
    claude_cache_hits = 0
    claude_api_calls = 0
    if scorer == "clip":
        resolve_optional_hf_token(search_dir=env_search_dir)
        print("  Loading CLIP model (first run downloads ~1.7GB)...")
        try:
            clip_model, clip_processor = load_clip_model()
        except Exception as e:
            print(f"  ⚠ CLIP unavailable: {e}")
            print(f"  💡 {_clip_setup_hint(e)}")
            print("  ↪ Falling back to technical-only ranking for this run.")
            for item in candidates:
                item.final_score = item.technical.get("composite", 0.0)
                item.one_line = "CLIP unavailable — ranked by technical score only"
            candidates.sort(key=lambda x: x.final_score, reverse=True)
            return candidates
    elif scorer == "claude":
        claude_api_key = resolve_anthropic_api_key(search_dir=env_search_dir)
        try:
            import anthropic

            claude_client = anthropic.Anthropic(api_key=claude_api_key)

            preferred_model = resolve_claude_model(cli_model=claude_model)
            last_error = None
            for candidate in _claude_model_candidates(preferred_model):
                try:
                    # Preflight once to avoid repeated 404s per image.
                    claude_client.messages.create(
                        model=candidate,
                        max_tokens=1,
                        messages=[{"role": "user", "content": "ok"}],
                    )
                    claude_model_used = candidate
                    if candidate != preferred_model:
                        print(f"  ↪ Claude model fallback: {preferred_model} -> {candidate}")
                    break
                except Exception as e:
                    last_error = e
                    if _is_model_not_found_error(e):
                        continue
                    raise

            if claude_model_used is None:
                raise RuntimeError(
                    f"No available Claude model found. Tried: {', '.join(_claude_model_candidates(preferred_model))}. "
                    f"Last error: {last_error}"
                )

            account_context = resolve_account_context(search_dir=env_search_dir)
            claude_base_prompt = build_vision_prompt(account_context)
            claude_prompt_hash = claude_prompt_sha256(claude_base_prompt)
        except Exception as e:
            print(f"  ⚠ Claude unavailable: {e}")
            print(f"  💡 {_claude_setup_hint(e)}")
            print("  ↪ Falling back to technical-only ranking for this run.")
            for item in candidates:
                item.final_score = item.technical.get("composite", 0.0)
                item.one_line = "Claude unavailable — ranked by technical score only"
            candidates.sort(key=lambda x: x.final_score, reverse=True)
            return candidates
    elif scorer == "ollama":
        try:
            ollama_base_url = resolve_ollama_base_url(search_dir=env_search_dir)
            ollama_model_used = resolve_ollama_model(search_dir=env_search_dir)
            ollama_keep_alive = resolve_ollama_keep_alive(search_dir=env_search_dir)
            request = Request(
                f"{ollama_base_url.rstrip('/')}/api/tags",
                headers={"Accept": "application/json"},
                method="GET",
            )
            with urlopen(request, timeout=min(60, ollama_timeout_seconds)):
                pass

            account_context = resolve_account_context(search_dir=env_search_dir)
            claude_base_prompt = build_vision_prompt(account_context)
            claude_prompt_hash = claude_prompt_sha256(claude_base_prompt)
        except Exception as e:
            print(f"  ⚠ Ollama unavailable: {e}")
            print(f"  💡 {_ollama_setup_hint(e)}")
            print("  ↪ Falling back to technical-only ranking for this run.")
            for item in candidates:
                item.final_score = item.technical.get("composite", 0.0)
                item.one_line = "Ollama unavailable — ranked by technical score only"
            candidates.sort(key=lambda x: x.final_score, reverse=True)
            return candidates

    if scorer == "ollama":
        progress_write = print
        progress_bar = None
        try:
            from tqdm.auto import tqdm

            progress_bar = tqdm(
                total=len(candidates),
                desc="  Ollama scoring",
                unit="img",
            )
            progress_write = tqdm.write
        except Exception as e:
            print(f"  ⚠ Progress bar unavailable: {e}")

        def finalize_item(item: ImageScore, vision: dict) -> None:
            vision_normalized = vision.get("total", 30) / 60.0
            base_score = item.technical["composite"] * 0.3 + vision_normalized * 0.7
            gate = _claude_crop_gate_multiplier(vision)
            item.vision = vision
            item.final_score = base_score * gate
            item.one_line = vision.get("one_line", "")

        def mark_failed(item: ImageScore, message: str) -> None:
            item.final_score = item.technical["composite"] * 0.3
            item.one_line = message

        def score_one_with_retry(item: ImageScore) -> dict:
            last_error = None
            for attempt in range(ollama_max_retries + 1):
                try:
                    return score_with_ollama(
                        item.path,
                        base_url=ollama_base_url or DEFAULT_OLLAMA_BASE_URL,
                        model=ollama_model_used or DEFAULT_OLLAMA_MODEL,
                        use_yolo_context=ollama_use_yolo_context,
                        prompt=claude_base_prompt,
                        timeout_seconds=ollama_timeout_seconds,
                        max_image_edge=ollama_max_image_edge,
                        jpeg_quality=ollama_jpeg_quality,
                        keep_alive=ollama_keep_alive,
                    )
                except Exception as e:
                    last_error = e
                    if attempt >= ollama_max_retries or not _is_retryable_ollama_error(e):
                        break
                    sleep_seconds = ollama_retry_backoff * (2**attempt)
                    time.sleep(sleep_seconds)
            raise RuntimeError(
                f"Ollama scoring failed after {ollama_max_retries + 1} attempt(s): {last_error}"
            ) from last_error

        scored = 0
        failed = 0
        consecutive_failures = 0
        stop_submissions = False
        next_index = 0
        total_items = len(candidates)
        pending: dict = {}

        with ThreadPoolExecutor(max_workers=ollama_concurrency) as executor:
            while next_index < total_items and len(pending) < ollama_concurrency:
                future = executor.submit(score_one_with_retry, candidates[next_index])
                pending[future] = next_index
                next_index += 1

            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    idx = pending.pop(future)
                    item = candidates[idx]
                    try:
                        vision = future.result()
                        finalize_item(item, vision)
                        scored += 1
                        consecutive_failures = 0
                    except Exception as e:
                        progress_write(f"  ⚠ Vision score failed for {item.path.name}: {e}")
                        mark_failed(item, "Vision scoring failed — ranked by technical score only")
                        failed += 1
                        consecutive_failures += 1
                    if progress_bar is not None:
                        progress_bar.update(1)

                while (
                    not stop_submissions
                    and next_index < total_items
                    and len(pending) < ollama_concurrency
                ):
                    if consecutive_failures >= ollama_circuit_breaker_errors:
                        stop_submissions = True
                        progress_write(
                            "  ⚠ Ollama circuit breaker opened due to consecutive failures; "
                            "remaining images will use technical-only fallback."
                        )
                        break
                    future = executor.submit(score_one_with_retry, candidates[next_index])
                    pending[future] = next_index
                    next_index += 1

        if stop_submissions and next_index < total_items:
            remaining = candidates[next_index:]
            for item in remaining:
                mark_failed(item, "Ollama circuit breaker active — ranked by technical score only")
            failed += len(remaining)
            if progress_bar is not None:
                progress_bar.update(len(remaining))

        if progress_bar is not None:
            progress_bar.close()

        yolo_label = "on" if ollama_use_yolo_context else "off"
        print(f"  🖥️  Ollama server: {ollama_base_url} | model: {ollama_model_used}")
        print(
            "  ⚙️  Ollama tuning: "
            f"timeout={ollama_timeout_seconds}s, max_edge={ollama_max_image_edge}px, "
            f"jpeg_quality={ollama_jpeg_quality}, keep_alive={ollama_keep_alive}, yolo={yolo_label}"
        )
        print(
            "  🔁 Ollama resilience: "
            f"concurrency={ollama_concurrency}, retries={ollama_max_retries}, "
            f"retry_backoff={ollama_retry_backoff:.2f}s, "
            f"circuit_breaker_errors={ollama_circuit_breaker_errors}"
        )
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        print(f"  ✅ Vision scoring: {scored} scored, {failed} failed")
        return candidates

    scored = 0
    failed = 0
    score_iter = candidates
    progress_write = print
    active_claude_model = claude_model_used or resolve_claude_model(cli_model=claude_model)
    if scorer == "claude":
        try:
            from tqdm.auto import tqdm

            score_iter = tqdm(
                candidates,
                total=len(candidates),
                desc=f"  {scorer.capitalize()} scoring",
                unit="img",
            )
            progress_write = tqdm.write
        except Exception as e:
            print(f"  ⚠ Progress bar unavailable: {e}")

    for item in score_iter:
        try:
            if scorer == "claude":
                source_for_cache = item.source_path or item.path
                source_sha256 = _file_sha256(source_for_cache)
                cached = load_claude_score_from_file_cache(
                    source_path=source_for_cache,
                    source_sha256=source_sha256,
                    model=active_claude_model,
                    prompt_sha256=claude_prompt_hash,
                )
                if cached is not None:
                    item.vision = cached
                    claude_cache_hits += 1
                else:
                    item.vision = score_with_claude(
                        item.path,
                        api_key=claude_api_key,
                        model=active_claude_model,
                        client=claude_client,
                        use_yolo_context=True,  # Enable YOLO context for Claude scoring
                        prompt=claude_base_prompt,
                    )
                    try:
                        save_claude_score_to_file_cache(
                            source_path=source_for_cache,
                            source_sha256=source_sha256,
                            model=active_claude_model,
                            prompt_sha256=claude_prompt_hash,
                            vision=item.vision,
                        )
                    except Exception as e:
                        progress_write(
                            f"  ⚠ Could not write cache for {source_for_cache.name}: {e}"
                        )
                    claude_api_calls += 1
            else:
                item.vision = score_with_clip(item.path, clip_model, clip_processor)

            # Final composite: 30% technical + 70% vision
            vision_normalized = item.vision.get("total", 30) / 60.0
            base_score = item.technical["composite"] * 0.3 + vision_normalized * 0.7
            if scorer == "claude":
                gate = _claude_crop_gate_multiplier(item.vision)
                item.final_score = base_score * gate
            else:
                item.final_score = base_score
            item.one_line = item.vision.get("one_line", "")
            scored += 1
        except Exception as e:
            progress_write(f"  ⚠ Vision score failed for {item.path.name}: {e}")
            item.final_score = item.technical["composite"] * 0.3
            item.one_line = "Vision scoring failed — ranked by technical score only"
            failed += 1

    if scorer == "claude" and hasattr(score_iter, "close"):
        score_iter.close()
    if scorer == "claude":
        print(f"  📦 Claude cache hits: {claude_cache_hits} | API calls: {claude_api_calls}")

    candidates.sort(key=lambda x: x.final_score, reverse=True)
    print(f"  ✅ Vision scoring: {scored} scored, {failed} failed")
    return candidates


# ---------------------------------------------------------------------------
# Stage 4: Smart crop to 1080x1440
# ---------------------------------------------------------------------------


def resolve_yolo_model_path(debug: bool = False) -> Path:
    """
    Resolve YOLO model path from env override or runtime cache.

    Priority:
      1. PICKINSTA_YOLO_MODEL (explicit local path)
      2. ~/.cache/pickinsta/models/yolov8n.pt (downloaded on first use)
    """
    override = os.environ.get(YOLO_MODEL_ENV_VAR)
    if override:
        model_path = Path(override).expanduser()
        if debug:
            print(f"Using YOLO model from {YOLO_MODEL_ENV_VAR}: {model_path}")
        return model_path

    cache_dir = Path.home() / ".cache" / "pickinsta" / "models"
    model_path = cache_dir / YOLO_MODEL_FILENAME
    if model_path.exists():
        return model_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        print(f"YOLO model not found; downloading to {model_path} ...")
    try:
        urlretrieve(YOLO_MODEL_URL, model_path)
    except Exception as e:
        raise RuntimeError(
            f"Could not download YOLO model to {model_path}. "
            f"Set {YOLO_MODEL_ENV_VAR} to a local model path to skip download."
        ) from e
    return model_path


def _load_yolo_model(debug: bool = False):
    """Load and cache YOLO model instance."""
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    from ultralytics import YOLO

    model_path = resolve_yolo_model_path(debug=debug)
    _YOLO_MODEL = YOLO(str(model_path))
    return _YOLO_MODEL


def _bbox_iou_xywh(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """IoU for two XYWH boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def _bbox_center_distance_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> float:
    """Center distance normalized by frame diagonal."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    dist = float(np.hypot(acx - bcx, acy - bcy))
    diag = float(np.hypot(max(1, img_w), max(1, img_h)))
    return dist / max(diag, 1.0)


def _bbox_union_xywh(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Union box for two XYWH boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return x1, y1, x2 - x1, y2 - y1


def _combine_rider_motorcycle_box(
    best_detection: tuple[int, int, int, int, str, float],
    detections: list[tuple[int, int, int, int, str, float]],
    *,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int, str, float]:
    """Combine nearby person + motorcycle detections into a full rider+bike subject box."""
    bx, by, bw, bh, bcls, bconf = best_detection
    if bcls not in {"person", "motorcycle"}:
        return best_detection

    counterpart_cls = "motorcycle" if bcls == "person" else "person"
    base_box = (bx, by, bw, bh)

    best_pair = None
    best_pair_score = 0.0
    for dx, dy, dw, dh, dcls, dconf in detections:
        if dcls != counterpart_cls:
            continue
        other_box = (dx, dy, dw, dh)
        iou = _bbox_iou_xywh(base_box, other_box)
        dist_ratio = _bbox_center_distance_ratio(base_box, other_box, img_w, img_h)
        # Require overlap or reasonably close proximity.
        if iou < 0.01 and dist_ratio > 0.35:
            continue
        pair_score = float(dconf) * (1.0 + iou * 1.8 - dist_ratio * 0.8)
        if pair_score > best_pair_score:
            best_pair_score = pair_score
            best_pair = (dx, dy, dw, dh, dcls, dconf)

    if best_pair is None:
        return best_detection

    ox, oy, ow, oh, _ocls, oconf = best_pair
    ux, uy, uw, uh = _bbox_union_xywh(base_box, (ox, oy, ow, oh))
    return ux, uy, uw, uh, "rider_motorcycle", max(bconf, oconf)


def yolo_detect_subject(img: np.ndarray, debug: bool = False):
    """
    Use YOLOv8 to detect subjects (people, vehicles, animals, etc.) in the image.
    Returns the bounding box of the most prominent detection, or None if nothing found.

    Args:
        img: OpenCV image (BGR format)
        debug: If True, print detection info

    Returns:
        Tuple of (x, y, w, h, class_name, confidence) or None
    """
    try:
        model = _load_yolo_model(debug=debug)
    except ImportError:
        if debug:
            print("YOLO not available (ultralytics not installed)")
        return None
    except Exception as e:
        if debug:
            print(f"YOLO model setup failed: {e}")
        return None

    try:
        # Run inference
        results = model(img, verbose=False)

        if not results or len(results) == 0:
            return None

        # Get detections from first result
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return None

        # Priority classes for subject detection (most likely to be the main subject)
        # COCO dataset class IDs:
        priority_classes = {
            0: "person",  # People are usually the main subject
            1: "bicycle",
            2: "car",
            3: "motorcycle",  # Perfect for your use case!
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            16: "dog",
            17: "cat",
            18: "horse",
        }

        # Find the best detection (prioritize certain classes, then by confidence and size)
        all_detections: list[tuple[int, int, int, int, str, float]] = []
        best_detection = None
        best_score = 0.0

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Calculate bounding box area
            area = (x2 - x1) * (y2 - y1)

            # Score = confidence * area_ratio * class_priority
            img_area = img.shape[0] * img.shape[1]
            area_ratio = area / img_area

            # Give priority to relevant classes; prioritize motorcycle slightly.
            if cls_id == 3:
                class_priority = 2.25
            elif cls_id in priority_classes:
                class_priority = 2.0
            else:
                class_priority = 1.0

            score = conf * area_ratio * class_priority

            class_name = priority_classes.get(cls_id, f"class_{cls_id}")
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            all_detections.append((x, y, w, h, class_name, conf))

            if score > best_score:
                best_score = score
                best_detection = (x, y, w, h, class_name, conf)

        if best_detection:
            best_detection = _combine_rider_motorcycle_box(
                best_detection,
                all_detections,
                img_w=img.shape[1],
                img_h=img.shape[0],
            )

        if debug and best_detection:
            x, y, w, h, class_name, conf = best_detection
            print(f"YOLO detected: {class_name} (conf={conf:.2f}) at bbox=({x}, {y}, {w}, {h})")

        return best_detection

    except Exception as e:
        if debug:
            print(f"YOLO detection failed: {e}")
        return None


def _classify_shot_type(subject_area_ratio: float) -> str:
    """Classify shot type from subject area as fraction of frame."""
    if subject_area_ratio >= 0.50:
        return "close-up"
    if subject_area_ratio >= 0.20:
        return "medium"
    if subject_area_ratio >= 0.10:
        return "environmental"
    if subject_area_ratio >= 0.05:
        return "scenic"
    return "extreme_wide"


def _guess_facing_direction(img: np.ndarray, sx: int, sy: int, sw: int, sh: int) -> str:
    """Guess whether the subject faces left, right, or is head-on.

    Uses multiple signals inside the subject bbox:
      1. Edge density in left vs right outer thirds (robust for side profiles)
      2. Directional Sobel energy as fallback
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    subj_region = gray[sy : sy + sh, sx : sx + sw]
    if subj_region.size == 0:
        return "unknown"

    # Heuristic 1: edge asymmetry near outer thirds.
    edges = cv2.Canny(subj_region, 60, 160)
    third = max(1, subj_region.shape[1] // 3)
    left_edge_energy = float(edges[:, :third].mean())
    right_edge_energy = float(edges[:, -third:].mean())
    if left_edge_energy > right_edge_energy * 1.05:
        return "left"
    if right_edge_energy > left_edge_energy * 1.05:
        return "right"

    # Heuristic 2: directional Sobel energy.
    sobel = cv2.Sobel(subj_region, cv2.CV_64F, 1, 0, ksize=3)
    pos_energy = float(np.maximum(sobel, 0).mean())
    neg_energy = float(np.maximum(-sobel, 0).mean())
    if pos_energy > neg_energy * 1.05:
        return "right"
    if neg_energy > pos_energy * 1.05:
        return "left"
    return "head-on"


def _ideal_subject_x(facing: str, shot_type: str) -> float:
    """Return ideal normalized x for the subject center, respecting lead room.

    Per composition rules:
      - rightward-facing: x ≈ 0.33 (60-70% space ahead on right)
      - leftward-facing:  x ≈ 0.67
      - head-on:          x ≈ 0.50
      - close-up/extreme: x ≈ 0.50 (less lead room needed)
    """
    if shot_type in ("close-up", "extreme_wide"):
        return 0.50
    if facing == "right":
        return 0.35
    if facing == "left":
        return 0.65
    return 0.50


def _ideal_subject_y(shot_type: str) -> float:
    """Return ideal normalized y for the subject center.

    Per composition rules:
      - low-angle / environmental: y ≈ 0.60-0.67 (lower-third area)
      - medium / eye-level:         y ≈ 0.50
      - close-up:                   y ≈ 0.50
    """
    if shot_type in ("environmental", "scenic", "extreme_wide"):
        return 0.62
    return 0.50


def _expand_subject_bbox(
    sx: int,
    sy: int,
    sw: int,
    sh: int,
    img_w: int,
    img_h: int,
    class_name: str = "",
) -> tuple[int, int, int, int]:
    """Expand a detected subject bbox to avoid overly-tight crops."""
    if class_name == "rider_motorcycle":
        pad_ratio_x = max(YOLO_BBOX_PAD_RATIO, 0.10)
        pad_ratio_y = max(YOLO_BBOX_PAD_RATIO, 0.08)
    elif class_name == "motorcycle":
        pad_ratio_x = max(YOLO_BBOX_PAD_RATIO, 0.08)
        pad_ratio_y = max(YOLO_BBOX_PAD_RATIO, 0.07)
    else:
        pad_ratio_x = YOLO_BBOX_PAD_RATIO
        pad_ratio_y = YOLO_BBOX_PAD_RATIO

    pad_x = max(YOLO_BBOX_PAD_MIN_PX, int(round(sw * pad_ratio_x)))
    pad_y = max(YOLO_BBOX_PAD_MIN_PX, int(round(sh * pad_ratio_y)))

    x0 = max(0, sx - pad_x)
    y0 = max(0, sy - pad_y)
    x1 = min(img_w, sx + sw + pad_x)
    y1 = min(img_h, sy + sh + pad_y)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _horizontal_margin_bounds(
    sx: int,
    sw: int,
    crop_w: int,
    frame_w: int,
    min_gap_px: int,
) -> Optional[tuple[int, int]]:
    """Return crop start bounds that keep subject inside with side breathing room."""
    if sw + 2 * min_gap_px > crop_w:
        return None
    min_start = max(0, sx + sw + min_gap_px - crop_w)
    max_start = min(frame_w - crop_w, sx - min_gap_px)
    if min_start <= max_start:
        return int(min_start), int(max_start)
    return None


def _subject_side_gap_ratios(crop_x: int, crop_w: int, sx: int, sw: int) -> tuple[float, float, float]:
    """Return (left_gap, right_gap, min_gap) normalized to crop width."""
    denom = max(1.0, float(crop_w))
    left_gap = (sx - crop_x) / denom
    right_gap = ((crop_x + crop_w) - (sx + sw)) / denom
    return left_gap, right_gap, min(left_gap, right_gap)


def _crop_uncertainty_flags(
    *,
    sx: int,
    sy: int,
    sw: int,
    sh: int,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    img_w: int,
    img_h: int,
) -> dict[str, object]:
    """Return crop uncertainty flags for deciding if padded fallback should be emitted."""
    left_gap_px = sx - crop_x
    right_gap_px = (crop_x + crop_w) - (sx + sw)
    top_gap_px = sy - crop_y
    bottom_gap_px = (crop_y + crop_h) - (sy + sh)
    min_gap_px = min(left_gap_px, right_gap_px, top_gap_px, bottom_gap_px)

    min_gap_ratio = min(
        left_gap_px / max(1.0, float(crop_w)),
        right_gap_px / max(1.0, float(crop_w)),
        top_gap_px / max(1.0, float(crop_h)),
        bottom_gap_px / max(1.0, float(crop_h)),
    )
    gap_px_threshold = max(
        CROP_UNCERTAIN_EDGE_GAP_MIN_PX,
        int(round(min(crop_w, crop_h) * CROP_UNCERTAIN_EDGE_GAP_RATIO)),
    )

    too_large_for_crop = sw > crop_w or sh > crop_h
    too_close_to_border = min_gap_px < gap_px_threshold or min_gap_ratio < CROP_UNCERTAIN_EDGE_GAP_RATIO
    clipped_subject = min_gap_px < 0
    subject_bbox_hits_frame_edge = sx <= 0 or sy <= 0 or (sx + sw) >= img_w or (sy + sh) >= img_h

    reasons: list[str] = []
    if too_large_for_crop:
        reasons.append("subject_larger_than_crop")
    if clipped_subject:
        reasons.append("subject_clipped_by_crop")
    if too_close_to_border:
        reasons.append("subject_too_close_to_crop_border")
    if subject_bbox_hits_frame_edge:
        reasons.append("subject_bbox_hits_image_edge")

    return {
        "too_large_for_crop": too_large_for_crop,
        "too_close_to_border": too_close_to_border,
        "clipped_subject": clipped_subject,
        "subject_bbox_hits_frame_edge": subject_bbox_hits_frame_edge,
        "edge_gap_px_threshold": int(gap_px_threshold),
        "min_subject_gap_px": int(min_gap_px),
        "min_subject_gap_ratio": float(min_gap_ratio),
        "uncertain_crop": len(reasons) > 0,
        "uncertain_crop_reasons": reasons,
    }


def _score_crop_candidate(
    img: np.ndarray,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    sx: int,
    sy: int,
    sw: int,
    sh: int,
    facing: str,
) -> float:
    """Score a candidate crop window against composition rules.

    Evaluates: subject on power-point, lead room ratio, subject not clipped.
    """
    # Subject center relative to the crop window
    subj_cx = (sx + sw / 2 - crop_x) / crop_w
    subj_cy = (sy + sh / 2 - crop_y) / crop_h

    # 1) Placement on power points (Thirds + Phi)
    placement = _composition_score(subj_cx, subj_cy)

    # 2) Lead room (horizontal space ahead of facing direction)
    if facing == "right":
        space_ahead = 1.0 - subj_cx
    elif facing == "left":
        space_ahead = subj_cx
    else:
        space_ahead = 0.65  # head-on: neutral

    if 0.55 <= space_ahead <= 0.75:
        lead_score = 1.0
    elif 0.45 <= space_ahead <= 0.85:
        lead_score = 0.7
    else:
        lead_score = 0.3

    # 3) Proportional clipping penalty (small clip != severe clip)
    overlap_left = max(sx, crop_x)
    overlap_top = max(sy, crop_y)
    overlap_right = min(sx + sw, crop_x + crop_w)
    overlap_bottom = min(sy + sh, crop_y + crop_h)
    overlap_w = max(0, overlap_right - overlap_left)
    overlap_h = max(0, overlap_bottom - overlap_top)
    overlap_ratio = (overlap_w * overlap_h) / max(1.0, float(sw * sh))
    clip_score = overlap_ratio**1.8

    # 4) Keep a small breathing gap between subject front and border.
    min_gap = MIN_FRONT_EDGE_GAP_RATIO
    left_gap = (sx - crop_x) / max(1.0, float(crop_w))
    right_gap = ((crop_x + crop_w) - (sx + sw)) / max(1.0, float(crop_w))
    if facing == "left":
        front_gap = left_gap
        rear_gap = right_gap
    elif facing == "right":
        front_gap = right_gap
        rear_gap = left_gap
    else:
        front_gap = min(left_gap, right_gap)
        rear_gap = front_gap
    front_gap_score = float(np.clip(front_gap / max(min_gap, 1e-6), 0.0, 1.0))
    rear_gap_score = float(np.clip(rear_gap / max(min_gap, 1e-6), 0.0, 1.0))
    edge_gap_score = 0.7 * front_gap_score + 0.3 * rear_gap_score

    return 0.15 * placement + 0.15 * lead_score + 0.55 * clip_score + 0.15 * edge_gap_score


def write_padded_full_subject(
    image_path: Path,
    output_path: Path,
    out_w: int = OUTPUT_WIDTH,
    out_h: int = OUTPUT_HEIGHT,
) -> Path:
    """Write an uncropped portrait variant by fitting the full image with blurred padding."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read {image_path}")

    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    fit_w = max(1, int(round(w * scale)))
    fit_h = max(1, int(round(h * scale)))

    fit_interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    fit = cv2.resize(img, (fit_w, fit_h), interpolation=fit_interp)

    bg = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=18, sigmaY=18)

    x0 = (out_w - fit_w) // 2
    y0 = (out_h - fit_h) // 2
    bg[y0 : y0 + fit_h, x0 : x0 + fit_w] = fit
    cv2.imwrite(str(output_path), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path


def smart_crop(
    image_path: Path,
    output_path: Path,
    out_w: int = OUTPUT_WIDTH,
    out_h: int = OUTPUT_HEIGHT,
    debug: bool = False,
    save_debug: bool = False,
    use_yolo: bool = True,
    meta_out: Optional[dict[str, object]] = None,
) -> Path:
    """Crop and resize image to out_w x out_h using composition rules.

    Steps (per composition rules algorithm):
      1. Detect subject with YOLO (fallback: saliency).
      2. Classify shot type from subject area.
      3. Determine facing direction and ideal placement (lead room,
         rule-of-thirds / Phi Grid power points).
      4. Generate candidate crop windows and score each.
      5. Select the highest-scoring crop that keeps the subject intact.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read {image_path}")

    h, w = img.shape[:2]
    target_ratio = out_w / out_h  # 0.75 for 1080x1440
    current_ratio = w / h

    # --- Step 1: Detect subject ---
    sx, sy, sw, sh = None, None, None, None
    detection_method = "none"
    class_name = ""
    conf = 0.0
    raw_bbox: Optional[tuple[int, int, int, int]] = None
    expanded_bbox: Optional[tuple[int, int, int, int]] = None

    if use_yolo:
        yolo_result = yolo_detect_subject(img, debug=debug)
        if yolo_result:
            sx, sy, sw, sh, class_name, conf = yolo_result
            raw_bbox = (sx, sy, sw, sh)
            sx, sy, sw, sh = _expand_subject_bbox(sx, sy, sw, sh, w, h, class_name=class_name)
            expanded_bbox = (sx, sy, sw, sh)
            detection_method = f"yolo ({class_name})"

    # Fallback to saliency
    if sx is None:
        if debug:
            print("YOLO found nothing, falling back to saliency detection...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            saliency_det = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency_det.computeSaliency(gray)
            if not success:
                raise RuntimeError("Saliency computation failed")
            saliency_map = (saliency_map * 255).astype(np.uint8)
        except Exception:
            saliency_map = cv2.Canny(gray, 50, 150)

        _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            sx, sy, sw, sh = cv2.boundingRect(largest)
            expanded_bbox = (sx, sy, sw, sh)
            detection_method = "saliency"
        else:
            # Absolute fallback: image center
            sx, sy, sw, sh = w // 4, h // 4, w // 2, h // 2
            expanded_bbox = (sx, sy, sw, sh)
            detection_method = "center_fallback"

    center_x = sx + sw // 2
    center_y = sy + sh // 2

    if debug:
        print(
            f"Detection: {detection_method} | bbox=({sx},{sy},{sw},{sh}) | center=({center_x},{center_y}) | img={w}x{h}"
        )
        debug_img = img.copy()
        cv2.rectangle(debug_img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 5)
        cv2.circle(debug_img, (center_x, center_y), 15, (0, 0, 255), -1)
        if class_name:
            cv2.putText(
                debug_img,
                f"{class_name}",
                (sx, sy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )

    # --- Step 2: Classify shot type ---
    subject_area_ratio = (sw * sh) / (w * h)
    shot_type = _classify_shot_type(subject_area_ratio)

    # --- Step 3: Determine facing direction and ideal placement ---
    facing = _guess_facing_direction(img, sx, sy, sw, sh)
    ideal_x_norm = _ideal_subject_x(facing, shot_type)
    ideal_y_norm = _ideal_subject_y(shot_type)

    if debug:
        print(
            f"Shot type: {shot_type} | Facing: {facing} | Ideal placement: ({ideal_x_norm:.2f}, {ideal_y_norm:.2f})"
        )

    # --- Step 4: Generate candidate crop windows and pick the best ---
    selected_candidate_name = "no_crop"
    crop_origin_x = 0
    crop_origin_y = 0
    crop_w = w
    crop_h = h

    if abs(current_ratio - target_ratio) < 0.01:
        # Already the right ratio — no crop needed
        cropped = img
    elif current_ratio > target_ratio:
        # Too wide → crop sides
        new_w = int(h * target_ratio)
        min_edge_gap_px = max(8, int(new_w * MIN_FRONT_EDGE_GAP_RATIO))
        margin_bounds = _horizontal_margin_bounds(sx, sw, new_w, w, min_edge_gap_px)

        # Candidate A: place subject at ideal x (rule-of-thirds with lead room)
        ideal_x_start = int(center_x - ideal_x_norm * new_w)
        ideal_x_start = max(0, min(ideal_x_start, w - new_w))

        # Candidate B: ensure full subject inclusion (old behavior)
        safe_x_start = center_x - new_w // 2
        # Adjust to include entire subject bbox
        if safe_x_start > sx:
            safe_x_start = sx
        if safe_x_start + new_w < sx + sw:
            safe_x_start = sx + sw - new_w
        safe_x_start = max(0, min(safe_x_start, w - new_w))

        # Candidate C: front-preserve with a breathing gap from the border.
        if facing == "left":
            front_preserve_x_start = sx - min_edge_gap_px
        elif facing == "right":
            front_preserve_x_start = sx + sw + min_edge_gap_px - new_w
        else:
            front_preserve_x_start = ideal_x_start
        front_preserve_x_start = max(0, min(front_preserve_x_start, w - new_w))

        # Candidate D: pure center
        center_x_start = max(0, (w - new_w) // 2)

        candidates = [
            ("front_preserve", front_preserve_x_start),
            ("ideal", ideal_x_start),
            ("safe", safe_x_start),
            ("center", center_x_start),
        ]

        # Keep candidates within side-gap-safe bounds when feasible.
        if margin_bounds is not None:
            lo, hi = margin_bounds
            bounded: list[tuple[str, int]] = []
            seen: set[tuple[str, int]] = set()
            for name, cx_start in candidates:
                bounded_x = int(np.clip(cx_start, lo, hi))
                key = (name, bounded_x)
                if key not in seen:
                    seen.add(key)
                    bounded.append(key)
            candidates = bounded

        # If the subject is wider than crop width, clipping is unavoidable.
        # Prefer preserving the subject's front side with a small border gap.
        if sw > new_w:
            # Close-up subjects are often quasi-symmetric in the detected box.
            # Direction inference is unstable here, so avoid one-sided crops.
            if shot_type == "close-up":
                best_name, best_x = "ideal", ideal_x_start
            else:
                best_name, best_x = "front_preserve", front_preserve_x_start
            best_score = _score_crop_candidate(img, best_x, 0, new_w, h, sx, sy, sw, sh, facing)
            if debug:
                if shot_type == "close-up":
                    print(
                        f"  Subject wider than crop ({sw}>{new_w}) and shot is close-up; "
                        "forcing centered ideal crop to avoid wrong-side clipping."
                    )
                else:
                    print(
                        f"  Subject wider than crop ({sw}>{new_w}); forcing front-preserve "
                        f"with min edge gap {min_edge_gap_px}px."
                    )
        else:
            best_name, best_x = "center", center_x_start
            best_score = -1.0
            front_score = None
            candidate_scores: dict[str, tuple[int, float]] = {}
            for name, cx_start in candidates:
                s = _score_crop_candidate(img, cx_start, 0, new_w, h, sx, sy, sw, sh, facing)
                candidate_scores[name] = (cx_start, s)
                if debug:
                    print(f"  Crop candidate '{name}': x_start={cx_start}, score={s:.3f}")
                if name == "front_preserve":
                    front_score = s
                if s > best_score:
                    best_score = s
                    best_name = name
                    best_x = cx_start

            # Front-preserve is preferred when it is near-optimal.
            if (
                front_score is not None
                and facing in {"left", "right"}
                and front_score >= (best_score - 0.05)
            ):
                best_name = "front_preserve"
                best_x = front_preserve_x_start
                best_score = front_score

            # If the top-scoring crop rides a border with a large subject, prefer
            # a near-scoring alternative with materially better side breathing room.
            subject_fill_ratio = sw / max(1.0, float(new_w))
            _left_gap, _right_gap, best_min_gap = _subject_side_gap_ratios(best_x, new_w, sx, sw)
            tight_gap = MIN_FRONT_EDGE_GAP_RATIO * EDGE_RISK_TIGHT_GAP_MULT
            roomy_gap = MIN_FRONT_EDGE_GAP_RATIO * EDGE_RISK_ALT_GAP_MULT
            if subject_fill_ratio >= EDGE_RISK_SUBJECT_FILL_RATIO and best_min_gap <= tight_gap:
                alternative_name = None
                alternative_x = None
                alternative_score = None
                alternative_min_gap = None
                for name, (cx_start, score) in candidate_scores.items():
                    if name == best_name:
                        continue
                    _lg, _rg, min_gap = _subject_side_gap_ratios(cx_start, new_w, sx, sw)
                    if min_gap < roomy_gap:
                        continue
                    if score < (best_score - EDGE_RISK_ALT_MAX_SCORE_DELTA):
                        continue
                    if (
                        alternative_score is None
                        or score > alternative_score
                        or (abs(score - alternative_score) < 1e-6 and min_gap > (alternative_min_gap or -1.0))
                    ):
                        alternative_name = name
                        alternative_x = cx_start
                        alternative_score = score
                        alternative_min_gap = min_gap
                if alternative_name is not None and alternative_x is not None and alternative_score is not None:
                    if debug:
                        print(
                            "  Edge-risk override: replacing "
                            f"'{best_name}' with '{alternative_name}' "
                            f"(best_min_gap={best_min_gap:.3f}, alt_min_gap={alternative_min_gap:.3f})."
                        )
                    best_name = alternative_name
                    best_x = alternative_x
                    best_score = alternative_score

        if debug:
            for name, cx_start in candidates:
                if sw > new_w and name != "front_preserve":
                    continue
                s = _score_crop_candidate(img, cx_start, 0, new_w, h, sx, sy, sw, sh, facing)
                print(f"  Crop candidate '{name}': x_start={cx_start}, score={s:.3f}")
            print(f"  Selected: '{best_name}' (x_start={best_x}, score={best_score:.3f})")

        selected_candidate_name = best_name
        crop_origin_x = best_x
        crop_origin_y = 0
        crop_w = new_w
        crop_h = h
        cropped = img[:, best_x : best_x + new_w]

    else:
        # Too tall → crop top/bottom
        new_h = int(w / target_ratio)
        min_top_gap_px = max(8, int(new_h * MIN_TOP_EDGE_GAP_RATIO))

        # Candidate A: place subject at ideal y
        ideal_y_start = int(center_y - ideal_y_norm * new_h)
        ideal_y_start = max(0, min(ideal_y_start, h - new_h))

        # Candidate B: ensure full subject inclusion
        safe_y_start = center_y - new_h // 2
        if safe_y_start > sy:
            safe_y_start = sy
        if safe_y_start + new_h < sy + sh:
            safe_y_start = sy + sh - new_h
        safe_y_start = max(0, min(safe_y_start, h - new_h))

        # Candidate C: preserve head/top edge with breathing room.
        top_preserve_y_start = sy - min_top_gap_px
        top_preserve_y_start = max(0, min(top_preserve_y_start, h - new_h))

        # Candidate C: pure center
        center_y_start = max(0, (h - new_h) // 2)

        candidates = [
            ("top_preserve", top_preserve_y_start),
            ("ideal", ideal_y_start),
            ("safe", safe_y_start),
            ("center", center_y_start),
        ]

        # If the subject is taller than crop height, clipping is unavoidable.
        # For rider/person portraits, preserve head/top first.
        if sh > new_h and class_name in {"person", "rider_motorcycle"}:
            best_name, best_y = "top_preserve", top_preserve_y_start
            best_score = _score_crop_candidate(img, 0, best_y, w, new_h, sx, sy, sw, sh, facing)
            if debug:
                print(
                    f"  Subject taller than crop ({sh}>{new_h}) for {class_name}; "
                    "forcing top-preserve to protect headroom."
                )
        else:
            best_name, best_y = "center", center_y_start
            best_score = -1.0
            for name, cy_start in candidates:
                s = _score_crop_candidate(img, 0, cy_start, w, new_h, sx, sy, sw, sh, facing)
                if debug:
                    print(f"  Crop candidate '{name}': y_start={cy_start}, score={s:.3f}")
                if s > best_score:
                    best_score = s
                    best_name = name
                    best_y = cy_start

        if debug:
            print(f"  Selected: '{best_name}' (y_start={best_y}, score={best_score:.3f})")

        selected_candidate_name = best_name
        crop_origin_x = 0
        crop_origin_y = best_y
        crop_w = w
        crop_h = new_h
        cropped = img[best_y : best_y + new_h, :]

    crop_meta = _crop_uncertainty_flags(
        sx=sx,
        sy=sy,
        sw=sw,
        sh=sh,
        crop_x=crop_origin_x,
        crop_y=crop_origin_y,
        crop_w=crop_w,
        crop_h=crop_h,
        img_w=w,
        img_h=h,
    )
    crop_meta.update(
        {
            "selected_candidate": selected_candidate_name,
            "crop_window_xywh": [int(crop_origin_x), int(crop_origin_y), int(crop_w), int(crop_h)],
        }
    )
    if meta_out is not None:
        meta_out.clear()
        meta_out.update(crop_meta)

    # Debug visualization with grid overlay
    if debug or save_debug:
        debug_crop = cropped.copy()
        ch_d, cw_d = debug_crop.shape[:2]
        # Draw rule-of-thirds grid
        for frac in (1 / 3, 2 / 3):
            gx = int(cw_d * frac)
            gy = int(ch_d * frac)
            cv2.line(debug_crop, (gx, 0), (gx, ch_d), (255, 255, 0), 1)
            cv2.line(debug_crop, (0, gy), (cw_d, gy), (255, 255, 0), 1)
        # Draw Phi grid
        for frac in (0.382, 0.618):
            gx = int(cw_d * frac)
            gy = int(ch_d * frac)
            cv2.line(debug_crop, (gx, 0), (gx, ch_d), (0, 255, 255), 1)
            cv2.line(debug_crop, (0, gy), (cw_d, gy), (0, 255, 255), 1)

        # Draw subject/object boxes in crop coordinates.
        if raw_bbox is not None:
            rx, ry, rw, rh = raw_bbox
            rx1 = int(np.clip(rx - crop_origin_x, 0, cw_d - 1))
            ry1 = int(np.clip(ry - crop_origin_y, 0, ch_d - 1))
            rx2 = int(np.clip(rx + rw - crop_origin_x, 0, cw_d - 1))
            ry2 = int(np.clip(ry + rh - crop_origin_y, 0, ch_d - 1))
            if rx2 > rx1 and ry2 > ry1:
                cv2.rectangle(debug_crop, (rx1, ry1), (rx2, ry2), (0, 140, 255), 2)
                cv2.putText(
                    debug_crop,
                    "raw box",
                    (rx1, max(16, ry1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 140, 255),
                    1,
                )
        if expanded_bbox is not None:
            ex, ey, ew, eh = expanded_bbox
            ex1 = int(np.clip(ex - crop_origin_x, 0, cw_d - 1))
            ey1 = int(np.clip(ey - crop_origin_y, 0, ch_d - 1))
            ex2 = int(np.clip(ex + ew - crop_origin_x, 0, cw_d - 1))
            ey2 = int(np.clip(ey + eh - crop_origin_y, 0, ch_d - 1))
            if ex2 > ex1 and ey2 > ey1:
                cv2.rectangle(debug_crop, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)
                cv2.putText(
                    debug_crop,
                    "subject box",
                    (ex1, max(32, ey1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        debug_path = output_path.parent / f"debug_yolo_{output_path.name}"
        cv2.imwrite(str(debug_path), debug_crop)

        # Full-frame debug with selected crop window and object boxes.
        source_debug = img.copy()
        if raw_bbox is not None:
            rx, ry, rw, rh = raw_bbox
            cv2.rectangle(source_debug, (rx, ry), (rx + rw, ry + rh), (0, 140, 255), 3)
            cv2.putText(
                source_debug,
                "raw box",
                (rx, max(20, ry - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 140, 255),
                2,
            )
        if expanded_bbox is not None:
            ex, ey, ew, eh = expanded_bbox
            cv2.rectangle(source_debug, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            cv2.putText(
                source_debug,
                "subject box",
                (ex, max(45, ey - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        cv2.rectangle(
            source_debug,
            (crop_origin_x, crop_origin_y),
            (crop_origin_x + crop_w, crop_origin_y + crop_h),
            (255, 0, 255),
            3,
        )
        cv2.putText(
            source_debug,
            f"crop: {selected_candidate_name}",
            (crop_origin_x + 8, max(30, crop_origin_y + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 255),
            2,
        )
        source_debug_path = output_path.parent / f"debug_yolo_source_{output_path.name}"
        cv2.imwrite(str(source_debug_path), source_debug)

        debug_meta_path = output_path.parent / f"debug_yolo_{output_path.name}.json"
        debug_meta = {
            "image_path": str(image_path),
            "output_path": str(output_path),
            "image_size": {"width": w, "height": h},
            "detection_method": detection_method,
            "class_name": class_name,
            "confidence": round(float(conf), 6),
            "shot_type": shot_type,
            "facing": facing,
            "subject_center_xy": [int(center_x), int(center_y)],
            "raw_bbox_xywh": list(raw_bbox) if raw_bbox is not None else None,
            "expanded_bbox_xywh": list(expanded_bbox) if expanded_bbox is not None else None,
            "crop_window_xywh": [int(crop_origin_x), int(crop_origin_y), int(crop_w), int(crop_h)],
            "selected_candidate": selected_candidate_name,
            "uncertain_crop": bool(crop_meta["uncertain_crop"]),
            "uncertain_crop_reasons": crop_meta["uncertain_crop_reasons"],
            "min_subject_gap_px": crop_meta["min_subject_gap_px"],
            "min_subject_gap_ratio": crop_meta["min_subject_gap_ratio"],
            "target_size": {"width": int(out_w), "height": int(out_h)},
        }
        debug_meta_path.write_text(json.dumps(debug_meta, indent=2), encoding="utf-8")

        if debug:
            print(f"Debug crop with grid saved to {debug_path}")
            print(f"Debug source overlay saved to {source_debug_path}")
            print(f"Debug metadata saved to {debug_meta_path}")

    # Resize to exact output dimensions
    final = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(output_path), final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path


def _prepare_claude_crop_first_candidates(
    candidates: list[ImageScore],
    *,
    work_folder: Path,
) -> list[ImageScore]:
    """Pre-crop candidates to 1080x1440 before Claude vision scoring."""
    prepared: list[ImageScore] = []
    crop_first_dir = work_folder / "claude_crop_first"
    crop_first_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(candidates, start=1):
        source_for_name = item.source_path or item.path
        pre_crop_path = crop_first_dir / f"{idx:04d}_{source_for_name.stem}.jpg"
        try:
            smart_crop(
                item.path,
                pre_crop_path,
                out_w=OUTPUT_WIDTH,
                out_h=OUTPUT_HEIGHT,
                debug=False,
                save_debug=False,
                use_yolo=True,
            )
            prepared.append(
                ImageScore(
                    path=pre_crop_path,
                    source_path=item.source_path or item.path,
                    technical=dict(item.technical),
                )
            )
        except Exception as e:
            print(f"  ⚠ Claude crop-first prepare failed for {source_for_name.name}: {e}")
            prepared.append(item)

    return prepared


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    input_folder: str,
    output_folder: str = "selected",
    top_n: int = 10,
    scorer: str = "clip",
    vision_candidates_pct: float = 0.5,
    claude_model: Optional[str] = None,
    score_all: bool = False,
    claude_crop_first: bool = False,
):
    """
    Full pipeline: resize → deduplicate → technical score → vision score → crop → output.

    Args:
        input_folder: Path to folder with raw event photos
        output_folder: Path for final 1080x1440 output images
        top_n: Number of top images to output
        scorer: "clip" (free/local), "claude" (best quality, API costs), or "ollama" (self-hosted)
        vision_candidates_pct: Send top N% of technically-scored images to vision scoring
        score_all: If True, send all technically-scored images to vision scoring
        claude_crop_first: If True with Claude scorer, pre-crop candidates before scoring
    """
    src = Path(input_folder)
    out = Path(output_folder)
    work = src.parent / f"{src.name}_work"

    if not src.exists():
        print(f"❌ Input folder not found: {src}")
        sys.exit(1)

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
    resized, source_map = resize_for_processing(src, work)
    if not resized:
        print("❌ No valid images found.")
        sys.exit(1)

    # --- Stage 1: Deduplicate ---
    print("\n🔍 Stage 1: Deduplicating...")
    unique = deduplicate(resized)

    # --- Stage 2: Technical scoring ---
    print("\n📊 Stage 2: Technical quality scoring...")
    scored = batch_technical_score(unique, source_map=source_map)

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
    print(f"\n🧠 Stage 3: Vision scoring {scope} candidates ({scorer})...")
    ranked = batch_vision_score(
        candidates,
        scorer=scorer,
        env_search_dir=src,
        claude_model=claude_model,
    )

    # --- Stage 4: Output top N cropped to 1080x1440 ---
    print(f"\n✂️  Stage 4: Cropping top {top_n} to {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}...")
    out.mkdir(parents=True, exist_ok=True)

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

    for i, item in enumerate(top):
        rank = i + 1
        output_stem = (item.source_path or item.path).stem
        dest = out / f"{rank:02d}_{output_stem}.jpg"
        dest_full = out / f"{rank:02d}_full_{output_stem}.jpg"
        display_name = (item.source_path or item.path).name
        padded_written = False
        crop_meta: dict[str, object] = {}
        try:
            smart_crop(item.path, dest, save_debug=True, meta_out=crop_meta)
        except Exception:
            # Fallback: simple center crop via PIL
            try:
                with Image.open(item.path) as img:
                    img = img.convert("RGB")
                    img = img.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS)
                    img.save(dest, "JPEG", quality=95)
                crop_meta = {
                    "uncertain_crop": True,
                    "uncertain_crop_reasons": ["smart_crop_failed_used_center_resize_fallback"],
                }
            except Exception as e2:
                print(f"  ⚠ Could not process {item.path.name}: {e2}")
                continue

        # Additional non-ranked full-subject portrait variant only when crop is uncertain.
        if bool(crop_meta.get("uncertain_crop", False)):
            try:
                write_padded_full_subject(item.source_path or item.path, dest_full)
                padded_written = True
            except Exception as e3:
                print(f"  ⚠ Could not write full-subject variant for {display_name}: {e3}")

        report.append(
            {
                "rank": rank,
                "filename": display_name,
                "final_score": round(item.final_score, 4),
                "technical_composite": round(item.technical.get("composite", 0), 4),
                "vision_total": item.vision.get("total", 0),
                "one_line": item.one_line,
                "output": dest.name,
                "output_full_subject": dest_full.name if padded_written else None,
                "uncertain_crop": bool(crop_meta.get("uncertain_crop", False)),
                "uncertain_crop_reasons": crop_meta.get("uncertain_crop_reasons", []),
            }
        )

        print(f"  #{rank}: {display_name} → {dest.name}")
        print(
            f"       Score: {item.final_score:.3f} | Tech: {item.technical.get('composite', 0):.3f} | Vision: {item.vision.get('total', 0)}"
        )
        print(f"       {item.one_line}")

    # Save report
    report_json_path = out / "selection_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
    report_md_path = out / "selection_report.md"
    write_markdown_report(
        report_md_path,
        input_folder=src,
        output_folder=out,
        scorer=scorer,
        top_n=top_n,
        selected_report=report,
        analyzed_items=ranked,
    )

    print(f"\n{'=' * 60}")
    print(f"🏆 Done! {len(top)} images saved to {out}/")
    print(f"🗂️  Work folder retained: {work}")
    print(f"📋 JSON Report: {report_json_path}")
    print(f"📝 Markdown Report: {report_md_path}")
    print(f"{'=' * 60}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help text on parse errors."""

    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, f"\n{self.prog}: error: {message}\n")


def main():
    parser = _HelpOnErrorArgumentParser(
        description="Select the best Instagram cover images from an event photo dump.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./input --top 10 --scorer clip
  %(prog)s ./input --output ./selected --scorer claude --top 5
  %(prog)s ./input --output ./selected --scorer claude --all
  %(prog)s ./input --output ./selected --scorer claude --all --claude-crop-first
  %(prog)s ./input --output ./selected --scorer clip --all
        """,
    )
    parser.add_argument("input", help="Path to folder containing event photos")
    parser.add_argument(
        "--output", "-o", default="selected", help="Output folder (default: selected)"
    )
    parser.add_argument(
        "--top", "-n", type=int, default=10, help="Number of top images to output (default: 10)"
    )
    parser.add_argument(
        "--scorer",
        "-s",
        choices=["clip", "claude", "ollama"],
        default="clip",
        help="Vision scorer: 'clip' (free/local), 'claude' (API), or 'ollama' (self-hosted)",
    )
    parser.add_argument(
        "--vision-pct",
        type=float,
        default=0.5,
        help="Fraction of technically-scored images to send to vision scoring (default: 0.5)",
    )
    parser.add_argument(
        "--claude-model",
        default=resolve_claude_model(),
        help=(
            f"Claude model id (default from ANTHROPIC_MODEL/CLAUDE_MODEL or {DEFAULT_CLAUDE_MODEL})"
        ),
    )
    parser.add_argument(
        "--all",
        "--claude-all",
        dest="score_all",
        action="store_true",
        help="Score all Stage 2 images (ignore --vision-pct).",
    )
    parser.add_argument(
        "--claude-crop-first",
        dest="claude_crop_first",
        action="store_true",
        help=(
            "For --scorer claude, pre-crop candidate images to 1080x1440 before "
            "Claude scoring to better align ranking with final crop quality."
        ),
    )

    args = parser.parse_args()

    run_pipeline(
        input_folder=args.input,
        output_folder=args.output,
        top_n=args.top,
        scorer=args.scorer,
        vision_candidates_pct=args.vision_pct,
        claude_model=args.claude_model,
        score_all=args.score_all,
        claude_crop_first=args.claude_crop_first,
    )


if __name__ == "__main__":
    main()
