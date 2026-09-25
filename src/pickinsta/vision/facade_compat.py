"""Small compatibility policies retained by the legacy selector facade."""

import hashlib
from pathlib import Path
from typing import Optional

from pickinsta.config import DEFAULT_CLAUDE_MODEL
from pickinsta.infrastructure.vision_cache import load_vision_score, save_vision_score
from pickinsta.vision.claude import CLAUDE_SCORING_JPEG_QUALITY, CLAUDE_SCORING_MAX_EDGE

VISION_CACHE_SCHEMA_VERSION = 2


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def claude_model_candidates(preferred: str) -> list[str]:
    candidates = [preferred]
    parts = preferred.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        candidates.append(parts[0])
    candidates.extend([DEFAULT_CLAUDE_MODEL, "claude-3-5-sonnet-latest"])
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def is_model_not_found_error(error: Exception) -> bool:
    text = str(error).lower()
    return "not_found_error" in text and "model" in text


def claude_prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def claude_cache_options(*, score_path: Path, source_path: Path) -> dict:
    return {
        "use_yolo_context": True,
        "max_image_edge": CLAUDE_SCORING_MAX_EDGE,
        "jpeg_quality": CLAUDE_SCORING_JPEG_QUALITY,
        "score_input": "source" if score_path == source_path else "preprocessed",
        "score_input_sha256": file_sha256(score_path),
    }


def load_claude_score_from_file_cache(
    *,
    source_path: Path,
    source_sha256: str,
    model: str,
    prompt_sha256: str,
    scorer: str = "claude",
    scoring_options: Optional[dict] = None,
    strict_model: bool = True,
) -> Optional[dict]:
    del strict_model
    return load_vision_score(
        source_path=source_path,
        schema_version=VISION_CACHE_SCHEMA_VERSION,
        source_sha256=source_sha256,
        scorer=scorer,
        model=model,
        prompt_sha256=prompt_sha256,
        scoring_options=scoring_options,
    )


def save_claude_score_to_file_cache(
    *,
    source_path: Path,
    source_sha256: str,
    model: str,
    prompt_sha256: str,
    vision: dict,
    scorer: str = "claude",
    scoring_options: Optional[dict] = None,
) -> None:
    save_vision_score(
        source_path=source_path,
        schema_version=VISION_CACHE_SCHEMA_VERSION,
        source_sha256=source_sha256,
        scorer=scorer,
        model=model,
        prompt_sha256=prompt_sha256,
        scoring_options=scoring_options,
        vision=vision,
    )


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def claude_crop_gate_multiplier(vision: dict) -> float:
    score = safe_float(vision.get("crop_4x5"))
    for ceiling, multiplier in ((4, 0.15), (5, 0.35), (6, 0.60), (7, 0.80)):
        if score <= ceiling:
            return multiplier
    return 1.0


def claude_setup_hint(error: Exception) -> str:
    text = str(error).lower()
    if "no module named" in text or "import" in text:
        return "Install Claude dependency in your active environment:\n  python -m pip install -e '.[claude]'"
    if is_model_not_found_error(error):
        return (
            f"Claude model not found for your account.\nTry: --claude-model {DEFAULT_CLAUDE_MODEL}"
        )
    return "Claude initialization failed. Verify dependency and API key setup."
