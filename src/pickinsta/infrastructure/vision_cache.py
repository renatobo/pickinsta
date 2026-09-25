"""Persistent cache identity and payload handling for vision scores."""

import json
from pathlib import Path
from typing import Optional

from pickinsta.infrastructure.filesystem import atomic_write_text


def cache_file_for_source(source_path: Path) -> Path:
    """Return the per-image vision cache path next to its source image."""
    return Path(str(source_path) + ".pickinsta.json")


def load_vision_score(
    *,
    source_path: Path,
    schema_version: int,
    source_sha256: str,
    scorer: str,
    model: str,
    prompt_sha256: str,
    scoring_options: Optional[dict] = None,
) -> Optional[dict]:
    """Load a score only when every result-affecting identity field matches."""
    cache_file = cache_file_for_source(source_path)
    if not cache_file.exists():
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None

    expected_identity = {
        "schema_version": schema_version,
        "source_sha256": source_sha256,
        "scorer": scorer,
        "model": model,
        "prompt_sha256": prompt_sha256,
        "scoring_options": scoring_options or {},
    }
    if not isinstance(payload, dict):
        return None
    if any(payload.get(key) != value for key, value in expected_identity.items()):
        return None

    vision = payload.get("vision")
    return vision if isinstance(vision, dict) else None


def save_vision_score(
    *,
    source_path: Path,
    schema_version: int,
    source_sha256: str,
    scorer: str,
    model: str,
    prompt_sha256: str,
    vision: dict,
    scoring_options: Optional[dict] = None,
) -> None:
    """Persist one vision score with its complete cache identity."""
    payload = {
        "schema_version": schema_version,
        "source_file": str(source_path),
        "source_sha256": source_sha256,
        "scorer": scorer,
        "model": model,
        "prompt_sha256": prompt_sha256,
        "scoring_options": scoring_options or {},
        "vision": vision,
    }
    atomic_write_text(cache_file_for_source(source_path), json.dumps(payload, indent=2))
