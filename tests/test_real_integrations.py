"""Opt-in compatibility probes for external services and heavyweight local models.

Every test in this module requires both ``PICKINSTA_RUN_REAL_INTEGRATIONS=1``
and a component-specific opt-in. Ordinary pytest runs make no network requests,
incur no API charges, and do not download models.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytest

from pickinsta.clip_scorer import load_clip_model, score_with_clip
from pickinsta.ig_image_selector import (
    DEFAULT_CLAUDE_MODEL,
    score_with_claude,
    score_with_ollama,
    yolo_detect_subject,
)

FIXTURE_IMAGE = Path(__file__).parent / "cropping" / "dsc5897_original.jpeg"
VISION_KEYS = {
    "subject_clarity",
    "lighting",
    "color_pop",
    "emotion",
    "scroll_stop",
    "crop_4x5",
    "total",
    "one_line",
}


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _require(component_flag: str) -> None:
    if not _enabled("PICKINSTA_RUN_REAL_INTEGRATIONS"):
        pytest.skip("set PICKINSTA_RUN_REAL_INTEGRATIONS=1 to enable real integrations")
    if not _enabled(component_flag):
        pytest.skip(f"set {component_flag}=1 to enable this compatibility probe")


def _assert_vision_contract(result: dict) -> None:
    assert VISION_KEYS <= result.keys()
    assert isinstance(result["total"], (int, float))
    assert 0 <= result["total"] <= 60
    assert isinstance(result["one_line"], str)


@pytest.mark.real_integration
@pytest.mark.anthropic_integration
def test_anthropic_vision_contract() -> None:
    """Make one paid request only after two explicit opt-ins and a credential check."""
    _require("PICKINSTA_RUN_ANTHROPIC_INTEGRATION")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    result = score_with_claude(
        FIXTURE_IMAGE,
        api_key=api_key,
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL),
        use_yolo_context=False,
    )
    _assert_vision_contract(result)


@pytest.mark.real_integration
@pytest.mark.ollama_integration
def test_ollama_vision_contract() -> None:
    """Call only an explicitly enabled Ollama endpoint; never invoke YOLO."""
    _require("PICKINSTA_RUN_OLLAMA_INTEGRATION")
    base_url = os.environ.get("PICKINSTA_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.environ.get("PICKINSTA_OLLAMA_MODEL", "").strip()
    if not model:
        pytest.skip("PICKINSTA_OLLAMA_MODEL must name an already installed vision model")

    result = score_with_ollama(
        FIXTURE_IMAGE,
        base_url=base_url,
        model=model,
        use_yolo_context=False,
        timeout_seconds=60,
        keep_alive="0",
    )
    _assert_vision_contract(result)


@pytest.mark.real_integration
@pytest.mark.clip_integration
def test_clip_vision_contract() -> None:
    """Load CLIP only when the caller explicitly permits heavyweight downloads."""
    _require("PICKINSTA_RUN_CLIP_INTEGRATION")
    if not _enabled("PICKINSTA_INTEGRATION_ALLOW_MODEL_DOWNLOADS"):
        pytest.skip("set PICKINSTA_INTEGRATION_ALLOW_MODEL_DOWNLOADS=1 to load CLIP")

    model, processor = load_clip_model()
    result = score_with_clip(FIXTURE_IMAGE, model=model, processor=processor)
    _assert_vision_contract(result)


@pytest.mark.real_integration
@pytest.mark.yolo_integration
def test_yolo_local_model_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run YOLO only against an explicit existing model path, preventing downloads."""
    _require("PICKINSTA_RUN_YOLO_INTEGRATION")
    model_path = Path(os.environ.get("PICKINSTA_YOLO_MODEL", "")).expanduser()
    if not str(model_path) or not model_path.is_file():
        pytest.skip("PICKINSTA_YOLO_MODEL must point to an existing local model file")
    monkeypatch.setenv("PICKINSTA_YOLO_MODEL", str(model_path))

    image = cv2.imread(str(FIXTURE_IMAGE))
    assert image is not None
    detection = yolo_detect_subject(image, debug=False)
    assert detection is None or (
        len(detection) == 6 and all(isinstance(value, (int, float, str)) for value in detection)
    )
