"""Intentional public surface of the legacy selector facade."""

from pickinsta import config
from pickinsta import ig_image_selector as selector
from pickinsta.models import ImageScore
from pickinsta.pipeline import vision_scoring
from pickinsta.reporting.gallery import generate_gallery, generate_gallery_index

PUBLIC_COMPATIBILITY_EXPORTS = {
    "ImageScore",
    "batch_technical_score",
    "batch_vision_score",
    "deduplicate",
    "generate_gallery",
    "generate_gallery_index",
    "main",
    "resize_for_processing",
    "run_dedup_only",
    "run_pipeline",
    "run_pipeline_recursive",
    "score_technical",
    "score_with_claude",
    "score_with_ollama",
    "smart_crop",
    "write_padded_full_subject",
    "yolo_detect_subject",
}


def test_selector_retains_intentional_public_compatibility_exports() -> None:
    assert not (PUBLIC_COMPATIBILITY_EXPORTS - vars(selector).keys())
    assert not (set(selector.__all__) - vars(selector).keys())


def test_selector_direct_aliases_point_to_extracted_owners() -> None:
    assert selector.ImageScore is ImageScore
    assert selector.generate_gallery is generate_gallery
    assert selector.generate_gallery_index is generate_gallery_index
    assert selector.resolve_claude_model is config.resolve_claude_model


def test_vision_collaborators_capture_current_selector_values(monkeypatch) -> None:
    def replacement(*_args, **_kwargs):
        return "patched"

    monkeypatch.setattr(selector, "score_with_clip", replacement)

    collaborators = vision_scoring.FacadeCollaborators.from_module(selector)

    assert collaborators.score_with_clip is replacement
