import json
from pathlib import Path

import pytest

import pickinsta.ig_image_selector as selector
from pickinsta.infrastructure.filesystem import (
    atomic_write_json,
    publish_managed_artifacts,
)
from pickinsta.infrastructure.vision_cache import load_vision_score, save_vision_score
from pickinsta.models import ImageScore


def test_legacy_module_reexports_domain_model() -> None:
    assert selector.ImageScore is ImageScore


def test_atomic_json_publication_replaces_complete_document(tmp_path) -> None:
    destination = tmp_path / "nested" / "document.json"

    atomic_write_json(destination, {"status": "complete"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "complete"}
    assert not list(destination.parent.glob(f".{destination.name}.*"))


def test_vision_cache_requires_complete_identity(tmp_path) -> None:
    source = tmp_path / "image.jpg"
    source.write_bytes(b"source")
    identity = {
        "source_path": source,
        "schema_version": 2,
        "source_sha256": "source-hash",
        "scorer": "claude",
        "model": "model-a",
        "prompt_sha256": "prompt-hash",
        "scoring_options": {"max_edge": 1024},
    }

    save_vision_score(**identity, vision={"total": 55})

    assert load_vision_score(**identity) == {"total": 55}
    assert load_vision_score(**{**identity, "model": "model-b"}) is None


@pytest.mark.parametrize("cache_content", ["{broken", "null", '"not an object"'])
def test_vision_cache_treats_corrupt_or_wrong_shape_payload_as_miss(
    tmp_path, cache_content
) -> None:
    source = tmp_path / "image.jpg"
    source.write_bytes(b"source")
    Path(str(source) + ".pickinsta.json").write_text(cache_content, encoding="utf-8")

    assert (
        load_vision_score(
            source_path=source,
            schema_version=2,
            source_sha256="source-hash",
            scorer="claude",
            model="model-a",
            prompt_sha256="prompt-hash",
        )
        is None
    )

    save_vision_score(
        source_path=source,
        schema_version=2,
        source_sha256="source-hash",
        scorer="claude",
        model="model-a",
        prompt_sha256="prompt-hash",
        vision={"total": 55},
    )
    assert load_vision_score(
        source_path=source,
        schema_version=2,
        source_sha256="source-hash",
        scorer="claude",
        model="model-a",
        prompt_sha256="prompt-hash",
    ) == {"total": 55}


def test_managed_publication_rejects_missing_completion_artifact(tmp_path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    (staging / "image.jpg").write_bytes(b"image")

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        publish_managed_artifacts(
            staging,
            output,
            ["image.jpg", "manifest.json"],
            completion_artifact="manifest.json",
        )

    assert not (output / "image.jpg").exists()
