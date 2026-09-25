from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pickinsta import ig_image_selector as selector
from pickinsta.pipeline import technical_scoring


def test_selector_technical_score_delegates_with_patchable_subject_mask(
    monkeypatch, tmp_path: Path
) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (80, 60), color=(90, 120, 150)).save(image_path)
    calls = []

    def no_subject(image: np.ndarray):
        calls.append(image.shape)
        return None

    monkeypatch.setattr(selector, "_detect_subject_mask", no_subject)

    through_selector = selector.score_technical(image_path)
    through_module = technical_scoring.score_technical(image_path, detect_subject_mask=no_subject)

    assert calls == [(60, 80, 3), (60, 80, 3)]
    assert through_selector == through_module


def test_detect_subject_mask_translates_detection_to_binary_region() -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)

    mask = technical_scoring.detect_subject_mask(
        image,
        lambda _image, debug=False: (2, 3, 4, 2, "motorcycle", 0.9),
    )

    assert mask is not None
    assert mask.dtype == np.uint8
    assert cv2.countNonZero(mask) == 8
    assert np.all(mask[3:5, 2:6] == 255)


def test_batch_technical_score_empty_input_does_not_resolve_workers() -> None:
    assert technical_scoring.batch_technical_score([]) == []
