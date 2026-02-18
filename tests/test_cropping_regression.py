from pathlib import Path

import cv2
import numpy as np
import pytest

import pickinsta.ig_image_selector as selector


FIXTURE_DIR = Path("tests/cropping")
CASE_BOXES = {
    "dsc5365": (610, 369, 2369, 1902, "motorcycle", 0.887),
    "dsc5897": (414, 860, 2258, 946, "motorcycle", 0.841),
    "dsc09054": (2809, 613, 1895, 3303, "rider_motorcycle", 0.873),
    "dsc5660": (9, 220, 1899, 2860, "rider_motorcycle", 0.7296050190925598),
}


def _normalized_mae(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(diff.mean() / 255.0)


@pytest.mark.parametrize(
    "case_id",
    ["dsc5365", "dsc5897", "dsc09054", "dsc5660"],
)
def test_smart_crop_regression_matches_front_preserve_reference(tmp_path, monkeypatch, case_id) -> None:
    ext = ".jpg" if case_id == "dsc09054" else ".jpeg"
    src = FIXTURE_DIR / f"{case_id}_original{ext}"
    expected = FIXTURE_DIR / f"{case_id}_front_preserve_expected.jpg"
    out = tmp_path / f"{case_id}_actual.jpg"

    assert src.exists(), f"Missing fixture image: {src}"
    assert expected.exists(), f"Missing expected crop image: {expected}"

    monkeypatch.setattr(
        selector,
        "yolo_detect_subject",
        lambda _img, debug=False, _box=CASE_BOXES[case_id]: _box,
    )

    selector.smart_crop(src, out, out_w=1080, out_h=1440, debug=False, use_yolo=True)

    expected_img = cv2.imread(str(expected))
    actual_img = cv2.imread(str(out))
    assert expected_img is not None
    assert actual_img is not None
    assert expected_img.shape == actual_img.shape == (1440, 1080, 3)

    mae = _normalized_mae(actual_img, expected_img)
    assert mae <= 0.035, (
        f"Crop drifted too far from front-preserve reference for {case_id}. "
        f"normalized_mae={mae:.4f} (limit=0.0350)"
    )
