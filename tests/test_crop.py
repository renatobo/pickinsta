import json
from pathlib import Path

import cv2
import numpy as np

import pickinsta.ig_image_selector as selector


def _write_test_image(path: Path, width: int = 1200, height: int = 800) -> None:
    """Create a synthetic RGB image with gradients and a high-contrast subject box."""
    x_grad = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    y_grad = np.tile(np.linspace(255, 0, height, dtype=np.uint8), (width, 1)).T
    img = np.dstack((x_grad, y_grad, np.full((height, width), 80, dtype=np.uint8)))
    cv2.rectangle(img, (250, 220), (680, 620), (255, 255, 255), -1)
    cv2.imwrite(str(path), img)


def test_smart_crop_writes_expected_dimensions_and_debug_file(tmp_path, monkeypatch) -> None:
    src = tmp_path / "source.jpg"
    out = tmp_path / "cropped.jpg"
    _write_test_image(src)

    monkeypatch.setattr(
        selector,
        "yolo_detect_subject",
        lambda _img, debug=False: (250, 220, 430, 400, "motorcycle", 0.94),
    )

    crop_meta = {}
    result = selector.smart_crop(
        src,
        out,
        out_w=108,
        out_h=144,
        debug=True,
        use_yolo=True,
        meta_out=crop_meta,
    )

    assert result == out
    assert out.exists()
    cropped = cv2.imread(str(out))
    assert cropped is not None
    assert cropped.shape[:2] == (144, 108)
    assert (tmp_path / "debug_yolo_cropped.jpg").exists()
    assert (tmp_path / "debug_yolo_source_cropped.jpg").exists()
    meta_path = tmp_path / "debug_yolo_cropped.jpg.json"
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["expanded_bbox_xywh"] is not None
    crop_window = payload["crop_window_xywh"]
    assert len(crop_window) == 4
    assert crop_window[2] == 600
    assert crop_window[3] == 800
    assert isinstance(payload["uncertain_crop"], bool)
    assert isinstance(payload["uncertain_crop_reasons"], list)
    assert "uncertain_crop" in crop_meta


def test_smart_crop_uses_center_fallback_when_no_subject_detected(tmp_path, monkeypatch) -> None:
    src = tmp_path / "fallback.jpg"
    out = tmp_path / "fallback_cropped.jpg"
    _write_test_image(src, width=900, height=1400)

    monkeypatch.setattr(selector, "yolo_detect_subject", lambda _img, debug=False: None)
    monkeypatch.setattr(selector.cv2, "findContours", lambda *_a, **_k: ([], None))

    selector.smart_crop(src, out, out_w=108, out_h=144, debug=False, use_yolo=True)

    cropped = cv2.imread(str(out))
    assert cropped is not None
    assert cropped.shape[:2] == (144, 108)


def test_smart_crop_saves_debug_overlay_without_verbose_debug(tmp_path, monkeypatch) -> None:
    src = tmp_path / "source.jpg"
    out = tmp_path / "cropped.jpg"
    _write_test_image(src)

    monkeypatch.setattr(
        selector,
        "yolo_detect_subject",
        lambda _img, debug=False: (250, 220, 430, 400, "motorcycle", 0.94),
    )

    selector.smart_crop(src, out, out_w=108, out_h=144, debug=False, save_debug=True, use_yolo=True)

    assert out.exists()
    assert (tmp_path / "debug_yolo_cropped.jpg").exists()
    assert (tmp_path / "debug_yolo_source_cropped.jpg").exists()
    assert (tmp_path / "debug_yolo_cropped.jpg.json").exists()


def test_smart_crop_prefers_front_preserve_with_edge_gap(tmp_path, monkeypatch) -> None:
    src = tmp_path / "front_preserve.jpg"
    out = tmp_path / "front_preserve_cropped.jpg"

    img = np.zeros((400, 1000, 3), dtype=np.uint8)
    # Left marker represents front side of subject.
    cv2.rectangle(img, (100, 80), (220, 320), (0, 255, 0), -1)
    # Right marker represents rear side of subject.
    cv2.rectangle(img, (480, 80), (600, 320), (0, 0, 255), -1)
    cv2.imwrite(str(src), img)

    monkeypatch.setattr(
        selector,
        "yolo_detect_subject",
        lambda _img, debug=False: (100, 80, 500, 240, "motorcycle", 0.95),
    )
    monkeypatch.setattr(selector, "_guess_facing_direction", lambda *_args, **_kwargs: "left")

    selector.smart_crop(src, out, out_w=300, out_h=400, debug=False, use_yolo=True)

    cropped = cv2.imread(str(out))
    assert cropped is not None
    assert cropped.shape[:2] == (400, 300)

    green_mask = (
        (cropped[:, :, 1] > 170)
        & (cropped[:, :, 0] < 120)
        & (cropped[:, :, 2] < 120)
    )
    green_cols = np.where(green_mask.any(axis=0))[0]
    assert green_cols.size > 0
    # Require a breathing gap from border for the front marker.
    assert green_cols.min() >= 8


def test_smart_crop_closeup_wide_subject_ignores_unstable_facing(tmp_path, monkeypatch) -> None:
    src = tmp_path / "closeup.jpg"
    out = tmp_path / "closeup_cropped.jpg"

    img = np.zeros((400, 1000, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 40), (900, 360), (220, 220, 220), -1)
    cv2.imwrite(str(src), img)

    monkeypatch.setattr(
        selector,
        "yolo_detect_subject",
        lambda _img, debug=False: (100, 40, 800, 320, "motorcycle", 0.95),
    )
    monkeypatch.setattr(selector, "_expand_subject_bbox", lambda sx, sy, sw, sh, *_a, **_k: (sx, sy, sw, sh))
    monkeypatch.setattr(selector, "_guess_facing_direction", lambda *_args, **_kwargs: "right")

    selector.smart_crop(src, out, out_w=300, out_h=400, debug=False, save_debug=True, use_yolo=True)

    meta_path = tmp_path / "debug_yolo_closeup_cropped.jpg.json"
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    # Close-up + sw>new_w should force centered ideal crop, not directional front-preserve.
    assert payload["selected_candidate"] == "ideal"
    assert payload["crop_window_xywh"][0] == 350


def test_smart_crop_tall_rider_subject_forces_top_preserve(tmp_path, monkeypatch) -> None:
    src = tmp_path / "tall_rider.jpg"
    out = tmp_path / "tall_rider_cropped.jpg"

    img = np.zeros((1200, 800, 3), dtype=np.uint8)
    cv2.rectangle(img, (180, 0), (620, 1180), (210, 210, 210), -1)
    cv2.imwrite(str(src), img)

    monkeypatch.setattr(
        selector,
        "yolo_detect_subject",
        lambda _img, debug=False: (180, 0, 440, 1180, "rider_motorcycle", 0.9),
    )
    monkeypatch.setattr(selector, "_expand_subject_bbox", lambda sx, sy, sw, sh, *_a, **_k: (sx, sy, sw, sh))
    monkeypatch.setattr(selector, "_guess_facing_direction", lambda *_args, **_kwargs: "head-on")

    selector.smart_crop(src, out, out_w=1080, out_h=1440, debug=False, save_debug=True, use_yolo=True)

    meta_path = tmp_path / "debug_yolo_tall_rider_cropped.jpg.json"
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["selected_candidate"] == "top_preserve"
    assert payload["crop_window_xywh"][1] == 0


def test_smart_crop_overrides_tight_edge_candidate_when_subject_is_wide(tmp_path, monkeypatch) -> None:
    src = tmp_path / "edge_override.jpg"
    out = tmp_path / "edge_override_cropped.jpg"

    img = np.zeros((400, 1200, 3), dtype=np.uint8)
    cv2.rectangle(img, (300, 80), (510, 320), (255, 255, 255), -1)
    cv2.imwrite(str(src), img)

    monkeypatch.setattr(
        selector,
        "yolo_detect_subject",
        lambda _img, debug=False: (300, 80, 210, 240, "rider_motorcycle", 0.95),
    )
    monkeypatch.setattr(selector, "_expand_subject_bbox", lambda sx, sy, sw, sh, *_a, **_k: (sx, sy, sw, sh))
    monkeypatch.setattr(selector, "_guess_facing_direction", lambda *_args, **_kwargs: "left")

    selector.smart_crop(src, out, out_w=300, out_h=400, debug=False, use_yolo=True)

    cropped = cv2.imread(str(out))
    assert cropped is not None
    white_mask = (
        (cropped[:, :, 0] > 240)
        & (cropped[:, :, 1] > 240)
        & (cropped[:, :, 2] > 240)
    )
    white_cols = np.where(white_mask.any(axis=0))[0]
    assert white_cols.size > 0
    # Edge-risk override should avoid a right-edge-tight crop.
    assert white_cols.max() <= 280


def test_write_padded_full_subject_writes_expected_dimensions(tmp_path) -> None:
    src = tmp_path / "source.jpg"
    out = tmp_path / "padded.jpg"
    _write_test_image(src, width=1200, height=800)

    result = selector.write_padded_full_subject(src, out, out_w=108, out_h=144)

    assert result == out
    assert out.exists()
    padded = cv2.imread(str(out))
    assert padded is not None
    assert padded.shape[:2] == (144, 108)


def test_expand_subject_bbox_adds_padding_and_clamps_bounds() -> None:
    sx, sy, sw, sh = selector._expand_subject_bbox(5, 6, 100, 80, img_w=140, img_h=90)
    assert sx == 0
    assert sy == 0
    assert sw == 117
    assert sh == 90


def test_horizontal_margin_bounds_returns_feasible_range_when_subject_can_fit() -> None:
    bounds = selector._horizontal_margin_bounds(
        sx=300,
        sw=200,
        crop_w=320,
        frame_w=1000,
        min_gap_px=8,
    )
    assert bounds is not None
    lo, hi = bounds
    assert lo <= hi
    # Any x in bounds keeps at least 8px left/right breathing room.
    test_x = hi
    assert 300 - test_x >= 8
    assert test_x + 320 - (300 + 200) >= 8


def test_combine_rider_motorcycle_box_merges_overlapping_person_and_bike() -> None:
    best = (300, 200, 180, 400, "person", 0.9)
    detections = [
        best,
        (330, 350, 280, 300, "motorcycle", 0.82),
    ]

    combined = selector._combine_rider_motorcycle_box(best, detections, img_w=1200, img_h=800)
    x, y, w, h, cls, conf = combined
    assert cls == "rider_motorcycle"
    assert conf >= 0.82
    assert x <= 300
    assert y <= 200
    assert x + w >= 610
    assert y + h >= 650


def test_combine_rider_motorcycle_box_keeps_best_when_counterpart_far() -> None:
    best = (300, 200, 180, 400, "person", 0.9)
    detections = [
        best,
        (900, 100, 180, 180, "motorcycle", 0.95),
    ]
    combined = selector._combine_rider_motorcycle_box(best, detections, img_w=1200, img_h=800)
    assert combined == best
