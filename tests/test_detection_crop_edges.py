"""Failure and extreme-input coverage for detection and crop modules."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pickinsta.detection import yolo
from pickinsta.models import ImageScore
from pickinsta.pipeline import crop_geometry, cropping


def _image(path: Path, *, width: int = 60, height: int = 40) -> None:
    assert cv2.imwrite(str(path), np.full((height, width, 3), 127, dtype=np.uint8))


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return _Tensor(self.value[index])

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _box(cls_id: int, confidence: float, xyxy: tuple[float, float, float, float]):
    return SimpleNamespace(cls=_Tensor([cls_id]), conf=_Tensor([confidence]), xyxy=_Tensor([xyxy]))


def test_failed_yolo_download_removes_partial_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(yolo.YOLO_MODEL_ENV_VAR, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    def fail_after_writing(_url, destination):
        Path(destination).write_bytes(b"partial")
        raise ConnectionError("connection reset")

    with pytest.raises(RuntimeError, match="Could not download"):
        yolo.resolve_yolo_model_path(downloader=fail_after_writing)

    assert not (tmp_path / ".cache/pickinsta/models" / yolo.YOLO_MODEL_FILENAME).exists()


@pytest.mark.parametrize("image", [None, np.array([]), np.zeros((0, 1, 3), dtype=np.uint8)])
def test_yolo_rejects_invalid_images_without_loading_model(image) -> None:
    calls = []
    assert (
        yolo.yolo_detect_subject(image, model_loader=lambda **_kwargs: calls.append(True)) is None
    )
    assert calls == []


def test_yolo_handles_loader_and_inference_failures() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    def bad_loader(**_kwargs):
        raise OSError("bad override")

    assert yolo.yolo_detect_subject(image, model_loader=bad_loader) is None
    assert (
        yolo.yolo_detect_subject(
            image,
            model_loader=lambda **_kwargs: (
                lambda *_args, **_kw: (_ for _ in ()).throw(RuntimeError())
            ),
        )
        is None
    )


def test_yolo_ignores_degenerate_boxes_and_labels_unknown_classes() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [_box(0, 0.99, (20, 20, 20, 80)), _box(99, 0.8, (10, 10, 70, 70))]

    def model(*_args, **_kwargs):
        return [SimpleNamespace(boxes=boxes)]

    assert yolo.yolo_detect_subject(image, model_loader=lambda **_kwargs: model) == (
        10,
        10,
        60,
        60,
        "class_99",
        pytest.approx(0.8),
    )


def test_yolo_combines_person_and_motorcycle_and_handles_missing_boxes() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [_box(0, 0.9, (10, 10, 40, 80)), _box(3, 0.8, (25, 45, 80, 90))]

    def model(*_args, **_kwargs):
        return [SimpleNamespace(boxes=boxes)]

    result = yolo.yolo_detect_subject(image, model_loader=lambda **_kwargs: model)
    assert result is not None and result[4] == "rider_motorcycle"

    def no_boxes(*_args, **_kwargs):
        return [SimpleNamespace()]

    assert yolo.yolo_detect_subject(image, model_loader=lambda **_kwargs: no_boxes) is None


@pytest.mark.parametrize(
    ("left", "right"),
    [((0, 0, 0, 10), (0, 0, 10, 10)), ((0, 0, -3, -4), (0, 0, -2, -5))],
)
def test_bbox_iou_returns_zero_for_degenerate_geometry(left, right) -> None:
    assert crop_geometry._bbox_iou_xywh(left, right) == 0.0


def test_crop_scoring_rejects_zero_sized_window() -> None:
    with pytest.raises(ValueError, match="crop dimensions"):
        crop_geometry._score_crop_candidate(
            np.zeros((2, 2, 3), dtype=np.uint8), 0, 0, 0, 2, 0, 0, 1, 1, "head-on"
        )


def test_uncertainty_flags_degenerate_subject_and_clipped_crop() -> None:
    flags = crop_geometry._crop_uncertainty_flags(
        sx=-2, sy=0, sw=20, sh=20, crop_x=0, crop_y=0, crop_w=10, crop_h=10, img_w=10, img_h=10
    )
    assert flags["too_large_for_crop"] is True
    assert flags["clipped_subject"] is True
    assert flags["subject_bbox_hits_frame_edge"] is True


@pytest.mark.parametrize(("width", "height"), [(1, 100), (100, 1), (2, 2)])
def test_smart_crop_handles_extreme_image_dimensions(
    tmp_path: Path, width: int, height: int
) -> None:
    source = tmp_path / f"{width}x{height}.png"
    output = tmp_path / f"{width}x{height}.jpg"
    _image(source, width=width, height=height)

    cropping.smart_crop(source, output, out_w=3, out_h=4, use_yolo=False)
    assert cv2.imread(str(output)).shape[:2] == (4, 3)


def test_smart_crop_falls_back_when_detector_raises_or_returns_invalid_result(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    _image(source)
    for index, detector in enumerate(
        (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("model crashed")),
            lambda *_args, **_kwargs: (1, 2),
            lambda *_args, **_kwargs: (1, 2, -3, 4, "person", 0.9),
        )
    ):
        output = tmp_path / f"result-{index}.jpg"
        cropping.smart_crop(source, output, out_w=30, out_h=40, detector=detector)
        assert output.exists()


def test_crop_functions_reject_unreadable_input_and_invalid_dimensions(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    with pytest.raises(ValueError, match="Cannot read"):
        cropping.smart_crop(missing, tmp_path / "out.jpg")
    with pytest.raises(ValueError, match="Cannot read"):
        cropping.write_padded_full_subject(missing, tmp_path / "padded.jpg")

    source = tmp_path / "source.jpg"
    _image(source)
    with pytest.raises(ValueError, match="output dimensions"):
        cropping.smart_crop(source, tmp_path / "out.jpg", out_w=0)
    with pytest.raises(ValueError, match="output dimensions"):
        cropping.write_padded_full_subject(source, tmp_path / "out.jpg", out_h=0)


def test_image_write_failure_is_not_reported_as_success(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    _image(source)
    monkeypatch.setattr(cropping.cv2, "imwrite", lambda *_args, **_kwargs: False)

    with pytest.raises(OSError, match="Could not write image"):
        cropping.smart_crop(source, tmp_path / "out.jpg", out_w=30, out_h=40, use_yolo=False)
    with pytest.raises(OSError, match="Could not write image"):
        cropping.write_padded_full_subject(source, tmp_path / "padded.jpg", out_w=30, out_h=40)


def test_debug_metadata_atomic_write_preserves_previous_file_on_failure(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "metadata.json"
    target.write_text('{"old": true}', encoding="utf-8")
    monkeypatch.setattr(
        Path, "replace", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("full"))
    )

    with pytest.raises(OSError, match="full"):
        cropping._atomic_write_json(target, {"new": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob(".metadata.json.*.tmp")) == []


def test_prepare_candidates_keeps_original_item_when_cropper_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    item = ImageScore(path=source, technical={"sharpness": 1.0})
    prepared = cropping._prepare_claude_crop_first_candidates(
        [item],
        work_folder=tmp_path,
        cropper=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )
    assert prepared == [item]


def test_crop_workers_use_fallback_and_report_total_failure(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    _image(source)

    def failure(*_args, **_kwargs):
        raise RuntimeError

    index, meta = cropping._crop_one_image((4, source, tmp_path / "fallback.jpg"), cropper=failure)
    assert index == 4
    assert meta["uncertain_crop"] is True
    assert (tmp_path / "fallback.jpg").exists()

    index, meta = cropping._crop_one_image_no_debug(
        (5, tmp_path / "missing.jpg", tmp_path / "failed.jpg"), cropper=failure
    )
    assert index == 5
    assert meta == {"_failed": True}
