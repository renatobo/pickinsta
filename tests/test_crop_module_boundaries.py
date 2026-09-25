"""Boundary and compatibility checks for extracted crop modules."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import cv2
import numpy as np

from pickinsta import ig_image_selector as selector
from pickinsta.detection import yolo
from pickinsta.pipeline import crop_geometry, cropping


def test_crop_geometry_has_only_numerical_dependencies() -> None:
    source = Path(crop_geometry.__file__).read_text(encoding="utf-8")
    imports = {
        node.names[0].name.split(".", 1)[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
    }
    imports.update(
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    )
    assert not imports.intersection(
        {"pathlib", "urllib", "ultralytics", "pickinsta", "os", "shutil"}
    )


def test_importing_geometry_does_not_load_ultralytics() -> None:
    assert "ultralytics" not in sys.modules


def test_selector_geometry_exports_are_exact_aliases() -> None:
    assert selector._score_crop_candidate is crop_geometry._score_crop_candidate
    assert selector._crop_uncertainty_flags is crop_geometry._crop_uncertainty_flags
    assert selector._combine_rider_motorcycle_box is yolo._combine_rider_motorcycle_box


def test_extracted_crop_uses_injected_detector(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    cv2.imwrite(str(source), np.full((400, 600, 3), 127, dtype=np.uint8))
    calls: list[bool] = []

    def detector(_image: np.ndarray, debug: bool = False):
        calls.append(debug)
        return 100, 50, 300, 250, "motorcycle", 0.9

    result = cropping.smart_crop(source, output, out_w=300, out_h=400, detector=detector)

    assert result == output
    assert calls == [False]
    assert cv2.imread(str(output)).shape[:2] == (400, 300)
