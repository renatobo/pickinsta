"""Lazy YOLO model acquisition and primary-subject detection."""

import logging
import os
import threading
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from pickinsta.config import YOLO_MODEL_ENV_VAR
from pickinsta.events import EventLevel, emit
from pickinsta.pipeline.crop_geometry import (
    _bbox_center_distance_ratio,
    _bbox_iou_xywh,
    _bbox_union_xywh,
)

YOLO_MODEL_FILENAME = "yolov8n.pt"
# Pin the upstream asset release so the default model does not silently change
# when Ultralytics publishes a new `latest` release.
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
_YOLO_MODEL = None
_YOLO_LOCK = threading.Lock()


def resolve_yolo_model_path(debug: bool = False, *, downloader=urlretrieve) -> Path:
    """Resolve an override or download the default model into the runtime cache."""
    override = os.environ.get(YOLO_MODEL_ENV_VAR)
    if override:
        model_path = Path(override).expanduser()
        if debug:
            emit(
                "yolo.model.override",
                f"Using YOLO model from {YOLO_MODEL_ENV_VAR}: {model_path}",
                stage="detection",
                path=str(model_path),
            )
        return model_path
    cache_dir = Path.home() / ".cache" / "pickinsta" / "models"
    model_path = cache_dir / YOLO_MODEL_FILENAME
    if model_path.exists():
        return model_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        emit(
            "yolo.model.download",
            f"YOLO model not found; downloading to {model_path} ...",
            stage="detection",
            path=str(model_path),
        )
    try:
        downloader(YOLO_MODEL_URL, model_path)
    except Exception as error:
        # Some downloaders create the destination before failing. Never leave a
        # partial model that the next run would mistake for a valid cache hit.
        model_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download YOLO model to {model_path}. Set {YOLO_MODEL_ENV_VAR} to a local model path to skip download."
        ) from error
    return model_path


def _load_yolo_model(debug: bool = False, *, path_resolver=resolve_yolo_model_path):
    """Load and process-cache YOLO; importing this module never imports Ultralytics."""
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    with _YOLO_LOCK:
        if _YOLO_MODEL is not None:
            return _YOLO_MODEL
        os.environ.setdefault("YOLO_VERBOSE", "false")
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        from ultralytics import YOLO

        _YOLO_MODEL = YOLO(str(path_resolver(debug=debug)), task="detect", verbose=False)
    return _YOLO_MODEL


def _combine_rider_motorcycle_box(
    best_detection: tuple[int, int, int, int, str, float],
    detections: list[tuple[int, int, int, int, str, float]],
    *,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int, str, float]:
    """Combine nearby person + motorcycle detections into a full rider+bike subject box."""
    bx, by, bw, bh, bcls, bconf = best_detection
    if bcls not in {"person", "motorcycle"}:
        return best_detection

    counterpart_cls = "motorcycle" if bcls == "person" else "person"
    base_box = (bx, by, bw, bh)

    best_pair = None
    best_pair_score = 0.0
    for dx, dy, dw, dh, dcls, dconf in detections:
        if dcls != counterpart_cls:
            continue
        other_box = (dx, dy, dw, dh)
        iou = _bbox_iou_xywh(base_box, other_box)
        dist_ratio = _bbox_center_distance_ratio(base_box, other_box, img_w, img_h)
        # Require overlap or reasonably close proximity.
        if iou < 0.01 and dist_ratio > 0.35:
            continue
        pair_score = float(dconf) * (1.0 + iou * 1.8 - dist_ratio * 0.8)
        if pair_score > best_pair_score:
            best_pair_score = pair_score
            best_pair = (dx, dy, dw, dh, dcls, dconf)

    if best_pair is None:
        return best_detection

    ox, oy, ow, oh, _ocls, oconf = best_pair
    ux, uy, uw, uh = _bbox_union_xywh(base_box, (ox, oy, ow, oh))
    return ux, uy, uw, uh, "rider_motorcycle", max(bconf, oconf)


def yolo_detect_subject(img: np.ndarray, debug: bool = False, *, model_loader=None):
    """
    Use YOLOv8 to detect subjects (people, vehicles, animals, etc.) in the image.
    Returns the bounding box of the most prominent detection, or None if nothing found.

    Args:
        img: OpenCV image (BGR format)
        debug: If True, print detection info

    Returns:
        Tuple of (x, y, w, h, class_name, confidence) or None
    """
    if not isinstance(img, np.ndarray) or img.ndim < 2 or img.size == 0:
        if debug:
            emit(
                "yolo.image.invalid",
                "YOLO detection skipped: invalid image",
                level=EventLevel.WARNING,
                stage="detection",
            )
        return None

    try:
        model = (model_loader or _load_yolo_model)(debug=debug)
    except ImportError:
        if debug:
            emit(
                "yolo.unavailable",
                "YOLO not available (ultralytics not installed)",
                level=EventLevel.WARNING,
                stage="detection",
            )
        return None
    except Exception as e:
        if debug:
            emit(
                "yolo.setup.failed",
                f"YOLO model setup failed: {e}",
                level=EventLevel.WARNING,
                stage="detection",
                error_type=type(e).__name__,
            )
        return None

    try:
        # Run inference
        results = model(img, verbose=False)

        if not results or len(results) == 0:
            return None

        # Get detections from first result
        result = results[0]
        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            return None

        # Priority classes for subject detection (most likely to be the main subject)
        # COCO dataset class IDs:
        priority_classes = {
            0: "person",  # People are usually the main subject
            1: "bicycle",
            2: "car",
            3: "motorcycle",  # Perfect for your use case!
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            16: "dog",
            17: "cat",
            18: "horse",
        }

        # Find the best detection (prioritize certain classes, then by confidence and size)
        all_detections: list[tuple[int, int, int, int, str, float]] = []
        best_detection = None
        best_score = 0.0

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Calculate bounding box area
            box_w = float(x2 - x1)
            box_h = float(y2 - y1)
            if not np.isfinite((x1, y1, x2, y2, conf)).all() or box_w <= 0 or box_h <= 0:
                continue
            area = box_w * box_h

            # Score = confidence * area_ratio * class_priority
            img_area = img.shape[0] * img.shape[1]
            area_ratio = area / img_area

            # Give priority to relevant classes; prioritize motorcycle slightly.
            if cls_id == 3:
                class_priority = 2.25
            elif cls_id in priority_classes:
                class_priority = 2.0
            else:
                class_priority = 1.0

            score = conf * area_ratio * class_priority

            class_name = priority_classes.get(cls_id, f"class_{cls_id}")
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            all_detections.append((x, y, w, h, class_name, conf))

            if score > best_score:
                best_score = score
                best_detection = (x, y, w, h, class_name, conf)

        if best_detection:
            best_detection = _combine_rider_motorcycle_box(
                best_detection,
                all_detections,
                img_w=img.shape[1],
                img_h=img.shape[0],
            )

        if debug and best_detection:
            x, y, w, h, class_name, conf = best_detection
            emit(
                "yolo.subject.detected",
                f"YOLO detected: {class_name} (conf={conf:.2f}) at bbox=({x}, {y}, {w}, {h})",
                stage="detection",
                subject_class=class_name,
                confidence=conf,
                bbox=(x, y, w, h),
            )

        return best_detection

    except Exception as e:
        if debug:
            emit(
                "yolo.detect.failed",
                f"YOLO detection failed: {e}",
                level=EventLevel.WARNING,
                stage="detection",
                error_type=type(e).__name__,
            )
        return None
