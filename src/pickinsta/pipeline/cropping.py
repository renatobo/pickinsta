"""Crop image I/O and orchestration with injected detection collaborators."""

import json
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from pickinsta.events import console_event
from pickinsta.models import ImageScore
from pickinsta.pipeline.crop_geometry import (
    _classify_shot_type,
    _crop_uncertainty_flags,
    _expand_subject_bbox,
    _guess_facing_direction,
    _horizontal_margin_bounds,
    _ideal_subject_x,
    _ideal_subject_y,
    _score_crop_candidate,
    _subject_side_gap_ratios,
)

print = partial(console_event, "cropping")

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1440
MIN_FRONT_EDGE_GAP_RATIO = 0.03
EDGE_RISK_SUBJECT_FILL_RATIO = 0.60
EDGE_RISK_TIGHT_GAP_MULT = 1.15
EDGE_RISK_ALT_GAP_MULT = 2.0
EDGE_RISK_ALT_MAX_SCORE_DELTA = 0.12
MIN_TOP_EDGE_GAP_RATIO = 0.03


def _write_image(path: Path, image: np.ndarray, params: list[int] | None = None) -> None:
    """Write an image or raise instead of silently accepting OpenCV failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, params or []):
        raise OSError(f"Could not write image to {path}")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Publish JSON metadata atomically so interrupted writes cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_padded_full_subject(
    image_path: Path,
    output_path: Path,
    out_w: int = OUTPUT_WIDTH,
    out_h: int = OUTPUT_HEIGHT,
) -> Path:
    """Write an uncropped portrait variant by fitting the full image with blurred padding."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read {image_path}")

    if out_w <= 0 or out_h <= 0:
        raise ValueError("output dimensions must be positive")
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    fit_w = max(1, int(round(w * scale)))
    fit_h = max(1, int(round(h * scale)))

    fit_interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    fit = cv2.resize(img, (fit_w, fit_h), interpolation=fit_interp)

    bg = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=18, sigmaY=18)

    x0 = (out_w - fit_w) // 2
    y0 = (out_h - fit_h) // 2
    bg[y0 : y0 + fit_h, x0 : x0 + fit_w] = fit
    _write_image(output_path, bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path


def smart_crop(
    image_path: Path,
    output_path: Path,
    out_w: int = OUTPUT_WIDTH,
    out_h: int = OUTPUT_HEIGHT,
    debug: bool = False,
    save_debug: bool = False,
    use_yolo: bool = True,
    meta_out: Optional[dict[str, object]] = None,
    *,
    detector: Callable[..., object] | None = None,
    guess_facing: Callable[..., str] = _guess_facing_direction,
    expand_bbox: Callable[..., tuple[int, int, int, int]] = _expand_subject_bbox,
) -> Path:
    """Crop and resize image to out_w x out_h using composition rules.

    Steps (per composition rules algorithm):
      1. Detect subject with YOLO (fallback: saliency).
      2. Classify shot type from subject area.
      3. Determine facing direction and ideal placement (lead room,
         rule-of-thirds / Phi Grid power points).
      4. Generate candidate crop windows and score each.
      5. Select the highest-scoring crop that keeps the subject intact.
    """
    if out_w <= 0 or out_h <= 0:
        raise ValueError("output dimensions must be positive")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read {image_path}")

    h, w = img.shape[:2]
    target_ratio = out_w / out_h  # 0.75 for 1080x1440
    current_ratio = w / h

    # --- Step 1: Detect subject ---
    sx, sy, sw, sh = None, None, None, None
    detection_method = "none"
    class_name = ""
    conf = 0.0
    raw_bbox: Optional[tuple[int, int, int, int]] = None
    expanded_bbox: Optional[tuple[int, int, int, int]] = None

    if use_yolo:
        try:
            yolo_result = detector(img, debug=debug) if detector is not None else None
            if yolo_result is not None:
                sx, sy, sw, sh, class_name, conf = yolo_result
                values = np.asarray((sx, sy, sw, sh, conf), dtype=float)
                if not np.isfinite(values).all() or sw <= 0 or sh <= 0:
                    raise ValueError("invalid detector bounding box")
                sx, sy, sw, sh = int(sx), int(sy), int(sw), int(sh)
                raw_bbox = (sx, sy, sw, sh)
                detection_method = f"yolo ({class_name})"
        except Exception as error:
            if debug:
                print(f"Ignoring invalid YOLO detection: {error}")
            sx, sy, sw, sh = None, None, None, None

    # Fallback to saliency
    if sx is None:
        if debug:
            print("YOLO found nothing, falling back to saliency detection...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            saliency_det = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency_det.computeSaliency(gray)
            if not success:
                raise RuntimeError("Saliency computation failed")
            saliency_map = (saliency_map * 255).astype(np.uint8)
        except Exception:
            saliency_map = cv2.Canny(gray, 50, 150)

        _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            sx, sy, sw, sh = cv2.boundingRect(largest)
            expanded_bbox = (sx, sy, sw, sh)
            detection_method = "saliency"
        else:
            # Absolute fallback: image center
            sx, sy, sw, sh = w // 4, h // 4, w // 2, h // 2
            expanded_bbox = (sx, sy, sw, sh)
            detection_method = "center_fallback"

    center_x = sx + sw // 2
    center_y = sy + sh // 2

    if detection_method.startswith("yolo"):
        facing = guess_facing(img, *raw_bbox) if raw_bbox is not None else "unknown"
        sx, sy, sw, sh = expand_bbox(
            sx,
            sy,
            sw,
            sh,
            w,
            h,
            class_name=class_name,
            facing=facing,
        )
        expanded_bbox = (sx, sy, sw, sh)
        center_x = sx + sw // 2
        center_y = sy + sh // 2
    else:
        facing = guess_facing(img, sx, sy, sw, sh)
        sx, sy, sw, sh = expand_bbox(sx, sy, sw, sh, w, h, facing=facing)
        expanded_bbox = (sx, sy, sw, sh)
        center_x = sx + sw // 2
        center_y = sy + sh // 2

    if debug:
        print(
            f"Detection: {detection_method} | bbox=({sx},{sy},{sw},{sh}) | center=({center_x},{center_y}) | img={w}x{h}"
        )
        debug_img = img.copy()
        cv2.rectangle(debug_img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 5)
        cv2.circle(debug_img, (center_x, center_y), 15, (0, 0, 255), -1)
        if class_name:
            cv2.putText(
                debug_img,
                f"{class_name}",
                (sx, sy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )

    # --- Step 2: Classify shot type ---
    subject_area_ratio = (sw * sh) / (w * h)
    shot_type = _classify_shot_type(subject_area_ratio)

    # --- Step 3: Determine facing direction and ideal placement ---
    facing = guess_facing(img, sx, sy, sw, sh)
    ideal_x_norm = _ideal_subject_x(facing, shot_type)
    ideal_y_norm = _ideal_subject_y(shot_type)

    if debug:
        print(
            f"Shot type: {shot_type} | Facing: {facing} | Ideal placement: ({ideal_x_norm:.2f}, {ideal_y_norm:.2f})"
        )

    # --- Step 4: Generate candidate crop windows and pick the best ---
    selected_candidate_name = "no_crop"
    crop_origin_x = 0
    crop_origin_y = 0
    crop_w = w
    crop_h = h

    if abs(current_ratio - target_ratio) < 0.01:
        # Already the right ratio — no crop needed
        cropped = img
    elif current_ratio > target_ratio:
        # Too wide → crop sides
        new_w = max(1, min(w, int(h * target_ratio)))
        min_edge_gap_px = max(8, int(new_w * MIN_FRONT_EDGE_GAP_RATIO))
        margin_bounds = _horizontal_margin_bounds(sx, sw, new_w, w, min_edge_gap_px)

        # Candidate A: place subject at ideal x (rule-of-thirds with lead room)
        ideal_x_start = int(center_x - ideal_x_norm * new_w)
        ideal_x_start = max(0, min(ideal_x_start, w - new_w))

        # Candidate B: ensure full subject inclusion (old behavior)
        safe_x_start = center_x - new_w // 2
        # Adjust to include entire subject bbox
        if safe_x_start > sx:
            safe_x_start = sx
        if safe_x_start + new_w < sx + sw:
            safe_x_start = sx + sw - new_w
        safe_x_start = max(0, min(safe_x_start, w - new_w))

        # Candidate C: front-preserve with a breathing gap from the border.
        if facing == "left":
            front_preserve_x_start = sx - min_edge_gap_px
        elif facing == "right":
            front_preserve_x_start = sx + sw + min_edge_gap_px - new_w
        else:
            front_preserve_x_start = ideal_x_start
        front_preserve_x_start = max(0, min(front_preserve_x_start, w - new_w))

        # Candidate D: pure center
        center_x_start = max(0, (w - new_w) // 2)

        candidates = [
            ("front_preserve", front_preserve_x_start),
            ("ideal", ideal_x_start),
            ("safe", safe_x_start),
            ("center", center_x_start),
        ]

        # Keep candidates within side-gap-safe bounds when feasible.
        if margin_bounds is not None:
            lo, hi = margin_bounds
            bounded: list[tuple[str, int]] = []
            seen: set[tuple[str, int]] = set()
            for name, cx_start in candidates:
                bounded_x = int(np.clip(cx_start, lo, hi))
                key = (name, bounded_x)
                if key not in seen:
                    seen.add(key)
                    bounded.append(key)
            candidates = bounded

        # If the subject is wider than crop width, clipping is unavoidable.
        # Prefer preserving the subject's front side with a small border gap.
        if sw > new_w:
            # Close-up subjects are often quasi-symmetric in the detected box.
            # Direction inference is unstable here, so avoid one-sided crops.
            if shot_type == "close-up":
                best_name, best_x = "ideal", ideal_x_start
            else:
                best_name, best_x = "front_preserve", front_preserve_x_start
            best_score = _score_crop_candidate(img, best_x, 0, new_w, h, sx, sy, sw, sh, facing)
            if debug:
                if shot_type == "close-up":
                    print(
                        f"  Subject wider than crop ({sw}>{new_w}) and shot is close-up; "
                        "forcing centered ideal crop to avoid wrong-side clipping."
                    )
                else:
                    print(
                        f"  Subject wider than crop ({sw}>{new_w}); forcing front-preserve "
                        f"with min edge gap {min_edge_gap_px}px."
                    )
        else:
            best_name, best_x = "center", center_x_start
            best_score = -1.0
            front_score = None
            candidate_scores: dict[str, tuple[int, float]] = {}
            for name, cx_start in candidates:
                s = _score_crop_candidate(img, cx_start, 0, new_w, h, sx, sy, sw, sh, facing)
                candidate_scores[name] = (cx_start, s)
                if debug:
                    print(f"  Crop candidate '{name}': x_start={cx_start}, score={s:.3f}")
                if name == "front_preserve":
                    front_score = s
                if s > best_score:
                    best_score = s
                    best_name = name
                    best_x = cx_start

            # Front-preserve is preferred when it is near-optimal.
            if (
                front_score is not None
                and facing in {"left", "right"}
                and front_score >= (best_score - 0.05)
            ):
                best_name = "front_preserve"
                best_x = front_preserve_x_start
                best_score = front_score

            # If the top-scoring crop rides a border with a large subject, prefer
            # a near-scoring alternative with materially better side breathing room.
            subject_fill_ratio = sw / max(1.0, float(new_w))
            _left_gap, _right_gap, best_min_gap = _subject_side_gap_ratios(best_x, new_w, sx, sw)
            tight_gap = MIN_FRONT_EDGE_GAP_RATIO * EDGE_RISK_TIGHT_GAP_MULT
            roomy_gap = MIN_FRONT_EDGE_GAP_RATIO * EDGE_RISK_ALT_GAP_MULT
            if subject_fill_ratio >= EDGE_RISK_SUBJECT_FILL_RATIO and best_min_gap <= tight_gap:
                alternative_name = None
                alternative_x = None
                alternative_score = None
                alternative_min_gap = None
                for name, (cx_start, score) in candidate_scores.items():
                    if name == best_name:
                        continue
                    _lg, _rg, min_gap = _subject_side_gap_ratios(cx_start, new_w, sx, sw)
                    if min_gap < roomy_gap:
                        continue
                    if score < (best_score - EDGE_RISK_ALT_MAX_SCORE_DELTA):
                        continue
                    if (
                        alternative_score is None
                        or score > alternative_score
                        or (
                            abs(score - alternative_score) < 1e-6
                            and min_gap > (alternative_min_gap or -1.0)
                        )
                    ):
                        alternative_name = name
                        alternative_x = cx_start
                        alternative_score = score
                        alternative_min_gap = min_gap
                if (
                    alternative_name is not None
                    and alternative_x is not None
                    and alternative_score is not None
                ):
                    if debug:
                        print(
                            "  Edge-risk override: replacing "
                            f"'{best_name}' with '{alternative_name}' "
                            f"(best_min_gap={best_min_gap:.3f}, alt_min_gap={alternative_min_gap:.3f})."
                        )
                    best_name = alternative_name
                    best_x = alternative_x
                    best_score = alternative_score

        if debug:
            for name, cx_start in candidates:
                if sw > new_w and name != "front_preserve":
                    continue
                s = _score_crop_candidate(img, cx_start, 0, new_w, h, sx, sy, sw, sh, facing)
                print(f"  Crop candidate '{name}': x_start={cx_start}, score={s:.3f}")
            print(f"  Selected: '{best_name}' (x_start={best_x}, score={best_score:.3f})")

        selected_candidate_name = best_name
        crop_origin_x = best_x
        crop_origin_y = 0
        crop_w = new_w
        crop_h = h
        cropped = img[:, best_x : best_x + new_w]

    else:
        # Too tall → crop top/bottom
        new_h = max(1, min(h, int(w / target_ratio)))
        min_top_gap_px = max(8, int(new_h * MIN_TOP_EDGE_GAP_RATIO))

        # Candidate A: place subject at ideal y
        ideal_y_start = int(center_y - ideal_y_norm * new_h)
        ideal_y_start = max(0, min(ideal_y_start, h - new_h))

        # Candidate B: ensure full subject inclusion
        safe_y_start = center_y - new_h // 2
        if safe_y_start > sy:
            safe_y_start = sy
        if safe_y_start + new_h < sy + sh:
            safe_y_start = sy + sh - new_h
        safe_y_start = max(0, min(safe_y_start, h - new_h))

        # Candidate C: preserve head/top edge with breathing room.
        top_preserve_y_start = sy - min_top_gap_px
        top_preserve_y_start = max(0, min(top_preserve_y_start, h - new_h))

        # Candidate C: pure center
        center_y_start = max(0, (h - new_h) // 2)

        candidates = [
            ("top_preserve", top_preserve_y_start),
            ("ideal", ideal_y_start),
            ("safe", safe_y_start),
            ("center", center_y_start),
        ]

        # If the subject is taller than crop height, clipping is unavoidable.
        # For rider/person portraits, preserve head/top first.
        if sh > new_h and class_name in {"person", "rider_motorcycle"}:
            best_name, best_y = "top_preserve", top_preserve_y_start
            best_score = _score_crop_candidate(img, 0, best_y, w, new_h, sx, sy, sw, sh, facing)
            if debug:
                print(
                    f"  Subject taller than crop ({sh}>{new_h}) for {class_name}; "
                    "forcing top-preserve to protect headroom."
                )
        else:
            best_name, best_y = "center", center_y_start
            best_score = -1.0
            for name, cy_start in candidates:
                s = _score_crop_candidate(img, 0, cy_start, w, new_h, sx, sy, sw, sh, facing)
                if debug:
                    print(f"  Crop candidate '{name}': y_start={cy_start}, score={s:.3f}")
                if s > best_score:
                    best_score = s
                    best_name = name
                    best_y = cy_start

        if debug:
            print(f"  Selected: '{best_name}' (y_start={best_y}, score={best_score:.3f})")

        selected_candidate_name = best_name
        crop_origin_x = 0
        crop_origin_y = best_y
        crop_w = w
        crop_h = new_h
        cropped = img[best_y : best_y + new_h, :]

    crop_meta = _crop_uncertainty_flags(
        sx=sx,
        sy=sy,
        sw=sw,
        sh=sh,
        crop_x=crop_origin_x,
        crop_y=crop_origin_y,
        crop_w=crop_w,
        crop_h=crop_h,
        img_w=w,
        img_h=h,
    )
    crop_meta.update(
        {
            "selected_candidate": selected_candidate_name,
            "crop_window_xywh": [int(crop_origin_x), int(crop_origin_y), int(crop_w), int(crop_h)],
        }
    )
    if meta_out is not None:
        meta_out.clear()
        meta_out.update(crop_meta)

    # Debug visualization with grid overlay
    if debug or save_debug:
        debug_crop = cropped.copy()
        ch_d, cw_d = debug_crop.shape[:2]
        # Draw rule-of-thirds grid
        for frac in (1 / 3, 2 / 3):
            gx = int(cw_d * frac)
            gy = int(ch_d * frac)
            cv2.line(debug_crop, (gx, 0), (gx, ch_d), (255, 255, 0), 1)
            cv2.line(debug_crop, (0, gy), (cw_d, gy), (255, 255, 0), 1)
        # Draw Phi grid
        for frac in (0.382, 0.618):
            gx = int(cw_d * frac)
            gy = int(ch_d * frac)
            cv2.line(debug_crop, (gx, 0), (gx, ch_d), (0, 255, 255), 1)
            cv2.line(debug_crop, (0, gy), (cw_d, gy), (0, 255, 255), 1)

        # Draw subject/object boxes in crop coordinates.
        if raw_bbox is not None:
            rx, ry, rw, rh = raw_bbox
            rx1 = int(np.clip(rx - crop_origin_x, 0, cw_d - 1))
            ry1 = int(np.clip(ry - crop_origin_y, 0, ch_d - 1))
            rx2 = int(np.clip(rx + rw - crop_origin_x, 0, cw_d - 1))
            ry2 = int(np.clip(ry + rh - crop_origin_y, 0, ch_d - 1))
            if rx2 > rx1 and ry2 > ry1:
                cv2.rectangle(debug_crop, (rx1, ry1), (rx2, ry2), (0, 140, 255), 2)
                cv2.putText(
                    debug_crop,
                    "raw box",
                    (rx1, max(16, ry1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 140, 255),
                    1,
                )
        if expanded_bbox is not None:
            ex, ey, ew, eh = expanded_bbox
            ex1 = int(np.clip(ex - crop_origin_x, 0, cw_d - 1))
            ey1 = int(np.clip(ey - crop_origin_y, 0, ch_d - 1))
            ex2 = int(np.clip(ex + ew - crop_origin_x, 0, cw_d - 1))
            ey2 = int(np.clip(ey + eh - crop_origin_y, 0, ch_d - 1))
            if ex2 > ex1 and ey2 > ey1:
                cv2.rectangle(debug_crop, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)
                cv2.putText(
                    debug_crop,
                    "subject box",
                    (ex1, max(32, ey1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        debug_path = output_path.parent / f"debug_yolo_{output_path.name}"
        _write_image(debug_path, debug_crop)

        # Full-frame debug with selected crop window and object boxes.
        source_debug = img.copy()
        if raw_bbox is not None:
            rx, ry, rw, rh = raw_bbox
            cv2.rectangle(source_debug, (rx, ry), (rx + rw, ry + rh), (0, 140, 255), 3)
            cv2.putText(
                source_debug,
                "raw box",
                (rx, max(20, ry - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 140, 255),
                2,
            )
        if expanded_bbox is not None:
            ex, ey, ew, eh = expanded_bbox
            cv2.rectangle(source_debug, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            cv2.putText(
                source_debug,
                "subject box",
                (ex, max(45, ey - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        cv2.rectangle(
            source_debug,
            (crop_origin_x, crop_origin_y),
            (crop_origin_x + crop_w, crop_origin_y + crop_h),
            (255, 0, 255),
            3,
        )
        cv2.putText(
            source_debug,
            f"crop: {selected_candidate_name}",
            (crop_origin_x + 8, max(30, crop_origin_y + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 255),
            2,
        )
        source_debug_path = output_path.parent / f"debug_yolo_source_{output_path.name}"
        _write_image(source_debug_path, source_debug)

        debug_meta_path = output_path.parent / f"debug_yolo_{output_path.name}.json"
        debug_meta = {
            "image_path": str(image_path),
            "output_path": str(output_path),
            "image_size": {"width": w, "height": h},
            "detection_method": detection_method,
            "class_name": class_name,
            "confidence": round(float(conf), 6),
            "shot_type": shot_type,
            "facing": facing,
            "subject_center_xy": [int(center_x), int(center_y)],
            "raw_bbox_xywh": list(raw_bbox) if raw_bbox is not None else None,
            "expanded_bbox_xywh": list(expanded_bbox) if expanded_bbox is not None else None,
            "crop_window_xywh": [int(crop_origin_x), int(crop_origin_y), int(crop_w), int(crop_h)],
            "selected_candidate": selected_candidate_name,
            "uncertain_crop": bool(crop_meta["uncertain_crop"]),
            "uncertain_crop_reasons": crop_meta["uncertain_crop_reasons"],
            "min_subject_gap_px": crop_meta["min_subject_gap_px"],
            "min_subject_gap_ratio": crop_meta["min_subject_gap_ratio"],
            "target_size": {"width": int(out_w), "height": int(out_h)},
        }
        _atomic_write_json(debug_meta_path, debug_meta)

        if debug:
            print(f"Debug crop with grid saved to {debug_path}")
            print(f"Debug source overlay saved to {source_debug_path}")
            print(f"Debug metadata saved to {debug_meta_path}")

    # Resize to exact output dimensions
    final = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    _write_image(output_path, final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path


def _prepare_claude_crop_first_candidates(
    candidates: list[ImageScore],
    *,
    work_folder: Path,
    cropper: Callable[..., Path] = smart_crop,
) -> list[ImageScore]:
    """Pre-crop candidates to 1080x1440 before Claude vision scoring."""
    prepared: list[ImageScore] = []
    crop_first_dir = work_folder / "claude_crop_first"
    crop_first_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(candidates, start=1):
        source_for_name = item.source_path or item.path
        pre_crop_path = crop_first_dir / f"{idx:04d}_{source_for_name.stem}.jpg"
        try:
            cropper(
                item.path,
                pre_crop_path,
                out_w=OUTPUT_WIDTH,
                out_h=OUTPUT_HEIGHT,
                debug=False,
                save_debug=False,
                use_yolo=True,
            )
            prepared.append(
                ImageScore(
                    path=pre_crop_path,
                    source_path=item.source_path or item.path,
                    technical=dict(item.technical),
                )
            )
        except Exception as e:
            print(f"  ⚠ Claude crop-first prepare failed for {source_for_name.name}: {e}")
            prepared.append(item)

    return prepared


def _crop_one_image_no_debug(
    args: tuple, *, cropper: Callable[..., Path] = smart_crop
) -> tuple[int, dict]:
    """Smart-crop one image without debug output. For dedup-only mode."""
    idx, src_path, dest_path = args
    meta: dict[str, object] = {}
    try:
        cropper(src_path, dest_path, save_debug=False, meta_out=meta)
    except Exception:
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS)
                img.save(dest_path, "JPEG", quality=95)
            meta = {"uncertain_crop": True}
        except Exception:
            meta = {"_failed": True}
    return (idx, meta)


def _crop_one_image(args: tuple, *, cropper: Callable[..., Path] = smart_crop) -> tuple[int, dict]:
    """Smart-crop one image for Stage 4. Module-level for ProcessPoolExecutor pickling."""
    idx, src_path, dest_path = args
    meta: dict[str, object] = {}
    try:
        cropper(src_path, dest_path, save_debug=True, meta_out=meta)
    except Exception:
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS)
                img.save(dest_path, "JPEG", quality=95)
            meta = {
                "uncertain_crop": True,
                "uncertain_crop_reasons": ["smart_crop_failed_used_center_resize_fallback"],
            }
        except Exception:
            meta = {"_failed": True}
    return (idx, meta)
