"""Pure subject-box and crop-window geometry.

No filesystem, network, model, or selector access belongs in this module.
"""

from typing import Optional

import cv2
import numpy as np

MIN_FRONT_EDGE_GAP_RATIO = 0.03
YOLO_BBOX_PAD_RATIO = 0.06
YOLO_BBOX_PAD_MIN_PX = 12
CROP_UNCERTAIN_EDGE_GAP_RATIO = 0.02
CROP_UNCERTAIN_EDGE_GAP_MIN_PX = 8


def composition_score(center_x_norm: float, center_y_norm: float) -> float:
    """Score placement against thirds and phi-grid power points."""
    thirds = ((0.333, 0.333), (0.667, 0.333), (0.333, 0.667), (0.667, 0.667))
    phi = ((0.382, 0.382), (0.618, 0.382), (0.382, 0.618), (0.618, 0.618))
    best_dist = min(np.hypot(center_x_norm - x, center_y_norm - y) for x, y in (*thirds, *phi))
    return float(np.exp(-0.5 * (best_dist / 0.12) ** 2))


def _bbox_iou_xywh(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """IoU for two XYWH boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def _bbox_center_distance_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> float:
    """Center distance normalized by frame diagonal."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    dist = float(np.hypot(acx - bcx, acy - bcy))
    diag = float(np.hypot(max(1, img_w), max(1, img_h)))
    return dist / max(diag, 1.0)


def _bbox_union_xywh(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Union box for two XYWH boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return x1, y1, x2 - x1, y2 - y1


def _classify_shot_type(subject_area_ratio: float) -> str:
    """Classify shot type from subject area as fraction of frame."""
    if subject_area_ratio >= 0.50:
        return "close-up"
    if subject_area_ratio >= 0.20:
        return "medium"
    if subject_area_ratio >= 0.10:
        return "environmental"
    if subject_area_ratio >= 0.05:
        return "scenic"
    return "extreme_wide"


def _guess_facing_direction(img: np.ndarray, sx: int, sy: int, sw: int, sh: int) -> str:
    """Guess whether the subject faces left, right, or is head-on.

    Uses multiple signals inside the subject bbox:
      1. Edge density in left vs right outer thirds (robust for side profiles)
      2. Directional Sobel energy as fallback
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    subj_region = gray[sy : sy + sh, sx : sx + sw]
    if subj_region.size == 0:
        return "unknown"

    # Heuristic 1: edge asymmetry near outer thirds.
    edges = cv2.Canny(subj_region, 60, 160)
    third = max(1, subj_region.shape[1] // 3)
    left_edge_energy = float(edges[:, :third].mean())
    right_edge_energy = float(edges[:, -third:].mean())
    if left_edge_energy > right_edge_energy * 1.05:
        return "left"
    if right_edge_energy > left_edge_energy * 1.05:
        return "right"

    # Heuristic 2: directional Sobel energy.
    sobel = cv2.Sobel(subj_region, cv2.CV_64F, 1, 0, ksize=3)
    pos_energy = float(np.maximum(sobel, 0).mean())
    neg_energy = float(np.maximum(-sobel, 0).mean())
    if pos_energy > neg_energy * 1.05:
        return "right"
    if neg_energy > pos_energy * 1.05:
        return "left"
    return "head-on"


def _ideal_subject_x(facing: str, shot_type: str) -> float:
    """Return ideal normalized x for the subject center, respecting lead room.

    Per composition rules:
      - rightward-facing: x ≈ 0.33 (60-70% space ahead on right)
      - leftward-facing:  x ≈ 0.67
      - head-on:          x ≈ 0.50
      - close-up/extreme: x ≈ 0.50 (less lead room needed)
    """
    if shot_type in ("close-up", "extreme_wide"):
        return 0.50
    if facing == "right":
        return 0.35
    if facing == "left":
        return 0.65
    return 0.50


def _ideal_subject_y(shot_type: str) -> float:
    """Return ideal normalized y for the subject center.

    Per composition rules:
      - low-angle / environmental: y ≈ 0.60-0.67 (lower-third area)
      - medium / eye-level:         y ≈ 0.50
      - close-up:                   y ≈ 0.50
    """
    if shot_type in ("environmental", "scenic", "extreme_wide"):
        return 0.62
    return 0.50


def _expand_subject_bbox(
    sx: int,
    sy: int,
    sw: int,
    sh: int,
    img_w: int,
    img_h: int,
    class_name: str = "",
    facing: str = "",
) -> tuple[int, int, int, int]:
    """Expand a detected subject bbox to avoid overly-tight crops.

    For motorcycles and rider+bike detections, bias the expansion toward the
    leading side so the front wheel/bumper is less likely to be clipped.
    """
    if class_name == "rider_motorcycle":
        pad_ratio_x = max(YOLO_BBOX_PAD_RATIO, 0.10)
        pad_ratio_y = max(YOLO_BBOX_PAD_RATIO, 0.08)
    elif class_name == "motorcycle":
        pad_ratio_x = max(YOLO_BBOX_PAD_RATIO, 0.08)
        pad_ratio_y = max(YOLO_BBOX_PAD_RATIO, 0.07)
    else:
        pad_ratio_x = YOLO_BBOX_PAD_RATIO
        pad_ratio_y = YOLO_BBOX_PAD_RATIO

    pad_x = max(YOLO_BBOX_PAD_MIN_PX, int(round(sw * pad_ratio_x)))
    pad_y = max(YOLO_BBOX_PAD_MIN_PX, int(round(sh * pad_ratio_y)))

    # Default to symmetric expansion.
    x0 = max(0, sx - pad_x)
    y0 = max(0, sy - pad_y)
    x1 = min(img_w, sx + sw + pad_x)
    y1 = min(img_h, sy + sh + pad_y)

    # Motorcycle crops need a little more room on the front side than on the rear.
    # This helps preserve the leading wheel when YOLO boxes are tight.
    if class_name in {"motorcycle", "rider_motorcycle"} and facing in {"left", "right"}:
        front_pad_x = max(YOLO_BBOX_PAD_MIN_PX, int(round(sw * 0.12)))
        rear_pad_x = max(4, int(round(sw * 0.04)))
        if facing == "right":
            x0 = max(0, sx - rear_pad_x)
            x1 = min(img_w, sx + sw + front_pad_x)
        else:
            x0 = max(0, sx - front_pad_x)
            x1 = min(img_w, sx + sw + rear_pad_x)

    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _horizontal_margin_bounds(
    sx: int,
    sw: int,
    crop_w: int,
    frame_w: int,
    min_gap_px: int,
) -> Optional[tuple[int, int]]:
    """Return crop start bounds that keep subject inside with side breathing room."""
    if sw + 2 * min_gap_px > crop_w:
        return None
    min_start = max(0, sx + sw + min_gap_px - crop_w)
    max_start = min(frame_w - crop_w, sx - min_gap_px)
    if min_start <= max_start:
        return int(min_start), int(max_start)
    return None


def _subject_side_gap_ratios(
    crop_x: int, crop_w: int, sx: int, sw: int
) -> tuple[float, float, float]:
    """Return (left_gap, right_gap, min_gap) normalized to crop width."""
    denom = max(1.0, float(crop_w))
    left_gap = (sx - crop_x) / denom
    right_gap = ((crop_x + crop_w) - (sx + sw)) / denom
    return left_gap, right_gap, min(left_gap, right_gap)


def _crop_uncertainty_flags(
    *,
    sx: int,
    sy: int,
    sw: int,
    sh: int,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    img_w: int,
    img_h: int,
) -> dict[str, object]:
    """Return crop uncertainty flags for deciding if padded fallback should be emitted."""
    left_gap_px = sx - crop_x
    right_gap_px = (crop_x + crop_w) - (sx + sw)
    top_gap_px = sy - crop_y
    bottom_gap_px = (crop_y + crop_h) - (sy + sh)
    min_gap_px = min(left_gap_px, right_gap_px, top_gap_px, bottom_gap_px)

    min_gap_ratio = min(
        left_gap_px / max(1.0, float(crop_w)),
        right_gap_px / max(1.0, float(crop_w)),
        top_gap_px / max(1.0, float(crop_h)),
        bottom_gap_px / max(1.0, float(crop_h)),
    )
    gap_px_threshold = max(
        CROP_UNCERTAIN_EDGE_GAP_MIN_PX,
        int(round(min(crop_w, crop_h) * CROP_UNCERTAIN_EDGE_GAP_RATIO)),
    )

    too_large_for_crop = sw > crop_w or sh > crop_h
    too_close_to_border = (
        min_gap_px < gap_px_threshold or min_gap_ratio < CROP_UNCERTAIN_EDGE_GAP_RATIO
    )
    clipped_subject = min_gap_px < 0
    subject_bbox_hits_frame_edge = sx <= 0 or sy <= 0 or (sx + sw) >= img_w or (sy + sh) >= img_h

    reasons: list[str] = []
    if too_large_for_crop:
        reasons.append("subject_larger_than_crop")
    if clipped_subject:
        reasons.append("subject_clipped_by_crop")
    if too_close_to_border:
        reasons.append("subject_too_close_to_crop_border")
    if subject_bbox_hits_frame_edge:
        reasons.append("subject_bbox_hits_image_edge")

    return {
        "too_large_for_crop": too_large_for_crop,
        "too_close_to_border": too_close_to_border,
        "clipped_subject": clipped_subject,
        "subject_bbox_hits_frame_edge": subject_bbox_hits_frame_edge,
        "edge_gap_px_threshold": int(gap_px_threshold),
        "min_subject_gap_px": int(min_gap_px),
        "min_subject_gap_ratio": float(min_gap_ratio),
        "uncertain_crop": len(reasons) > 0,
        "uncertain_crop_reasons": reasons,
    }


def _score_crop_candidate(
    img: np.ndarray,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    sx: int,
    sy: int,
    sw: int,
    sh: int,
    facing: str,
) -> float:
    """Score a candidate crop window against composition rules.

    Evaluates: subject on power-point, lead room ratio, subject not clipped.
    """
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("crop dimensions must be positive")

    # Subject center relative to the crop window
    subj_cx = (sx + sw / 2 - crop_x) / crop_w
    subj_cy = (sy + sh / 2 - crop_y) / crop_h

    # 1) Placement on power points (Thirds + Phi)
    placement = composition_score(subj_cx, subj_cy)

    # 2) Lead room (horizontal space ahead of facing direction)
    if facing == "right":
        space_ahead = 1.0 - subj_cx
    elif facing == "left":
        space_ahead = subj_cx
    else:
        space_ahead = 0.65  # head-on: neutral

    if 0.55 <= space_ahead <= 0.75:
        lead_score = 1.0
    elif 0.45 <= space_ahead <= 0.85:
        lead_score = 0.7
    else:
        lead_score = 0.3

    # 3) Proportional clipping penalty (small clip != severe clip)
    overlap_left = max(sx, crop_x)
    overlap_top = max(sy, crop_y)
    overlap_right = min(sx + sw, crop_x + crop_w)
    overlap_bottom = min(sy + sh, crop_y + crop_h)
    overlap_w = max(0, overlap_right - overlap_left)
    overlap_h = max(0, overlap_bottom - overlap_top)
    overlap_ratio = (overlap_w * overlap_h) / max(1.0, float(sw * sh))
    clip_score = overlap_ratio**1.8

    # 4) Keep a small breathing gap between subject front and border.
    min_gap = MIN_FRONT_EDGE_GAP_RATIO
    left_gap = (sx - crop_x) / max(1.0, float(crop_w))
    right_gap = ((crop_x + crop_w) - (sx + sw)) / max(1.0, float(crop_w))
    if facing == "left":
        front_gap = left_gap
        rear_gap = right_gap
    elif facing == "right":
        front_gap = right_gap
        rear_gap = left_gap
    else:
        front_gap = min(left_gap, right_gap)
        rear_gap = front_gap
    front_gap_score = float(np.clip(front_gap / max(min_gap, 1e-6), 0.0, 1.0))
    rear_gap_score = float(np.clip(rear_gap / max(min_gap, 1e-6), 0.0, 1.0))
    edge_gap_score = 0.7 * front_gap_score + 0.3 * rear_gap_score

    return 0.15 * placement + 0.15 * lead_score + 0.55 * clip_score + 0.15 * edge_gap_score
