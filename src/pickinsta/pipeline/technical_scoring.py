"""Technical image-quality scoring and cache orchestration."""

from __future__ import annotations

import concurrent.futures
import hashlib
import inspect
import json
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from pickinsta.config import WorkerSettings, resolve_worker_settings
from pickinsta.events import EventLevel, emit
from pickinsta.infrastructure.filesystem import atomic_write_text
from pickinsta.models import ImageScore
from pickinsta.pipeline.worker_tuning import bounded_worker_count

TECHNICAL_CACHE_SCHEMA_VERSION = 2
SubjectDetector = Callable[[np.ndarray], Optional[np.ndarray]]
TechnicalScorer = Callable[[Path], dict]
CacheObserver = Callable[[bool], None]


def detect_subject_mask(
    img: np.ndarray, detect_subject: Callable[..., object]
) -> Optional[np.ndarray]:
    """Return a binary mask around the primary detected subject."""
    detection = detect_subject(img, debug=False)
    if detection is None:
        return None
    x, y, w, h, _cls, _conf = detection
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    return mask


def composition_score(center_x_norm: float, center_y_norm: float) -> float:
    """Score subject placement against thirds and phi-grid power points."""
    thirds = ((0.333, 0.333), (0.667, 0.333), (0.333, 0.667), (0.667, 0.667))
    phi = ((0.382, 0.382), (0.618, 0.382), (0.382, 0.618), (0.618, 0.618))
    best_dist = min(np.hypot(center_x_norm - x, center_y_norm - y) for x, y in (*thirds, *phi))
    return float(np.exp(-0.5 * (best_dist / 0.12) ** 2))


def horizon_tilt_penalty(gray: np.ndarray) -> float:
    """Estimate horizon tilt and return 1.0 for a level image."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return 0.8
    angles = []
    for line in lines:
        coordinates = np.asarray(line).reshape(-1)
        if coordinates.size < 4:
            continue
        x1, y1, x2, y2 = coordinates[:4]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 15 or angle > 165:
            angles.append(min(angle, 180 - angle))
    if not angles:
        return 0.8
    median_tilt = float(np.median(angles))
    return 1.0 if median_tilt <= 2.0 else max(0.0, 1.0 - (median_tilt - 2.0) / 10.0)


def lead_room_score(img: np.ndarray, subject_mask: Optional[np.ndarray]) -> float:
    """Estimate whether the detected subject has useful space ahead."""
    if subject_mask is None:
        return 0.5
    _h, width = img.shape[:2]
    cols = np.where(subject_mask.any(axis=0))[0]
    if len(cols) == 0:
        return 0.5
    center_x = float(cols.mean()) / width
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gradient[subject_mask == 0] = 0
    space_ahead = 1.0 - center_x if gradient.mean() >= 0 else center_x
    if 0.55 <= space_ahead <= 0.75:
        return 1.0
    if 0.45 <= space_ahead <= 0.85:
        return 0.7
    return 0.3


def colorfulness_metric(img: np.ndarray) -> float:
    """Return the normalized Hasler-Suesstrunk colorfulness metric."""
    blue, green, red = (
        img[:, :, 0].astype(float),
        img[:, :, 1].astype(float),
        img[:, :, 2].astype(float),
    )
    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue
    sigma = np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2)
    mean = np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
    return min((sigma + 0.3 * mean) / 80.0, 1.0)


def score_technical(image_path: Path, *, detect_subject_mask: SubjectDetector) -> dict:
    """Score an image on seven normalized technical-quality metrics."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    scores: dict[str, float] = {}
    subject_mask = detect_subject_mask(img)
    bg_mask = cv2.bitwise_not(subject_mask) if subject_mask is not None else None

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    if subject_mask is not None:
        subject_laplacian = laplacian.copy()
        subject_laplacian[subject_mask == 0] = 0
        subject_pixels = max(1, int(subject_mask.sum() / 255))
        subject_variance = float((subject_laplacian**2).sum() / subject_pixels)
    else:
        subject_variance = float(laplacian.var())
    scores["sharpness"] = min(subject_variance / (500.0 * (width / 1080.0)), 1.0)

    if subject_mask is not None and bg_mask is not None:
        bg_laplacian = laplacian.copy()
        bg_laplacian[bg_mask == 0] = 0
        bg_pixels = max(1, int(bg_mask.sum() / 255))
        bg_variance = float((bg_laplacian**2).sum() / bg_pixels)
        scores["background_sep"] = min(subject_variance / max(bg_variance, 1e-6) / 5.0, 1.0)
    else:
        scores["background_sep"] = 0.5

    if subject_mask is not None:
        rows, cols = np.where(subject_mask > 0)
        center = (
            (float(cols.mean()) / width, float(rows.mean()) / height) if len(rows) else (0.5, 0.5)
        )
    else:
        center = (0.5, 0.5)
    scores["composition"] = (
        0.50 * composition_score(*center)
        + 0.25 * horizon_tilt_penalty(gray)
        + 0.25 * lead_room_score(img, subject_mask)
    )

    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_pixels = histogram.sum()
    clipping = min((histogram[:5].sum() + histogram[250:].sum()) / total_pixels, 0.04) / 0.04
    luminance = float(gray.mean())
    luminance_score = 1.0 if 90 <= luminance <= 170 else 0.7 if 60 <= luminance <= 200 else 0.4
    scores["lighting"] = luminance_score * (1.0 - clipping)

    colorfulness = colorfulness_metric(img)
    if subject_mask is not None and bg_mask is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        subject_hue = hsv[:, :, 0][subject_mask > 0].mean() if (subject_mask > 0).any() else 0
        bg_hue = hsv[:, :, 0][bg_mask > 0].mean() if (bg_mask > 0).any() else 0
        hue_difference = abs(float(subject_hue) - float(bg_hue))
        if hue_difference > 90:
            hue_difference = 180 - hue_difference
        hue_contrast = min(hue_difference / 60.0, 1.0)
    else:
        hue_contrast = 0.5
    scores["color_harmony"] = 0.6 * colorfulness + 0.4 * hue_contrast

    edges = cv2.Canny(gray, 50, 150)
    if bg_mask is not None:
        bg_edges = edges.copy()
        bg_edges[bg_mask == 0] = 0
        edge_density = float(bg_edges.sum() / 255) / max(1, int(bg_mask.sum() / 255))
    else:
        edge_density = float(edges.sum() / 255) / (height * width)
    scores["visual_clutter"] = max(0.0, 1.0 - edge_density * 10.0)

    contrast = min(gray.std() / 80.0, 1.0)
    saturation = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean() / 255.0
    scores["aesthetic"] = 0.5 * contrast + 0.5 * (1.0 - abs(saturation - 0.5) * 2.0)
    weights = {
        "sharpness": 0.18,
        "background_sep": 0.12,
        "composition": 0.20,
        "lighting": 0.18,
        "color_harmony": 0.13,
        "visual_clutter": 0.12,
        "aesthetic": 0.07,
    }
    scores["composite"] = sum(scores[key] * weight for key, weight in weights.items())
    return scores


def print_score_distribution(results: list[ImageScore]) -> None:
    """Print summary statistics and a histogram for technical composites."""
    if not results:
        emit(
            "technical.distribution.empty",
            "  📈 Score distribution (n=0): no successful scores",
            stage="technical_scoring",
        )
        return
    values = np.array([item.technical.get("composite", 0.0) for item in results])
    emit(
        "technical.distribution.header",
        f"  📈 Score distribution (n={len(values)}):",
        stage="technical_scoring",
        count=len(values),
    )
    emit(
        "technical.distribution.stats",
        f"     min={values.min():.3f}  max={values.max():.3f}  mean={values.mean():.3f}  median={float(np.median(values)):.3f}  std={values.std():.3f}",
        stage="technical_scoring",
    )
    metrics = (
        "sharpness",
        "background_sep",
        "composition",
        "lighting",
        "color_harmony",
        "visual_clutter",
        "aesthetic",
    )
    averages = {
        name: np.mean([item.technical.get(name, 0.0) for item in results]) for name in metrics
    }
    emit(
        "technical.distribution.metrics",
        "     metric avgs: " + ", ".join(f"{name}={averages[name]:.2f}" for name in metrics),
        stage="technical_scoring",
    )
    counts, edges = np.histogram(values, bins=10, range=(0.0, 1.0))
    maximum = int(counts.max()) if counts.max() > 0 else 1
    emit("technical.distribution.histogram", "     ┌" + "─" * 48 + "┐", stage="technical_scoring")
    for index, count in enumerate(counts):
        bar = "█" * int(count / maximum * 30)
        emit(
            "technical.distribution.bin",
            f"     │ {edges[index]:.1f}–{edges[index + 1]:.1f} │ {bar:<30} {count:>3} │",
            stage="technical_scoring",
            bin=index,
            count=int(count),
        )
    emit("technical.distribution.histogram", "     └" + "─" * 48 + "┘", stage="technical_scoring")


def tech_cache_path(image_path: Path) -> Path:
    return image_path.with_suffix(image_path.suffix + ".techscore.json")


@lru_cache(maxsize=1)
def technical_algorithm_fingerprint() -> str:
    """Identify the technical algorithm and numerical dependencies."""
    functions = (
        score_technical,
        composition_score,
        horizon_tilt_penalty,
        lead_room_score,
        colorfulness_metric,
    )
    identity = {
        "implementation": "\n".join(inspect.getsource(fn) for fn in functions),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_tech_cache(
    image_path: Path, *, fingerprint: Callable[[], str] = technical_algorithm_fingerprint
) -> Optional[dict]:
    cache = tech_cache_path(image_path)
    if not cache.exists():
        return None
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != TECHNICAL_CACHE_SCHEMA_VERSION
    ):
        return None
    try:
        if (
            payload.get("algorithm_fingerprint") != fingerprint()
            or payload.get("mtime") != image_path.stat().st_mtime
        ):
            return None
    except OSError:
        return None
    scores = payload.get("scores")
    return scores if isinstance(scores, dict) else None


def json_safe_cache_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe_cache_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_cache_value(item) for item in value]
    return value


def save_tech_cache(
    image_path: Path,
    scores: dict,
    *,
    fingerprint: Callable[[], str] = technical_algorithm_fingerprint,
) -> None:
    payload = {
        "schema_version": TECHNICAL_CACHE_SCHEMA_VERSION,
        "algorithm_fingerprint": fingerprint(),
        "mtime": image_path.stat().st_mtime,
        "scores": json_safe_cache_value(scores),
    }
    try:
        atomic_write_text(tech_cache_path(image_path), json.dumps(payload))
    except Exception:
        pass


def score_technical_with_cache(
    image_path: Path,
    *,
    scorer: TechnicalScorer = score_technical,
    loader=load_tech_cache,
    saver=save_tech_cache,
) -> Optional[tuple[Path, dict]]:
    try:
        cached = loader(image_path)
        if cached is not None:
            return image_path, cached
        scores = scorer(image_path)
        saver(image_path, scores)
        return image_path, scores
    except Exception:
        return None


def batch_technical_score(
    images: list[Path],
    source_map: Optional[dict[Path, Path]] = None,
    worker_settings: WorkerSettings | None = None,
    *,
    loader=load_tech_cache,
    score_cached=score_technical_with_cache,
    distribution=print_score_distribution,
    cache_observer: CacheObserver | None = None,
    failure_observer: Callable[[Path, str, str], None] | None = None,
) -> list[ImageScore]:
    if not images:
        return []
    worker_settings = worker_settings or resolve_worker_settings()
    results: list[ImageScore] = []
    cached_results, to_score = [], []
    for image_path in images:
        cached = loader(image_path)
        if cache_observer is not None:
            cache_observer(cached is not None)
        (cached_results if cached is not None else to_score).append(
            (image_path, cached) if cached is not None else image_path
        )
    for image_path, scores in cached_results:
        results.append(
            ImageScore(
                path=image_path,
                source_path=source_map.get(image_path) if source_map else None,
                technical=scores,
            )
        )
    if to_score:
        workers = bounded_worker_count("thread", len(to_score), worker_settings)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for image_path, result in zip(to_score, pool.map(score_cached, to_score), strict=True):
                if result is None:
                    emit(
                        "technical.image.failed",
                        "  ⚠ Tech score failed for an image",
                        level=EventLevel.WARNING,
                        stage="technical_scoring",
                    )
                    if failure_observer is not None:
                        failure_observer(image_path, "technical_scoring", "technical score failed")
                    continue
                image_path, scores = result
                results.append(
                    ImageScore(
                        path=image_path,
                        source_path=source_map.get(image_path) if source_map else None,
                        technical=scores,
                    )
                )
    results.sort(key=lambda item: item.technical.get("composite", 0), reverse=True)
    if not results:
        emit(
            "technical.all.failed",
            "  ⚠ Technical scoring failed for all images",
            level=EventLevel.WARNING,
            stage="technical_scoring",
        )
        distribution(results)
        return []
    emit(
        "technical.complete",
        f"  ✅ Technical scoring complete ({len(cached_results)}/{len(results)} cached). Top: {results[0].path.name} ({results[0].technical['composite']:.3f})",
        stage="technical_scoring",
        scored=len(results),
        cached=len(cached_results),
        top=results[0].path.name,
    )
    distribution(results)
    return results
