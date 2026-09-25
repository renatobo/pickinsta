"""Perceptual-hash grouping for the deduplication stage."""

from __future__ import annotations

import concurrent.futures
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np
from PIL import Image

from pickinsta.config import WorkerSettings, resolve_worker_settings
from pickinsta.events import EventLevel, emit
from pickinsta.pipeline.worker_tuning import bounded_worker_count

DEDUP_THRESHOLD = 8
HIST_DEDUP_THRESHOLD = 0.92
HIST_DEDUP_TEMPORAL_THRESHOLD = 0.80
HIST_DEDUP_TEMPORAL_ORB_THRESHOLD = 0.60
HIST_DEDUP_THUMB_SIZE = (256, 256)
BURST_MAX_INTERVAL_SEC = 3.0
ORB_MATCH_THRESHOLD = 0.25

Histogram = np.ndarray | None
Descriptors = np.ndarray | None
DedupFeatures = tuple[Path, Histogram, float | None, float, Descriptors]


class PerceptualHash(Protocol):
    """Hash value whose subtraction returns a Hamming distance."""

    def __sub__(self, other: object, /) -> int: ...


@dataclass
class HashGroupingStats:
    """Optional operation counts for profiling hash grouping."""

    queries: int = 0
    distance_comparisons: int = 0


@dataclass
class _MetricNode:
    value: PerceptualHash
    group_index: int
    children: dict[int, _MetricNode]


class _HammingMetricIndex:
    """BK-tree over group representatives using Hamming distance."""

    def __init__(self, stats: HashGroupingStats) -> None:
        self._root: _MetricNode | None = None
        self._stats = stats

    def add(self, value: PerceptualHash, group_index: int) -> None:
        node = self._root
        if node is None:
            self._root = _MetricNode(value, group_index, {})
            return

        while True:
            distance = self._distance(value, node.value)
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _MetricNode(value, group_index, {})
                return
            node = child

    def first_within(self, value: PerceptualHash, threshold: int) -> int | None:
        self._stats.queries += 1
        if self._root is None or threshold < 0:
            return None

        earliest: int | None = None
        pending = [self._root]
        while pending:
            node = pending.pop()
            distance = self._distance(value, node.value)
            if distance <= threshold and (earliest is None or node.group_index < earliest):
                earliest = node.group_index
            lower = distance - threshold
            upper = distance + threshold
            pending.extend(child for edge, child in node.children.items() if lower <= edge <= upper)
        return earliest

    def _distance(self, left: PerceptualHash, right: PerceptualHash) -> int:
        self._stats.distance_comparisons += 1
        return abs(left - right)


def group_perceptual_hashes(
    images: list[Path],
    path_hash_map: dict[Path, PerceptualHash],
    threshold: int,
    *,
    stats: HashGroupingStats | None = None,
) -> dict[PerceptualHash, list[Path]]:
    """Group hashes while preserving the legacy first-match ordering semantics."""
    operation_stats = stats if stats is not None else HashGroupingStats()
    index = _HammingMetricIndex(operation_stats)
    representatives: list[PerceptualHash] = []
    members: list[list[Path]] = []

    for image_path in images:
        image_hash = path_hash_map.get(image_path)
        if image_hash is None:
            continue
        group_index = index.first_within(image_hash, threshold)
        if group_index is not None:
            members[group_index].append(image_path)
            continue

        group_index = len(representatives)
        representatives.append(image_hash)
        members.append([image_path])
        index.add(image_hash, group_index)

    return dict(zip(representatives, members))


def image_histogram(img_path: Path) -> Histogram:
    """Compute a normalized color histogram for burst-shot deduplication."""
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            return None
        thumbnail = cv2.resize(image, HIST_DEDUP_THUMB_SIZE)
        histogram = cv2.calcHist([thumbnail], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(histogram, histogram)
        return histogram
    except Exception:
        return None


def exif_timestamp(img_path: Path) -> float | None:
    """Extract EXIF DateTimeOriginal as epoch seconds, if present."""
    try:
        from PIL.ExifTags import TAGS

        with Image.open(img_path) as image:
            raw = image._getexif()
            if not raw:
                return None
            tagged = {TAGS.get(key, key): value for key, value in raw.items()}
            datetime_text = tagged.get("DateTimeOriginal") or tagged.get("DateTime")
            if not datetime_text:
                return None
            # Pillow's public TAGS mapping does not include these standard EXIF
            # subsecond tags in every supported release, so retain numeric fallbacks.
            subseconds = (
                tagged.get("SubSecTimeOriginal")
                or tagged.get("SubsecTimeOriginal")
                or tagged.get(0x9291)
                or tagged.get("SubSecTime")
                or tagged.get("SubsecTime")
                or tagged.get(0x9290)
                or "0"
            )
            timestamp = datetime.strptime(str(datetime_text), "%Y:%m:%d %H:%M:%S").timestamp()
            return timestamp + float(f"0.{subseconds}")
    except Exception:
        return None


def quick_sharpness(img_path: Path) -> float:
    """Return Laplacian variance used to select a group representative."""
    try:
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 0.0
        return float(cv2.Laplacian(image, cv2.CV_64F).var())
    except Exception:
        return 0.0


def compute_phash(img_path: Path) -> tuple[Path, object] | None:
    """Compute one perceptual hash in a process-pool-safe function."""
    try:
        import imagehash

        with Image.open(img_path) as image:
            return img_path, imagehash.phash(image, hash_size=16)
    except Exception:
        return None


def compute_orb_descriptors(img_path: Path) -> Descriptors:
    """Compute ORB descriptors used to confirm burst membership."""
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            return None
        thumbnail = cv2.resize(image, (512, 512))
        gray = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
        _, descriptors = cv2.ORB_create(500).detectAndCompute(gray, None)
        return descriptors
    except Exception:
        return None


def orb_match_ratio(left: Descriptors, right: Descriptors) -> float:
    """Compute the fraction of ORB matches below the legacy distance limit."""
    if left is None or right is None:
        return 0.0
    try:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(left, right)
        if not matches:
            return 0.0
        return sum(match.distance < 50 for match in matches) / len(matches)
    except Exception:
        return 0.0


def compute_dedup_features(
    img_path: Path,
    *,
    histogram_function: Callable[[Path], Histogram] = image_histogram,
    timestamp_function: Callable[[Path], float | None] = exif_timestamp,
    sharpness_function: Callable[[Path], float] = quick_sharpness,
    orb_function: Callable[[Path], Descriptors] = compute_orb_descriptors,
) -> DedupFeatures:
    """Compute all second-pass burst features for one image."""
    return (
        img_path,
        histogram_function(img_path),
        timestamp_function(img_path),
        sharpness_function(img_path),
        orb_function(img_path),
    )


def _group_bursts(
    features: list[DedupFeatures],
    orb_ratio_function: Callable[[Descriptors, Descriptors], float],
) -> list[list[DedupFeatures]]:
    features.sort(key=lambda entry: entry[2] if entry[2] is not None else float("inf"))
    groups: list[list[DedupFeatures]] = []
    bucket_width = BURST_MAX_INTERVAL_SEC
    timed_buckets: dict[int, set[int]] = defaultdict(set)
    untimed_groups: list[int] = []

    def bucket(timestamp: float) -> int:
        return math.floor(timestamp / bucket_width)

    for entry in features:
        _, histogram, timestamp, _, descriptors = entry
        if histogram is None:
            groups.append([entry])
            continue
        placed = False
        if timestamp is None:
            candidate_indexes = untimed_groups
        else:
            first_bucket = bucket(timestamp - BURST_MAX_INTERVAL_SEC)
            last_bucket = bucket(timestamp)
            candidate_indexes = sorted(
                index
                for bucket_index in range(first_bucket, last_bucket + 1)
                for index in timed_buckets.get(bucket_index, ())
            )

        for group_index in candidate_indexes:
            group = groups[group_index]
            _, last_histogram, last_timestamp, _, last_descriptors = group[-1]
            if timestamp is not None and last_timestamp is not None:
                time_to_last = abs(timestamp - last_timestamp)
                if time_to_last > BURST_MAX_INTERVAL_SEC:
                    continue
            elif timestamp is not None or last_timestamp is not None:
                continue
            if last_histogram is None or last_histogram.size == 0:
                continue
            correlation = cv2.compareHist(last_histogram, histogram, cv2.HISTCMP_CORREL)
            temporal = (
                timestamp is not None
                and last_timestamp is not None
                and time_to_last <= BURST_MAX_INTERVAL_SEC
            )
            orb_ratio = orb_ratio_function(last_descriptors, descriptors)
            if temporal and orb_ratio >= ORB_MATCH_THRESHOLD:
                if correlation < HIST_DEDUP_TEMPORAL_ORB_THRESHOLD:
                    continue
            elif temporal:
                if correlation < HIST_DEDUP_TEMPORAL_THRESHOLD:
                    continue
                if orb_ratio < ORB_MATCH_THRESHOLD:
                    continue
            else:
                if correlation < HIST_DEDUP_THRESHOLD:
                    continue
                if orb_ratio < ORB_MATCH_THRESHOLD:
                    continue
            group.append(entry)
            if last_timestamp is None:
                # Untimed groups remain isolated from timestamped images.
                pass
            else:
                timed_buckets[bucket(last_timestamp)].discard(group_index)
                timed_buckets[bucket(timestamp)].add(group_index)
            placed = True
            break
        if not placed:
            group_index = len(groups)
            groups.append([entry])
            if timestamp is None:
                untimed_groups.append(group_index)
            else:
                timed_buckets[bucket(timestamp)].add(group_index)
    return groups


def deduplicate(
    images: list[Path],
    threshold: int = DEDUP_THRESHOLD,
    worker_settings: WorkerSettings | None = None,
    *,
    phash_function: Callable[[Path], tuple[Path, object] | None] = compute_phash,
    sharpness_function: Callable[[Path], float] = quick_sharpness,
    features_function: Callable[[Path], DedupFeatures] = compute_dedup_features,
    orb_ratio_function: Callable[[Descriptors, Descriptors], float] = orb_match_ratio,
) -> tuple[list[Path], dict[Path, list[Path]]]:
    """Remove perceptual duplicates and burst frames with stable legacy ordering."""
    if not images:
        return [], {}

    settings = worker_settings or resolve_worker_settings()
    worker_count = bounded_worker_count("process", len(images), settings)
    path_hash_map: dict[Path, PerceptualHash] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
        for result in pool.map(phash_function, images):
            if result is None:
                emit(
                    "dedup.hash.failed",
                    "  ⚠ Hash failed for an image",
                    level=EventLevel.WARNING,
                    stage="deduplication",
                )
                continue
            img_path, image_hash = result
            path_hash_map[img_path] = image_hash  # type: ignore[assignment]

    hash_groups = group_perceptual_hashes(images, path_hash_map, threshold)
    sharpness_cache: dict[Path, float] = {}
    multi_groups = [group for group in hash_groups.values() if len(group) > 1]
    if multi_groups:
        multi_paths = [path for group in multi_groups for path in group]
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
            for path, sharpness in zip(
                multi_paths, pool.map(sharpness_function, multi_paths), strict=False
            ):
                sharpness_cache[path] = sharpness

    representative_by_path: dict[Path, Path] = {}
    hash_removed = 0
    for group in hash_groups.values():
        best = (
            group[0]
            if len(group) == 1
            else max(group, key=lambda path: sharpness_cache.get(path, 0.0))
        )
        representative_by_path.update(dict.fromkeys(group, best))
        hash_removed += len(group) - 1

    # A failed perceptual hash must degrade deduplication, not remove the image.
    # Build the representatives in source order so failed hashes remain stable and
    # a sharper representative still occupies its group's first input position.
    hash_unique: list[Path] = []
    emitted: set[Path] = set()
    for image_path in images:
        representative = representative_by_path.get(image_path, image_path)
        if representative not in emitted:
            hash_unique.append(representative)
            emitted.add(representative)

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
        features = list(pool.map(features_function, hash_unique))
    burst_groups = _group_bursts(features, orb_ratio_function)

    unique: list[Path] = []
    burst_map: dict[Path, list[Path]] = {}
    burst_removed = 0
    for group in burst_groups:
        best_path = max(group, key=lambda entry: entry[3])[0]
        paths = [entry[0] for entry in group]
        unique.append(best_path)
        if len(paths) > 1:
            burst_map[best_path] = paths
        burst_removed += len(group) - 1

    emit(
        "dedup.complete",
        f"  ✅ Dedup: {len(images)} → {len(unique)} unique "
        f"({hash_removed} hash dupes, {burst_removed} burst dupes removed)",
        stage="deduplication",
        input_count=len(images),
        unique_count=len(unique),
        hash_removed=hash_removed,
        burst_removed=burst_removed,
    )
    return unique, burst_map
