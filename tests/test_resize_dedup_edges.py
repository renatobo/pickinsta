from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from pickinsta import config
from pickinsta.pipeline import deduplication, resize


class _SequentialExecutor:
    created_with: list[int] = []

    def __init__(self, max_workers: int):
        self.created_with.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def map(self, function, items):
        return map(function, items)


@dataclass(frozen=True)
class _Hash:
    value: int

    def __sub__(self, other: object) -> int:
        if not isinstance(other, _Hash):
            return NotImplemented
        return (self.value ^ other.value).bit_count()


def _features(
    path: Path,
    histogram: np.ndarray | None = None,
    timestamp: float | None = None,
    sharpness: float = 0.0,
    descriptors: np.ndarray | None = None,
) -> deduplication.DedupFeatures:
    return path, histogram, timestamp, sharpness, descriptors


def test_resize_worker_handles_corrupt_input_and_normalizes_output(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    assert resize.resize_one_image((corrupt, tmp_path / "ignored.jpg")) is None

    source = tmp_path / "rgba.png"
    destination = tmp_path / "nested" / "out.jpg"
    destination.parent.mkdir()
    Image.new("RGBA", (2400, 1200), (255, 0, 0, 100)).save(source)
    assert resize.resize_one_image((source, destination)) == (destination, source)
    with Image.open(destination) as output:
        assert output.mode == "RGB"
        assert output.size == (1920, 960)


def test_resize_worker_accepts_read_only_input(tmp_path: Path) -> None:
    source = tmp_path / "read-only.jpg"
    destination = tmp_path / "output.jpg"
    Image.new("RGB", (32, 24), "blue").save(source)
    source.chmod(0o444)

    try:
        assert resize.resize_one_image((source, destination)) == (destination, source)
        assert source.stat().st_mode & 0o222 == 0
        with Image.open(destination) as output:
            assert output.size == (32, 24)
    finally:
        source.chmod(0o644)


def test_resize_empty_folder_does_not_create_executor(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _SequentialExecutor.created_with.clear()
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SequentialExecutor)

    assert resize.resize_for_processing(source, tmp_path / "work") == ([], {})
    assert _SequentialExecutor.created_with == []


def test_resize_skips_failed_worker_and_keeps_sorted_reuse_map(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    work = tmp_path / "work"
    source.mkdir()
    work.mkdir()
    for name in ("z.jpg", "a.jpg", "a.png"):
        (source / name).write_bytes(name.encode())
    reused = work / "z.jpg"
    reused.write_bytes(b"cached")
    reused.touch()
    _SequentialExecutor.created_with.clear()
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SequentialExecutor)

    def worker(pair: tuple[Path, Path]):
        return None if pair[0].suffix == ".png" else (pair[1], pair[0])

    outputs, source_map = resize.resize_for_processing(
        source,
        work,
        config.WorkerSettings(9, 1, None),
        resize_worker=worker,
    )

    assert [path.name for path in outputs] == ["a.jpg", "z.jpg"]
    assert source_map == {work / "a.jpg": source / "a.jpg", reused: source / "z.jpg"}
    assert _SequentialExecutor.created_with == [2]


class _ExifImage:
    def __init__(self, payload: dict[int, object] | None):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def _getexif(self):
        return self.payload


def test_exif_timestamp_supports_subseconds_and_rejects_malformed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        deduplication.Image,
        "open",
        lambda _path: _ExifImage({36867: "2026:07:18 12:00:00", 37521: "125"}),
    )
    timestamp = deduplication.exif_timestamp(tmp_path / "image.jpg")
    assert timestamp is not None
    assert timestamp % 1 == 0.125

    monkeypatch.setattr(
        deduplication.Image,
        "open",
        lambda _path: _ExifImage({36867: "not-a-date", 37521: "n/a"}),
    )
    assert deduplication.exif_timestamp(tmp_path / "image.jpg") is None


def test_low_level_features_degrade_safely_when_images_or_opencv_fail(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "missing.jpg"
    monkeypatch.setattr(deduplication.cv2, "imread", lambda *_args, **_kwargs: None)
    assert deduplication.image_histogram(path) is None
    assert deduplication.quick_sharpness(path) == 0.0
    assert deduplication.compute_orb_descriptors(path) is None
    assert deduplication.compute_phash(path) is None
    assert deduplication.orb_match_ratio(None, np.ones((1, 2))) == 0.0

    monkeypatch.setattr(
        deduplication.cv2,
        "BFMatcher",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("opencv")),
    )
    descriptors = np.ones((1, 2), dtype=np.uint8)
    assert deduplication.orb_match_ratio(descriptors, descriptors) == 0.0


def test_burst_grouping_applies_temporal_and_non_temporal_thresholds(
    monkeypatch, tmp_path: Path
) -> None:
    paths = [tmp_path / f"{index}.jpg" for index in range(4)]
    histogram = np.ones((2, 2), dtype=np.float32)
    correlations = iter([0.7, 0.95, 0.91])
    monkeypatch.setattr(deduplication.cv2, "compareHist", lambda *_args: next(correlations))
    entries = [
        _features(paths[0], histogram, 1.0, descriptors=histogram),
        # Temporal + strong ORB accepts the lower 0.60 correlation threshold.
        _features(paths[1], histogram, 2.0, descriptors=histogram),
        # A timestamp mixed with no timestamp can never join.
        _features(paths[2], histogram, None, descriptors=histogram),
        # No timestamps require the normal 0.92 histogram threshold.
        _features(paths[3], histogram, None, descriptors=histogram),
    ]

    groups = deduplication._group_bursts(entries, lambda *_args: 1.0)

    assert [[entry[0] for entry in group] for group in groups] == [
        paths[:2],
        paths[2:],
    ]


def test_burst_grouping_keeps_missing_or_empty_histograms_separate(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.jpg" for index in range(3)]
    groups = deduplication._group_bursts(
        [
            _features(paths[0], None),
            _features(paths[1], np.array([])),
            _features(paths[2], np.ones((1, 1))),
        ],
        lambda *_args: 1.0,
    )
    assert [[entry[0] for entry in group] for group in groups] == [
        [paths[0]],
        [paths[1]],
        [paths[2]],
    ]


def test_deduplicate_preserves_all_images_when_every_hash_fails(
    monkeypatch, tmp_path: Path
) -> None:
    paths = [tmp_path / name for name in ("c.jpg", "a.jpg", "b.jpg")]
    _SequentialExecutor.created_with.clear()
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SequentialExecutor)

    unique, bursts = deduplication.deduplicate(
        paths,
        worker_settings=config.WorkerSettings(2, 1, None),
        phash_function=lambda _path: None,
        features_function=lambda path: _features(path),
    )

    assert unique == paths
    assert bursts == {}
    assert _SequentialExecutor.created_with == [2, 2]


def test_deduplicate_preserves_failed_hash_position_and_group_representative(
    monkeypatch, tmp_path: Path
) -> None:
    paths = [tmp_path / name for name in ("first.jpg", "failed.jpg", "better.jpg")]
    hashes = {paths[0]: _Hash(1), paths[2]: _Hash(1)}
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SequentialExecutor)

    unique, _ = deduplication.deduplicate(
        paths,
        threshold=0,
        worker_settings=config.WorkerSettings(3, 1, None),
        phash_function=lambda path: (path, hashes[path]) if path in hashes else None,
        sharpness_function=lambda path: 10.0 if path == paths[2] else 1.0,
        features_function=lambda path: _features(path),
    )

    assert unique == [paths[2], paths[1]]


def test_hash_grouping_is_deterministic_and_partitions_hashed_inputs(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index:02}.jpg" for index in range(40)]
    hashes = {path: _Hash((index * 17) % 256) for index, path in enumerate(paths)}

    first = deduplication.group_perceptual_hashes(paths, hashes, threshold=2)
    second = deduplication.group_perceptual_hashes(paths, hashes, threshold=2)

    assert first == second
    flattened = [path for members in first.values() for path in members]
    assert len(flattened) == len(paths)
    assert set(flattened) == set(paths)
    assert all(members == sorted(members) for members in first.values())
