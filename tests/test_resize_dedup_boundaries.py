from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import pickinsta.ig_image_selector as selector
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


def test_resize_module_and_selector_preserve_collisions_mapping_and_reuse(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (2400, 1200), "red").save(source / "photo.jpg")
    Image.new("RGB", (800, 1600), "blue").save(source / "photo.png")
    (source / "ignored.txt").write_text("not an image", encoding="utf-8")
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SequentialExecutor)
    settings = config.WorkerSettings(2, 2, None)

    direct = resize.resize_for_processing(source, tmp_path / "direct", settings)
    compatibility = selector.resize_for_processing(source, tmp_path / "compat", settings)

    assert [path.name for path in direct[0]] == ["photo.jpg", "photo_1.jpg"]
    assert [path.name for path in compatibility[0]] == ["photo.jpg", "photo_1.jpg"]
    assert [path.name for path in direct[1].values()] == ["photo.jpg", "photo.png"]
    assert [path.name for path in compatibility[1].values()] == ["photo.jpg", "photo.png"]
    assert max(max(Image.open(path).size) for path in direct[0]) <= 1920

    reused, reused_map = resize.resize_for_processing(source, tmp_path / "direct", settings)
    assert reused == direct[0]
    assert reused_map == direct[1]


def test_selector_resize_wrapper_keeps_patchable_worker(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.jpg").write_bytes(b"input")
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SequentialExecutor)
    calls: list[tuple[Path, Path]] = []

    def fake_worker(pair: tuple[Path, Path]):
        calls.append(pair)
        return pair[1], pair[0]

    monkeypatch.setattr(selector, "_resize_one_image", fake_worker)
    resized, source_map = selector.resize_for_processing(
        source, tmp_path / "work", config.WorkerSettings(1, 1, None)
    )

    assert calls == [(source / "one.jpg", tmp_path / "work" / "one.jpg")]
    assert resized == [tmp_path / "work" / "one.jpg"]
    assert source_map == {resized[0]: source / "one.jpg"}


def test_resize_rejects_extreme_pixel_dimensions_before_transpose(
    monkeypatch, tmp_path: Path
) -> None:
    class HugeImage:
        size = (200_000, 200_000)
        info = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(resize.Image, "open", lambda _path: HugeImage())
    assert resize.resize_one_image((tmp_path / "huge.jpg", tmp_path / "out.jpg")) is None


def test_dedup_module_and_selector_match_burst_order_and_representative(
    monkeypatch, tmp_path: Path
) -> None:
    paths = [tmp_path / name for name in ("first.jpg", "second.jpg", "third.jpg")]
    hashes = dict(zip(paths, (_Hash(0), _Hash(0), _Hash((1 << 64) - 1)), strict=True))
    sharpness = {paths[0]: 1.0, paths[1]: 5.0, paths[2]: 9.0}
    timestamps = {paths[1]: 20.0, paths[2]: 18.0}
    histogram = np.ones((2, 2), dtype=np.float32)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SequentialExecutor)
    monkeypatch.setattr(deduplication.cv2, "compareHist", lambda *_args: 1.0)

    def phash(path: Path):
        return path, hashes[path]

    def features(path: Path):
        return path, histogram, timestamps[path], sharpness[path], histogram

    direct = deduplication.deduplicate(
        paths,
        threshold=0,
        worker_settings=config.WorkerSettings(2, 2, None),
        phash_function=phash,
        sharpness_function=sharpness.__getitem__,
        features_function=features,
        orb_ratio_function=lambda *_args: 1.0,
    )

    monkeypatch.setattr(selector, "_compute_phash", phash)
    monkeypatch.setattr(selector, "_quick_sharpness", sharpness.__getitem__)
    monkeypatch.setattr(selector, "_compute_dedup_features", features)
    monkeypatch.setattr(selector, "_orb_match_ratio", lambda *_args: 1.0)
    compatibility = selector.deduplicate(
        paths, threshold=0, worker_settings=config.WorkerSettings(2, 2, None)
    )

    assert direct == compatibility == ([paths[2]], {paths[2]: [paths[2], paths[1]]})


def test_dedup_extracts_burst_features_once_per_hash_representative(
    monkeypatch, tmp_path: Path
) -> None:
    paths = [tmp_path / f"{index}.jpg" for index in range(64)]
    feature_calls: list[Path] = []
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _SequentialExecutor)

    def features(path: Path):
        feature_calls.append(path)
        return path, None, None, 0.0, None

    unique, bursts = deduplication.deduplicate(
        paths,
        threshold=0,
        worker_settings=config.WorkerSettings(4, 4, None),
        phash_function=lambda path: (path, _Hash(int(path.stem))),
        features_function=features,
    )

    assert unique == paths
    assert bursts == {}
    assert feature_calls == paths


def test_extracted_local_stages_keep_upper_layers_and_models_out() -> None:
    forbidden = {
        "pickinsta.ig_image_selector",
        "pickinsta.reporting",
        "pickinsta.orchestration",
        "transformers",
        "torch",
        "anthropic",
        "ultralytics",
    }
    for module_path in (Path(resize.__file__), Path(deduplication.__file__)):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        assert not any(
            name == blocked or name.startswith(f"{blocked}.")
            for name in imported
            for blocked in forbidden
        )
