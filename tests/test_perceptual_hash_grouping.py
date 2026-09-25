from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from pickinsta.pipeline.deduplication import HashGroupingStats, group_perceptual_hashes


@dataclass(frozen=True)
class BitHash:
    value: int

    def __sub__(self, other: object) -> int:
        if not isinstance(other, BitHash):
            return NotImplemented
        return (self.value ^ other.value).bit_count()


def group_reference(
    images: list[Path], path_hash_map: dict[Path, BitHash], threshold: int
) -> dict[BitHash, list[Path]]:
    groups: dict[BitHash, list[Path]] = {}
    for image in images:
        image_hash = path_hash_map.get(image)
        if image_hash is None:
            continue
        for representative, group in groups.items():
            if abs(image_hash - representative) <= threshold:
                group.append(image)
                break
        else:
            groups[image_hash] = [image]
    return groups


@pytest.mark.parametrize("threshold", [-1, 0, 1, 3, 8, 16])
@pytest.mark.parametrize("seed", range(10))
def test_indexed_grouping_matches_randomized_linear_reference(seed: int, threshold: int) -> None:
    randomizer = random.Random(seed)
    images = [Path(f"{index}.jpg") for index in range(80)]
    # Eight-bit values deliberately produce collisions and overlapping match radii.
    path_hash_map = {image: BitHash(randomizer.randrange(256)) for image in images}

    actual = group_perceptual_hashes(images, path_hash_map, threshold)

    assert actual == group_reference(images, path_hash_map, threshold)


def test_ambiguous_match_uses_first_created_group_not_tree_traversal_order() -> None:
    images = [Path(name) for name in ("first.jpg", "second.jpg", "ambiguous.jpg")]
    hashes = {
        images[0]: BitHash(0b0000),
        images[1]: BitHash(0b1111),
        images[2]: BitHash(0b0011),
    }

    groups = group_perceptual_hashes(images, hashes, threshold=2)

    assert list(groups.values()) == [[images[0], images[2]], [images[1]]]


def test_stats_are_optional_and_count_queries_and_metric_comparisons() -> None:
    images = [Path(f"{index}.jpg") for index in range(3)]
    hashes = dict(zip(images, (BitHash(0), BitHash(255), BitHash(1))))
    stats = HashGroupingStats()

    group_perceptual_hashes(images, hashes, threshold=1, stats=stats)

    assert stats.queries == 3
    assert stats.distance_comparisons > 0
