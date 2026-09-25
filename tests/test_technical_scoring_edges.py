import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pickinsta.config import WorkerSettings
from pickinsta.models import ImageScore
from pickinsta.pipeline import technical_scoring


def _write_image(path: Path, color: tuple[int, int, int] = (80, 120, 160)) -> None:
    Image.new("RGB", (32, 24), color=color).save(path)


def test_load_cache_returns_none_for_missing_malformed_and_wrong_shapes(tmp_path: Path) -> None:
    image_path = tmp_path / "source.jpg"
    _write_image(image_path)
    cache_path = technical_scoring.tech_cache_path(image_path)

    assert technical_scoring.load_tech_cache(image_path, fingerprint=lambda: "fp") is None

    for content in ("not json", "[]", json.dumps({"schema_version": 2, "scores": []})):
        cache_path.write_text(content, encoding="utf-8")
        assert technical_scoring.load_tech_cache(image_path, fingerprint=lambda: "fp") is None


def test_cache_round_trip_normalizes_numpy_values(tmp_path: Path) -> None:
    image_path = tmp_path / "source.jpg"
    _write_image(image_path)
    scores = {
        "composite": np.float32(0.75),
        "nested": {"count": np.int64(3)},
        "sequence": (np.bool_(True), np.float64(0.25)),
    }

    technical_scoring.save_tech_cache(image_path, scores, fingerprint=lambda: "stable")

    payload = json.loads(technical_scoring.tech_cache_path(image_path).read_text())
    assert payload["scores"] == {
        "composite": pytest.approx(0.75),
        "nested": {"count": 3},
        "sequence": [True, 0.25],
    }
    assert (
        technical_scoring.load_tech_cache(image_path, fingerprint=lambda: "stable")
        == payload["scores"]
    )


def test_cache_rejects_stale_source_schema_and_fingerprint(tmp_path: Path) -> None:
    image_path = tmp_path / "source.jpg"
    _write_image(image_path)
    technical_scoring.save_tech_cache(
        image_path, {"composite": 0.5}, fingerprint=lambda: "original"
    )
    cache_path = technical_scoring.tech_cache_path(image_path)

    assert technical_scoring.load_tech_cache(image_path, fingerprint=lambda: "different") is None

    payload = json.loads(cache_path.read_text())
    payload["schema_version"] += 1
    cache_path.write_text(json.dumps(payload))
    assert technical_scoring.load_tech_cache(image_path, fingerprint=lambda: "original") is None

    payload["schema_version"] = technical_scoring.TECHNICAL_CACHE_SCHEMA_VERSION
    payload["mtime"] -= 1
    cache_path.write_text(json.dumps(payload))
    assert technical_scoring.load_tech_cache(image_path, fingerprint=lambda: "original") is None


def test_load_cache_degrades_when_source_was_removed(tmp_path: Path) -> None:
    image_path = tmp_path / "source.jpg"
    _write_image(image_path)
    technical_scoring.save_tech_cache(image_path, {"composite": 0.5}, fingerprint=lambda: "stable")
    image_path.unlink()

    assert technical_scoring.load_tech_cache(image_path, fingerprint=lambda: "stable") is None


def test_save_cache_degrades_without_replacing_existing_data(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "source.jpg"
    _write_image(image_path)
    cache_path = technical_scoring.tech_cache_path(image_path)
    cache_path.write_text("existing", encoding="utf-8")

    def fail_write(_path: Path, _content: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(technical_scoring, "atomic_write_text", fail_write)
    technical_scoring.save_tech_cache(image_path, {"composite": 0.9}, fingerprint=lambda: "fp")

    assert cache_path.read_text(encoding="utf-8") == "existing"


def test_score_with_cache_distinguishes_hit_miss_and_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "source.jpg"
    calls: list[str] = []

    hit = technical_scoring.score_technical_with_cache(
        image_path,
        loader=lambda _path: {"composite": 0.8},
        scorer=lambda _path: pytest.fail("cache hit must not score"),
        saver=lambda *_args: pytest.fail("cache hit must not save"),
    )
    assert hit == (image_path, {"composite": 0.8})

    miss = technical_scoring.score_technical_with_cache(
        image_path,
        loader=lambda _path: None,
        scorer=lambda _path: calls.append("score") or {"composite": 0.6},
        saver=lambda _path, _scores: calls.append("save"),
    )
    assert miss == (image_path, {"composite": 0.6})
    assert calls == ["score", "save"]

    assert (
        technical_scoring.score_technical_with_cache(
            image_path,
            loader=lambda _path: None,
            scorer=lambda _path: (_ for _ in ()).throw(ValueError("unreadable")),
        )
        is None
    )


def test_score_technical_rejects_unreadable_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not read"):
        technical_scoring.score_technical(
            tmp_path / "missing.jpg", detect_subject_mask=lambda _image: None
        )


def test_metric_helpers_handle_neutral_inputs() -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    gray = np.zeros((24, 32), dtype=np.uint8)
    empty_mask = np.zeros((24, 32), dtype=np.uint8)

    assert technical_scoring.composition_score(0.333, 0.333) == pytest.approx(1.0)
    assert technical_scoring.horizon_tilt_penalty(gray) == 0.8
    assert technical_scoring.lead_room_score(image, None) == 0.5
    assert technical_scoring.lead_room_score(image, empty_mask) == 0.5
    assert technical_scoring.colorfulness_metric(image) == 0.0
    assert technical_scoring.detect_subject_mask(image, lambda _image, debug=False: None) is None


def test_score_technical_outputs_finite_normalized_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "flat.jpg"
    _write_image(image_path)

    scores = technical_scoring.score_technical(
        image_path, detect_subject_mask=lambda image: np.zeros(image.shape[:2], np.uint8)
    )

    assert set(scores) == {
        "sharpness",
        "background_sep",
        "composition",
        "lighting",
        "color_harmony",
        "visual_clutter",
        "aesthetic",
        "composite",
    }
    assert all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in scores.values())


def test_distribution_handles_one_value(capsys) -> None:
    technical_scoring.print_score_distribution(
        [ImageScore(path=Path("one.jpg"), technical={"composite": 0.5})]
    )

    output = capsys.readouterr().out
    assert "n=1" in output
    assert "min=0.500" in output
    assert "median=0.500" in output


def test_distribution_handles_no_successful_scores(capsys) -> None:
    technical_scoring.print_score_distribution([])

    assert "n=0" in capsys.readouterr().out


def test_batch_bounds_workers_sorts_results_and_preserves_source_map(monkeypatch) -> None:
    images = [Path("low.jpg"), Path("high.jpg"), Path("cached.jpg")]
    source = Path("original.jpg")
    observed: list[tuple[str, int, WorkerSettings]] = []

    def bounded(kind: str, count: int, settings: WorkerSettings) -> int:
        observed.append((kind, count, settings))
        return 1

    monkeypatch.setattr(technical_scoring, "bounded_worker_count", bounded)
    scores = {"low.jpg": 0.1, "high.jpg": 0.9}
    results = technical_scoring.batch_technical_score(
        images,
        source_map={Path("high.jpg"): source},
        worker_settings=WorkerSettings(2, 7, None),
        loader=lambda path: {"composite": 0.5} if path.name == "cached.jpg" else None,
        score_cached=lambda path: (path, {"composite": scores[path.name]}),
        distribution=lambda _results: None,
    )

    assert observed == [("thread", 2, WorkerSettings(2, 7, None))]
    assert [result.path.name for result in results] == ["high.jpg", "cached.jpg", "low.jpg"]
    assert results[0].source_path == source


def test_batch_observer_receives_cache_hits_and_misses() -> None:
    images = [Path("hit.jpg"), Path("miss.jpg")]
    observations: list[bool] = []

    technical_scoring.batch_technical_score(
        images,
        worker_settings=WorkerSettings(1, 1, None),
        loader=lambda path: {"composite": 0.8} if path.name == "hit.jpg" else None,
        score_cached=lambda path: (path, {"composite": 0.4}),
        distribution=lambda _results: None,
        cache_observer=observations.append,
    )

    assert observations == [True, False]


def test_batch_all_failures_returns_empty_and_reports_distribution(capsys) -> None:
    distributions: list[list[ImageScore]] = []
    failures: list[tuple[Path, str, str]] = []

    result = technical_scoring.batch_technical_score(
        [Path("broken.jpg")],
        worker_settings=WorkerSettings(1, 1, None),
        loader=lambda _path: None,
        score_cached=lambda _path: None,
        distribution=distributions.append,
        failure_observer=lambda *issue: failures.append(issue),
    )

    assert result == []
    assert distributions == [[]]
    assert failures == [(Path("broken.jpg"), "technical_scoring", "technical score failed")]
    assert "failed for all images" in capsys.readouterr().out


def test_algorithm_fingerprint_is_stable_and_dependency_sensitive(monkeypatch) -> None:
    technical_scoring.technical_algorithm_fingerprint.cache_clear()
    first = technical_scoring.technical_algorithm_fingerprint()
    assert first == technical_scoring.technical_algorithm_fingerprint()
    assert len(first) == 64

    monkeypatch.setattr(technical_scoring.cv2, "__version__", "future-opencv")
    technical_scoring.technical_algorithm_fingerprint.cache_clear()
    assert technical_scoring.technical_algorithm_fingerprint() != first
    technical_scoring.technical_algorithm_fingerprint.cache_clear()
