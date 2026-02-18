import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import pickinsta.ig_image_selector as selector
from pickinsta.ig_image_selector import ImageScore


def _write_image(path: Path, width: int = 400, height: int = 300) -> None:
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_run_pipeline_writes_outputs_and_reports(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "selected"
    work_dir = tmp_path / "input_work"
    input_dir.mkdir()
    work_dir.mkdir()

    source_a = input_dir / "a.jpg"
    source_b = input_dir / "b.jpg"
    work_a = work_dir / "a.jpg"
    work_b = work_dir / "b.jpg"

    _write_image(source_a)
    _write_image(source_b)
    _write_image(work_a)
    _write_image(work_b)

    def fake_resize_for_processing(src_folder, work_folder):
        assert src_folder == input_dir
        assert work_folder == input_dir.parent / f"{input_dir.name}_work"
        return [work_a, work_b], {work_a: source_a, work_b: source_b}

    def fake_batch_technical_score(images, source_map=None):
        assert images == [work_a, work_b]
        return [
            ImageScore(path=work_a, source_path=source_a, technical={"composite": 0.50}),
            ImageScore(path=work_b, source_path=source_b, technical={"composite": 0.40}),
        ]

    def fake_batch_vision_score(candidates, scorer="clip", **_kwargs):
        assert scorer == "clip"
        candidates[0].vision = {"total": 54}
        candidates[0].one_line = "Strong color and clean subject isolation"
        candidates[0].final_score = 0.80
        if len(candidates) > 1:
            candidates[1].vision = {"total": 39}
            candidates[1].one_line = "Decent but weaker framing"
            candidates[1].final_score = 0.55
        return candidates

    def fake_smart_crop(image_path, output_path, **_kwargs):
        assert image_path in {work_a, work_b}
        meta_out = _kwargs.get("meta_out")
        if isinstance(meta_out, dict):
            meta_out.update(
                {
                    "uncertain_crop": True,
                    "uncertain_crop_reasons": ["test_forced_uncertain"],
                }
            )
        _write_image(output_path, width=1080, height=1440)
        return output_path

    monkeypatch.setattr(selector, "resize_for_processing", fake_resize_for_processing)
    monkeypatch.setattr(selector, "deduplicate", lambda images: images)
    monkeypatch.setattr(selector, "batch_technical_score", fake_batch_technical_score)
    monkeypatch.setattr(selector, "batch_vision_score", fake_batch_vision_score)
    monkeypatch.setattr(selector, "smart_crop", fake_smart_crop)

    report = selector.run_pipeline(
        input_folder=str(input_dir),
        output_folder=str(output_dir),
        top_n=1,
        scorer="clip",
    )

    assert len(report) == 1
    assert report[0]["filename"] == "a.jpg"
    assert report[0]["output"] == f"01_{work_a.stem}.jpg"
    assert report[0]["output_full_subject"] == f"01_full_{work_a.stem}.jpg"

    output_image = output_dir / report[0]["output"]
    assert output_image.exists()
    output_full = output_dir / report[0]["output_full_subject"]
    assert output_full.exists()

    report_json = output_dir / "selection_report.json"
    report_md = output_dir / "selection_report.md"
    assert report_json.exists()
    assert report_md.exists()

    loaded = json.loads(report_json.read_text(encoding="utf-8"))
    assert loaded[0]["rank"] == 1
    assert loaded[0]["vision_total"] == 54
    assert loaded[0]["uncertain_crop"] is True
    assert "Top Selected Outputs" in report_md.read_text(encoding="utf-8")


def test_run_pipeline_skips_padded_variant_when_crop_is_confident(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "selected"
    work_dir = tmp_path / "input_work"
    input_dir.mkdir()
    work_dir.mkdir()

    source_a = input_dir / "a.jpg"
    work_a = work_dir / "a.jpg"
    _write_image(source_a)
    _write_image(work_a)

    monkeypatch.setattr(
        selector,
        "resize_for_processing",
        lambda _src, _work: ([work_a], {work_a: source_a}),
    )
    monkeypatch.setattr(selector, "deduplicate", lambda images: images)
    monkeypatch.setattr(
        selector,
        "batch_technical_score",
        lambda _images, source_map=None: [
            ImageScore(path=work_a, source_path=source_a, technical={"composite": 0.6})
        ],
    )

    def fake_batch_vision_score(candidates, scorer="clip", **_kwargs):
        candidates[0].vision = {"total": 51}
        candidates[0].one_line = "Good subject"
        candidates[0].final_score = 0.7
        return candidates

    def fake_smart_crop(image_path, output_path, **kwargs):
        meta_out = kwargs.get("meta_out")
        if isinstance(meta_out, dict):
            meta_out.update({"uncertain_crop": False, "uncertain_crop_reasons": []})
        _write_image(output_path, width=1080, height=1440)
        return output_path

    monkeypatch.setattr(selector, "batch_vision_score", fake_batch_vision_score)
    monkeypatch.setattr(selector, "smart_crop", fake_smart_crop)

    report = selector.run_pipeline(
        input_folder=str(input_dir),
        output_folder=str(output_dir),
        top_n=1,
        scorer="clip",
    )

    assert len(report) == 1
    assert report[0]["output_full_subject"] is None
    assert report[0]["uncertain_crop"] is False
    assert not any(p.name.startswith("01_full_") for p in output_dir.glob("*.jpg"))


def test_run_pipeline_missing_input_exits(tmp_path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(SystemExit):
        selector.run_pipeline(input_folder=str(missing))


def test_run_pipeline_claude_prefers_crop_safe_outputs(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "selected"
    work_dir = tmp_path / "input_work"
    input_dir.mkdir()
    work_dir.mkdir()

    source_a = input_dir / "a.jpg"
    source_b = input_dir / "b.jpg"
    source_c = input_dir / "c.jpg"
    work_a = work_dir / "a.jpg"
    work_b = work_dir / "b.jpg"
    work_c = work_dir / "c.jpg"
    for p in (source_a, source_b, source_c, work_a, work_b, work_c):
        _write_image(p)

    def fake_resize_for_processing(src_folder, work_folder):
        return [work_a, work_b, work_c], {work_a: source_a, work_b: source_b, work_c: source_c}

    def fake_batch_technical_score(images, source_map=None):
        return [
            ImageScore(path=work_a, source_path=source_a, technical={"composite": 0.5}),
            ImageScore(path=work_b, source_path=source_b, technical={"composite": 0.5}),
            ImageScore(path=work_c, source_path=source_c, technical={"composite": 0.5}),
        ]

    def fake_batch_vision_score(candidates, scorer="claude", **_kwargs):
        assert scorer == "claude"
        # b.jpg has the highest score, but low crop_4x5 and should be deprioritized for output.
        candidates[0].vision = {"total": 54, "crop_4x5": 9}
        candidates[0].one_line = "A high crop confidence"
        candidates[0].final_score = 0.90
        candidates[1].vision = {"total": 58, "crop_4x5": 5}
        candidates[1].one_line = "B low crop confidence"
        candidates[1].final_score = 0.95
        candidates[2].vision = {"total": 50, "crop_4x5": 8}
        candidates[2].one_line = "C high crop confidence"
        candidates[2].final_score = 0.85
        return candidates

    crop_calls = []

    def fake_smart_crop(image_path, output_path, **kwargs):
        crop_calls.append({"image": image_path.name, "save_debug": kwargs.get("save_debug")})
        _write_image(output_path, width=1080, height=1440)
        return output_path

    monkeypatch.setattr(selector, "resize_for_processing", fake_resize_for_processing)
    monkeypatch.setattr(selector, "deduplicate", lambda images: images)
    monkeypatch.setattr(selector, "batch_technical_score", fake_batch_technical_score)
    monkeypatch.setattr(selector, "batch_vision_score", fake_batch_vision_score)
    monkeypatch.setattr(selector, "smart_crop", fake_smart_crop)

    report = selector.run_pipeline(
        input_folder=str(input_dir),
        output_folder=str(output_dir),
        top_n=2,
        scorer="claude",
        score_all=True,
    )

    assert [row["filename"] for row in report] == ["a.jpg", "c.jpg"]
    assert [call["save_debug"] for call in crop_calls] == [True, True]


def test_run_pipeline_claude_crop_first_mode_precrops_before_scoring(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "selected"
    work_dir = tmp_path / "input_work"
    input_dir.mkdir()
    work_dir.mkdir()

    source_a = input_dir / "a.jpg"
    source_b = input_dir / "b.jpg"
    work_a = work_dir / "a.jpg"
    work_b = work_dir / "b.jpg"
    for p in (source_a, source_b, work_a, work_b):
        _write_image(p)

    def fake_resize_for_processing(src_folder, work_folder):
        return [work_a, work_b], {work_a: source_a, work_b: source_b}

    def fake_batch_technical_score(images, source_map=None):
        return [
            ImageScore(path=work_a, source_path=source_a, technical={"composite": 0.5}),
            ImageScore(path=work_b, source_path=source_b, technical={"composite": 0.4}),
        ]

    def fake_batch_vision_score(candidates, scorer="claude", **_kwargs):
        assert scorer == "claude"
        # In crop-first mode, Claude sees pre-cropped candidate files.
        assert all(c.path.parent.name == "claude_crop_first" for c in candidates)
        assert [c.source_path.name for c in candidates] == ["a.jpg", "b.jpg"]
        candidates[0].vision = {"total": 54, "crop_4x5": 9}
        candidates[0].one_line = "A"
        candidates[0].final_score = 0.9
        candidates[1].vision = {"total": 40, "crop_4x5": 8}
        candidates[1].one_line = "B"
        candidates[1].final_score = 0.7
        return candidates

    crop_calls = []

    def fake_smart_crop(image_path, output_path, **kwargs):
        crop_calls.append({"src_parent": image_path.parent.name, "save_debug": kwargs.get("save_debug")})
        _write_image(output_path, width=1080, height=1440)
        return output_path

    monkeypatch.setattr(selector, "resize_for_processing", fake_resize_for_processing)
    monkeypatch.setattr(selector, "deduplicate", lambda images: images)
    monkeypatch.setattr(selector, "batch_technical_score", fake_batch_technical_score)
    monkeypatch.setattr(selector, "batch_vision_score", fake_batch_vision_score)
    monkeypatch.setattr(selector, "smart_crop", fake_smart_crop)

    report = selector.run_pipeline(
        input_folder=str(input_dir),
        output_folder=str(output_dir),
        top_n=1,
        scorer="claude",
        score_all=True,
        claude_crop_first=True,
    )

    assert len(report) == 1
    assert report[0]["filename"] == "a.jpg"
    # First two crop calls are pre-crop stage; final call is output stage with debug.
    assert crop_calls[0]["src_parent"] == "input_work"
    assert crop_calls[1]["src_parent"] == "input_work"
    assert crop_calls[2]["src_parent"] == "claude_crop_first"
    assert crop_calls[2]["save_debug"] is True
