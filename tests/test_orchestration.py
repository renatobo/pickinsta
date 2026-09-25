import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import pickinsta.pipeline.orchestration as orchestration
from pickinsta.config import WorkerSettings
from pickinsta.infrastructure.filesystem import atomic_write_json, publish_managed_artifacts
from pickinsta.pipeline.orchestration import (
    FacadeCollaborators,
    RecursiveCollaborators,
    RunContext,
    discover_recursive_input_folders,
    run_dedup_only,
    run_pipeline,
    run_pipeline_recursive,
)
from pickinsta.telemetry import RunTelemetry, build_run_manifest


def test_public_use_case_wrappers_dispatch_to_extracted_implementations(monkeypatch) -> None:
    calls = []
    collaborators = object()
    monkeypatch.setattr(
        orchestration,
        "_run_dedup_only_bound",
        lambda *args, **kwargs: calls.append(("dedup", args, kwargs)),
    )
    monkeypatch.setattr(
        orchestration,
        "_run_pipeline_bound",
        lambda *args, **kwargs: calls.append(("full", args, kwargs)),
    )

    run_dedup_only("input", collaborators=collaborators, marker=1)
    run_pipeline("input", collaborators=collaborators, marker=2)

    assert calls == [
        ("dedup", ("input",), {"collaborators": collaborators, "marker": 1}),
        ("full", ("input",), {"collaborators": collaborators, "marker": 2}),
    ]


def test_run_context_resolves_single_and_recursive_work_paths(tmp_path: Path) -> None:
    settings = WorkerSettings(process_workers=1, thread_workers=2, opencv_threads=0)
    source = tmp_path / "photos"
    output = tmp_path / "selected"

    single = RunContext.resolve(str(source), str(output), None, settings)
    recursive = RunContext.resolve(str(source), str(output), None, settings, recursive=True)

    assert single.input_path == source
    assert single.output_path == output
    assert single.work_path == tmp_path / "photos_work"
    assert recursive.work_path == tmp_path / "selected_work"


def test_recursive_discovery_returns_only_leaf_image_folders(tmp_path: Path) -> None:
    event = tmp_path / "event"
    child = event / "day-one"
    sibling = event / "day-two"
    child.mkdir(parents=True)
    sibling.mkdir()
    (event / "cover.jpg").write_bytes(b"not decoded during discovery")
    (child / "one.JPG").write_bytes(b"image")
    (sibling / "notes.txt").write_text("none")

    assert discover_recursive_input_folders(tmp_path, frozenset({".jpg"})) == [child]


def test_context_and_collaborator_records_are_immutable(tmp_path: Path) -> None:
    settings = WorkerSettings(process_workers=1, thread_workers=2, opencv_threads=0)
    context = RunContext.resolve(str(tmp_path), str(tmp_path / "out"), None, settings)
    recursive = RecursiveCollaborators(
        resolve_workers=lambda: settings,
        configure_opencv=lambda _settings: None,
        run_pipeline=lambda **_kwargs: [],
        write_summary=lambda **_kwargs: (Path("summary.json"), Path("summary.md")),
        generate_gallery_index=lambda _path: [],
        supported_extensions=frozenset({".jpg"}),
    )

    with pytest.raises(FrozenInstanceError):
        context.output_path = tmp_path / "elsewhere"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        recursive.supported_extensions = frozenset()  # type: ignore[misc]
    assert FacadeCollaborators.__dataclass_params__.frozen is True


@pytest.mark.parametrize("missing", [True, False])
def test_recursive_exits_for_missing_or_image_free_input(tmp_path: Path, missing: bool) -> None:
    root = tmp_path / "missing"
    if not missing:
        root.mkdir()
        (root / "notes.txt").write_text("no images", encoding="utf-8")
    settings = WorkerSettings(process_workers=1, thread_workers=1, opencv_threads=0)
    collaborators = RecursiveCollaborators(
        resolve_workers=lambda: settings,
        configure_opencv=lambda _settings: None,
        run_pipeline=lambda **_kwargs: pytest.fail("pipeline must not run"),
        write_summary=lambda **_kwargs: pytest.fail("summary must not run"),
        generate_gallery_index=lambda _path: pytest.fail("gallery must not run"),
        supported_extensions=frozenset({".jpg"}),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_pipeline_recursive(str(root), collaborators=collaborators)

    assert exc_info.value.code == 1


def test_recursive_continues_after_folder_failure_and_writes_summary(tmp_path: Path) -> None:
    root = tmp_path / "input"
    good = root / "a-good"
    bad = root / "b-bad"
    good.mkdir(parents=True)
    bad.mkdir()
    (good / "one.jpg").write_bytes(b"image")
    (bad / "two.jpg").write_bytes(b"image")
    settings = WorkerSettings(process_workers=1, thread_workers=1, opencv_threads=0)
    captured: dict = {}

    def run_folder(**kwargs):
        if kwargs["input_folder"] == str(bad):
            raise RuntimeError("broken folder")
        return [{"final_score": 0.75, "uncertain_crop": True}]

    def write_summary(**kwargs):
        captured.update(kwargs)
        return tmp_path / "recursive.json", tmp_path / "recursive.md"

    collaborators = RecursiveCollaborators(
        resolve_workers=lambda: settings,
        configure_opencv=lambda _settings: None,
        run_pipeline=run_folder,
        write_summary=write_summary,
        generate_gallery_index=lambda _path: [],
        supported_extensions=frozenset({".jpg"}),
    )

    summaries = run_pipeline_recursive(
        str(root), str(tmp_path / "output"), collaborators=collaborators
    )

    assert [item["status"] for item in summaries] == ["complete", "failed"]
    assert summaries[0]["selected_count"] == 1
    assert summaries[1]["error"] == "broken folder"
    assert captured["folder_summaries"] == summaries


def _facade(tmp_path: Path, **overrides) -> FacadeCollaborators:
    settings = WorkerSettings(process_workers=1, thread_workers=1, opencv_threads=0)
    source = tmp_path / "input" / "photo.jpg"
    work_image = tmp_path / "work" / "photo.jpg"
    source.parent.mkdir(exist_ok=True)
    work_image.parent.mkdir(exist_ok=True)
    source.write_bytes(b"source")
    work_image.write_bytes(b"work")
    item = SimpleNamespace(
        path=work_image,
        source_path=source,
        technical={"composite": 0.8},
        vision={"total": 8, "crop_4x5": 9},
        final_score=0.8,
        one_line="good",
        burst_group=None,
        burst_selected_by="",
    )

    def crop(args):
        index, _path, destination = args
        destination.write_bytes(b"crop")
        return index, {}

    def markdown(path, **_kwargs):
        path.write_text("report", encoding="utf-8")

    def gallery(folder, _source):
        path = folder / "index.html"
        path.write_text("gallery", encoding="utf-8")
        return path

    values = {
        "RunTelemetry": RunTelemetry,
        "resolve_worker_settings": lambda: settings,
        "configure_opencv_threads": lambda _settings: None,
        "resize_for_processing": lambda *_args, **_kwargs: ([work_image], {work_image: source}),
        "deduplicate": lambda paths, **_kwargs: (paths, {}),
        "batch_technical_score": lambda *_args, **_kwargs: [item],
        "batch_vision_score": lambda candidates, **_kwargs: candidates,
        "score_technical_with_cache": lambda path: (path, {"composite": 0.8}),
        "prepare_claude_crop_first_candidates": lambda candidates, **_kwargs: candidates,
        "crop_one_image_no_debug": crop,
        "crop_one_image": crop,
        "generate_dedup_gallery": lambda *_args: None,
        "generate_gallery": gallery,
        "write_markdown_report": markdown,
        "atomic_write_json": atomic_write_json,
        "publish_managed_artifacts": publish_managed_artifacts,
        "build_run_manifest": build_run_manifest,
        "bounded_worker_count": lambda _kind, count, _settings: max(1, count),
        "call_worker_stage": lambda function, *args, **kwargs: function(*args, **kwargs),
        "call_with_supported_kwargs": lambda function, *args, **kwargs: function(*args, **kwargs),
        "build_vision_prompt": lambda _context: "prompt",
        "resolve_account_context": lambda **_kwargs: "context",
        "claude_prompt_sha256": lambda _prompt: "hash",
        "resolve_claude_model": lambda: "model",
        "claude_cache_options": lambda **_kwargs: {},
        "load_claude_score_from_file_cache": lambda **_kwargs: None,
        "file_sha256": lambda _path: "sha",
        "safe_float": lambda value, default=0.0: float(value) if value is not None else default,
        "Image": object(),
        "max_resize_px": 1920,
        "output_width": 1080,
        "output_height": 1440,
        "min_crop_score": 7.0,
    }
    values.update(overrides)
    return FacadeCollaborators(**values)


def test_full_pipeline_empty_resize_exits_before_downstream_stages(tmp_path: Path) -> None:
    collaborators = _facade(
        tmp_path,
        resize_for_processing=lambda *_args, **_kwargs: ([], {}),
        deduplicate=lambda *_args, **_kwargs: pytest.fail("dedup must not run"),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_pipeline(str(tmp_path / "input"), collaborators=collaborators)

    assert exc_info.value.code == 1


def test_dedup_pipeline_empty_resize_exits_before_downstream_stages(tmp_path: Path) -> None:
    collaborators = _facade(
        tmp_path,
        resize_for_processing=lambda *_args, **_kwargs: ([], {}),
        deduplicate=lambda *_args, **_kwargs: pytest.fail("dedup must not run"),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_dedup_only(str(tmp_path / "input"), collaborators=collaborators)

    assert exc_info.value.code == 1


def test_dedup_pipeline_transactionally_publishes_manifest_and_preserves_unrelated(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    unrelated = output / "keep.txt"
    unrelated.write_text("mine", encoding="utf-8")
    publication: dict = {}

    def gallery(folder, _items):
        (folder / "index.html").write_text("gallery", encoding="utf-8")

    def publish(staging, destination, names, *, completion_artifact):
        publication["completion"] = completion_artifact
        publication["manifest_exists"] = (staging / completion_artifact).exists()
        publish_managed_artifacts(
            staging, destination, names, completion_artifact=completion_artifact
        )

    run_dedup_only(
        str(tmp_path / "input"),
        str(output),
        collaborators=_facade(
            tmp_path,
            generate_dedup_gallery=gallery,
            publish_managed_artifacts=publish,
        ),
    )

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert publication == {"completion": "run_manifest.json", "manifest_exists": True}
    assert manifest["status"] == "complete"
    assert manifest["processed"] == 1
    assert manifest["failed"] == 0
    assert unrelated.read_text(encoding="utf-8") == "mine"
    assert not list(output.glob(".pickinsta-run-*"))


@pytest.mark.parametrize("failure", [OSError("publication failed"), KeyboardInterrupt()])
def test_dedup_publication_failure_cleans_staging_and_preserves_old_output(
    tmp_path: Path, failure: BaseException
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old = output / "index.html"
    old.write_text("old", encoding="utf-8")

    def fail_publish(*_args, **_kwargs):
        raise failure

    with pytest.raises(type(failure)):
        run_dedup_only(
            str(tmp_path / "input"),
            str(output),
            collaborators=_facade(tmp_path, publish_managed_artifacts=fail_publish),
        )

    assert old.read_text(encoding="utf-8") == "old"
    assert not list(output.glob(".pickinsta-run-*"))


def test_dedup_manifest_records_copy_failures_as_degraded(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(
        "pickinsta.pipeline.orchestration.shutil.copy2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy failed")),
    )

    run_dedup_only(str(tmp_path / "input"), str(output), collaborators=_facade(tmp_path))

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "degraded"
    assert manifest["processed"] == 1
    assert manifest["skipped"] == 0
    assert manifest["failed"] == 2
    assert [issue["stage"] for issue in manifest["issues"]] == ["copy_hd", "copy_full"]
    assert manifest["warnings"] == manifest["issues"]


def test_dedup_disambiguates_colliding_source_stems(tmp_path: Path) -> None:
    first = tmp_path / "work" / "first.jpg"
    second = tmp_path / "work" / "second.jpg"
    source_one = tmp_path / "input" / "one" / "photo.jpg"
    source_two = tmp_path / "input" / "two" / "photo.jpg"
    for path in (first, second, source_one, source_two):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    source_map = {first: source_one, second: source_two}
    output = tmp_path / "output"

    run_dedup_only(
        str(tmp_path / "input"),
        str(output),
        collaborators=_facade(
            tmp_path,
            resize_for_processing=lambda *_args, **_kwargs: ([first, second], source_map),
        ),
    )

    assert (output / "photo_cropped.jpg").exists()
    assert (output / "photo_2_cropped.jpg").exists()


def test_full_pipeline_publishes_manifest_last_and_preserves_unrelated_file(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    unrelated = output / "keep.txt"
    unrelated.write_text("mine", encoding="utf-8")
    publication: dict = {}

    def publish(staging, destination, names, *, completion_artifact):
        publication["names"] = list(names)
        publication["completion"] = completion_artifact
        assert (staging / completion_artifact).exists()
        publish_managed_artifacts(
            staging, destination, names, completion_artifact=completion_artifact
        )

    report = run_pipeline(
        str(tmp_path / "input"),
        str(output),
        top_n=1,
        collaborators=_facade(tmp_path, publish_managed_artifacts=publish),
    )

    assert len(report) == 1
    assert publication["completion"] == "run_manifest.json"
    assert "run_manifest.json" in publication["names"]
    assert unrelated.read_text(encoding="utf-8") == "mine"
    assert (output / "run_manifest.json").exists()


def test_full_pipeline_publication_failure_removes_staging_and_keeps_old_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    old = output / "selection_report.json"
    old.write_text("old", encoding="utf-8")

    captured_manifest: dict = {}

    def fail_publish(staging, *_args, **_kwargs):
        captured_manifest.update(
            json.loads((staging / "run_manifest.json").read_text(encoding="utf-8"))
        )
        raise OSError("publication failed")

    with pytest.raises(OSError, match="publication failed"):
        run_pipeline(
            str(tmp_path / "input"),
            str(output),
            top_n=1,
            collaborators=_facade(tmp_path, publish_managed_artifacts=fail_publish),
        )

    assert old.read_text(encoding="utf-8") == "old"
    assert not list(output.glob(".pickinsta-run-*"))
    assert set(captured_manifest["stage_timings_seconds"]) == {
        "resize",
        "deduplication",
        "technical_scoring",
        "burst_reevaluation",
        "vision_scoring",
        "output_generation",
        "reporting",
    }


def test_full_pipeline_manifest_counts_degraded_copy_issues(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"

    def fail_copy(_source, destination):
        raise OSError(f"cannot copy {Path(destination).name}")

    monkeypatch.setattr("pickinsta.pipeline.orchestration.shutil.copy2", fail_copy)
    report = run_pipeline(
        str(tmp_path / "input"), str(output), top_n=1, collaborators=_facade(tmp_path)
    )

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(report) == 1
    assert manifest["status"] == "degraded"
    assert manifest["processed"] == 1
    assert manifest["skipped"] == 0
    assert manifest["failed"] == 2
    assert [issue["stage"] for issue in manifest["issues"]] == ["copy_hd", "copy_full"]
    assert manifest["warnings"] == manifest["issues"]
