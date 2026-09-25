from pathlib import Path

import pytest

import pickinsta.infrastructure.filesystem as filesystem


def _stage(staging: Path, name: str, content: str) -> None:
    path = staging / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_missing_staged_artifact_rolls_back_already_published_files(tmp_path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    _stage(staging, "image.jpg", "new image")
    (output / "image.jpg").write_text("old image", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing.json"):
        filesystem.publish_managed_artifacts(
            staging,
            output,
            ["image.jpg", "missing.json", "manifest.json"],
            completion_artifact="manifest.json",
        )

    assert (output / "image.jpg").read_text(encoding="utf-8") == "old image"
    assert not (output / "missing.json").exists()


def test_destination_directory_is_rejected_without_disturbing_it(tmp_path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    _stage(staging, "gallery", "new gallery")
    _stage(staging, "manifest.json", "complete")
    destination = output / "gallery"
    destination.mkdir()
    (destination / "user-file").write_text("keep", encoding="utf-8")

    with pytest.raises(IsADirectoryError, match="managed artifact path is not a file"):
        filesystem.publish_managed_artifacts(
            staging,
            output,
            ["gallery", "manifest.json"],
            completion_artifact="manifest.json",
        )

    assert (destination / "user-file").read_text(encoding="utf-8") == "keep"
    assert not (output / "manifest.json").exists()


def test_initial_artifact_publish_failure_restores_its_previous_version(
    monkeypatch, tmp_path
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    _stage(staging, "image.jpg", "new image")
    _stage(staging, "manifest.json", "complete")
    destination = output / "image.jpg"
    destination.write_text("old image", encoding="utf-8")
    real_replace = filesystem.os.replace

    def fail_new_image(source, target) -> None:
        if Path(source) == staging / "image.jpg" and Path(target) == destination:
            raise OSError("publish failed")
        real_replace(source, target)

    monkeypatch.setattr(filesystem.os, "replace", fail_new_image)

    with pytest.raises(OSError, match="publish failed"):
        filesystem.publish_managed_artifacts(
            staging,
            output,
            ["image.jpg", "manifest.json"],
            completion_artifact="manifest.json",
        )

    assert destination.read_text(encoding="utf-8") == "old image"
    assert not (output / "manifest.json").exists()


def test_initial_new_artifact_publish_failure_leaves_no_destination(monkeypatch, tmp_path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    _stage(staging, "image.jpg", "new image")
    _stage(staging, "manifest.json", "complete")
    destination = output / "image.jpg"

    def fail_replace(source, target) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(filesystem.os, "replace", fail_replace)

    with pytest.raises(OSError, match="publish failed"):
        filesystem.publish_managed_artifacts(
            staging,
            output,
            ["image.jpg", "manifest.json"],
            completion_artifact="manifest.json",
        )

    assert not destination.exists()
    assert not (output / "manifest.json").exists()


def test_rollback_restores_completed_replacements_in_reverse_order(monkeypatch, tmp_path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    for name in ("first.txt", "second.txt", "manifest.json"):
        _stage(staging, name, f"new {name}")
        (output / name).write_text(f"old {name}", encoding="utf-8")
    real_replace = filesystem.os.replace
    restores: list[str] = []

    def fail_completion(source, target) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if source_path == staging / "manifest.json" and target_path == output / "manifest.json":
            raise OSError("completion publish failed")
        if source_path.parent.name.startswith(".previous-") and target_path.parent == output:
            restores.append(target_path.name)
        real_replace(source, target)

    monkeypatch.setattr(filesystem.os, "replace", fail_completion)

    with pytest.raises(OSError, match="completion publish failed"):
        filesystem.publish_managed_artifacts(
            staging,
            output,
            ["first.txt", "manifest.json", "second.txt"],
            completion_artifact="manifest.json",
        )

    assert restores == ["manifest.json", "second.txt", "first.txt"]
    for name in ("first.txt", "second.txt", "manifest.json"):
        assert (output / name).read_text(encoding="utf-8") == f"old {name}"


def test_nested_managed_artifact_is_published(tmp_path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    _stage(staging, "images/selected.jpg", "image")
    _stage(staging, "metadata/manifest.json", "complete")

    generation = filesystem.publish_managed_artifacts(
        staging,
        output,
        ["images/selected.jpg", "metadata/manifest.json"],
        completion_artifact="metadata/manifest.json",
    )

    assert (output / "images/selected.jpg").read_text(encoding="utf-8") == "image"
    assert (output / "metadata/manifest.json").read_text(encoding="utf-8") == "complete"
    assert (generation / "images/selected.jpg").read_text(encoding="utf-8") == "image"
    assert filesystem.resolve_current_run_directory(output) == generation


def test_failed_pointer_advance_keeps_previous_generation_authoritative(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "output"
    output.mkdir()

    def publish(staging: Path, content: str) -> Path:
        staging.mkdir()
        _stage(staging, "selection_report.json", content)
        _stage(staging, "run_manifest.json", content)
        return filesystem.publish_managed_artifacts(
            staging,
            output,
            ["selection_report.json", "run_manifest.json"],
            completion_artifact="run_manifest.json",
        )

    first = publish(tmp_path / "first", "first")
    pointer = output / "current_run.json"
    previous_pointer = pointer.read_text(encoding="utf-8")
    original_writer = filesystem.atomic_write_json

    def fail_pointer(path: Path, value: object) -> None:
        if path == pointer:
            raise OSError("pointer update failed")
        original_writer(path, value)

    monkeypatch.setattr(filesystem, "atomic_write_json", fail_pointer)
    with pytest.raises(OSError, match="pointer update failed"):
        publish(tmp_path / "second", "second")

    assert pointer.read_text(encoding="utf-8") == previous_pointer
    assert filesystem.resolve_current_run_directory(output) == first
    assert (first / "selection_report.json").read_text(encoding="utf-8") == "first"


def test_run_pointer_rejects_paths_outside_generation_root(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run_manifest.json").write_text("{}", encoding="utf-8")
    (output / "current_run.json").write_text(
        '{"generation":"../outside","manifest":"../outside/run_manifest.json"}',
        encoding="utf-8",
    )

    assert filesystem.resolve_current_run_directory(output) is None


def test_missing_completion_artifact_preserves_previous_managed_files(tmp_path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    _stage(staging, "image.jpg", "new image")
    (output / "image.jpg").write_text("old image", encoding="utf-8")
    (output / "manifest.json").write_text("old manifest", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        filesystem.publish_managed_artifacts(
            staging,
            output,
            ["image.jpg", "manifest.json"],
            completion_artifact="manifest.json",
        )

    assert (output / "image.jpg").read_text(encoding="utf-8") == "old image"
    assert (output / "manifest.json").read_text(encoding="utf-8") == "old manifest"


def test_atomic_writer_failure_preserves_destination_and_removes_temporary_file(
    monkeypatch, tmp_path
) -> None:
    destination = tmp_path / "result.txt"
    destination.write_text("old", encoding="utf-8")

    def fail_replace(source, target) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(filesystem.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        filesystem.atomic_write_text(destination, "new")

    assert destination.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(f".{destination.name}.*")) == []


def test_atomic_writer_interrupt_preserves_destination_and_removes_temporary_file(
    monkeypatch, tmp_path
) -> None:
    destination = tmp_path / "result.txt"
    destination.write_text("old", encoding="utf-8")

    def interrupt_replace(source, target) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(filesystem.os, "replace", interrupt_replace)

    with pytest.raises(KeyboardInterrupt):
        filesystem.atomic_write_text(destination, "new")

    assert destination.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(f".{destination.name}.*")) == []


def test_publication_interrupt_rolls_back_and_does_not_publish_completion(
    monkeypatch, tmp_path
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    for name in ("first.txt", "second.txt", "manifest.json"):
        _stage(staging, name, f"new {name}")
    (output / "first.txt").write_text("old first", encoding="utf-8")
    real_replace = filesystem.os.replace

    def interrupt_second(source, target) -> None:
        if Path(source) == staging / "second.txt":
            raise KeyboardInterrupt
        real_replace(source, target)

    monkeypatch.setattr(filesystem.os, "replace", interrupt_second)

    with pytest.raises(KeyboardInterrupt):
        filesystem.publish_managed_artifacts(
            staging,
            output,
            ["first.txt", "second.txt", "manifest.json"],
            completion_artifact="manifest.json",
        )

    assert (output / "first.txt").read_text(encoding="utf-8") == "old first"
    assert not (output / "second.txt").exists()
    assert not (output / "manifest.json").exists()


def test_atomic_writer_rejects_parent_path_conflict_without_partial_output(tmp_path) -> None:
    parent = tmp_path / "not-a-directory"
    parent.write_text("user data", encoding="utf-8")
    destination = parent / "result.txt"

    with pytest.raises(FileExistsError):
        filesystem.atomic_write_text(destination, "new")

    assert parent.read_text(encoding="utf-8") == "user data"


def test_publication_rejects_nested_output_path_conflict_without_completion(tmp_path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    _stage(staging, "nested/image.jpg", "new image")
    _stage(staging, "manifest.json", "complete")
    conflict = output / "nested"
    conflict.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError):
        filesystem.publish_managed_artifacts(
            staging,
            output,
            ["nested/image.jpg", "manifest.json"],
            completion_artifact="manifest.json",
        )

    assert conflict.read_text(encoding="utf-8") == "user data"
    assert not (output / "manifest.json").exists()


def test_publication_preserves_unrelated_output_files(tmp_path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    _stage(staging, "image.jpg", "new image")
    _stage(staging, "manifest.json", "complete")
    unrelated = output / "user-notes.txt"
    unrelated.write_text("do not touch", encoding="utf-8")

    filesystem.publish_managed_artifacts(
        staging,
        output,
        ["image.jpg", "manifest.json"],
        completion_artifact="manifest.json",
    )

    assert unrelated.read_text(encoding="utf-8") == "do not touch"
