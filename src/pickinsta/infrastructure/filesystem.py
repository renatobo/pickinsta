"""Atomic file publication and rollback-safe managed artifact operations."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from shutil import copy2 as _copy_file_metadata
from typing import Optional


def atomic_write_text(path: Path, content: str) -> None:
    """Write text without exposing a partially written destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: object) -> None:
    """Serialize a value as indented JSON and publish it atomically."""
    atomic_write_text(path, json.dumps(value, indent=2) + "\n")


def publish_managed_artifacts(
    staging_folder: Path,
    output_folder: Path,
    artifact_names: list[str],
    *,
    completion_artifact: str,
) -> Path:
    """Publish an immutable run generation and atomically advance its pointer.

    Flat files remain as compatibility mirrors. Consumers that need a
    consistent snapshot should resolve ``current_run.json`` and read from the
    referenced immutable generation.
    """
    for name in artifact_names:
        relative_name = Path(name)
        source = (staging_folder / relative_name).resolve()
        if (
            relative_name.is_absolute()
            or ".." in relative_name.parts
            or not source.is_relative_to(staging_folder.resolve())
            or not source.is_file()
        ):
            raise FileNotFoundError(f"staged artifact is missing: {name}")
    ordered_names = [name for name in artifact_names if name != completion_artifact]
    ordered_names.append(completion_artifact)
    runs_folder = output_folder / ".pickinsta-runs"
    runs_folder.mkdir(parents=True, exist_ok=True)
    generation = Path(tempfile.mkdtemp(prefix="run-", dir=runs_folder))
    try:
        for name in artifact_names:
            source = staging_folder / name
            destination = generation / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_file_metadata(source, destination)
    except BaseException:
        shutil.rmtree(generation, ignore_errors=True)
        raise

    backup_folder = Path(tempfile.mkdtemp(prefix=".previous-", dir=runs_folder))
    replaced: list[tuple[Path, Optional[Path]]] = []
    try:
        for name in ordered_names:
            source = staging_folder / name
            if not source.is_file():
                raise FileNotFoundError(f"staged artifact is missing: {name}")
            destination = output_folder / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup: Optional[Path] = None
            if destination.exists():
                if not destination.is_file():
                    raise IsADirectoryError(f"managed artifact path is not a file: {destination}")
                backup = backup_folder / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
            try:
                os.replace(source, destination)
            except BaseException:
                if backup is not None:
                    os.replace(backup, destination)
                raise
            replaced.append((destination, backup))
    except BaseException:
        for destination, backup in reversed(replaced):
            destination.unlink(missing_ok=True)
            if backup is not None and backup.exists():
                os.replace(backup, destination)
        shutil.rmtree(generation, ignore_errors=True)
        shutil.rmtree(backup_folder, ignore_errors=True)
        raise
    shutil.rmtree(backup_folder, ignore_errors=True)
    try:
        atomic_write_json(
            output_folder / "current_run.json",
            {
                "generation": generation.relative_to(output_folder).as_posix(),
                "manifest": f"{generation.relative_to(output_folder).as_posix()}/{completion_artifact}",
            },
        )
    except BaseException:
        shutil.rmtree(generation, ignore_errors=True)
        raise
    return generation


def resolve_current_run_directory(output_folder: Path) -> Optional[Path]:
    """Resolve and validate the immutable directory named by the run pointer."""
    pointer = output_folder / "current_run.json"
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
        relative = Path(data["generation"])
        base = output_folder.resolve()
        generation = (base / relative).resolve()
        manifest = (base / Path(data["manifest"])).resolve()
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != ".pickinsta-runs"
            or not generation.is_relative_to(base)
            or not manifest.is_relative_to(generation)
            or not manifest.is_file()
        ):
            return None
        return generation
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
