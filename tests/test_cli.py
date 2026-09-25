from __future__ import annotations

import tomllib
from pathlib import Path

from pickinsta import cli


def _operations(**overrides):
    def unused(**_kwargs):
        raise AssertionError("unexpected CLI dispatch")

    return cli.CliCollaborators(
        run_pipeline=overrides.get("run_pipeline", unused),
        run_pipeline_recursive=overrides.get("run_pipeline_recursive", unused),
        run_dedup_only=overrides.get("run_dedup_only", unused),
    )


def test_cli_main_dispatches_without_using_process_arguments(tmp_path: Path) -> None:
    captured = {}
    operations = _operations(run_pipeline=lambda **kwargs: captured.update(kwargs))

    cli.main(
        [str(tmp_path), "--top", "4", "--scorer", "ollama", "--rescore"],
        collaborators=operations,
    )

    assert captured["input_folder"] == str(tmp_path)
    assert captured["top_n"] == 4
    assert captured["scorer"] == "ollama"
    assert captured["rescore"] is True


def test_cli_main_dispatches_dedup_only(tmp_path: Path) -> None:
    captured = {}
    operations = _operations(run_dedup_only=lambda **kwargs: captured.update(kwargs))

    cli.main([str(tmp_path), "--dedup-only", "--work", "work"], collaborators=operations)

    assert captured == {
        "input_folder": str(tmp_path),
        "output_folder": "selected",
        "work_folder": "work",
    }


def test_project_console_script_points_to_cli_module() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["scripts"]["pickinsta"] == "pickinsta.cli:main"
