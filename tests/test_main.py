from pickinsta import __version__
import sys

import pytest

import pickinsta.ig_image_selector as selector


def test_version() -> None:
    assert __version__ == "1.0.0"


def test_main_parses_cli_and_invokes_run_pipeline(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return []

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"

    monkeypatch.setattr(selector, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pickinsta",
            str(input_dir),
            "--output",
            str(output_dir),
            "--top",
            "3",
            "--scorer",
            "claude",
            "--vision-pct",
            "0.75",
            "--claude-model",
            "claude-test-model",
            "--all",
            "--claude-crop-first",
        ],
    )

    selector.main()

    assert captured == {
        "input_folder": str(input_dir),
        "output_folder": str(output_dir),
        "top_n": 3,
        "scorer": "claude",
        "vision_candidates_pct": 0.75,
        "claude_model": "claude-test-model",
        "score_all": True,
        "claude_crop_first": True,
    }


def test_main_unrecognized_argument_prints_full_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["pickinsta", "./input", "--bogus"])

    with pytest.raises(SystemExit) as exc:
        selector.main()

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "options:" in stderr
    assert "--vision-pct" in stderr
    assert "--claude-crop-first" in stderr
    assert "error: unrecognized arguments: --bogus" in stderr


def test_main_missing_input_prints_full_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["pickinsta"])

    with pytest.raises(SystemExit) as exc:
        selector.main()

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "positional arguments:" in stderr
    assert "input" in stderr
    assert "--scorer" in stderr
    assert "error: the following arguments are required: input" in stderr
