from pathlib import Path

import pytest

import pickinsta.ig_image_selector as selector


def test_read_env_file_parses_comments_exports_and_quotes(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "export ANTHROPIC_API_KEY='abc123'",
                'HF_TOKEN="hf_test"',
                "INVALID_LINE",
                "  EMPTY = spaced-value  ",
            ]
        ),
        encoding="utf-8",
    )

    parsed = selector._read_env_file(env_file)

    assert parsed["ANTHROPIC_API_KEY"] == "abc123"
    assert parsed["HF_TOKEN"] == "hf_test"
    assert parsed["EMPTY"] == "spaced-value"
    assert "INVALID_LINE" not in parsed


def test_resolve_anthropic_api_key_prefers_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env_value")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=file_value", encoding="utf-8")

    resolved = selector.resolve_anthropic_api_key(search_dir=tmp_path)

    assert resolved == "env_value"


def test_resolve_anthropic_api_key_reads_cwd_then_search_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cwd = tmp_path / "cwd"
    search = tmp_path / "search"
    cwd.mkdir()
    search.mkdir()
    monkeypatch.chdir(cwd)
    (cwd / ".env").write_text("ANTHROPIC_API_KEY=cwd_value", encoding="utf-8")
    (search / ".env").write_text("ANTHROPIC_API_KEY=search_value", encoding="utf-8")

    assert selector.resolve_anthropic_api_key(search_dir=search) == "cwd_value"

    (cwd / ".env").unlink()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert selector.resolve_anthropic_api_key(search_dir=search) == "search_value"


def test_resolve_anthropic_api_key_raises_when_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not found"):
        selector.resolve_anthropic_api_key(search_dir=tmp_path / "missing")


def test_resolve_optional_hf_token_sets_both_env_names(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    env_dir = tmp_path / "envdir"
    env_dir.mkdir()
    (env_dir / ".env").write_text("HUGGINGFACE_HUB_TOKEN=hf_123", encoding="utf-8")

    token = selector.resolve_optional_hf_token(search_dir=env_dir)

    assert token == "hf_123"
    assert selector.os.environ["HF_TOKEN"] == "hf_123"
    assert selector.os.environ["HUGGINGFACE_HUB_TOKEN"] == "hf_123"


def test_resolve_claude_model_precedence(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "anthropic-env-model")
    monkeypatch.setenv("CLAUDE_MODEL", "legacy-env-model")
    assert selector.resolve_claude_model("cli-model") == "cli-model"
    assert selector.resolve_claude_model() == "anthropic-env-model"

    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert selector.resolve_claude_model() == "legacy-env-model"


def test_claude_model_candidates_include_alias_and_fallbacks() -> None:
    models = selector._claude_model_candidates("claude-sonnet-4-5-20250514")
    assert models[0] == "claude-sonnet-4-5-20250514"
    assert "claude-sonnet-4-5" in models
    assert "claude-3-5-sonnet-latest" in models
    assert len(models) == len(set(models))


def test_claude_cache_round_trip_and_validation(tmp_path) -> None:
    source = tmp_path / "image.jpg"
    source.write_bytes(b"fake image bytes")
    source_sha = selector._file_sha256(source)
    model = "claude-sonnet-4-6"
    prompt_sha = "prompt-hash-123"
    vision = {"total": 42, "one_line": "Strong contrast and composition"}

    selector.save_claude_score_to_file_cache(
        source_path=source,
        source_sha256=source_sha,
        model=model,
        prompt_sha256=prompt_sha,
        vision=vision,
    )

    loaded = selector.load_claude_score_from_file_cache(
        source_path=source,
        source_sha256=source_sha,
        model=model,
        prompt_sha256=prompt_sha,
    )
    assert loaded == vision

    assert selector.load_claude_score_from_file_cache(
        source_path=source,
        source_sha256=source_sha,
        model="different-model",
        prompt_sha256=prompt_sha,
    ) is None

    assert selector.load_claude_score_from_file_cache(
        source_path=source,
        source_sha256=source_sha,
        model=model,
        prompt_sha256="different-prompt-hash",
    ) is None

    source.write_bytes(b"new bytes")
    new_sha = selector._file_sha256(source)
    assert selector.load_claude_score_from_file_cache(
        source_path=source,
        source_sha256=new_sha,
        model=model,
        prompt_sha256=prompt_sha,
    ) is None


def test_resolve_yolo_model_path_prefers_env_override(monkeypatch, tmp_path) -> None:
    override = tmp_path / "custom-yolo.pt"
    monkeypatch.setenv(selector.YOLO_MODEL_ENV_VAR, str(override))

    resolved = selector.resolve_yolo_model_path()

    assert resolved == override


def test_resolve_yolo_model_path_downloads_to_runtime_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(selector.YOLO_MODEL_ENV_VAR, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    calls = {"n": 0}

    def fake_urlretrieve(url, destination):
        calls["n"] += 1
        assert url == selector.YOLO_MODEL_URL
        Path(destination).write_bytes(b"fake-weights")
        return destination, None

    monkeypatch.setattr(selector, "urlretrieve", fake_urlretrieve)

    resolved = selector.resolve_yolo_model_path()

    expected = tmp_path / ".cache" / "pickinsta" / "models" / selector.YOLO_MODEL_FILENAME
    assert resolved == expected
    assert expected.exists()
    assert calls["n"] == 1

    # Second call should reuse the cached file and skip download.
    assert selector.resolve_yolo_model_path() == expected
    assert calls["n"] == 1
