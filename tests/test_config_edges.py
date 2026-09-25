from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import pickinsta.config as config

CONFIG_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "ANTHROPIC_MODEL",
    "CLAUDE_MODEL",
    config.ACCOUNT_CONTEXT_ENV_VAR,
    config.PICKINSTA_OLLAMA_BASE_URL_ENV_VAR,
    config.PICKINSTA_OLLAMA_MODEL_ENV_VAR,
    config.OLLAMA_TIMEOUT_ENV_VAR,
    config.OLLAMA_MAX_EDGE_ENV_VAR,
    config.OLLAMA_JPEG_QUALITY_ENV_VAR,
    config.OLLAMA_KEEP_ALIVE_ENV_VAR,
    config.OLLAMA_USE_YOLO_ENV_VAR,
    config.OLLAMA_CONCURRENCY_ENV_VAR,
    config.OLLAMA_MAX_RETRIES_ENV_VAR,
    config.OLLAMA_BACKOFF_BASE_ENV_VAR,
    config.OLLAMA_CIRCUIT_BREAKER_ENV_VAR,
    config.MAX_WORKERS_ENV_VAR,
    config.PROCESS_WORKERS_ENV_VAR,
    config.THREAD_WORKERS_ENV_VAR,
    config.OPENCV_THREADS_ENV_VAR,
)


@pytest.fixture
def isolated_config_env(monkeypatch, tmp_path) -> Path:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    for name in CONFIG_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return cwd


def test_read_env_file_handles_unreadable_and_empty_key(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("=ignored\nexport =also-ignored\nVALID=kept", encoding="utf-8")
    assert config._read_env_file(env_file) == {"VALID": "kept"}

    def fail_read_text(self, *args, **kwargs):
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "read_text", fail_read_text)
    assert config._read_env_file(env_file) == {}


def test_env_file_search_deduplicates_same_directory(isolated_config_env) -> None:
    env_file = isolated_config_env / ".env"
    env_file.write_text("VALUE=one", encoding="utf-8")
    assert config._env_files(isolated_config_env) == [env_file]


def test_string_resolution_precedence_and_empty_values(
    monkeypatch, tmp_path, isolated_config_env, capsys
) -> None:
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (isolated_config_env / ".env").write_text(
        f"{config.ACCOUNT_CONTEXT_ENV_VAR}=   ", encoding="utf-8"
    )
    (search_dir / ".env").write_text(
        f"{config.ACCOUNT_CONTEXT_ENV_VAR}=search context", encoding="utf-8"
    )

    assert config.resolve_account_context(search_dir) == "search context"
    assert "Loaded PICKINSTA_ACCOUNT_CONTEXT" in capsys.readouterr().out
    monkeypatch.setenv(config.ACCOUNT_CONTEXT_ENV_VAR, " process context ")
    assert config.resolve_account_context(search_dir) == "process context"


def test_cwd_env_file_precedes_search_dir(monkeypatch, tmp_path, isolated_config_env) -> None:
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (isolated_config_env / ".env").write_text(
        f"{config.PICKINSTA_OLLAMA_MODEL_ENV_VAR}=cwd-model", encoding="utf-8"
    )
    (search_dir / ".env").write_text(
        f"{config.PICKINSTA_OLLAMA_MODEL_ENV_VAR}=search-model", encoding="utf-8"
    )
    assert config.resolve_ollama_model(search_dir) == "cwd-model"
    assert config.resolve_ollama_model(search_dir) == "cwd-model"


def test_credentials_ignore_whitespace_and_report_missing(
    monkeypatch, tmp_path, isolated_config_env
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    monkeypatch.setenv("HF_TOKEN", "  ")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "\t")
    (isolated_config_env / ".env").write_text("ANTHROPIC_API_KEY=  \nHF_TOKEN=  ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not found"):
        config.resolve_anthropic_api_key(tmp_path / "missing")
    assert config.resolve_optional_hf_token(tmp_path / "missing") is None


@pytest.mark.parametrize("source_name", ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"])
def test_existing_optional_token_propagates_to_both_names(
    monkeypatch, isolated_config_env, source_name
) -> None:
    monkeypatch.setenv(source_name, " token-value ")
    assert config.resolve_optional_hf_token() == "token-value"
    assert config.os.environ["HF_TOKEN"] == "token-value"
    assert config.os.environ["HUGGINGFACE_HUB_TOKEN"] == "token-value"


def test_claude_model_defaults_when_all_sources_absent(isolated_config_env) -> None:
    assert config.resolve_claude_model() == config.DEFAULT_CLAUDE_MODEL


@pytest.mark.parametrize(
    ("resolver", "env_name", "bad_value", "expected"),
    [
        (config.resolve_ollama_timeout_seconds, config.OLLAMA_TIMEOUT_ENV_VAR, "bad", 300),
        (config.resolve_ollama_max_image_edge, config.OLLAMA_MAX_EDGE_ENV_VAR, "3.5", 1024),
        (config.resolve_ollama_jpeg_quality, config.OLLAMA_JPEG_QUALITY_ENV_VAR, "", 80),
        (config.resolve_ollama_concurrency, config.OLLAMA_CONCURRENCY_ENV_VAR, "none", 2),
        (config.resolve_ollama_max_retries, config.OLLAMA_MAX_RETRIES_ENV_VAR, "2x", 2),
        (
            config.resolve_ollama_retry_backoff_seconds,
            config.OLLAMA_BACKOFF_BASE_ENV_VAR,
            "quickly",
            0.75,
        ),
        (
            config.resolve_ollama_circuit_breaker_errors,
            config.OLLAMA_CIRCUIT_BREAKER_ENV_VAR,
            "many",
            6,
        ),
    ],
)
def test_malformed_and_empty_numeric_values_use_defaults(
    monkeypatch, isolated_config_env, resolver, env_name, bad_value, expected
) -> None:
    monkeypatch.setenv(env_name, bad_value)
    assert resolver() == expected


@pytest.mark.parametrize(
    ("resolver", "env_name", "low", "low_expected", "high", "high_expected"),
    [
        (config.resolve_ollama_timeout_seconds, config.OLLAMA_TIMEOUT_ENV_VAR, "0", 30, "999", 999),
        (
            config.resolve_ollama_max_image_edge,
            config.OLLAMA_MAX_EDGE_ENV_VAR,
            "1",
            256,
            "4096",
            4096,
        ),
        (
            config.resolve_ollama_jpeg_quality,
            config.OLLAMA_JPEG_QUALITY_ENV_VAR,
            "1",
            30,
            "100",
            95,
        ),
        (config.resolve_ollama_concurrency, config.OLLAMA_CONCURRENCY_ENV_VAR, "0", 1, "99", 16),
        (config.resolve_ollama_max_retries, config.OLLAMA_MAX_RETRIES_ENV_VAR, "-1", 0, "99", 8),
        (
            config.resolve_ollama_retry_backoff_seconds,
            config.OLLAMA_BACKOFF_BASE_ENV_VAR,
            "0",
            0.05,
            "99",
            10.0,
        ),
        (
            config.resolve_ollama_circuit_breaker_errors,
            config.OLLAMA_CIRCUIT_BREAKER_ENV_VAR,
            "0",
            1,
            "99",
            50,
        ),
    ],
)
def test_numeric_values_are_clamped(
    monkeypatch,
    isolated_config_env,
    resolver,
    env_name,
    low,
    low_expected,
    high,
    high_expected,
) -> None:
    monkeypatch.setenv(env_name, low)
    assert resolver() == low_expected
    monkeypatch.setenv(env_name, high)
    assert resolver() == high_expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("YES", True),
        ("on", True),
        ("1", True),
        ("FALSE", False),
        ("no", False),
        ("0", False),
        ("maybe", False),
        ("", False),
    ],
)
def test_boolean_spellings_and_malformed_default(
    monkeypatch, isolated_config_env, value, expected
) -> None:
    monkeypatch.setenv(config.OLLAMA_USE_YOLO_ENV_VAR, value)
    assert config.resolve_ollama_use_yolo_context() is expected


def test_settings_aggregate_every_resolver_and_are_immutable(
    monkeypatch, isolated_config_env
) -> None:
    monkeypatch.setenv(config.PICKINSTA_OLLAMA_BASE_URL_ENV_VAR, "http://ollama.test")
    monkeypatch.setenv(config.PICKINSTA_OLLAMA_MODEL_ENV_VAR, "vision:latest")
    monkeypatch.setenv(config.OLLAMA_TIMEOUT_ENV_VAR, "60")
    monkeypatch.setenv(config.OLLAMA_MAX_EDGE_ENV_VAR, "512")
    monkeypatch.setenv(config.OLLAMA_JPEG_QUALITY_ENV_VAR, "75")
    monkeypatch.setenv(config.OLLAMA_KEEP_ALIVE_ENV_VAR, "30m")
    monkeypatch.setenv(config.OLLAMA_USE_YOLO_ENV_VAR, "yes")
    monkeypatch.setenv(config.OLLAMA_CONCURRENCY_ENV_VAR, "3")
    monkeypatch.setenv(config.OLLAMA_MAX_RETRIES_ENV_VAR, "4")
    monkeypatch.setenv(config.OLLAMA_BACKOFF_BASE_ENV_VAR, "1.25")
    monkeypatch.setenv(config.OLLAMA_CIRCUIT_BREAKER_ENV_VAR, "9")

    settings = config.resolve_ollama_settings()
    assert settings == config.OllamaSettings(
        base_url="http://ollama.test",
        model="vision:latest",
        timeout_seconds=60,
        max_image_edge=512,
        jpeg_quality=75,
        keep_alive="30m",
        use_yolo_context=True,
        concurrency=3,
        max_retries=4,
        retry_backoff_seconds=1.25,
        circuit_breaker_errors=9,
    )
    with pytest.raises(FrozenInstanceError):
        settings.timeout_seconds = 90
