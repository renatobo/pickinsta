"""Environment-backed configuration for Pickinsta integrations.

This module owns configuration resolution only.  Callers may continue importing
the historical names from :mod:`pickinsta.ig_image_selector`, which re-exports
this API for compatibility.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OLLAMA_MODEL = "qwen2.5vl:7b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_ACCOUNT_CONTEXT = "Ducati motorcycle enthusiast account"

ACCOUNT_CONTEXT_ENV_VAR = "PICKINSTA_ACCOUNT_CONTEXT"
PICKINSTA_OLLAMA_BASE_URL_ENV_VAR = "PICKINSTA_OLLAMA_BASE_URL"
PICKINSTA_OLLAMA_MODEL_ENV_VAR = "PICKINSTA_OLLAMA_MODEL"
OLLAMA_TIMEOUT_ENV_VAR = "PICKINSTA_OLLAMA_TIMEOUT_SEC"
OLLAMA_MAX_EDGE_ENV_VAR = "PICKINSTA_OLLAMA_MAX_IMAGE_EDGE"
OLLAMA_JPEG_QUALITY_ENV_VAR = "PICKINSTA_OLLAMA_JPEG_QUALITY"
OLLAMA_KEEP_ALIVE_ENV_VAR = "PICKINSTA_OLLAMA_KEEP_ALIVE"
OLLAMA_USE_YOLO_ENV_VAR = "PICKINSTA_OLLAMA_USE_YOLO_CONTEXT"
OLLAMA_CONCURRENCY_ENV_VAR = "PICKINSTA_OLLAMA_CONCURRENCY"
OLLAMA_MAX_RETRIES_ENV_VAR = "PICKINSTA_OLLAMA_MAX_RETRIES"
OLLAMA_BACKOFF_BASE_ENV_VAR = "PICKINSTA_OLLAMA_RETRY_BACKOFF_SEC"
OLLAMA_CIRCUIT_BREAKER_ENV_VAR = "PICKINSTA_OLLAMA_CIRCUIT_BREAKER_ERRORS"
YOLO_MODEL_ENV_VAR = "PICKINSTA_YOLO_MODEL"
MAX_WORKERS_ENV_VAR = "PICKINSTA_MAX_WORKERS"
PROCESS_WORKERS_ENV_VAR = "PICKINSTA_PROCESS_WORKERS"
THREAD_WORKERS_ENV_VAR = "PICKINSTA_THREAD_WORKERS"
OPENCV_THREADS_ENV_VAR = "PICKINSTA_OPENCV_THREADS"
MAX_LOCAL_WORKERS = 256


def _read_env_file(env_path: Path) -> dict[str, str]:
    """Parse the simple ``KEY=value`` subset used by Pickinsta .env files."""
    values: dict[str, str] = {}
    try:
        content = env_path.read_text(encoding="utf-8")
    except OSError:
        return values

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _env_files(search_dir: Optional[Path]) -> list[Path]:
    candidates = [Path.cwd() / ".env"]
    if search_dir is not None:
        candidates.append(search_dir / ".env")
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and candidate.exists():
            seen.add(resolved)
            result.append(candidate)
    return result


def _resolve_env_string(var_name: str, search_dir: Optional[Path] = None) -> Optional[str]:
    env_value = (os.environ.get(var_name) or "").strip()
    if env_value:
        return env_value
    for env_file in _env_files(search_dir):
        value = (_read_env_file(env_file).get(var_name) or "").strip()
        if value:
            os.environ.setdefault(var_name, value)
            print(f"  📝 Loaded {var_name} from {env_file}")
            return value
    return None


def resolve_anthropic_api_key(search_dir: Optional[Path] = None) -> str:
    env_value = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if env_value:
        return env_value
    for env_file in _env_files(search_dir):
        key = (_read_env_file(env_file).get("ANTHROPIC_API_KEY") or "").strip()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            print(f"  🔐 Loaded ANTHROPIC_API_KEY from {env_file}")
            return key
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. Set it in the environment or add it to a .env file."
    )


def resolve_optional_hf_token(search_dir: Optional[Path] = None) -> Optional[str]:
    existing = (os.environ.get("HF_TOKEN") or "").strip() or (
        os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""
    ).strip()
    if existing:
        os.environ["HF_TOKEN"] = existing
        os.environ["HUGGINGFACE_HUB_TOKEN"] = existing
        return existing
    for env_file in _env_files(search_dir):
        values = _read_env_file(env_file)
        token = (values.get("HF_TOKEN") or "").strip() or (
            values.get("HUGGINGFACE_HUB_TOKEN") or ""
        ).strip()
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token
            print(f"  🔐 Loaded HF_TOKEN from {env_file} (optional)")
            return token
    return None


def resolve_claude_model(cli_model: Optional[str] = None) -> str:
    return (
        cli_model
        or os.environ.get("ANTHROPIC_MODEL")
        or os.environ.get("CLAUDE_MODEL")
        or DEFAULT_CLAUDE_MODEL
    )


def _resolve_env_int(var_name: str, default: int) -> int:
    try:
        return int((os.environ.get(var_name) or "").strip() or default)
    except ValueError:
        return default


def _resolve_env_float(var_name: str, default: float) -> float:
    try:
        return float((os.environ.get(var_name) or "").strip() or default)
    except ValueError:
        return default


def _resolve_env_bool(var_name: str, default: bool) -> bool:
    raw = (os.environ.get(var_name) or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def resolve_ollama_base_url(search_dir: Optional[Path] = None) -> str:
    return (
        _resolve_env_string(PICKINSTA_OLLAMA_BASE_URL_ENV_VAR, search_dir)
        or DEFAULT_OLLAMA_BASE_URL
    )


def resolve_ollama_model(search_dir: Optional[Path] = None) -> str:
    return _resolve_env_string(PICKINSTA_OLLAMA_MODEL_ENV_VAR, search_dir) or DEFAULT_OLLAMA_MODEL


def resolve_ollama_timeout_seconds() -> int:
    return max(30, _resolve_env_int(OLLAMA_TIMEOUT_ENV_VAR, 300))


def resolve_ollama_max_image_edge() -> int:
    return max(256, _resolve_env_int(OLLAMA_MAX_EDGE_ENV_VAR, 1024))


def resolve_ollama_jpeg_quality() -> int:
    return min(95, max(30, _resolve_env_int(OLLAMA_JPEG_QUALITY_ENV_VAR, 80)))


def resolve_ollama_keep_alive(search_dir: Optional[Path] = None) -> str:
    return _resolve_env_string(OLLAMA_KEEP_ALIVE_ENV_VAR, search_dir) or "10m"


def resolve_ollama_use_yolo_context() -> bool:
    return _resolve_env_bool(OLLAMA_USE_YOLO_ENV_VAR, False)


def resolve_ollama_concurrency() -> int:
    return min(16, max(1, _resolve_env_int(OLLAMA_CONCURRENCY_ENV_VAR, 2)))


def resolve_ollama_max_retries() -> int:
    return min(8, max(0, _resolve_env_int(OLLAMA_MAX_RETRIES_ENV_VAR, 2)))


def resolve_ollama_retry_backoff_seconds() -> float:
    return min(10.0, max(0.05, _resolve_env_float(OLLAMA_BACKOFF_BASE_ENV_VAR, 0.75)))


def resolve_ollama_circuit_breaker_errors() -> int:
    return min(50, max(1, _resolve_env_int(OLLAMA_CIRCUIT_BREAKER_ENV_VAR, 6)))


def resolve_account_context(search_dir: Optional[Path] = None) -> str:
    return _resolve_env_string(ACCOUNT_CONTEXT_ENV_VAR, search_dir) or DEFAULT_ACCOUNT_CONTEXT


@dataclass(frozen=True)
class WorkerSettings:
    """Immutable limits for local CPU-bound and native OpenCV work."""

    process_workers: int
    thread_workers: int
    opencv_threads: int | None


def _resolve_worker_cap(var_name: str, default: int, *, allow_zero: bool = False) -> int:
    raw = (os.environ.get(var_name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = -1
    minimum = 0 if allow_zero else 1
    if minimum <= value <= MAX_LOCAL_WORKERS:
        return value
    print(f"  ⚠ Configuration warning: {var_name}={raw!r} is invalid; using {default}.")
    return default


def resolve_worker_settings(cpu_count: int | None = None) -> WorkerSettings:
    """Resolve local worker limits at run time so embedders can change the environment."""
    logical_cpus = cpu_count if cpu_count is not None else os.cpu_count()
    # Image decode/model workloads are memory hungry. Keep the implicit default
    # conservative; deployments can opt into a higher cap explicitly.
    default_cap = min(8, MAX_LOCAL_WORKERS, max(1, logical_cpus or 4))
    shared_cap = _resolve_worker_cap(MAX_WORKERS_ENV_VAR, default_cap)
    process_cap = _resolve_worker_cap(PROCESS_WORKERS_ENV_VAR, shared_cap)
    thread_cap = _resolve_worker_cap(THREAD_WORKERS_ENV_VAR, shared_cap)
    opencv_threads = _resolve_worker_cap(OPENCV_THREADS_ENV_VAR, 0, allow_zero=True)
    return WorkerSettings(
        process_workers=process_cap,
        thread_workers=thread_cap,
        opencv_threads=opencv_threads or None,
    )


@dataclass(frozen=True)
class OllamaSettings:
    """Immutable snapshot of Ollama connection and request tuning."""

    base_url: str
    model: str
    timeout_seconds: int
    max_image_edge: int
    jpeg_quality: int
    keep_alive: str
    use_yolo_context: bool
    concurrency: int
    max_retries: int
    retry_backoff_seconds: float
    circuit_breaker_errors: int


def resolve_ollama_settings(search_dir: Optional[Path] = None) -> OllamaSettings:
    return OllamaSettings(
        base_url=resolve_ollama_base_url(search_dir),
        model=resolve_ollama_model(search_dir),
        timeout_seconds=resolve_ollama_timeout_seconds(),
        max_image_edge=resolve_ollama_max_image_edge(),
        jpeg_quality=resolve_ollama_jpeg_quality(),
        keep_alive=resolve_ollama_keep_alive(search_dir),
        use_yolo_context=resolve_ollama_use_yolo_context(),
        concurrency=resolve_ollama_concurrency(),
        max_retries=resolve_ollama_max_retries(),
        retry_backoff_seconds=resolve_ollama_retry_backoff_seconds(),
        circuit_breaker_errors=resolve_ollama_circuit_breaker_errors(),
    )
