"""Structured timing and runtime evidence for a Pickinsta run manifest."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide resource.
    resource = None  # type: ignore[assignment]

DEFAULT_PACKAGES = (
    "pickinsta",
    "Pillow",
    "opencv-python-headless",
    "numpy",
    "ImageHash",
)

_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = frozenset(
    {
        "apikey",
        "authorization",
        "cookie",
        "password",
        "secret",
        "token",
    }
)
_MAX_MANIFEST_DEPTH = 50


class RunTelemetry:
    """Collect low-overhead, process-local telemetry for one pipeline run."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._stage_seconds: dict[str, float] = {}
        self._stage_invocations: dict[str, int] = {}
        self._cache_counts: dict[str, dict[str, int]] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure one stage invocation, including invocations that raise."""
        if not name:
            raise ValueError("stage name must not be empty")
        started = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - started
            if not math.isfinite(elapsed) or elapsed < 0:
                elapsed = 0.0
            self._stage_seconds[name] = self._stage_seconds.get(name, 0.0) + elapsed
            self._stage_invocations[name] = self._stage_invocations.get(name, 0) + 1

    def record_cache(self, name: str, *, hit: bool) -> None:
        """Record one lookup against a named cache."""
        if not name:
            raise ValueError("cache name must not be empty")
        counts = self._cache_counts.setdefault(name, {"hits": 0, "misses": 0})
        counts["hits" if hit else "misses"] += 1

    def manifest_fields(self) -> dict[str, object]:
        """Return the timing and cache fields intended for ``run_manifest.json``."""
        caches: dict[str, dict[str, int | float]] = {}
        for name, counts in sorted(self._cache_counts.items()):
            lookups = counts["hits"] + counts["misses"]
            caches[name] = {
                **counts,
                "hit_ratio": counts["hits"] / lookups if lookups else 0.0,
            }
        return {
            "stage_timings_seconds": {
                name: round(seconds, 6) for name, seconds in sorted(self._stage_seconds.items())
            },
            "stage_invocations": dict(sorted(self._stage_invocations.items())),
            "caches": caches,
        }


def runtime_metadata(package_names: Sequence[str] = DEFAULT_PACKAGES) -> dict[str, object]:
    """Describe the runtime used for a run in JSON-native values."""
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    peak_memory_bytes = None
    if resource is not None:
        try:
            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if math.isfinite(max_rss) and max_rss >= 0:
                peak_memory_bytes = int(max_rss if sys.platform == "darwin" else max_rss * 1024)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "peak_memory_bytes": peak_memory_bytes,
        "packages": packages,
    }


def build_run_manifest(
    base: Mapping[str, object],
    telemetry: RunTelemetry,
    *,
    configuration: Mapping[str, object],
    warnings: Sequence[Mapping[str, object]] = (),
    package_names: Sequence[str] = DEFAULT_PACKAGES,
) -> dict[str, object]:
    """Add operational evidence to the existing manifest and verify JSON safety."""
    manifest = _json_safe(
        {
            **base,
            "configuration": configuration,
            **telemetry.manifest_fields(),
            "runtime": runtime_metadata(package_names),
            "warnings": warnings,
        }
    )
    json.dumps(manifest, allow_nan=False)
    return manifest


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    if _depth > _MAX_MANIFEST_DEPTH:
        raise ValueError(f"manifest nesting exceeds {_MAX_MANIFEST_DEPTH} levels")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest floats must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            safe_key = str(key)
            normalized[safe_key] = (
                _REDACTED if _is_sensitive_key(safe_key) else _json_safe(item, _depth=_depth + 1)
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, _depth=_depth + 1) for item in value]
    raise TypeError(f"manifest value is not JSON-safe: {type(value).__name__}")


def _is_sensitive_key(key: str) -> bool:
    normalized = "".join(character for character in key.casefold() if character.isalnum())
    return any(normalized == name or normalized.endswith(name) for name in _SENSITIVE_KEYS)
