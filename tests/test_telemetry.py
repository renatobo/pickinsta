import importlib.metadata
import json
from pathlib import Path

import pytest

from pickinsta.telemetry import RunTelemetry, build_run_manifest, runtime_metadata


class Clock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def test_stage_timings_accumulate_and_record_failed_invocations() -> None:
    telemetry = RunTelemetry(clock=Clock(10.0, 11.25, 20.0, 19.0))

    with telemetry.stage("score"):
        pass
    with pytest.raises(RuntimeError, match="failed"):
        with telemetry.stage("score"):
            raise RuntimeError("failed")

    assert telemetry.manifest_fields() == {
        "stage_timings_seconds": {"score": 1.25},
        "stage_invocations": {"score": 2},
        "caches": {},
    }


def test_nested_and_repeated_stages_are_measured_independently() -> None:
    telemetry = RunTelemetry(clock=Clock(0.0, 1.0, 2.5, 4.0, 10.0, 12.0))

    with telemetry.stage("outer"):
        with telemetry.stage("inner"):
            pass
    with telemetry.stage("outer"):
        pass

    fields = telemetry.manifest_fields()
    assert fields["stage_timings_seconds"] == {"inner": 1.5, "outer": 6.0}
    assert fields["stage_invocations"] == {"inner": 1, "outer": 2}


@pytest.mark.parametrize("end", [float("nan"), float("inf"), float("-inf")])
def test_stage_timing_clamps_non_finite_clock_results(end: float) -> None:
    telemetry = RunTelemetry(clock=Clock(1.0, end))

    with telemetry.stage("clock_anomaly"):
        pass

    assert telemetry.manifest_fields()["stage_timings_seconds"] == {"clock_anomaly": 0.0}


def test_stage_and_cache_names_must_not_be_empty() -> None:
    telemetry = RunTelemetry()

    with pytest.raises(ValueError, match="stage name"):
        with telemetry.stage(""):
            pass
    with pytest.raises(ValueError, match="cache name"):
        telemetry.record_cache("", hit=True)


def test_cache_counts_and_ratios_are_stable_and_sorted() -> None:
    telemetry = RunTelemetry()
    telemetry.record_cache("vision", hit=False)
    telemetry.record_cache("technical", hit=True)
    telemetry.record_cache("technical", hit=False)
    telemetry.record_cache("technical", hit=True)

    assert telemetry.manifest_fields()["caches"] == {
        "technical": {"hits": 2, "misses": 1, "hit_ratio": 2 / 3},
        "vision": {"hits": 0, "misses": 1, "hit_ratio": 0.0},
    }


def test_runtime_metadata_reports_installed_and_missing_packages(monkeypatch) -> None:
    def version(name: str) -> str:
        if name == "missing":
            raise importlib.metadata.PackageNotFoundError(name)
        return "1.2.3"

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr("pickinsta.telemetry.platform.python_version", lambda: "3.test")
    monkeypatch.setattr("pickinsta.telemetry.platform.python_implementation", lambda: "CPython")
    monkeypatch.setattr("pickinsta.telemetry.platform.platform", lambda: "test-platform")
    monkeypatch.setattr("pickinsta.telemetry.os.cpu_count", lambda: 8)
    monkeypatch.setattr(
        "pickinsta.telemetry.resource.getrusage", lambda _who: type("R", (), {"ru_maxrss": 5})()
    )
    monkeypatch.setattr("pickinsta.telemetry.sys.platform", "linux")

    assert runtime_metadata(("present", "missing")) == {
        "python": "3.test",
        "implementation": "CPython",
        "platform": "test-platform",
        "cpu_count": 8,
        "peak_memory_bytes": 5120,
        "packages": {"present": "1.2.3", "missing": None},
    }


def test_runtime_metadata_uses_byte_rss_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(
        "pickinsta.telemetry.resource.getrusage", lambda _who: type("R", (), {"ru_maxrss": 7})()
    )
    monkeypatch.setattr("pickinsta.telemetry.sys.platform", "darwin")

    assert runtime_metadata(())["peak_memory_bytes"] == 7


def test_runtime_metadata_tolerates_platform_without_resource(monkeypatch) -> None:
    monkeypatch.setattr("pickinsta.telemetry.resource", None)

    assert runtime_metadata(())["peak_memory_bytes"] is None


@pytest.mark.parametrize("max_rss", [-1, float("nan")])
def test_runtime_metadata_ignores_invalid_resource_values(monkeypatch, max_rss) -> None:
    monkeypatch.setattr(
        "pickinsta.telemetry.resource.getrusage",
        lambda _who: type("R", (), {"ru_maxrss": max_rss})(),
    )

    assert runtime_metadata(())["peak_memory_bytes"] is None


def test_runtime_metadata_tolerates_resource_errors(monkeypatch) -> None:
    def unavailable(_who):
        raise OSError("resource accounting unavailable")

    monkeypatch.setattr("pickinsta.telemetry.resource.getrusage", unavailable)

    assert runtime_metadata(())["peak_memory_bytes"] is None


def test_manifest_builder_preserves_contract_and_normalizes_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "pickinsta.telemetry.runtime_metadata", lambda _packages: {"python": "test"}
    )
    telemetry = RunTelemetry(clock=Clock(1.0, 1.5))
    with telemetry.stage("deduplicate"):
        pass
    telemetry.record_cache("technical", hit=True)

    manifest = build_run_manifest(
        {"schema_version": 1, "status": "complete", "issues": []},
        telemetry,
        configuration={"input": tmp_path, "models": ("one", "two")},
        warnings=({"stage": "vision", "artifact": Path("image.jpg")},),
        package_names=("pickinsta",),
    )

    assert manifest["schema_version"] == 1
    assert manifest["configuration"] == {"input": str(tmp_path), "models": ["one", "two"]}
    assert manifest["warnings"] == [{"stage": "vision", "artifact": "image.jpg"}]
    assert manifest["runtime"] == {"python": "test"}
    json.dumps(manifest)


def test_manifest_builder_rejects_unsupported_values(monkeypatch) -> None:
    monkeypatch.setattr("pickinsta.telemetry.runtime_metadata", lambda _packages: {})

    with pytest.raises(TypeError, match="object"):
        build_run_manifest({}, RunTelemetry(), configuration={"bad": object()})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_manifest_builder_rejects_non_finite_values(monkeypatch, value: float) -> None:
    monkeypatch.setattr("pickinsta.telemetry.runtime_metadata", lambda _packages: {})

    with pytest.raises(ValueError, match="finite"):
        build_run_manifest({"score": value}, RunTelemetry(), configuration={})


def test_manifest_builder_redacts_sensitive_keys_recursively(monkeypatch) -> None:
    monkeypatch.setattr("pickinsta.telemetry.runtime_metadata", lambda _packages: {})

    manifest = build_run_manifest(
        {"authorization": "Bearer base-secret"},
        RunTelemetry(),
        configuration={
            "anthropic_api_key": "config-secret",
            "nested": {"access-token": "nested-secret", "model": "safe"},
        },
        warnings=({"context": {"password": "warning-secret", "detail": "safe"}},),
    )

    assert manifest["authorization"] == "[REDACTED]"
    assert manifest["configuration"] == {
        "anthropic_api_key": "[REDACTED]",
        "nested": {"access-token": "[REDACTED]", "model": "safe"},
    }
    assert manifest["warnings"] == [{"context": {"password": "[REDACTED]", "detail": "safe"}}]
    serialized = json.dumps(manifest)
    assert "base-secret" not in serialized
    assert "config-secret" not in serialized
    assert "nested-secret" not in serialized
    assert "warning-secret" not in serialized


def test_manifest_builder_rejects_excessive_nesting(monkeypatch) -> None:
    monkeypatch.setattr("pickinsta.telemetry.runtime_metadata", lambda _packages: {})
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(60):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child

    with pytest.raises(ValueError, match="nesting"):
        build_run_manifest({}, RunTelemetry(), configuration=nested)
