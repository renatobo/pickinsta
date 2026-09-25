import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parent / "benchmarks" / "offline_stage_benchmark.py"
SPEC = importlib.util.spec_from_file_location("offline_stage_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_percentile_interpolates_and_validates() -> None:
    assert benchmark.percentile([1.0, 2.0, 3.0], 50) == 2.0
    assert benchmark.percentile([1.0, 2.0, 3.0], 95) == pytest.approx(2.9)
    with pytest.raises(ValueError):
        benchmark.percentile([], 95)


def test_summary_schema() -> None:
    summary = benchmark.summarize([0.1, 0.2, 0.3])
    assert summary == {
        "repetitions": 3,
        "median_seconds": 0.2,
        "p95_seconds": pytest.approx(0.29),
        "min_seconds": 0.1,
        "max_seconds": 0.3,
    }


def test_offline_benchmark_report_schema(tmp_path) -> None:
    report = benchmark.run_benchmark(workspace=tmp_path, repetitions=3, image_count=2)
    assert report["schema_version"] == 1
    assert report["benchmark"] == "offline-local-stages"
    assert report["configuration"]["modes"] == ["cold", "warm", "cached"]
    assert report["corpus"]["kind"] == "deterministically-generated"
    assert len(report["corpus"]["files"]) == 2
    assert "python" in report["environment"]
    assert "opencv-python-headless" in report["environment"]["packages"]

    for mode in report["configuration"]["modes"]:
        data = report["modes"][mode]
        assert len(data["samples"]) == 3
        assert data["failure_count"] == 0
        assert set(data["summary"]) == {
            "resize_seconds",
            "technical_score_seconds",
            "total_seconds",
        }
        assert data["summary"]["total_seconds"]["repetitions"] == 3
        expected_hits = 2 if mode == "cached" else 0
        assert all(sample["technical_cache_hits"] == expected_hits for sample in data["samples"])

    json.dumps(report)


def test_report_guard_accepts_complete_cache_reuse(tmp_path) -> None:
    report = benchmark.run_benchmark(workspace=tmp_path, repetitions=3, image_count=2)

    assert benchmark.validate_report(report) == []
    assert benchmark.validate_report(report, max_cached_to_warm_ratio=0.75) == []


def test_report_guard_reports_invariant_violations(tmp_path) -> None:
    report = benchmark.run_benchmark(workspace=tmp_path, repetitions=3, image_count=1)
    report["modes"]["warm"]["failure_count"] = 1
    report["modes"]["cached"]["samples"][0]["technical_cache_hits"] = 0

    assert benchmark.validate_report(report) == [
        "warm mode reported failures",
        "cached mode did not report a technical-cache hit for every image",
    ]


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.5])
def test_report_guard_rejects_invalid_ratio(tmp_path, ratio) -> None:
    report = benchmark.run_benchmark(workspace=tmp_path, repetitions=3, image_count=1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        benchmark.validate_report(report, max_cached_to_warm_ratio=ratio)
