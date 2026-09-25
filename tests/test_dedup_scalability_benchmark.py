import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parent / "benchmarks" / "dedup_scalability_benchmark.py"
SPEC = importlib.util.spec_from_file_location("dedup_scalability_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_unique_grouping_reduces_comparisons_substantially() -> None:
    report = benchmark.run_benchmark(sizes=(64,), repetitions=3)
    result = report["results"][0]

    assert result["reference_comparison_count"] == 64 * 63 // 2
    assert result["comparison_reduction_ratio"] >= 0.5
    assert result["group_count"] == 64


def test_report_is_json_serializable_and_records_before_after_counts() -> None:
    report = benchmark.run_benchmark(sizes=(16, 32, 64), repetitions=3)

    assert report["schema_version"] == 2
    assert report["scenario"] == "all-unique-256-bit-perceptual-hashes"
    assert benchmark.validate_report(report) == []
    assert [result["reference_comparison_count"] for result in report["results"]] == [
        120,
        496,
        2016,
    ]
    assert all(len(result["indexed_samples_seconds"]) == 3 for result in report["results"])
    json.dumps(report)


@pytest.mark.parametrize("repetitions", [0, 1, 2])
def test_report_requires_at_least_three_repetitions(repetitions: int) -> None:
    with pytest.raises(ValueError, match="at least 3"):
        benchmark.run_benchmark(sizes=(8,), repetitions=repetitions)
