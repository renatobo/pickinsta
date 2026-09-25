from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import pickinsta.config as config
import pickinsta.ig_image_selector as selector
from pickinsta.pipeline import worker_tuning

WORKER_ENV_NAMES = (
    config.MAX_WORKERS_ENV_VAR,
    config.PROCESS_WORKERS_ENV_VAR,
    config.THREAD_WORKERS_ENV_VAR,
    config.OPENCV_THREADS_ENV_VAR,
)


@pytest.fixture(autouse=True)
def isolated_worker_env(monkeypatch):
    for name in WORKER_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_worker_settings_default_to_logical_cpu_count_and_are_frozen() -> None:
    settings = config.resolve_worker_settings(cpu_count=4)
    assert settings == config.WorkerSettings(4, 4, None)
    with pytest.raises(FrozenInstanceError):
        settings.thread_workers = 2  # type: ignore[misc]


def test_implicit_worker_default_is_capped_for_memory_heavy_images() -> None:
    settings = config.resolve_worker_settings(cpu_count=64)
    assert settings.process_workers == settings.thread_workers == 8


def test_shared_worker_cap_applies_to_both_pool_kinds(monkeypatch) -> None:
    monkeypatch.setenv(config.MAX_WORKERS_ENV_VAR, "2")
    settings = config.resolve_worker_settings(cpu_count=8)
    assert settings.process_workers == settings.thread_workers == 2


def test_stage_specific_worker_caps_override_shared_cap(monkeypatch) -> None:
    monkeypatch.setenv(config.MAX_WORKERS_ENV_VAR, "2")
    monkeypatch.setenv(config.PROCESS_WORKERS_ENV_VAR, "3")
    monkeypatch.setenv(config.THREAD_WORKERS_ENV_VAR, "5")
    assert config.resolve_worker_settings(cpu_count=8) == config.WorkerSettings(3, 5, None)


@pytest.mark.parametrize("bad_value", ["", "nope", "0", "-1", "257"])
def test_invalid_worker_cap_falls_back_with_warning(monkeypatch, capsys, bad_value) -> None:
    monkeypatch.setenv(config.MAX_WORKERS_ENV_VAR, bad_value)
    settings = config.resolve_worker_settings(cpu_count=4)
    assert settings.process_workers == settings.thread_workers == 4
    if bad_value:
        assert "Configuration warning" in capsys.readouterr().out


def test_invalid_stage_override_falls_back_to_shared_cap(monkeypatch, capsys) -> None:
    monkeypatch.setenv(config.MAX_WORKERS_ENV_VAR, "3")
    monkeypatch.setenv(config.PROCESS_WORKERS_ENV_VAR, "zero")
    settings = config.resolve_worker_settings(cpu_count=8)
    assert settings.process_workers == settings.thread_workers == 3
    assert config.PROCESS_WORKERS_ENV_VAR in capsys.readouterr().out


def test_bounded_worker_count_uses_kind_cap_and_available_work() -> None:
    settings = config.WorkerSettings(2, 4, None)
    assert worker_tuning.bounded_worker_count("process", 8, settings) == 2
    assert worker_tuning.bounded_worker_count("thread", 8, settings) == 4
    assert worker_tuning.bounded_worker_count("thread", 3, settings) == 3
    assert worker_tuning.bounded_worker_count("process", 0, settings) == 1


@pytest.mark.parametrize(("raw", "effective"), [(None, None), ("0", None), ("1", 1), ("256", 256)])
def test_opencv_thread_resolution(monkeypatch, raw, effective) -> None:
    if raw is not None:
        monkeypatch.setenv(config.OPENCV_THREADS_ENV_VAR, raw)
    assert config.resolve_worker_settings(cpu_count=4).opencv_threads == effective


def test_configure_opencv_threads_only_applies_explicit_setting(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(worker_tuning.cv2, "setNumThreads", calls.append)
    worker_tuning.configure_opencv_threads(config.WorkerSettings(2, 2, None))
    worker_tuning.configure_opencv_threads(config.WorkerSettings(2, 2, 1))
    assert calls == [1]


class _ExecutorSpy:
    created_with: list[int] = []

    def __init__(self, max_workers: int):
        self.created_with.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def map(self, function, items):
        return map(function, items)


def test_resize_stage_uses_process_worker_limit(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    work = tmp_path / "work"
    source.mkdir()
    for index in range(3):
        (source / f"{index}.jpg").write_bytes(b"image")
    _ExecutorSpy.created_with = []
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _ExecutorSpy)
    monkeypatch.setattr(selector, "_resize_one_image", lambda pair: (pair[1], pair[0]))

    selector.resize_for_processing(source, work, config.WorkerSettings(2, 7, None))

    assert _ExecutorSpy.created_with == [2]


def test_technical_stage_uses_thread_worker_limit(monkeypatch, tmp_path) -> None:
    paths = [tmp_path / f"{index}.jpg" for index in range(3)]
    _ExecutorSpy.created_with = []
    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", _ExecutorSpy)
    monkeypatch.setattr(selector, "_load_tech_cache", lambda _path: None)
    monkeypatch.setattr(
        selector,
        "_score_technical_with_cache",
        lambda path: (path, {"composite": float(path.stem)}),
    )

    results = selector.batch_technical_score(
        paths, worker_settings=config.WorkerSettings(7, 2, None)
    )

    assert _ExecutorSpy.created_with == [2]
    assert [item.path for item in results] == list(reversed(paths))


def test_worker_stage_adapter_preserves_legacy_injected_callable() -> None:
    calls: list[tuple[Path, Path]] = []

    def legacy_stage(source: Path, work: Path):
        calls.append((source, work))
        return "ok"

    result = selector._call_worker_stage(
        legacy_stage,
        Path("source"),
        Path("work"),
        worker_settings=config.WorkerSettings(1, 1, None),
    )

    assert result == "ok"
    assert calls == [(Path("source"), Path("work"))]
