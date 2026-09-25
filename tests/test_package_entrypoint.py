import importlib
import importlib.metadata
import runpy
import sys
from types import ModuleType

import pytest

import pickinsta


def _cli_stub(main):
    module = ModuleType("pickinsta.cli")
    module.main = main
    return module


def test_public_version_matches_installed_package_metadata() -> None:
    assert pickinsta.__version__ == importlib.metadata.version("pickinsta")


def test_importing_module_entrypoint_has_no_cli_side_effect(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "pickinsta.__main__", raising=False)
    monkeypatch.delitem(sys.modules, "pickinsta.cli", raising=False)
    monkeypatch.delitem(sys.modules, "pickinsta.ig_image_selector", raising=False)

    imported = importlib.import_module("pickinsta.__main__")

    assert "pickinsta.ig_image_selector" not in sys.modules
    assert "pickinsta.cli" not in sys.modules
    assert callable(imported._run)


def test_python_m_pickinsta_delegates_to_cli_main(monkeypatch) -> None:
    calls = []
    cli = _cli_stub(lambda: calls.append("main"))
    monkeypatch.delitem(sys.modules, "pickinsta.__main__", raising=False)
    monkeypatch.setitem(sys.modules, "pickinsta.cli", cli)

    runpy.run_module("pickinsta.__main__", run_name="__main__")

    assert calls == ["main"]


def test_python_m_pickinsta_propagates_cli_exit(monkeypatch) -> None:
    def exit_main() -> None:
        raise SystemExit(23)

    cli = _cli_stub(exit_main)
    monkeypatch.delitem(sys.modules, "pickinsta.__main__", raising=False)
    monkeypatch.setitem(sys.modules, "pickinsta.cli", cli)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("pickinsta.__main__", run_name="__main__")

    assert exc_info.value.code == 23
