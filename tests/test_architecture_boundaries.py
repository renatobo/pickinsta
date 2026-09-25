"""Executable dependency rules for the production package.

The import collector deliberately uses only the AST.  This keeps the guard fast and
ensures optional model dependencies are not imported merely to validate architecture.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

import pytest

PACKAGE = "pickinsta"
PACKAGE_ROOT = Path(__file__).parents[1] / "src" / PACKAGE
STAGE_NAMESPACES = frozenset({"pipeline", "vision", "detection"})


def _module_name(package_root: Path, source: Path) -> str:
    relative = source.relative_to(package_root)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PACKAGE, *parts)) if parts else PACKAGE


def _discover_modules(package_root: Path) -> dict[str, Path]:
    return {_module_name(package_root, path): path for path in package_root.rglob("*.py")}


def _absolute_from_base(importer: str, level: int, module: str | None) -> str:
    if level == 0:
        return module or ""

    # Relative imports are resolved from the importing module's package.  One dot
    # stays in that package, two dots move to its parent, and so on.
    package_parts = importer.split(".")[:-1]
    retained = package_parts[: len(package_parts) - (level - 1)]
    if module:
        retained.extend(module.split("."))
    return ".".join(retained)


def _internal_imports(
    importer: str, tree: ast.AST, known_modules: set[str]
) -> Iterable[tuple[str, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE or alias.name.startswith(f"{PACKAGE}."):
                    yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_from_base(importer, node.level, node.module)
            if base != PACKAGE and not base.startswith(f"{PACKAGE}."):
                continue
            for alias in node.names:
                candidate = f"{base}.{alias.name}" if base else alias.name
                # ``from package import module`` creates an edge to the concrete
                # module when it exists; imported attributes retain the base edge.
                yield (candidate if candidate in known_modules else base), node.lineno


def _cycle_violations(graph: dict[str, set[str]]) -> list[str]:
    violations: list[str] = []
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(module: str) -> None:
        visited.add(module)
        active.append(module)
        active_set.add(module)
        for target in sorted(graph[module]):
            if target not in graph:
                continue
            if target not in visited:
                visit(target)
            elif target in active_set:
                start = active.index(target)
                cycle = active[start:] + [target]
                canonical = " -> ".join(cycle)
                if f"import cycle: {canonical}" not in violations:
                    violations.append(f"import cycle: {canonical}")
        active.pop()
        active_set.remove(module)

    for module in sorted(graph):
        if module not in visited:
            visit(module)
    return violations


def architecture_violations(package_root: Path) -> list[str]:
    modules = _discover_modules(package_root)
    known_modules = set(modules)
    graph: dict[str, set[str]] = {module: set() for module in modules}
    violations: list[str] = []

    for importer, path in sorted(modules.items()):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for target, line in _internal_imports(importer, tree, known_modules):
            graph[importer].add(target)
            location = f"{path.relative_to(package_root)}:{line}"
            importer_tail = importer.removeprefix(f"{PACKAGE}.")
            target_tail = target.removeprefix(f"{PACKAGE}.")
            importer_namespace = importer_tail.split(".", 1)[0]
            target_namespace = target_tail.split(".", 1)[0]

            if target == f"{PACKAGE}.ig_image_selector" and importer not in {
                f"{PACKAGE}.__main__",
                f"{PACKAGE}.cli",
            }:
                violations.append(f"{location}: {importer} imports legacy selector")

            if importer in {f"{PACKAGE}.config", f"{PACKAGE}.models"}:
                violations.append(f"{location}: {importer} imports package module {target}")

            if importer_namespace == "infrastructure" and target_namespace in {
                "pipeline",
                "vision",
                "detection",
                "reporting",
                "orchestration",
                "cli",
            }:
                violations.append(f"{location}: infrastructure imports upper layer {target}")

            if importer_namespace in STAGE_NAMESPACES and (
                target_namespace in {"reporting", "cli", "orchestration"}
                or target_tail.endswith(".orchestration")
            ):
                violations.append(f"{location}: stage imports upper layer {target}")

            if target == f"{PACKAGE}.reporting" and importer != f"{PACKAGE}.reporting":
                violations.append(f"{location}: production code imports reporting package root")

    violations.extend(_cycle_violations(graph))
    return violations


def _write_module(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_production_package_respects_architecture_boundaries() -> None:
    assert architecture_violations(PACKAGE_ROOT) == []


@pytest.mark.parametrize(
    ("relative_path", "source", "message"),
    [
        ("models.py", "from pickinsta import config\n", "imports package module"),
        (
            "infrastructure/cache.py",
            "from pickinsta.pipeline import scoring\n",
            "infrastructure imports upper layer",
        ),
        (
            "pipeline/scoring.py",
            "from pickinsta.reporting import markdown\n",
            "stage imports upper layer",
        ),
        (
            "vision/client.py",
            "from pickinsta import ig_image_selector\n",
            "imports legacy selector",
        ),
        (
            "telemetry.py",
            "from pickinsta import reporting\n",
            "imports reporting package root",
        ),
    ],
)
def test_guard_rejects_forbidden_edges(
    tmp_path: Path, relative_path: str, source: str, message: str
) -> None:
    package_root = tmp_path / PACKAGE
    _write_module(package_root, "__init__.py", "")
    # Add possible import targets so ``from package import module`` resolves as
    # it does in the production graph.
    _write_module(package_root, "config.py", "")
    _write_module(package_root, "ig_image_selector.py", "")
    _write_module(package_root, "pipeline/scoring.py", "")
    _write_module(package_root, "reporting/__init__.py", "")
    _write_module(package_root, "reporting/markdown.py", "")
    _write_module(package_root, relative_path, source)

    assert any(message in violation for violation in architecture_violations(package_root))


def test_guard_rejects_relative_import_cycle(tmp_path: Path) -> None:
    package_root = tmp_path / PACKAGE
    _write_module(package_root, "__init__.py", "")
    _write_module(package_root, "alpha.py", "from . import beta\n")
    _write_module(package_root, "beta.py", "from .alpha import VALUE\nVALUE = 1\n")

    assert any(
        violation.startswith("import cycle:") for violation in architecture_violations(package_root)
    )


def test_selector_import_is_limited_to_entrypoints(tmp_path: Path) -> None:
    package_root = tmp_path / PACKAGE
    _write_module(package_root, "__init__.py", "")
    _write_module(package_root, "ig_image_selector.py", "")
    _write_module(package_root, "__main__.py", "from pickinsta import ig_image_selector\n")
    _write_module(package_root, "cli.py", "from . import ig_image_selector\n")

    assert architecture_violations(package_root) == []
