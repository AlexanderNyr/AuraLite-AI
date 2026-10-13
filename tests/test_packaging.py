"""Packaging contract: what `pip install auralite-ai` really gets.

Every CI job installs the project in editable mode, which puts the whole checkout
on ``sys.path``. A module that ``pyproject.toml`` forgets to ship is therefore
invisible to the test suite — that is exactly how ``auralite_cli`` (the target of
the ``auralite`` console script) and ``image_engine`` (the GUI's Image tab) went
missing from the wheel. These tests close the gap from three sides:

* statically, in both directions: every top-level module/package in the checkout
  is declared, and every declared one exists;
* by building a real wheel (no network, no isolation) and reading its file list;
* by checking that the test suite's own imports are declared in the ``test``
  extra, so a clean CI runner can collect it (PyYAML and, on Python 3.10,
  ``tomli`` used to arrive only by accident).
"""
from __future__ import annotations

import ast
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
SETUPTOOLS = PYPROJECT["tool"]["setuptools"]

#: Top-level files/directories that are not part of the runtime distribution.
NOT_SHIPPED_MODULES = {"setup", "conftest"}
NOT_SHIPPED_DIRS = {"tests", "benchmarks", "scripts", "docs", "build", "dist"}


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _checkout_modules() -> set[str]:
    return {p.stem for p in REPO_ROOT.glob("*.py")} - NOT_SHIPPED_MODULES


def _checkout_packages() -> set[str]:
    packages = set()
    for init in REPO_ROOT.rglob("__init__.py"):
        rel = init.parent.relative_to(REPO_ROOT)
        if not rel.parts or rel.parts[0] in NOT_SHIPPED_DIRS:
            continue
        if (REPO_ROOT / rel.parts[0] / "pyvenv.cfg").exists():
            continue  # a virtualenv inside the checkout (venv/, env/, ...)
        if any(part in {"__pycache__", "site-packages", "node_modules"} or part.startswith(".")
               for part in rel.parts):
            continue
        packages.add(".".join(rel.parts))
    return packages


# ----------------------------------------------------------------------
#  Static contract (milliseconds)
# ----------------------------------------------------------------------
def test_every_top_level_module_is_shipped():
    declared = set(SETUPTOOLS["py-modules"])
    on_disk = _checkout_modules()
    assert on_disk - declared == set(), (
        "top-level modules missing from [tool.setuptools] py-modules — a wheel "
        f"install would not contain them: {sorted(on_disk - declared)}")
    assert declared - on_disk == set(), f"declared but absent: {sorted(declared - on_disk)}"


def test_every_package_is_shipped():
    declared = set(SETUPTOOLS["packages"])
    on_disk = _checkout_packages()
    assert on_disk - declared == set(), f"packages missing: {sorted(on_disk - declared)}"
    assert declared - on_disk == set(), f"declared but absent: {sorted(declared - on_disk)}"


def test_console_scripts_target_shipped_modules():
    scripts = PYPROJECT["project"]["scripts"]
    assert "auralite" in scripts
    shipped = set(SETUPTOOLS["py-modules"]) | {p.split(".")[0] for p in SETUPTOOLS["packages"]}
    for name, target in scripts.items():
        module_name, _, attr = target.partition(":")
        assert module_name.split(".")[0] in shipped, f"{name} -> {target} is not shipped"
        assert callable(getattr(importlib.import_module(module_name), attr)), target


def test_license_metadata_uses_spdx():
    project = PYPROJECT["project"]
    assert project["license"] == "Apache-2.0"          # SPDX string, not a table
    for pattern in project["license-files"]:
        assert list(REPO_ROOT.glob(pattern)), f"license file {pattern} is missing"
    requires = " ".join(PYPROJECT["build-system"]["requires"])
    assert re.search(r"setuptools>=(7[7-9]|[89]\d|\d{3,})", requires), requires


def test_cli_reports_its_version(capsys):
    import auralite_cli
    from model_engine import __version__

    with pytest.raises(SystemExit) as exc:
        auralite_cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"auralite {__version__}"


# ----------------------------------------------------------------------
#  The test suite must be collectable on a clean `pip install -e .[test]`
# ----------------------------------------------------------------------
def _module_level_imports(tree: ast.Module) -> set[str]:
    """Unconditional imports executed at collection time (not in try/if/def)."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _declared_closure() -> set[str]:
    """Distributions pip installs for `.[test]`: base + test extra, transitively."""
    try:
        from packaging.requirements import Requirement
    except ImportError:  # pragma: no cover - packaging ships with pytest
        pytest.skip("packaging is not installed")
    project = PYPROJECT["project"]
    # (requirement, extras requested by the parent) pairs still to visit
    pending = [(r, ()) for r in project["dependencies"]]
    pending += [(r, ()) for r in project["optional-dependencies"]["test"]]
    seen: set[str] = set()
    while pending:
        spec, parent_extras = pending.pop()
        req = Requirement(spec)
        if req.marker is not None and not any(
                req.marker.evaluate({"extra": extra}) for extra in ("", *parent_extras)):
            continue
        name = _canonical(req.name)
        key = f"{name}[{','.join(sorted(req.extras))}]"
        if key in seen:
            continue
        seen.update({name, key})
        try:
            children = metadata.requires(req.name) or []
        except metadata.PackageNotFoundError:
            continue  # not installed here: its deps cannot be walked, the name still counts
        pending.extend((child, tuple(req.extras)) for child in children)
    return seen


def _import_name_providers() -> dict[str, set[str]]:
    """import name -> canonical distribution names, for every installed distribution.

    `importlib.metadata.packages_distributions()` on Python 3.10 reads only
    `top_level.txt`, which wheels built by meson-python, pdm, hatchling & co. do
    not have (numpy, fastapi, ...); 3.11+ falls back to the RECORD file list.
    Do the same here so the check behaves identically on every interpreter.
    """
    providers: dict[str, set[str]] = {}
    for dist in metadata.distributions():
        name = _canonical(dist.metadata["Name"] or "")
        if not name:
            continue
        top = set((dist.read_text("top_level.txt") or "").split())
        if not top:
            for file in dist.files or ():
                parts = file.parts
                if not parts or parts[0] == ".." or parts[0].endswith((".dist-info", ".data")):
                    continue
                if len(parts) > 1:
                    top.add(parts[0])                       # a package directory
                elif parts[0].endswith((".py", ".so", ".pyd")):
                    top.add(parts[0].split(".")[0])         # a top-level module
        for module in top:
            providers.setdefault(module, set()).add(name)
    return providers


def test_test_suite_imports_are_declared():
    first_party = (_checkout_modules() | {p.split(".")[0] for p in _checkout_packages()}
                   | {"tests"} | {p.stem for p in (REPO_ROOT / "tests").glob("*.py")})
    stdlib = set(sys.stdlib_module_names)
    providers = _import_name_providers()
    declared = _declared_closure()

    undeclared = {}
    for path in sorted((REPO_ROOT / "tests").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _module_level_imports(tree) - stdlib - first_party - {"__future__"}:
            if not providers.get(module, set()) & declared:
                undeclared.setdefault(module, []).append(path.name)
    assert not undeclared, (
        "test modules import packages that `pip install -e .[test]` does not install "
        f"(add them to the `test` extra or use pytest.importorskip): {undeclared}")


# ----------------------------------------------------------------------
#  A real wheel (built offline from a copy of the checkout)
# ----------------------------------------------------------------------
def _copy_project(dest: pathlib.Path) -> pathlib.Path:
    project = dest / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", *PYPROJECT["project"]["license-files"]):
        shutil.copy2(REPO_ROOT / name, project / name)
    for module in SETUPTOOLS["py-modules"]:
        shutil.copy2(REPO_ROOT / f"{module}.py", project / f"{module}.py")
    for top in sorted({p.split(".")[0] for p in SETUPTOOLS["packages"]}):
        shutil.copytree(REPO_ROOT / top, project / top,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return project


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> zipfile.ZipFile:
    setuptools = pytest.importorskip("setuptools")
    major = int(setuptools.__version__.split(".")[0])
    if major < 77:
        pytest.skip(f"setuptools {setuptools.__version__} < 77 cannot build SPDX metadata")
    tmp = tmp_path_factory.mktemp("wheel")
    project = _copy_project(tmp)
    out = tmp / "dist"
    out.mkdir()
    run = subprocess.run(
        [sys.executable, "-c",
         "import sys, setuptools.build_meta as b; print(b.build_wheel(sys.argv[1]))", str(out)],
        cwd=project, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    wheel = out / run.stdout.strip().splitlines()[-1]
    with zipfile.ZipFile(wheel) as zf:
        yield zf


def test_wheel_contains_every_runtime_module(built_wheel):
    names = set(built_wheel.namelist())
    for module in sorted(_checkout_modules()):
        assert f"{module}.py" in names, f"{module}.py is not in the wheel"
    for package in sorted(_checkout_packages()):
        init = "/".join(package.split(".")) + "/__init__.py"
        assert init in names, f"package {package} is not in the wheel"
    assert "model_engine/py.typed" in names


def test_wheel_metadata_entry_points_and_license(built_wheel):
    names = built_wheel.namelist()
    dist_info = next(n.split("/")[0] for n in names if n.split("/")[0].endswith(".dist-info"))
    entry_points = built_wheel.read(f"{dist_info}/entry_points.txt").decode()
    assert "auralite = auralite_cli:main" in entry_points
    meta = built_wheel.read(f"{dist_info}/METADATA").decode()
    assert "License-Expression: Apache-2.0" in meta
    assert f"{dist_info}/licenses/LICENSE" in names


# ----------------------------------------------------------------------
#  scripts/smoke_installed_wheel.py (used by the CI `package` and release jobs)
# ----------------------------------------------------------------------
def _load_smoke():
    spec = importlib.util.spec_from_file_location(
        "smoke_installed_wheel", REPO_ROOT / "scripts" / "smoke_installed_wheel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _installed_from_a_wheel() -> bool:
    try:
        dist = metadata.distribution("auralite-ai")
    except metadata.PackageNotFoundError:
        return False
    location = pathlib.Path(dist.locate_file("")).resolve()
    if location == REPO_ROOT or REPO_ROOT in location.parents:
        return False  # the egg-info a local `python -m build` leaves in the checkout
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    return not direct_url.get("dir_info", {}).get("editable", False)


def test_smoke_script_refuses_a_checkout():
    """From a source checkout the distribution is absent or editable: exit code 2."""
    if _installed_from_a_wheel():
        pytest.skip("auralite-ai is installed from a wheel here; the CI package job covers it")
    smoke = _load_smoke()
    saved = list(sys.path)
    try:
        assert smoke.main() == 2
    finally:
        sys.path[:] = saved


class _FakeEntryPoint:
    group = "console_scripts"
    name = "auralite"
    value = "auralite_cli:main"

    def load(self):
        raise ModuleNotFoundError("No module named 'auralite_cli'", name="auralite_cli")


class _FakeDist:
    version = "0.0.0"
    entry_points = [_FakeEntryPoint()]


def test_smoke_script_reports_a_dead_console_script():
    smoke = _load_smoke()
    problems, notes = [], []
    smoke._check_console_scripts(_FakeDist(), problems, notes)
    assert problems and "auralite_cli" in problems[0] and "does not load" in problems[0]
