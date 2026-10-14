"""Packaging contract: what `pip install auralite-ai` really gets.

Every CI job installs the project in editable mode, which puts the whole checkout
on ``sys.path``. A module that ``pyproject.toml`` forgets to ship is therefore
invisible to the test suite — that is exactly how ``auralite_cli`` (the target of
the ``auralite`` console script) and ``image_engine`` (the GUI's Image tab) went
missing from the 2.6.4 wheel. Since 2.7.0 everything lives in the one
``auralite`` package and the flat names are checkout-only aliases. These tests
close the gap from three sides:

* statically: the package is discovered (no hand-kept module list), every
  directory in it is a regular package, and the repository root holds nothing
  but the deprecated aliases — a new top-level module would not be shipped;
* by building a real wheel (no network, no isolation) and reading its file list:
  ``auralite/`` and nothing else at the top level, every module included;
* by checking that the test suite's own imports are declared in the ``test``
  extra, so a clean CI runner can collect it (PyYAML used to arrive only by
  accident).
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

import tomllib

from auralite import _compat

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "auralite"
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
SETUPTOOLS = PYPROJECT["tool"]["setuptools"]

#: Repository-root modules that are deprecated aliases (checkout only, never shipped).
ROOT_ALIASES = set(_compat.RENAMED)
#: Other top-level entries that are not part of the runtime distribution.
NOT_SHIPPED_MODULES = {"setup", "conftest"}
NOT_SHIPPED_DIRS = {"tests", "benchmarks", "scripts", "docs", "build", "dist"}


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _package_modules() -> set[str]:
    """Every ``.py`` file of the package, as a wheel path (``auralite/x/y.py``)."""
    return {path.relative_to(REPO_ROOT).as_posix() for path in PACKAGE.rglob("*.py")
            if "__pycache__" not in path.parts}


def _package_dirs() -> set[pathlib.Path]:
    return {path.parent for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts}


def _root_packages() -> set[str]:
    """Top-level directories of the checkout that are Python packages."""
    names = set()
    for init in REPO_ROOT.glob("*/__init__.py"):
        name = init.parent.name
        if name in NOT_SHIPPED_DIRS or name.startswith("."):
            continue
        if (init.parent / "pyvenv.cfg").exists():
            continue  # a virtualenv inside the checkout (venv/, env/, ...)
        names.add(name)
    return names


# ----------------------------------------------------------------------
#  Static contract (milliseconds)
# ----------------------------------------------------------------------
def test_one_package_is_discovered():
    assert "py-modules" not in SETUPTOOLS, "2.7.0 ships no top-level modules"
    find = SETUPTOOLS["packages"]["find"]
    assert find["include"] == ["auralite", "auralite.*"]
    # Namespace discovery would silently ship (or drop) half-made packages.
    assert find["namespaces"] is False


def test_every_package_directory_is_a_regular_package():
    """With `namespaces = false` a directory without __init__.py is not shipped."""
    missing = sorted(str(d.relative_to(REPO_ROOT)) for d in _package_dirs()
                     if not (d / "__init__.py").exists())
    assert not missing, f"directories that would be left out of the wheel: {missing}"


def test_repository_root_holds_only_the_deprecated_aliases():
    """A new top-level module or package would be invisible to the wheel."""
    root_modules = {p.stem for p in REPO_ROOT.glob("*.py")} - NOT_SHIPPED_MODULES
    assert root_modules == ROOT_ALIASES, (
        "top-level modules must be the deprecated aliases from auralite/_compat.py — "
        "new code belongs in the auralite package: "
        f"unexpected {sorted(root_modules - ROOT_ALIASES)}, "
        f"missing {sorted(ROOT_ALIASES - root_modules)}")
    assert _root_packages() == {"auralite"}, _root_packages()
    assert "kernels" not in ROOT_ALIASES     # Hugging Face's name: no alias at all


def test_entry_points_target_the_package():
    scripts = PYPROJECT["project"]["scripts"]
    gui_scripts = PYPROJECT["project"]["gui-scripts"]
    assert scripts == {"auralite": "auralite.cli:main"}
    assert gui_scripts == {"auralite-gui": "auralite.gui.app:main"}
    for target in (*scripts.values(), *gui_scripts.values()):
        module_name, _, attr = target.partition(":")
        if module_name.startswith("auralite.gui"):
            pytest.importorskip("tkinter")
        assert callable(getattr(importlib.import_module(module_name), attr)), target


def test_license_metadata_uses_spdx():
    project = PYPROJECT["project"]
    assert project["license"] == "Apache-2.0"          # SPDX string, not a table
    for pattern in project["license-files"]:
        assert list(REPO_ROOT.glob(pattern)), f"license file {pattern} is missing"
    requires = " ".join(PYPROJECT["build-system"]["requires"])
    assert re.search(r"setuptools>=(7[7-9]|[89]\d|\d{3,})", requires), requires


def test_cli_reports_its_version(capsys):
    import auralite
    import auralite.cli as auralite_cli
    from auralite.engine import __version__

    assert __version__ == auralite.__version__ == PYPROJECT["project"]["version"]
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
    first_party = ({"auralite", "tests"} | ROOT_ALIASES
                   | {p.stem for p in (REPO_ROOT / "tests").glob("*.py")})
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
    shutil.copytree(PACKAGE, project / "auralite",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The aliases sit next to the package in a checkout; copy them too, so the
    # wheel is built from the same layout and has to leave them out on its own.
    for alias in sorted(ROOT_ALIASES):
        shutil.copy2(REPO_ROOT / f"{alias}.py", project / f"{alias}.py")
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


def test_wheel_top_level_is_only_auralite(built_wheel):
    """The PyPI contract: no generic top-level names (`server`, `kernels`, …)."""
    top = {name.split("/")[0] for name in built_wheel.namelist()}
    dist_info = {name for name in top if name.endswith(".dist-info")}
    assert len(dist_info) == 1
    assert top - dist_info == {"auralite"}, sorted(top)


def test_wheel_contains_every_module(built_wheel):
    names = set(built_wheel.namelist())
    missing = sorted(_package_modules() - names)
    assert not missing, f"not in the wheel: {missing}"
    assert "auralite/py.typed" in names
    assert "auralite/_compat.py" in names       # the alias machinery itself ships


def test_wheel_metadata_entry_points_and_license(built_wheel):
    names = built_wheel.namelist()
    dist_info = next(n.split("/")[0] for n in names if n.split("/")[0].endswith(".dist-info"))
    entry_points = built_wheel.read(f"{dist_info}/entry_points.txt").decode()
    assert "auralite = auralite.cli:main" in entry_points
    assert "[gui_scripts]" in entry_points
    assert "auralite-gui = auralite.gui.app:main" in entry_points
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
    value = "auralite.cli:main"

    def load(self):
        raise ModuleNotFoundError("No module named 'auralite.cli'", name="auralite.cli")


class _FakeDist:
    version = "0.0.0"
    entry_points = [_FakeEntryPoint()]


def test_smoke_script_reports_a_dead_console_script():
    smoke = _load_smoke()
    problems, notes = [], []
    smoke._check_console_scripts(_FakeDist(), problems, notes)
    assert problems and "auralite.cli" in problems[0] and "does not load" in problems[0]


# ----------------------------------------------------------------------
#  v2.6.6: install without compiling — extras and requirements.txt
# ----------------------------------------------------------------------
def _load_print_requirements():
    spec = importlib.util.spec_from_file_location(
        "print_requirements", REPO_ROOT / "scripts" / "print_requirements.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _requirement_names(requirements) -> set[str]:
    return {_canonical(re.split(r"[\s\[<>=!~;]", req.strip(), maxsplit=1)[0])
            for req in requirements}


def test_requirements_txt_is_exactly_the_core_dependencies():
    lines = [line.strip() for line in
             (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    pinned = [line for line in lines if line and not line.startswith("#")]
    assert pinned == PYPROJECT["project"]["dependencies"]


def test_no_default_install_path_compiles_llama_cpp():
    """PyPI has only the sdist of llama-cpp-python: it must stay opt-in (`gguf`)."""
    extras = PYPROJECT["project"]["optional-dependencies"]
    assert _requirement_names(extras["gguf"]) == {"llama-cpp-python"}
    for name, reqs in extras.items():
        if name != "gguf":
            assert "llama-cpp-python" not in _requirement_names(reqs), name


def test_bitsandbytes_is_only_needed_for_qlora():
    extras = PYPROJECT["project"]["optional-dependencies"]
    assert "bitsandbytes" not in _requirement_names(extras["hf"])
    qlora = extras["qlora"]
    assert "auralite-ai[hf]" in qlora
    assert any(req.startswith("bitsandbytes") and "Darwin" in req for req in qlora)


def test_print_requirements_expands_extras_and_self_references(capsys):
    script = _load_print_requirements()
    project = PYPROJECT["project"]
    core = script.requirements(project, [])
    assert core == project["dependencies"]
    qlora = script.requirements(project, ["qlora"])
    assert not [req for req in qlora if req.startswith("auralite-ai")]   # never itself
    assert {"transformers", "peft", "accelerate", "bitsandbytes"} <= _requirement_names(qlora)
    assert len(qlora) == len(set(qlora))
    assert script.main(["serve"]) == 0
    out = capsys.readouterr().out.split()
    assert "fastapi>=0.111" in out and "torch>=2.1" in out
    with pytest.raises(SystemExit, match="unknown extra"):
        script.requirements(project, ["no-such-extra"])


def test_print_requirements_works_where_the_dockerfile_puts_it(tmp_path):
    """The images copy pyproject.toml and the script into one directory."""
    for name in ("pyproject.toml", "scripts/print_requirements.py"):
        shutil.copy2(REPO_ROOT / name, tmp_path / pathlib.Path(name).name)
    out = subprocess.run([sys.executable, "print_requirements.py", "serve"], cwd=tmp_path,
                         capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    assert "fastapi>=0.111" in out.stdout.split()
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.count("print_requirements.py --pyproject pyproject.toml") == 2


# ----------------------------------------------------------------------
#  2.7.0: import strings handed to other tools must name the package
# ----------------------------------------------------------------------
def test_serve_command_runs_the_packaged_app(monkeypatch, tmp_path):
    """`auralite serve` used to start uvicorn with "server.openai_server:app",
    which only resolves in a checkout (via the deprecated alias)."""
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    import auralite.cli as cli

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    monkeypatch.setenv("AURALITE_MODEL", "")
    assert cli.main(["serve", "--model", str(tmp_path / "m.pt"), "--quiet"]) == 0
    (app_ref, kwargs), = calls
    assert app_ref == cli.SERVER_APP == "auralite.server.openai_server:app"
    assert kwargs["workers"] == 1
    module_name, _, attr = app_ref.partition(":")
    assert getattr(importlib.import_module(module_name), attr) is not None


def test_deployment_files_use_the_package_paths():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.count('"auralite.server.openai_server:app"') == 2
    assert "server.openai_server:app\"" not in dockerfile.replace(
        "auralite.server.openai_server:app\"", "")
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "from model_engine import" not in ci
    assert " server.openai_server:app" not in ci
    build = (REPO_ROOT / "build_exe.bat").read_text(encoding="utf-8")
    assert "--collect-submodules auralite" in build
    for old in ("model_engine", "kernels", "--collect-submodules gui", "--collect-submodules server"):
        assert old not in build, old
