#!/usr/bin/env python3
"""Smoke-test an *installed* AuraLite distribution (CI ``package`` job / release gate).

Run it with the interpreter of a clean virtualenv into which the wheel (or the
sdist) was installed — never from an editable install or the source tree:

    python -m venv /tmp/wheel-venv
    /tmp/wheel-venv/bin/python -m pip install "$(ls dist/*.whl)[serve]" \\
        --extra-index-url https://download.pytorch.org/whl/cpu
    cd /tmp && /tmp/wheel-venv/bin/python "$GITHUB_WORKSPACE/scripts/smoke_installed_wheel.py"

It checks what ``twine check`` and an editable install cannot see:

* the distribution installs exactly one top-level name, ``auralite`` (2.7.0) —
  none of the deprecated flat names (``model_engine``, ``server``, …) that only
  exist as aliases in a source checkout;
* every module of the package imports, from the installed files (not from a
  checkout that happens to be on ``sys.path``); a missing *optional* dependency
  is reported as a note, anything else is a failure;
* every console script resolves, and ``auralite --version`` / ``--help`` exit 0;
  the ``auralite-gui`` entry point resolves (it is not started);
* the versions reported by the metadata, the package, the CLI and the API server
  agree.

Why it exists: until v2.6.5 ``pyproject.toml`` forgot ``auralite_cli`` and
``image_engine``. The wheel built fine, ``twine check`` passed, CI was green
(editable installs see the whole checkout) — and ``pip install auralite-ai``
produced an ``auralite`` command that died with ``ModuleNotFoundError`` and a GUI
without its Image tab.
"""
from __future__ import annotations

import importlib
import json
import os
import pathlib
import pkgutil
import shutil
import subprocess
import sys
from importlib import metadata

DIST_NAME = "auralite-ai"
SOURCE_TREE = pathlib.Path(__file__).resolve().parent.parent

#: The one top-level name of the distribution (2.7.0). tests/test_packaging.py
#: enforces the same statically; this is the check on the built artifact itself.
EXPECTED_TOP_LEVEL = {"auralite"}

#: Modules a usable install cannot lack: the CLI, the engines behind the GUI tabs,
#: the server.
REQUIRED_MODULES = {"auralite.cli", "auralite.engine", "auralite.image_engine",
                    "auralite.gui.app", "auralite.server.openai_server",
                    "auralite.agent", "auralite.quantization", "auralite.export"}

#: Pre-2.7 top-level names. They must not be importable from an installed
#: AuraLite: on PyPI they would collide with other distributions.
DEPRECATED_ALIASES = {"model_engine", "auralite_cli", "gui_app", "gui", "server", "agent",
                      "kernels", "quantization", "quantization_backends", "export", "eval",
                      "evaluation", "image_engine", "hf_integration", "chat_interface",
                      "web_tools"}

#: Imports that may legitimately be missing: system modules of a headless/slim
#: interpreter and optional dependencies (extras). Anything else that fails to
#: import is a packaging bug.
OPTIONAL_SYSTEM_MODULES = {"tkinter", "_tkinter"}
OPTIONAL_DEPENDENCIES = {"fastapi", "uvicorn", "starlette", "pydantic", "httpx", "transformers",
                         "peft", "accelerate", "bitsandbytes", "llama_cpp", "onnx", "onnxruntime",
                         "onnxscript", "safetensors", "lm_eval", "sentence_transformers", "faiss",
                         "torchao", "tiktoken", "sentencepiece", "vllm"}


def _inside(path: str | os.PathLike | None, root: pathlib.Path) -> bool:
    if not path:
        return False
    try:
        pathlib.Path(path).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _in_checkout(path: str | os.PathLike | None) -> bool:
    """True for paths inside the source checkout but outside this interpreter's prefix.

    (A virtualenv created *inside* the checkout is still an installed environment.)
    """
    prefix = pathlib.Path(sys.prefix).resolve()
    return _inside(path, SOURCE_TREE) and not _inside(path, prefix)


def _isolate_from_source_tree() -> None:
    """Drop checkout paths from sys.path so a stray checkout cannot mask a missing file."""
    sys.path[:] = [entry for entry in sys.path if not _in_checkout(entry or os.getcwd())]


def _top_level_names(dist: metadata.Distribution) -> list[str]:
    names: set[str] = set()
    top_level = dist.read_text("top_level.txt")
    if top_level:
        names.update(line.strip() for line in top_level.splitlines() if line.strip())
    for file in dist.files or ():
        parts = file.parts
        if not parts or parts[0].endswith((".dist-info", ".data")) or parts[0] == "..":
            continue
        if len(parts) == 1 and parts[0].endswith(".py"):
            names.add(parts[0][:-3])
        elif len(parts) > 1 and parts[-1] == "__init__.py" and len(parts) == 2:
            names.add(parts[0])
    return sorted(names)


def _package_modules(problems: list[str]) -> list[str]:
    """Every module of the installed ``auralite`` package (``__main__`` excluded)."""
    try:
        package = importlib.import_module("auralite")
    except ImportError as exc:
        problems.append(f"import auralite failed: {exc!r}")
        return []
    names = ["auralite"]
    for info in pkgutil.walk_packages(package.__path__, "auralite.",
                                      onerror=lambda name: problems.append(
                                          f"cannot scan {name}: {sys.exc_info()[1]!r}")):
        if info.name.rsplit(".", 1)[-1] != "__main__":
            names.append(info.name)
    return sorted(names)


def _check_imports(dist: metadata.Distribution, problems: list[str], notes: list[str]) -> None:
    location = pathlib.Path(dist.locate_file("")).resolve()
    top_level = set(_top_level_names(dist))       # from the distribution's own RECORD
    for alias in sorted(top_level & DEPRECATED_ALIASES):
        problems.append(f"the deprecated alias `{alias}` is shipped: it belongs to the "
                        "source checkout only")
    if top_level - DEPRECATED_ALIASES != EXPECTED_TOP_LEVEL:
        problems.append(f"top-level names {sorted(top_level)}, expected only "
                        f"{sorted(EXPECTED_TOP_LEVEL)}")
    names = _package_modules(problems)
    for missing in sorted(REQUIRED_MODULES - set(names)):
        problems.append(f"{missing} is not shipped in the distribution")
    skipped = 0
    for name in names:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:  # includes ModuleNotFoundError
            missing = (getattr(exc, "name", None) or "").split(".")[0]
            if missing in OPTIONAL_SYSTEM_MODULES | OPTIONAL_DEPENDENCIES:
                notes.append(f"skip {name}: needs {missing} (optional, not installed)")
                skipped += 1
                continue
            problems.append(f"import {name} failed: {exc!r}")
            continue
        origin = getattr(module, "__file__", None)
        if _in_checkout(origin):
            problems.append(f"{name} was imported from the source tree ({origin}), "
                            "not from the installed distribution")
        elif origin and not _inside(origin, location):
            problems.append(f"{name} resolved outside the distribution: {origin}")
    notes.append(f"ok   {len(names) - skipped} of {len(names)} modules import "
                 f"({skipped} need optional extras)")


def _check_console_scripts(dist: metadata.Distribution, problems: list[str],
                           notes: list[str]) -> None:
    scripts = [ep for ep in dist.entry_points if ep.group == "console_scripts"]
    if not scripts:
        problems.append("no console_scripts entry points (expected `auralite`)")
    bin_dir = pathlib.Path(sys.executable).parent
    for ep in scripts:
        try:
            ep.load()
        except Exception as exc:  # the v2.6.4 wheel failed exactly here
            problems.append(f"console script {ep.name} = {ep.value} does not load: {exc!r}")
            continue
        exe = shutil.which(ep.name, path=str(bin_dir))
        if exe is None:
            problems.append(f"console script {ep.name} is not installed in {bin_dir}")
            continue
        for args in (["--version"], ["--help"]):
            run = subprocess.run([exe, *args], capture_output=True, text=True,
                                 timeout=300, cwd=pathlib.Path.home())
            if run.returncode != 0:
                problems.append(f"`{ep.name} {' '.join(args)}` exited {run.returncode}: "
                                f"{(run.stderr or run.stdout).strip()[-400:]}")
            elif args == ["--version"] and dist.version not in run.stdout:
                problems.append(f"`{ep.name} --version` printed {run.stdout.strip()!r}, "
                                f"expected {dist.version}")
            else:
                notes.append(f"ok   {ep.name} {' '.join(args)}")


def _check_gui_scripts(dist: metadata.Distribution, problems: list[str],
                       notes: list[str]) -> None:
    """``auralite-gui`` must resolve; it is never started (it would open a window)."""
    gui_scripts = [ep for ep in dist.entry_points if ep.group == "gui_scripts"]
    if not gui_scripts:
        problems.append("no gui_scripts entry points (expected `auralite-gui`)")
    for ep in gui_scripts:
        try:
            ep.load()
        except ImportError as exc:
            missing = (getattr(exc, "name", None) or "").split(".")[0]
            if missing in OPTIONAL_SYSTEM_MODULES:
                notes.append(f"skip gui script {ep.name}: needs {missing}")
                continue
            problems.append(f"gui script {ep.name} = {ep.value} does not load: {exc!r}")
        except Exception as exc:  # noqa: BLE001 - any failure is a broken entry point
            problems.append(f"gui script {ep.name} = {ep.value} does not load: {exc!r}")
        else:
            notes.append(f"ok   gui script {ep.name} resolves")


def _check_versions(dist: metadata.Distribution, problems: list[str], notes: list[str]) -> None:
    import auralite
    import auralite.engine

    for name, value in (("auralite", auralite.__version__),
                        ("auralite.engine", auralite.engine.__version__)):
        if value != dist.version:
            problems.append(f"{name}.__version__={value} but the metadata says {dist.version}")
    try:
        import fastapi  # noqa: F401  (only installed with the `serve` extra)
    except ImportError:
        notes.append("skip server version: install the wheel with [serve] to check it")
        return
    from auralite.server import openai_server

    if openai_server._AURALITE_VERSION != dist.version:
        problems.append(f"server reports {openai_server._AURALITE_VERSION}, "
                        f"metadata says {dist.version}")
    else:
        notes.append(f"ok   versions agree ({dist.version})")


def main() -> int:
    _isolate_from_source_tree()
    try:
        dist = metadata.distribution(DIST_NAME)
    except metadata.PackageNotFoundError:
        print(f"FAIL: {DIST_NAME} is not installed in {sys.prefix}", file=sys.stderr)
        return 2
    try:
        direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    except ValueError:
        direct_url = {}
    if direct_url.get("dir_info", {}).get("editable"):
        print("FAIL: this is an editable install — the smoke test needs a built wheel/sdist",
              file=sys.stderr)
        return 2

    problems: list[str] = []
    notes: list[str] = []
    _check_imports(dist, problems, notes)
    _check_console_scripts(dist, problems, notes)
    _check_gui_scripts(dist, problems, notes)
    _check_versions(dist, problems, notes)

    print(f"{DIST_NAME} {dist.version} at {dist.locate_file('')}")
    for note in notes:
        print(f"  {note}")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print("OK: the installed distribution is complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
