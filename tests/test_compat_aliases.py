"""The pre-2.7 flat import names keep working in a checkout — as aliases, not copies.

2.7.0 moved every module into the ``auralite`` package. The repository-root
shims (``model_engine.py``, ``server.py``, …) and ``auralite/_compat.py`` keep old
scripts, notebooks and ``mock.patch("model_engine._legacy.…")`` targets working
until 2.8. What matters is *identity*: a second copy of a module would give two
``AuraLiteEngine`` classes, broken ``isinstance`` checks and patches that silently
miss. Each check runs in a fresh interpreter, because an alias is process-wide
state (and a warning is only raised by the first import).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from auralite import _compat

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _run(code: str, cwd: pathlib.Path = REPO_ROOT, extra_path: str | None = None,
         args: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    paths = [p for p in (extra_path, str(REPO_ROOT), env.get("PYTHONPATH")) if p]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.pop("PYTHONWARNINGS", None)
    cmd = [sys.executable, *(["-c", textwrap.dedent(code)] if code else []), *args]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=300)


def _json(run: subprocess.CompletedProcess):
    assert run.returncode == 0, run.stderr[-3000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_every_alias_is_the_new_module_and_warns_once():
    result = _json(_run("""
        import importlib, json, warnings
        from auralite import _compat

        out = {}
        for old, new in _compat.RENAMED.items():
            if old == "gui_app":          # a launcher with its own namespace, see below
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                module = importlib.import_module(old)
                again = importlib.import_module(old)
            messages = [str(w.message) for w in caught
                        if issubclass(w.category, DeprecationWarning)]
            out[old] = {"same": module is importlib.import_module(new) and again is module,
                        "warnings": len(messages),
                        "names_new": all(new in m and "2.8" in m for m in messages)}
        print(json.dumps(out))
    """))
    expected = set(_compat.RENAMED) - {"gui_app"}
    assert set(result) == expected
    for old, row in result.items():
        assert row["same"], f"{old} is a copy, not an alias of {_compat.RENAMED[old]}"
        assert row["warnings"] == 1, f"{old}: {row['warnings']} DeprecationWarnings"
        assert row["names_new"], f"{old}: the warning must name the replacement"


def test_the_warning_points_at_the_importing_line(tmp_path):
    script = tmp_path / "old_script.py"
    script.write_text("import model_engine  # line 1\n", encoding="utf-8")
    run = _run("", args=("-W", "always::DeprecationWarning", str(script)))
    assert run.returncode == 0, run.stderr
    assert f"{script}:1: DeprecationWarning" in run.stderr, run.stderr


def test_submodules_resolve_to_the_same_objects():
    result = _json(_run("""
        import json, warnings
        warnings.simplefilter("ignore", DeprecationWarning)
        from unittest import mock

        import model_engine.layers as old_layers
        import auralite.engine.layers as new_layers
        from model_engine import AuraLiteEngine as OldEngine
        from auralite import AuraLiteEngine as NewEngine
        import gui.tabs.model as old_tab
        import auralite.gui.tabs.model as new_tab
        import server.openai_server as old_srv
        import auralite.server.openai_server as new_srv
        import auralite.engine._legacy as legacy

        sentinel = object()
        with mock.patch("model_engine._legacy.logger", sentinel):
            patched = legacy.logger is sentinel
        print(json.dumps({
            "layers": old_layers is new_layers,
            "engine": OldEngine is NewEngine,
            "gui": old_tab is new_tab,
            "server": old_srv is new_srv,
            "spec": new_layers.__spec__.name,
            "patch_hits_the_real_module": patched,
        }))
    """))
    assert result == {"layers": True, "engine": True, "gui": True, "server": True,
                      "spec": "auralite.engine.layers", "patch_hits_the_real_module": True}


def test_kernels_has_no_alias():
    """`kernels` is Hugging Face's package name; ours is only `auralite.kernels`."""
    assert "kernels" not in _compat.RENAMED
    assert not (REPO_ROOT / "kernels.py").exists()
    assert not (REPO_ROOT / "kernels" / "__init__.py").exists()
    run = _run("""
        import importlib.util, json
        spec = importlib.util.find_spec("kernels")
        print(json.dumps(None if spec is None else spec.origin))
    """)
    origin = _json(run)
    assert origin is None or str(REPO_ROOT) not in origin


def test_a_users_own_package_with_an_old_name_is_left_alone(tmp_path):
    """The alias finder only redirects names that *are* aliases right now."""
    pkg = tmp_path / "server"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("MINE = True\n", encoding="utf-8")
    (pkg / "routes.py").write_text("ROUTES = ['/mine']\n", encoding="utf-8")
    result = _json(_run("""
        import json, sys, warnings
        warnings.simplefilter("ignore", DeprecationWarning)
        import model_engine           # installs the alias finder
        import server, server.routes  # the user's package (first on sys.path)
        print(json.dumps([getattr(server, "MINE", False), server.routes.ROUTES]))
    """, cwd=tmp_path, extra_path=str(tmp_path)))
    assert result == [True, ["/mine"]]


def test_new_name_translates_dotted_paths():
    assert _compat.new_name("model_engine") == "auralite.engine"
    assert _compat.new_name("model_engine._legacy.AuraLiteEngine") == \
        "auralite.engine._legacy.AuraLiteEngine"
    assert _compat.new_name("gui.tabs.model") == "auralite.gui.tabs.model"
    with pytest.raises(KeyError):
        _compat.new_name("kernels.sampling")


@pytest.mark.parametrize("launcher, args, expect", [
    ("auralite_cli.py", ("--version",), "auralite "),
    ("eval.py", ("--help",), "--model"),
])
def test_checkout_launchers_still_work_without_warnings(launcher, args, expect):
    run = _run("", args=("-W", "always::DeprecationWarning", launcher, *args))
    assert run.returncode == 0, run.stderr[-2000:]
    assert expect in run.stdout
    assert "DeprecationWarning" not in run.stderr, run.stderr


def test_python_dash_m_auralite():
    run = _run("", args=("-m", "auralite", "--version"))
    assert run.returncode == 0, run.stderr[-2000:]
    import auralite

    assert run.stdout.strip() == f"auralite {auralite.__version__}"


def test_version_and_help_do_not_import_torch():
    """`auralite --version` used to import the whole engine (and torch) first."""
    for args in (("--version",), ("--help",)):
        run = _run("", args=("-X", "importtime", "-m", "auralite", *args))
        assert run.returncode == 0, run.stderr[-2000:]
        imported = {line.rsplit("|", 1)[-1].strip() for line in run.stderr.splitlines()
                    if line.startswith("import time:") and "|" in line}
        assert "auralite.cli" in imported
        assert "torch" not in imported and "auralite.engine" not in imported, args


def test_top_level_package_is_lazy_and_lists_its_api():
    run = _json(_run("""
        import json, sys
        import auralite
        heavy = sorted(m for m in ("torch", "auralite.engine", "auralite.gui")
                       if m in sys.modules)
        names = [n for n in auralite.__all__ if n != "__version__"]
        resolved = all(getattr(auralite, n) is not None for n in names)
        print(json.dumps({"heavy_on_import": heavy, "resolved": resolved,
                          "engine_module": auralite.AuraLiteEngine.__module__}))
    """))
    assert run == {"heavy_on_import": [], "resolved": True,
                   "engine_module": "auralite.engine._legacy"}
    import auralite

    with pytest.raises(AttributeError):
        auralite.does_not_exist  # noqa: B018
    assert "AuraLiteEngine" in dir(auralite)


def test_alias_machinery_in_process(monkeypatch):
    """The same machinery in this process, with a throwaway name (the checks
    above run in subprocesses, which coverage does not see)."""
    import importlib
    import importlib.util

    import auralite.kernels
    import auralite.kernels.sampling as real_sampling

    probe = "auralite_alias_probe"
    monkeypatch.setitem(_compat.RENAMED, probe, "auralite.kernels")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))     # finder removed after
    try:
        with pytest.warns(DeprecationWarning, match="auralite.kernels"):
            module = _compat.alias_module(probe)
        assert module is sys.modules[probe] is auralite.kernels
        sub = importlib.import_module(f"{probe}.sampling")
        assert sub is real_sampling
        assert sub.__spec__.name == "auralite.kernels.sampling"   # spec restored
        assert _compat.new_name(f"{probe}.sampling") == "auralite.kernels.sampling"
        _compat.alias_module(probe, warn=False)                    # idempotent
        assert sum(isinstance(f, _compat._AliasFinder) for f in sys.meta_path) == 1
        assert importlib.util.find_spec(f"{probe}.does_not_exist") is None
    finally:
        for name in [n for n in sys.modules if n == probe or n.startswith(probe + ".")]:
            del sys.modules[name]
