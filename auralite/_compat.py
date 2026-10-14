"""Deprecated import names from AuraLite <= 2.6 (the flat layout).

Until 2.6 every module lived at the top level of the repository and was
installed as a top-level name: ``model_engine``, ``server``, ``export``,
``kernels``, … Generic names like these collide with other distributions once
the project is on PyPI (Hugging Face ships a ``kernels`` package), so 2.7.0
moved everything into the single ``auralite`` package.

A source checkout keeps the old names working for the 2.7.x series through
small shim modules in the repository root (``model_engine.py``, ``server.py``,
…). Each shim calls :func:`alias_module`, which

* emits one :class:`DeprecationWarning` that names the replacement;
* puts the *real* module into ``sys.modules`` under the old name, so
  ``model_engine is auralite.engine`` — not a copy;
* installs :class:`_AliasFinder`, which resolves submodules the same way:
  ``import model_engine.layers`` returns ``auralite.engine.layers`` itself.
  Without it Python would load ``auralite/engine/layers.py`` a second time
  under the old name, and classes, ``isinstance`` checks and
  ``mock.patch("model_engine._legacy.…")`` would silently stop matching.

The shims are deliberately *not* part of the wheel: an installed AuraLite
(PyPI, a release wheel, the Docker image) only provides ``auralite``.
``kernels`` has no alias at all — that name belongs to Hugging Face.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import threading
import warnings
from types import ModuleType

__all__ = ["RENAMED", "REMOVED_IN", "alias_module", "new_name"]

#: Old top-level import name → module in the ``auralite`` package (2.7.0).
RENAMED: dict[str, str] = {
    "model_engine": "auralite.engine",
    "auralite_cli": "auralite.cli",
    "gui_app": "auralite.gui.app",
    "gui": "auralite.gui",
    "server": "auralite.server",
    "agent": "auralite.agent",
    "quantization": "auralite.quantization",
    "quantization_backends": "auralite.quantization_backends",
    "export": "auralite.export",
    "eval": "auralite.eval",
    "evaluation": "auralite.evaluation",
    "image_engine": "auralite.image_engine",
    "hf_integration": "auralite.hf_integration",
    "chat_interface": "auralite.chat_interface",
    "web_tools": "auralite.web_tools",
}

#: The release that deletes the shims.
REMOVED_IN = "3.0"

_finder_lock = threading.Lock()


def new_name(old: str) -> str:
    """Translate an old dotted name (``model_engine.layers``) to the new one."""
    head, dot, rest = old.partition(".")
    if head not in RENAMED:
        raise KeyError(old)
    return RENAMED[head] + dot + rest


class _AliasLoader(importlib.abc.Loader):
    """Hands out the already-imported real module instead of executing code."""

    def __init__(self, target: str) -> None:
        self.target = target
        self._real_spec = None

    def create_module(self, spec):  # noqa: D102
        module = importlib.import_module(self.target)
        self._real_spec = module.__spec__
        return module

    def exec_module(self, module) -> None:  # noqa: D102
        # `module_from_spec` stamps the alias spec onto the module; put the
        # real one back so reload(), pickle and `python -m` keep working.
        if self._real_spec is not None:
            module.__spec__ = self._real_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolves ``<old>.<sub>`` to the module object of ``<new>.<sub>``."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: D102
        head, dot, _ = fullname.partition(".")
        if not dot or head not in RENAMED:
            return None
        if sys.modules.get(head) is not sys.modules.get(RENAMED[head]):
            return None   # the old name is not an alias (e.g. a user's own `gui` package)
        real = new_name(fullname)
        try:
            real_spec = importlib.util.find_spec(real)
        except (ImportError, ValueError):
            return None
        if real_spec is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname, _AliasLoader(real),
            is_package=real_spec.submodule_search_locations is not None)


def _install_finder() -> None:
    with _finder_lock:
        if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
            # First, so the regular path finder never loads a second copy.
            sys.meta_path.insert(0, _AliasFinder())


def alias_module(old: str, *, warn: bool = True) -> ModuleType:
    """Make ``old`` an alias of its new module and return that module.

    Called by the repository-root shims (``model_engine.py`` & co.) with their
    own ``__name__``. ``warn=False`` serves script launchers such as
    ``python gui_app.py``, which stay supported in a checkout.
    """
    target = RENAMED[old]
    if warn:
        # stacklevel 3 → the line that imported the shim (importlib's own
        # frames are skipped by the warnings machinery).
        warnings.warn(
            f"`{old}` is a deprecated alias of `{target}` and will be removed in "
            f"AuraLite {REMOVED_IN}; import `{target}` instead.",
            DeprecationWarning, stacklevel=3)
    module = importlib.import_module(target)
    _install_finder()
    sys.modules[old] = module
    return module
