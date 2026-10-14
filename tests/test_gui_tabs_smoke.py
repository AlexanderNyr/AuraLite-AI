"""Smoke tests for every GUI tab (roadmap P1.5) and the revived stub modules (P1.9).

The GUI cannot be started in CI (no display), so the whole tkinter surface is
replaced by recording doubles — the same approach `tests/test_gui_export_extras.py`
took for the Export tab, applied here to *all* nine tabs:

* `AIApp.__init__` builds every tab, and this suite fails if a tab stops
  building, loses a widget other tabs depend on, or starts relying on tkinter
  behaviour the doubles do not provide;
* the handlers that touch the engine are called for real (no model / trained
  model / after a chat turn) so a refactor that renames a widget or breaks an
  early-return shows up immediately;
* the v2.6.5 features that live in the app — language switching, session
  restore/save, the async bridge — are checked through their real code paths.

Nothing here asserts pixel output: these are *smoke* tests, they catch crashes
and broken wiring, not visual regressions.
"""
from __future__ import annotations

import os
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
tkinter = pytest.importorskip("tkinter", reason="tkinter not available (headless environment)")
# Import the stdlib dialog modules *before* the widget doubles are installed:
# `tkinter.simpledialog` subclasses `Toplevel` at import time, so patching it
# first would break the import of `tkinter.filedialog` (which `gui.common` needs).
# They are only needed for their import side effect, never accessed directly.
import importlib as _importlib  # noqa: E402

_importlib.import_module("tkinter.filedialog")
_importlib.import_module("tkinter.simpledialog")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._gui_harness import (  # noqa: E402
    _DIALOG_PATCHES,
    _TK_PATCHES,
    DummyTk,
    FakeAxis,
    MockText,
    MockVar,
    make_mock,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY_TEXT = ("AuraLite trains small models on CPU. " * 12)


# ----------------------------------------------------------------------
#  tkinter doubles
# ----------------------------------------------------------------------
@pytest.fixture()
def app(tmp_path, monkeypatch):
    """A fully built AIApp on mocked tkinter, with all state files redirected."""
    import tkinter as _tk
    from tkinter import ttk as _ttk

    monkeypatch.setenv("AURALITE_GUI_CONFIG", str(tmp_path / "gui_config.json"))
    monkeypatch.setenv("AURALITE_GUI_SESSION", str(tmp_path / "gui_session.json"))
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path))

    with ExitStack() as stack:
        for target, value in _TK_PATCHES.items():
            module_name, _, attr = target.rpartition(".")
            module = {"tkinter": _tk, "tkinter.ttk": _ttk}[module_name]
            stack.enter_context(patch.object(module, attr, value))
        stack.enter_context(patch("auralite.gui.common.HAS_MATPLOTLIB", False))
        for target, value in _DIALOG_PATCHES.items():
            module_name, _, attr = target.rpartition(".")
            module = {"messagebox": sys.modules["tkinter.messagebox"],
                      "filedialog": sys.modules["tkinter.filedialog"]}[module_name]
            stack.enter_context(patch.object(module, attr, value))

        from auralite.gui.app import AIApp
        from auralite.gui import session as gui_session

        # `SESSION_PATH`/`DEFAULT_CONFIG_PATH` are computed at import time, so
        # point the modules at the temp dir explicitly.
        stack.enter_context(patch.object(gui_session, "SESSION_PATH", tmp_path / "gui_session.json"))
        instance = AIApp(DummyTk())
        instance._tmp_path = tmp_path
        yield instance


# ----------------------------------------------------------------------
#  P1.5 — every tab builds and survives its handlers
# ----------------------------------------------------------------------
def test_all_nine_tabs_are_built(app):
    assert app._tab_names == ["train", "image", "gen", "chat", "model", "quant",
                              "eval", "export", "console"]
    for name in app._tab_names:
        assert hasattr(app, f"tab_{name}"), f"tab_{name} missing"
    assert app.notebook.add.call_count == 9
    assert app.notebook.pack.called


def test_tab_titles_come_from_the_catalogue(app):
    from auralite.gui.i18n import tr

    titles = [call.kwargs.get("text") for call in app.notebook.add.call_args_list]
    assert titles == [tr(f"tab.{name}") for name in app._tab_names]


def test_key_widgets_exist_for_each_tab(app):
    """Names other tabs (and the tests below) reach for through `self`."""
    for attr in ("status_label", "console_text", "result_text", "chat_text",
                 "chat_input", "chat_send_btn", "loss_text", "model_file_label", "model_info",
                 "chat_plot_ax", "chat_insights_btn", "chat_attention_btn",
                 "progress_var", "progress_detail_var", "notebook", "device_label", "loss_text",
                 "train_btn",
                 "param_label", "language_var", "bridge"):
        assert hasattr(app, attr), f"AIApp lost `{attr}`"


def test_console_redirect_is_installed_and_restored(tmp_path, monkeypatch):
    """`AIApp.__init__` must wrap stdout/stderr; `_on_close` must restore them.

    Built outside the shared fixture: pytest re-installs its own capture streams
    right after fixture setup, which would hide what the app did.
    """
    from contextlib import ExitStack

    import tkinter as _tk
    from tkinter import ttk as _ttk

    from auralite.gui.common import ConsoleRedirector

    monkeypatch.setenv("AURALITE_GUI_CONFIG", str(tmp_path / "cfg.json"))
    monkeypatch.setenv("AURALITE_GUI_SESSION", str(tmp_path / "sess.json"))
    with ExitStack() as stack:
        for target, value in _TK_PATCHES.items():
            module_name, _, attr = target.rpartition(".")
            module = {"tkinter": _tk, "tkinter.ttk": _ttk}[module_name]
            stack.enter_context(patch.object(module, attr, value))
        stack.enter_context(patch("auralite.gui.common.HAS_MATPLOTLIB", False))
        from auralite.gui.app import AIApp

        original_out, original_err = sys.stdout, sys.stderr
        instance = AIApp(DummyTk())
        assert isinstance(sys.stdout, ConsoleRedirector)
        assert isinstance(sys.stderr, ConsoleRedirector)
        instance._on_close()
        assert sys.stdout is original_out and sys.stderr is original_err


@pytest.mark.parametrize("handler", [
    "_refresh_model_info",
    "_clear_chat_history",
    "_plot_token_distribution",
    "_plot_attention_map",
    "_toggle_dark_mode",
])
def test_handlers_without_a_model_do_not_raise(app, handler):
    """Every no-model guard must return early instead of crashing a tab."""
    getattr(app, handler)()


def test_result_view_roundtrip(app):
    app.result_text.content = ""
    app._display_result("hello world")
    assert app._get_result_text() == "hello world"
    app._copy_result()
    assert "hello world" in app.root.clipboard
    app._clear_result()
    assert app._get_result_text() == ""


def test_chat_insights_plot_the_last_turn(app):
    """Token bars + attention heatmap on real engine data, fake axis."""
    app.engine.train(TINY_TEXT, {
        "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_kv_heads": 2,
        "n_layers": 2, "seq_length": 16, "batch_size": 8, "lr": 1e-2, "epochs": 1,
        "val_split": 0.0, "seed": 0, "amp_dtype": "none",
    })
    app.chat_history = [
        {"role": "user", "content": "What does AuraLite do?"},
        {"role": "assistant", "content": "It trains small models on CPU."},
    ]
    axis = FakeAxis()
    app.chat_plot_ax = axis
    app.chat_plot_canvas = None

    app._plot_token_distribution()
    assert "bar" in axis.kinds

    app.chat_attention_layer_var.set("1")
    axis.kinds.clear()
    app._plot_attention_map()
    assert "imshow" in axis.kinds


def test_chat_insights_hint_when_matplotlib_is_missing(app):
    app.engine.model = object()          # a "model" is loaded …
    app.chat_plot_ax = None              # … but there is no canvas (no matplotlib)
    app.chat_history = [{"role": "assistant", "content": "hi"}]
    app._plot_token_distribution()       # must inform instead of raising
    app._plot_attention_map()


def test_language_switch_relabels_tabs(app):
    from auralite.gui.i18n import tr

    app.notebook.tab.reset_mock()
    app._on_language_change("English")
    assert app.language == "en"
    assert app.gui_config.language == "en"
    assert tr("tab.train") == " Train "
    assert app.notebook.tab.call_count == len(app._tab_names)
    for call in app.notebook.tab.call_args_list:
        assert call.kwargs["text"].startswith(" ")


def test_language_switch_unknown_name_falls_back_to_english(app):
    app._on_language_change("Klingon")
    assert app.language == "en"
    assert app.gui_config.language == "en"


# ----------------------------------------------------------------------
#  P1.9 — session persistence + async bridge inside the app
# ----------------------------------------------------------------------
def test_session_is_saved_and_restored(app, tmp_path):
    import tkinter as _tk
    from tkinter import ttk as _ttk

    from auralite.gui.app import AIApp
    from auralite.gui import session as gui_session

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TINY_TEXT, encoding="utf-8")
    app.selected_file_path = str(corpus)
    app._save_session(quiet=True)

    saved = gui_session.load_session(tmp_path / "gui_session.json")
    assert saved["selected_file_path"] == str(corpus)
    assert saved["version"] == gui_session.SESSION_VERSION
    assert saved["saved_at"]                      # timestamp present

    with ExitStack() as stack:
        for target, value in _TK_PATCHES.items():
            module_name, _, attr = target.rpartition(".")
            module = {"tkinter": _tk, "tkinter.ttk": _ttk}[module_name]
            stack.enter_context(patch.object(module, attr, value))
        stack.enter_context(patch("auralite.gui.common.HAS_MATPLOTLIB", False))
        stack.enter_context(patch.object(gui_session, "SESSION_PATH",
                                         tmp_path / "gui_session.json"))
        second = AIApp(DummyTk())
    assert second.selected_file_path == str(corpus)


def test_restore_is_skipped_when_disabled(app, tmp_path):
    import tkinter as _tk
    from tkinter import ttk as _ttk

    from auralite.gui.app import AIApp
    from auralite.gui import session as gui_session

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("x", encoding="utf-8")
    gui_session.save_session({"selected_file_path": str(corpus)},
                             tmp_path / "gui_session.json")

    with ExitStack() as stack:
        for target, value in _TK_PATCHES.items():
            module_name, _, attr = target.rpartition(".")
            module = {"tkinter": _tk, "tkinter.ttk": _ttk}[module_name]
            stack.enter_context(patch.object(module, attr, value))
        stack.enter_context(patch("auralite.gui.common.HAS_MATPLOTLIB", False))
        stack.enter_context(patch.object(gui_session, "SESSION_PATH",
                                         tmp_path / "gui_session.json"))
        from auralite.gui.config import GuiConfig

        stack.enter_context(patch("auralite.gui.app.load_config",
                                  lambda *a, **k: GuiConfig(restore_session=False)))
        instance = AIApp(DummyTk())
    assert instance.selected_file_path is None


def test_restore_ignores_missing_files(app):
    import auralite.gui.session as gui_session

    assert gui_session.restore_to_app(app, {"selected_file_path": "/nope/missing.txt"}) == []


def test_on_close_saves_everything_and_stops_background_work(app, tmp_path):
    app._save_session = MagicMock(wraps=app._save_session)
    app._on_close()
    assert app._save_session.called
    assert (tmp_path / "gui_config.json").exists()
    assert app.root.destroyed is True
    assert app.bridge._closed is True
    assert app.stop_event.is_set()


def test_async_bridge_runs_blocking_work_on_the_tk_thread(app):
    results = []
    app.bridge.call(lambda: 6 * 7, callback=results.append)
    for _ in range(50):                  # the executor needs a moment
        app.bridge.poll()
        if results:
            break
        __import__("time").sleep(0.02)
    assert results == [42]


def test_async_bridge_reports_errors_instead_of_raising(app):
    seen = []
    app.bridge.call(lambda: 1 / 0, callback=lambda result, error: seen.append(error))
    for _ in range(50):
        app.bridge.poll()
        if seen:
            break
        __import__("time").sleep(0.02)
    assert isinstance(seen[0], ZeroDivisionError)
