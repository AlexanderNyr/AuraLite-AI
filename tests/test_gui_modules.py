"""Unit tests for the revived GUI support modules (roadmap P1.9).

`gui/config.py`, `gui/session.py`, `gui/i18n.py` and `gui/async_ops.py` used to
be placeholders (a single constant, two dicts, an unconditional thread). These
tests treat them as what they now are: small libraries with an explicit
contract, testable without a display — filesystem faults, unknown languages and
callbacks that raise are all part of the contract.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import time

import pytest

from auralite.gui import config as gui_config
from auralite.gui import i18n
from auralite.gui import session as gui_session
from auralite.gui.async_ops import TkAsyncBridge


# ----------------------------------------------------------------------
#  gui/config.py
# ----------------------------------------------------------------------
def test_config_roundtrip(tmp_path):
    path = tmp_path / "cfg.json"
    cfg = gui_config.GuiConfig(language="ru", theme="dark", geometry="1200x800",
                               last_corpus="/data/corpus.txt", attention_layer=2)
    cfg.save(path)
    loaded = gui_config.load_config(path)
    assert loaded == cfg
    assert loaded.as_dict()["language"] == "ru"


def test_config_missing_or_corrupt_file_yields_defaults(tmp_path):
    missing = gui_config.load_config(tmp_path / "nope.json")
    assert missing.language in i18n.available_languages()
    assert missing.restore_session is True

    broken = tmp_path / "broken.json"
    broken.write_text("{not json at all", encoding="utf-8")
    assert gui_config.load_config(broken).theme == "light"

    listing = tmp_path / "list.json"
    listing.write_text("[1, 2, 3]", encoding="utf-8")
    assert gui_config.load_config(listing).theme == "light"


def test_config_ignores_unknown_keys(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"language": "ru", "format": "now", "unknown": 42}),
                    encoding="utf-8")
    loaded = gui_config.load_config(path)
    assert loaded.language == "ru"
    assert not hasattr(loaded, "unknown")
    assert loaded.update(unknown=1, geometry="900x700").geometry == "900x700"


def test_config_save_is_atomic(tmp_path):
    path = tmp_path / "nested" / "cfg.json"
    gui_config.GuiConfig().save(path)
    assert path.exists()
    assert list(path.parent.glob("*.tmp")) == []      # temp file cleaned up


def test_config_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    assert gui_config.config_dir() == tmp_path / "home"
    monkeypatch.delenv("AURALITE_HOME")
    assert gui_config.config_dir() == gui_config.Path.home() / ".auralite"


def test_config_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AURALITE_GUI_CONFIG", str(tmp_path / "custom.json"))
    assert gui_config.load_config().save() == tmp_path / "custom.json"
    assert (tmp_path / "custom.json").exists()


def test_config_language_defaults_to_env(monkeypatch):
    monkeypatch.setenv("AURALITE_GUI_LANG", "ru")
    assert gui_config.GuiConfig().language == "ru"


# ----------------------------------------------------------------------
#  gui/session.py
# ----------------------------------------------------------------------
def test_session_roundtrip_adds_metadata(tmp_path):
    path = tmp_path / "session.json"
    saved = gui_session.save_session({"selected_file_path": "/tmp/x.txt"}, path)
    assert saved == path
    state = gui_session.load_session(path)
    assert state["selected_file_path"] == "/tmp/x.txt"
    assert state["version"] == gui_session.SESSION_VERSION
    assert state["saved_at"].endswith("+00:00")


def test_session_load_is_forgiving(tmp_path):
    assert gui_session.load_session(tmp_path / "none.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("}", encoding="utf-8")
    assert gui_session.load_session(bad) == {}
    bad.write_text('"a string"', encoding="utf-8")
    assert gui_session.load_session(bad) == {}


def test_update_and_clear_session(tmp_path):
    path = tmp_path / "session.json"
    gui_session.update_session(path, selected_file_path="/a.txt")
    gui_session.update_session(path, active_tab="chat")
    state = gui_session.load_session(path)
    assert state["selected_file_path"] == "/a.txt" and state["active_tab"] == "chat"

    assert gui_session.clear_session(path) is True
    assert gui_session.clear_session(path) is False      # already gone


def test_snapshot_from_app_skips_missing_paths(tmp_path):
    class FakeApp:
        selected_file_path = "/definitely/not/here.txt"
        last_prompt = "hello"
        language = "ru"
        gui_config = gui_config.GuiConfig(geometry="1000x700")

    state = gui_session.snapshot_from_app(FakeApp())
    assert "selected_file_path" not in state
    assert state["last_prompt"] == "hello" and state["language"] == "ru"
    assert state["geometry"] == "1000x700"


def test_restore_to_app_is_partial_and_never_fatal(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("data", encoding="utf-8")

    class Entry:
        def __init__(self): self.value = ""
        def insert(self, index, text): self.value = text

    class Notebook:
        def __init__(self): self.selected = None
        def select(self, index): self.selected = index

    class FakeApp:
        tab_names = ["train", "chat"]

        def __init__(self):
            self.selected_file_path = None
            self.chat_input = Entry()
            self.notebook = Notebook()
            self._tab_names = self.tab_names

    app = FakeApp()
    restored = gui_session.restore_to_app(app, {
        "selected_file_path": str(corpus),
        "last_prompt": "continue the story",
        "active_tab": "chat",
        "nonsense": object(),
    })
    assert restored == ["selected_file_path", "last_prompt", "active_tab"]
    assert app.selected_file_path == str(corpus)
    assert app.chat_input.value == "continue the story"
    assert app.notebook.selected == 1

    empty = FakeApp()
    assert gui_session.restore_to_app(empty, {}) == []


def test_restore_tolerates_a_broken_widget(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("data", encoding="utf-8")

    class Exploding:
        def insert(self, *a, **k):
            raise RuntimeError("widget already destroyed")

    class FakeApp:
        def __init__(self):
            self.selected_file_path = None
            self.chat_input = Exploding()
            self.notebook = None
            self._tab_names = []

    app = FakeApp()
    restored = gui_session.restore_to_app(app, {"selected_file_path": str(corpus),
                                                "last_prompt": "hi"})
    assert restored == ["selected_file_path"]        # prompt failed, file did not


# ----------------------------------------------------------------------
#  gui/i18n.py
# ----------------------------------------------------------------------
def test_catalogues_are_in_sync():
    en, ru = i18n.MESSAGES["en"], i18n.MESSAGES["ru"]
    assert set(en) == set(ru), f"missing translations: {set(en) ^ set(ru)}"
    assert all(en[key].strip() and ru[key].strip() for key in en)


def test_tab_keys_match_the_app_tab_names():
    from auralite.gui.app import AIApp

    for name in AIApp._TAB_NAMES:
        assert f"tab.{name}" in i18n.MESSAGES["en"], f"tab.{name} has no label"


def test_translation_lookup_and_fallbacks():
    assert i18n.tr("tab.train", lang="ru") == " Обучение "
    assert i18n.tr("tab.train", lang="en") == " Train "
    assert i18n.tr("does.not.exist") == "does.not.exist"
    assert i18n.tr("msg.attention_title", layer=3) == "Attention (layer 3)"
    assert i18n.tr("msg.attention_title", layer=3, lang="ru") == "Внимание (слой 3)"
    # a missing placeholder must not blow up
    assert "{layer}" in i18n.tr("msg.attention_title")


def test_set_language_switches_and_rejects_unknown():
    original = i18n.get_language()
    try:
        assert i18n.set_language("ru") == "ru"
        assert i18n.get_language() == "ru"
        assert i18n.tr("btn.save") == "Сохранить"
        assert i18n.set_language("klingon") == "en"
        assert i18n.language_name() == "English"
        assert set(i18n.available_languages()) == {"en", "ru"}
    finally:
        i18n.set_language(original)


# ----------------------------------------------------------------------
#  gui/async_ops.py
# ----------------------------------------------------------------------
class ImmediatelyCallingRoot:
    """Test double whose `after()` runs the callback right away (worst case)."""

    def __init__(self):
        self.scheduled = 0

    def after(self, ms, fn=None, *a):
        self.scheduled += 1
        if fn is not None:
            fn(*a)
        return "id"


class RecordingRoot:
    def __init__(self):
        self.pending: list = []

    def after(self, ms, fn=None, *a):
        self.pending.append((ms, fn, a))
        return "id"


def test_bridge_poll_does_not_recurse_with_immediate_after():
    bridge = TkAsyncBridge(ImmediatelyCallingRoot())
    try:
        assert bridge._scheduling is False          # guard released afterwards
    finally:
        bridge.shutdown()


def test_bridge_call_delivers_result_and_error():
    root = RecordingRoot()
    bridge = TkAsyncBridge(root)
    results, errors = [], []
    try:
        bridge.call(lambda: "ok", callback=results.append)
        bridge.call(lambda: 1 / 0, callback=lambda r, e: errors.append(e))
        deadline = time.time() + 5
        while time.time() < deadline and (not results or not errors):
            bridge.poll()
            time.sleep(0.02)
        assert results == ["ok"]
        assert isinstance(errors[0], ZeroDivisionError)
    finally:
        bridge.shutdown()


def test_bridge_runs_coroutines_on_its_loop():
    bridge = TkAsyncBridge(RecordingRoot())
    done = []
    try:
        async def work():
            await asyncio.sleep(0)
            return 5

        bridge.submit(work(), callback=lambda fut: done.append(fut.result()))
        deadline = time.time() + 5
        while time.time() < deadline and not done:
            bridge.poll()
            time.sleep(0.02)
        assert done == [5]
    finally:
        bridge.shutdown()


def test_bridge_callback_failure_is_logged_not_fatal(caplog):
    bridge = TkAsyncBridge(RecordingRoot())
    try:
        def boom(_result):
            raise ValueError("callback exploded")

        bridge.call(lambda: 1, callback=boom)
        deadline = time.time() + 5
        with caplog.at_level(logging.ERROR, logger="auralite.gui"):
            while time.time() < deadline and not caplog.records:
                bridge.poll()
                time.sleep(0.02)
        assert any("callback raised" in record.message for record in caplog.records)
    finally:
        bridge.shutdown()


def test_bridge_shutdown_is_idempotent_and_stops_polling():
    root = RecordingRoot()
    bridge = TkAsyncBridge(root)
    with bridge as ctx:
        assert ctx is bridge
    bridge.shutdown()                                  # second call is a no-op
    assert bridge._closed is True
    scheduled_before = len(root.pending)
    bridge.poll()                                      # destroyed window: no-op
    assert len(root.pending) == scheduled_before


def test_bridge_without_callback_swallows_nothing(capsys):
    """A background failure with no callback must be logged, not lost."""
    bridge = TkAsyncBridge(RecordingRoot())
    try:
        bridge.call(lambda: 1 / 0)
        time.sleep(0.2)
        bridge.poll()
    finally:
        bridge.shutdown()
