"""2.28: the interfaces in Russian - one catalogue for the Studio and the classic interface.

The catalogue (``auralite/locale/ru.json``) is keyed by the English text. These tests read
every string the code marks (``tr``, ``trn``, ``N_``, ``translate`` and the classic
interface's named messages) and check that the catalogue has each one, and nothing else;
that a translation keeps the placeholders of its English text, and a plural its three
Russian forms; then the language choice (saved choice, ``$AURALITE_LANG``, the computer's
language), and both interfaces built in Russian.
"""
from __future__ import annotations

import ast
import json
import os
import re
import string
import subprocess
import sys
from pathlib import Path

import pytest

import auralite
from auralite import i18n

PACKAGE = Path(auralite.__file__).resolve().parent
CATALOGUE = PACKAGE / "locale" / "ru.json"
MARKERS = {"tr", "N_", "trn", "trc", "translate"}
RUSSIAN = "\u0420\u0443\u0441\u0441\u043a\u0438\u0439"


def _literal(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _marked() -> tuple[set[str], set[str]]:
    """``(keys, plural_keys)``: every string the code marks, as the catalogue keys it."""
    keys: set[str] = set()
    plurals: set[str] = set()
    contexts: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in MARKERS and node.args):
                continue
            if node.func.id == "trn":
                many = _literal(node.args[1]) if len(node.args) > 1 else None
                if many is not None:
                    plurals.add(many)
                continue
            if node.func.id == "trc":             # "context|text"
                context = _literal(node.args[0])
                text = _literal(node.args[1]) if len(node.args) > 1 else None
                if context is not None and text is not None:
                    contexts.add(f"{context}|{text}")
                continue
            index = 1 if node.func.id == "translate" else 0
            text = _literal(node.args[index]) if len(node.args) > index else None
            if text is not None:
                keys.add(text)
    from auralite.gui.i18n import MESSAGES
    keys |= set(MESSAGES["en"].values())
    keys -= set(MESSAGES["en"])                 # a message's name is looked up by its text
    out = set()
    for text in keys:
        core = i18n.split_frame(text)[1] or text
        if core.strip():
            out.add(core)
    from auralite import runs
    out |= {f"data kind|{kind}" for kind in runs.DATA_KINDS}
    return out | contexts | plurals, plurals


def _fields(text: str) -> list[tuple[str, str, str]]:
    try:
        return sorted((name, spec or "", conv or "")
                      for _lit, name, spec, conv in string.Formatter().parse(text)
                      if name is not None)
    except ValueError as exc:                    # a literal brace: {"messages": ...}
        return [("!", str(exc), "")]


@pytest.fixture(scope="module")
def table() -> dict:
    return json.loads(CATALOGUE.read_text(encoding="utf-8"))


@pytest.fixture()
def russian():
    i18n.set_language("ru")
    yield
    i18n.set_language("en")


# ----------------------------------------------------------------------
#  the catalogue
# ----------------------------------------------------------------------
def test_the_catalogue_translates_every_marked_string_and_nothing_else(table):
    keys, _plurals = _marked()
    missing = sorted(keys - set(table))
    unused = sorted(set(table) - keys)
    assert not missing, f"{len(missing)} strings without a translation: {missing[:20]}"
    assert not unused, f"{len(unused)} translations nothing uses: {unused[:20]}"


def test_a_translation_keeps_its_placeholders_and_line_breaks(table):
    wrong = []
    for key, value in table.items():
        for form in value if isinstance(value, list) else [value]:
            if _fields(form) != _fields(key) or form.count("\n") != key.count("\n"):
                wrong.append((key, form))
            elif not form.strip():
                wrong.append((key, form))
    assert not wrong, wrong[:10]


def test_plurals_have_the_three_russian_forms(table):
    _keys, plurals = _marked()
    for key in plurals:
        assert isinstance(table[key], list) and len(table[key]) == 3, key
    lists = {key for key, value in table.items() if isinstance(value, list)}
    assert lists == plurals                     # a plain string is never a list


def _dummy(spec: str):
    kind = spec[-1:] if spec else ""
    if kind in ("e", "E", "f", "F", "g", "G", "%"):
        return 1234.5
    if kind == "s":
        return "x"
    return 1234


def test_every_translation_formats_where_its_english_does(table, russian):
    formatter = i18n._FORMATTER
    for key, value in table.items():
        fields = _fields(key)
        if not fields or not all(name.isidentifier() for name, _spec, _conv in fields):
            continue                            # no values, or literal braces: never formatted
        values = {name: _dummy(spec) for name, spec, _conv in fields}
        key.format(**values)
        for form in value if isinstance(value, list) else [value]:
            formatter.format(form, **values)


def test_the_catalogue_file_is_ascii_and_sorted():
    raw = CATALOGUE.read_bytes()
    assert raw.isascii()                        # the repository's files are ASCII
    data = json.loads(raw)
    assert list(data) == sorted(data)           # stable diffs


def test_every_agent_tool_is_described_in_the_interfaces_language():
    from auralite.agent import TOOL_REGISTRY
    from auralite.studio.agentmode import TOOL_WORDS
    assert set(TOOL_WORDS) == {str(tool.description) for tool in TOOL_REGISTRY.values()}


def test_the_classic_messages_all_have_translations(table):
    from auralite.gui.i18n import MESSAGES
    for name, text in MESSAGES["en"].items():
        assert i18n._lookup("ru", text) is not None, name


# ----------------------------------------------------------------------
#  the shared core
# ----------------------------------------------------------------------
def test_russian_plural_forms():
    cases = {1: 0, 2: 1, 4: 1, 5: 2, 11: 2, 12: 2, 14: 2, 20: 2, 21: 0, 22: 1, 25: 2,
             101: 0, 111: 2, 112: 2, 0: 2, 1.5: 1}
    assert {n: i18n.plural_index("ru", n) for n in cases} == cases
    assert [i18n.plural_index("en", n) for n in (0, 1, 2)] == [1, 0, 1]


def test_trn_picks_the_form_and_tr_keeps_the_english_without_a_language(russian):
    assert i18n.trn("{n:,} dialog", "{n:,} dialogs", 1) == "1 \u0434\u0438\u0430\u043b\u043e\u0433"
    assert i18n.trn("{n:,} dialog", "{n:,} dialogs", 3) == "3 \u0434\u0438\u0430\u043b\u043e\u0433\u0430"
    assert i18n.trn("{n:,} dialog", "{n:,} dialogs", 1200) == "1\u00a0200 \u0434\u0438\u0430\u043b\u043e\u0433\u043e\u0432"
    i18n.set_language("en")
    assert i18n.trn("{n:,} dialog", "{n:,} dialogs", 1) == "1 dialog"
    assert i18n.trn("{n:,} dialog", "{n:,} dialogs", 1200) == "1,200 dialogs"
    assert i18n.tr("Open a model\u2026") == "Open a model\u2026"


def test_tr_of_a_plural_key_gives_its_many_form(russian):
    assert i18n.tr("{n:,} dialogs", n=7) == "7 \u0434\u0438\u0430\u043b\u043e\u0433\u043e\u0432"


def test_the_engines_progress_lines_in_russian(russian):
    from auralite.studio.progress import engine_text
    assert engine_text("Epoch 2/5 \u2014 validation\u2026") == "\u042d\u043f\u043e\u0445\u0430 2/5 \u2014 \u0432\u0430\u043b\u0438\u0434\u0430\u0446\u0438\u044f\u2026"
    assert engine_text("Tokenizing the corpus \u00b7 1,200 / 5,000 tokens (24%) \u00b7 ~3 min left") == \
        "\u0422\u043e\u043a\u0435\u043d\u0438\u0437\u0430\u0446\u0438\u044f \u043a\u043e\u0440\u043f\u0443\u0441\u0430 \u00b7 1,200 / 5,000 \u0442\u043e\u043a\u0435\u043d\u043e\u0432 (24%) \u00b7 \u043e\u0441\u0442\u0430\u043b\u043e\u0441\u044c ~3 \u043c\u0438\u043d"
    assert engine_text("Training the BPE tokenizer (vocab 4,096)") == \
        "\u041e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 \u0442\u043e\u043a\u0435\u043d\u0438\u0437\u0430\u0442\u043e\u0440\u0430 BPE (\u0441\u043b\u043e\u0432\u0430\u0440\u044c 4,096)"
    i18n.set_language("en")
    assert engine_text("Epoch 2/5 \u2014 validation\u2026") == "Epoch 2/5 \u2014 validation\u2026"


def test_durations_and_sizes_in_russian(russian):
    from auralite.gui.common import _fmt_duration
    from auralite.studio import library
    assert _fmt_duration(75) == "1 \u043c\u0438\u043d 15 \u0441"
    assert _fmt_duration(75, english=True) == "1m 15s"      # the console's lines
    assert library.format_size(10) == "10 \u0411"
    assert library.format_size(3 * (1 << 30)) == "3.0 \u0413\u0411"
    from auralite import runs
    assert runs._bytes(10) == "10 \u0411" and runs._bytes(3 * (1 << 20)) == "3.0 \u041c\u0411"
    i18n.set_language("en")
    assert library.format_size(10) == "10 B" and _fmt_duration(75) == "1m 15s"


def test_russian_parts_thousands_with_a_no_break_space(russian):
    assert i18n.tr("Parameters: {n:,}", n=1234567) == "\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b: 1\u00a0234\u00a0567"
    assert i18n.tr("Speed: {speed:,.0f} GFLOP/s", speed=76.4) == "\u0421\u043a\u043e\u0440\u043e\u0441\u0442\u044c: 76 \u0413\u0424\u041b\u041e\u041f/\u0441"
    assert i18n.number(1234567) == "1\u00a0234\u00a0567"          # a number on its own
    assert i18n.number(12345.6, ",.1f") == "12\u00a0345.6"
    i18n.set_language("en")
    assert i18n.number(1234567) == "1,234,567"


def test_a_frame_around_a_string_stays_as_it_is(russian):
    assert i18n.tr("  \u25b6 Continue it  ") == "  \u25b6 \u041f\u0440\u043e\u0434\u043e\u043b\u0436\u0438\u0442\u044c \u0435\u0433\u043e  "
    assert i18n.tr("Not in the catalogue") == "Not in the catalogue"


def test_a_broken_translation_shows_the_english(russian, monkeypatch):
    table = dict(i18n.catalogue("ru"))
    table["Hello {name}"] = "\u041f\u0440\u0438\u0432\u0435\u0442 {nme}"
    monkeypatch.setitem(i18n._catalogues, "ru", table)
    assert i18n.tr("Hello {name}", name="Ann") == "Hello Ann"


def test_trc_reads_the_word_in_its_context(russian):
    assert i18n.trc("data kind", "dialogs") == "\u0434\u0438\u0430\u043b\u043e\u0433\u0438"
    assert i18n.tr("dialogs") == "\u0434\u0438\u0430\u043b\u043e\u0433\u043e\u0432"   # the unit, after a number
    assert i18n.trc("no such context", "dialogs") == "\u0434\u0438\u0430\u043b\u043e\u0433\u043e\u0432"
    i18n.set_language("en")
    assert i18n.trc("data kind", "dialogs") == "dialogs"


def test_resolve_language_order(monkeypatch):
    monkeypatch.delenv("AURALITE_LANG", raising=False)
    monkeypatch.setattr(i18n, "system_language", lambda: "ru")
    assert i18n.resolve_language() == "ru"                     # the computer's language
    assert i18n.resolve_language("en") == "en"                 # a saved choice wins
    monkeypatch.setenv("AURALITE_LANG", "en")
    assert i18n.resolve_language() == "en"                     # then the environment
    assert i18n.resolve_language("ru_RU") == "ru"
    assert i18n.resolve_language("RU") == "ru"
    assert i18n.resolve_language("klingon") == "en"
    assert i18n.set_language("ru") == "ru" and i18n.language() == "ru"
    assert i18n.language_name() == RUSSIAN
    i18n.set_language("en")


def test_system_language_reads_lang(monkeypatch):
    # 2.28.2: $LANG is what other systems read; on Windows the display language comes first
    # (2.28.1) - and a Windows runner's is English
    monkeypatch.setattr(sys, "platform", "linux")
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANG", "ru_RU.UTF-8")
    assert i18n.system_language() == "ru"
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    assert i18n.system_language() == "en"
    monkeypatch.setenv("LANGUAGE", "ru:en")
    assert i18n.system_language() == "ru"


def test_system_language_reads_the_windows_interface_language(monkeypatch):
    import ctypes
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    class Kernel32:
        langid = 0x0419

        def GetUserDefaultUILanguage(self):  # noqa: N802 - the Windows name
            return self.langid

    kernel = Kernel32()
    monkeypatch.setattr(ctypes, "windll", type("WinDLL", (), {"kernel32": kernel})(),
                        raising=False)
    assert i18n.system_language() == "ru"
    kernel.langid = 0x0409
    assert i18n.system_language() == "en"


def test_commands_stay_english_whatever_the_computer_speaks(tmp_path):
    env = dict(os.environ, AURALITE_LANG="ru", LANG="ru_RU.UTF-8", AURALITE_HOME=str(tmp_path))
    code = ("import auralite.cli, auralite.runs as r, auralite.i18n as i;"
            "print(i.language(), r._duration(4000.0))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, timeout=300, check=True).stdout.split()
    assert out == ["en", "1", "h", "06", "min"]


def test_run_names_and_details_in_russian(russian, tmp_path, monkeypatch):
    from auralite import runs
    name = runs._auto_name("finetune", "lora", {"label": "Qwen3.5-0.8B"},
                           {"paths": ["/data/chats.jsonl"]})
    assert name == "\u0414\u043e\u043e\u0431\u0443\u0447\u0435\u043d\u0438\u0435: Qwen3.5-0.8B \u043d\u0430 chats.jsonl"
    i18n.set_language("en")
    assert runs._auto_name("finetune", "lora", {"label": "Qwen3.5-0.8B"},
                           {"paths": ["/data/chats.jsonl"]}) == \
        "Fine-tune Qwen3.5-0.8B on chats.jsonl"                  # English as it was


def test_the_data_line_says_the_kind_and_the_count(russian, tmp_path, monkeypatch):
    from auralite import runs
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path))
    recorder = runs.RunRecorder(kind="finetune", method="qlora", source="cli",
                                base={"spec": "hf:Org/M", "label": "M"},
                                data={"paths": ["/d/chats.jsonl"], "kind": "dialogs",
                                      "records": 1200})
    lines = dict(runs.describe_run(recorder.fail(MemoryError("out of memory"))))
    assert lines["\u0414\u0430\u043d\u043d\u044b\u0435"] == "chats.jsonl (\u0434\u0438\u0430\u043b\u043e\u0433\u0438, 1\u00a0200 \u0434\u0438\u0430\u043b\u043e\u0433\u043e\u0432)"
    assert lines["\u0421\u0442\u0430\u0442\u0443\u0441"] == "\u043d\u0435 \u0443\u0434\u0430\u043b\u0441\u044f"


# ----------------------------------------------------------------------
#  the Studio
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    from tests._integration import require
    widgets = require("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = widgets.QApplication.instance() or widgets.QApplication([])
    yield app


@pytest.fixture()
def studio_dialogs():
    from auralite.studio.qt import common
    seen: dict = {"confirm": [], "answer": False}
    common.DIALOGS.update(
        error=lambda friendly: None, inform=lambda title, text: None,
        confirm=lambda title, text: seen["confirm"].append((title, text)) or seen["answer"],
        save=lambda title: "", open=lambda title: "", dir=lambda title: "")
    yield seen
    common.DIALOGS.clear()


def _window(tmp_path, monkeypatch, **settings):
    from auralite.engine import AuraLiteEngine
    from auralite.studio.qt.window import MainWindow
    from auralite.studio.settings import StudioSettings
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    return MainWindow(engine=AuraLiteEngine(), settings=StudioSettings(first_run_done=True,
                                                                        **settings),
                      detect=False, capture_streams=False, save_on_close=False,
                      auto_speed_test=False)


def _shown_texts(window) -> list[str]:
    from PySide6.QtWidgets import QAbstractButton, QGroupBox, QLabel, QLineEdit, QWidget
    texts = []
    for widget in window.findChildren(QWidget):
        if isinstance(widget, (QLabel, QAbstractButton)):
            texts.append(widget.text())
        elif isinstance(widget, QGroupBox):
            texts.append(widget.title())
        elif isinstance(widget, QLineEdit):
            texts.append(widget.placeholderText())
        texts.append(widget.toolTip())
    out = []
    for text in texts:
        if not text:
            continue
        out.append(text)
        if "<" in text:                          # rich text: its words, part by part
            plain = re.sub(r"<[^>]+>", "", text)
            out += [part.strip() for part in plain.split(" \u00b7 ") if part.strip()]
    return out


def test_the_studio_window_speaks_russian(qapp, studio_dialogs, tmp_path, monkeypatch, russian,
                                          table):
    from auralite.studio.qt.window import TAB_TITLES
    window = _window(tmp_path, monkeypatch, language="ru")
    try:
        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert titles == [table[title] for _key, title in TAB_TITLES]
        menus = [action.text() for action in window.menuBar().actions()]
        assert menus == ["&\u0424\u0430\u0439\u043b", "&\u0412\u0438\u0434", "&\u0418\u043d\u0441\u0442\u0440\u0443\u043c\u0435\u043d\u0442\u044b", "&\u0421\u043f\u0440\u0430\u0432\u043a\u0430"]
        assert window.language_actions["ru"].isChecked()
        calculator = window.chat_tab.tool_boxes["calculate"].text()   # the model reads English
        assert calculator.startswith("calculate - " + table["Evaluate a mathematical expression "
                                                            "safely."])
        english = []
        for text in _shown_texts(window):
            core = i18n.split_frame(text)[1]
            if isinstance(table.get(core), str) and table[core] != core:
                english.append(text)
        assert not english, f"shown in English: {english[:15]}"
    finally:
        window.close()


def test_choose_language_saves_it_and_offers_a_restart(qapp, studio_dialogs, tmp_path,
                                                       monkeypatch):
    from auralite.studio.settings import load_settings
    window = _window(tmp_path, monkeypatch)
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: started.append((cmd, kw)))
    try:
        assert window.language_actions[""].isChecked()          # automatic
        window.choose_language("ru")                           # restart? - no
        assert load_settings().language == "ru" and window.settings.language == "ru"
        assert window.language_actions["ru"].isChecked()
        assert len(studio_dialogs["confirm"]) == 1 and started == []
        window.choose_language("")                             # English again: no restart
        assert load_settings().language == "" and len(studio_dialogs["confirm"]) == 1
        studio_dialogs["answer"] = True
        window.choose_language("ru")                           # restart? - yes
        assert len(started) == 1
        command, options = started[0]
        assert command[:3] == [sys.executable, "-m", "auralite.studio"]
        assert options["stdin"] == subprocess.DEVNULL
        assert not window.isVisible()
    finally:
        window.close()


def test_the_import_line_says_megabytes_in_the_language(qapp, russian):
    from auralite.studio.qt.tab_model import import_text
    info = {"stage": "download", "file": "model.safetensors", "bytes": 5e6, "total": 1.5e9}
    assert import_text(info) == "\u0421\u043a\u0430\u0447\u0438\u0432\u0430\u043d\u0438\u0435 model.safetensors: 5 / 1\u00a0500 \u041c\u0411"
    i18n.set_language("en")
    assert import_text(info) == "Downloading model.safetensors: 5 / 1,500 MB"
    assert import_text({"stage": "download", "file": "f", "bytes": 2e6}) == \
        "Downloading f: 2 MB"


class _Screen:
    """A screen of the given width for the Studio window's _fit_to_tabs."""

    def __init__(self, width: int) -> None:
        self.width = width

    def availableGeometry(self):
        from PySide6.QtCore import QRect
        return QRect(0, 0, self.width, 1000)


def test_the_studio_window_fits_its_tabs_when_the_screen_has_room(qapp, studio_dialogs, tmp_path,
                                                                  monkeypatch, russian):
    from PySide6.QtCore import QSize
    window = _window(tmp_path, monkeypatch, language="ru")
    try:
        bar = window.tabs.tabBar()
        bar.sizeHint = lambda: QSize(1400, 30)
        window.screen = lambda: _Screen(1920)
        window._fit_to_tabs()
        assert window.width() == 1416
        window.resize(1080, 800)
        window.screen = lambda: _Screen(1280)           # as wide as the screen allows
        window._fit_to_tabs()
        assert window.width() == 1240
        bar.sizeHint = lambda: QSize(900, 30)           # fits already: as it was
        window.resize(1080, 800)
        window._fit_to_tabs()
        assert window.width() == 1080
    finally:
        window.close()


def test_a_saved_studio_window_narrower_than_its_tabs_widens(qapp, studio_dialogs, tmp_path,
                                                            monkeypatch, russian):
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QTabBar

    from auralite.studio.qt.window import MainWindow
    first = _window(tmp_path, monkeypatch, language="ru")
    first.resize(1000, 700)
    saved = bytes(first.saveGeometry().toHex().data()).decode()
    first.close()
    monkeypatch.setattr(MainWindow, "screen", lambda self: _Screen(1920))
    monkeypatch.setattr(QTabBar, "sizeHint", lambda self: QSize(1400, 30))
    window = _window(tmp_path, monkeypatch, language="ru", geometry=saved)
    try:
        assert (window.width(), window.height()) == (1416, 700)    # the saved height kept
    finally:
        window.close()


def test_restart_command_starts_the_studio_as_it_was_started(monkeypatch):
    from auralite.studio import app as studio_app
    command = studio_app.restart_command(["--foo", studio_app.CHILD, studio_app.SELF_TEST])
    assert command == [sys.executable, "-m", "auralite.studio", "--foo"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert studio_app.restart_command([]) == [sys.executable]


def test_qt_speaks_the_language_in_its_own_dialogs(qapp, russian):
    from PySide6.QtCore import QCoreApplication

    from auralite.studio.qt.window import install_qt_translations
    translator = install_qt_translations(qapp)
    if translator is None:
        pytest.skip("this Qt build has no Russian translations")
    try:
        assert QCoreApplication.translate("QPlatformTheme", "Cancel") == "\u041e\u0442\u043c\u0435\u043d\u0430"
    finally:
        qapp.removeTranslator(translator)
    i18n.set_language("en")
    assert install_qt_translations(qapp) is None


def test_the_self_test_names_the_languages(tmp_path, monkeypatch, capsys):
    """The app's smoke test reads this line: a frozen app must carry the catalogue."""
    import re

    from auralite.studio import app as studio_app
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path))
    monkeypatch.setattr(studio_app, "qt_problem", lambda: "AuraLite Studio needs PySide6.")
    assert studio_app.self_test() == 2
    assert re.search(r"\n  Languages: English, ru [1-9][\d,]* strings\n",
                     capsys.readouterr().out)


def test_the_packages_carry_the_catalogue():
    root = PACKAGE.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"locale/*.json"' in pyproject
    spec = (root / "packaging" / "app" / "auralite_app.spec").read_text(encoding="utf-8")
    assert '"auralite/locale"' in spec


def test_studio_settings_default_to_the_computers_language():
    from auralite.studio.settings import StudioSettings
    assert StudioSettings().language == ""


# ----------------------------------------------------------------------
#  the classic interface
# ----------------------------------------------------------------------
def _classic(tmp_path, monkeypatch, config):
    pytest.importorskip("tkinter", reason="tkinter not available")
    import importlib
    importlib.import_module("tkinter.filedialog")
    importlib.import_module("tkinter.simpledialog")
    from tests._gui_harness import build_gui_app
    return build_gui_app(tmp_path, monkeypatch, config=config, restore_session=False)


def test_the_classic_interface_speaks_russian(tmp_path, monkeypatch, table):
    import tkinter.messagebox as mb

    from auralite.gui.config import GuiConfig
    from auralite.gui.i18n import tr
    with _classic(tmp_path, monkeypatch, GuiConfig(language="ru", language_chosen=True)) as app:
        assert app.language == "ru"
        titles = [call.kwargs.get("text") for call in app.notebook.add.call_args_list]
        assert titles == [tr(f"tab.{name}") for name in app._tab_names]
        assert "\u041e\u0431\u0443\u0447\u0435\u043d\u0438\u0435" in titles[1] and "\u0411\u0438\u0431\u043b\u0438\u043e\u0442\u0435\u043a\u0430" in titles[10]
        assert app.language_var.get() == RUSSIAN
        mb.showwarning.reset_mock()
        app.engine.model = None
        app._send_chat_message()
        assert mb.showwarning.call_args.args == (table["No Model"],
                                                table["Please load or train a model first."])


def test_the_classic_labels_are_russian(tmp_path, monkeypatch, table):
    from unittest.mock import MagicMock

    from auralite.gui.config import GuiConfig
    labels = []

    def label(*args, **kwargs):
        labels.append(kwargs.get("text"))
        return MagicMock()

    pytest.importorskip("tkinter", reason="tkinter not available")
    from tests import _gui_harness
    monkeypatch.setitem(_gui_harness._TK_PATCHES, "tkinter.ttk.Label", label)
    with _classic(tmp_path, monkeypatch, GuiConfig(language="ru", language_chosen=True)):
        pass
    for english in ("Passes over the data:", "Steps at most (0 = no limit):", "QAT LR:",
                    "LoRA LR:", "Prompt weight:"):
        assert table[english] in labels, english
    assert table["RAM: {used}"].format(used="\u2014") in labels


def test_the_classic_window_fits_its_tabs_when_the_screen_has_room(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from auralite.gui.config import GuiConfig
    with _classic(tmp_path, monkeypatch, GuiConfig()) as app:
        app.notebook.winfo_reqwidth = lambda: 1400
        app.root.winfo_screenwidth = lambda: 1920
        app.root.geometry = MagicMock()
        app._window_size = (800, 600)
        app._fit_window_to_tabs()
        app.root.geometry.assert_called_once_with("1432x600")
        app.root.winfo_screenwidth = lambda: 1280            # as wide as the screen allows
        app._window_size = (800, 600)
        app._fit_window_to_tabs()
        assert app.root.geometry.call_args.args == ("1208x600",)
        app.root.geometry.reset_mock()
        app.notebook.winfo_reqwidth = lambda: 700            # fits already: as it was
        app._window_size = (800, 600)
        app._fit_window_to_tabs()
        app.root.geometry.assert_not_called()


def test_a_narrow_screen_brings_the_classic_tab_titles_closer(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, call

    from auralite.gui import app as gui_app
    from auralite.gui.config import GuiConfig
    assert gui_app._tab_padding_for(232, 12) == 4      # 10 px off each side of 12 titles
    assert gui_app._tab_padding_for(100, 12) == 9
    assert gui_app._tab_padding_for(0, 12) == 14 and gui_app._tab_padding_for(-50, 12) == 14
    assert gui_app._tab_padding_for(10_000, 12) == 4   # never under 4
    with _classic(tmp_path, monkeypatch, GuiConfig()) as app:
        style = MagicMock()
        monkeypatch.setattr(gui_app.ttk, "Style", lambda *args, **kwargs: style)
        widths = iter([1408, 1168])                    # the Russian row, then closer
        app.notebook.winfo_reqwidth = lambda: next(widths)
        app.root.winfo_screenwidth = lambda: 1280
        app.root.geometry = MagicMock()
        app._window_size = (800, 600)
        app._fit_window_to_tabs()
        style.configure.assert_called_once_with("TNotebook.Tab", padding=(4, 6))
        assert app._tab_padding == 4
        app.root.geometry.assert_called_once_with("1200x600")
        style.reset_mock()
        app._apply_theme(light=False)                  # the theme keeps them closer
        assert call("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(4, 6)) in \
            style.configure.call_args_list
        style.reset_mock()
        app.notebook.winfo_reqwidth = lambda: 1100     # a row that fits: no change
        app._window_size = (1200, 600)
        app._fit_window_to_tabs()
        style.configure.assert_not_called()


def test_a_column_fits_its_heading(monkeypatch):
    import tkinter.font as tkfont

    from auralite.gui.common import heading_width

    class Font:
        def measure(self, text):
            return 10 * len(text)

    monkeypatch.setattr(tkfont, "nametofont", lambda name: Font())
    assert heading_width("\u041f\u043e\u043c\u0435\u0449\u0430\u0435\u0442\u0441\u044f") == 124
    assert heading_width("Fits") == 80                       # at least the minimum
    monkeypatch.setattr(tkfont, "nametofont", lambda name: 1 / 0)
    assert heading_width("\u041f\u043e\u043c\u0435\u0449\u0430\u0435\u0442\u0441\u044f", 90) == 96             # no Tk: by the letters


def test_a_language_saved_before_2_28_is_not_a_choice(tmp_path, monkeypatch):
    from auralite.gui import config as gui_config
    path = tmp_path / "gui_config.json"
    path.write_text(json.dumps({"language": "ru", "theme": "dark"}), encoding="utf-8")
    loaded = gui_config.load_config(path)
    assert loaded.language == "ru" and loaded.language_chosen is False
    with _classic(tmp_path, monkeypatch, loaded) as app:
        assert app.language == "en"                 # $AURALITE_LANG (the suite: English)
        assert app.language_var.get() == "Automatic"
        assert app.language_choice.get() == ""


def test_the_classic_restart_starts_it_again_and_closes(tmp_path, monkeypatch):
    from auralite.gui.config import GuiConfig
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: started.append(cmd))
    with _classic(tmp_path, monkeypatch, GuiConfig()) as app:
        closed = []
        monkeypatch.setattr(type(app), "_on_close", lambda self: closed.append(True))
        assert app.restart() is True
        assert started == [[sys.executable, "-m", "auralite.gui"]] and closed == [True]


def test_the_classic_interface_waits_while_busy(tmp_path, monkeypatch):
    import tkinter.messagebox as mb

    from auralite.gui.config import GuiConfig
    with _classic(tmp_path, monkeypatch, GuiConfig()) as app:
        monkeypatch.setattr(type(app), "_busy_with", lambda self: "training")
        restarted = []
        monkeypatch.setattr(type(app), "restart", lambda self: restarted.append(True))
        mb.showinfo.reset_mock()
        app._on_language_change("ru")
        assert restarted == [] and mb.showinfo.called
        assert app.gui_config.language == "ru" and app.gui_config.language_chosen
