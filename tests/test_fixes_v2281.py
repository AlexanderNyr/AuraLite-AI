"""2.28.1: every tab scrolls to its end - in the Studio and in the classic interface.

The Studio: Chat, Generate, Library and Log were not in scroll areas. Generate needs 849 px,
so the window could not be shorter than 929 px - taller than a 768 or 800 px screen: the
bottom of every tab, the scrolled ones too, lay below the screen and no scroll bar could
bring it up (a maximized window got the screen's height until the next layout made it grow
again). Now every tab is a :class:`~auralite.studio.qt.common.ScrollPage`; these tests size
the window down and check that each tab's content can be scrolled to its end - also after
sections open and labels grow - and that the window opens within a small screen.

The classic interface: a tab whose content was once shorter than the window got the
window's height, fixed, and content that grew afterwards (a result, a section) lay below
the scroll region. Now ``_sync_scroll`` sizes the tab to its content or the window, and the
tab shown is measured again every 0.4 s.

And on Windows the display language comes before ``$LANG``, which Git Bash sets to English.
"""
from __future__ import annotations

import os
import sys

import pytest

from auralite import i18n


@pytest.fixture()
def qapp():
    from tests._integration import require
    widgets = require("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return widgets.QApplication.instance() or widgets.QApplication([])


@pytest.fixture()
def studio_dialogs():
    from auralite.studio.qt import common
    common.DIALOGS.update(error=lambda friendly: None, inform=lambda title, text: None,
                          confirm=lambda title, text: True, save=lambda title: "",
                          open=lambda title: "", dir=lambda title: "")
    yield
    common.DIALOGS.clear()


def _window(tmp_path, monkeypatch, **settings):
    from auralite.engine import AuraLiteEngine
    from auralite.studio.qt.window import MainWindow
    from auralite.studio.settings import StudioSettings
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    return MainWindow(engine=AuraLiteEngine(),
                      settings=StudioSettings(first_run_done=True, **settings),
                      detect=False, capture_streams=False, save_on_close=False,
                      auto_speed_test=False)


def _settle(app, rounds: int = 6) -> None:
    from PySide6.QtCore import QCoreApplication
    for _ in range(rounds):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, 0)


def _reach(area) -> tuple[int, int]:
    """``(needed, reachable)``: the height the page needs at the area's width, and how far
    the area can show it (its height plus the scroll range)."""
    from PySide6.QtWidgets import QWidget
    page, viewport = area.widget(), area.viewport()
    layout = page.layout()
    needed = (layout.totalHeightForWidth(QWidget.width(viewport)) if layout.hasHeightForWidth()
              else layout.totalMinimumSize().height())
    return needed, QWidget.height(viewport) + area.verticalScrollBar().maximum()


def _unreachable(app, window) -> list[str]:
    from auralite.studio.qt.window import TAB_TITLES
    short = []
    for index, (key, _title) in enumerate(TAB_TITLES):
        window.tabs.setCurrentIndex(index)
        _settle(app)
        needed, reachable = _reach(window.tabs.widget(index))
        if needed > reachable + 1:
            short.append(f"{key}: needs {needed}, shows {reachable}")
    return short


def test_every_tab_scrolls_and_none_sets_the_window_height(qapp, studio_dialogs, tmp_path,
                                                           monkeypatch):
    from auralite.studio.qt.common import ScrollPage
    from auralite.studio.qt.window import TAB_TITLES
    window = _window(tmp_path, monkeypatch)
    try:
        areas = [window.tabs.widget(i) for i in range(window.tabs.count())]
        assert all(isinstance(area, ScrollPage) for area in areas)
        assert [area.page() for area in areas] == [window.pages[key] for key, _ in TAB_TITLES]
        assert window.minimumSizeHint().height() < 400        # was 929: Generate's 849 px
        window.goto("generate")
        assert window.current_page() is window.generate_tab
    finally:
        window.close()


@pytest.mark.parametrize("size", [(1280, 720), (1024, 560)])
def test_every_tab_can_be_scrolled_to_its_end_on_a_small_screen(qapp, studio_dialogs, tmp_path,
                                                                monkeypatch, size):
    window = _window(tmp_path, monkeypatch)
    try:
        window.show()
        window.resize(*size)
        _settle(qapp)
        assert (window.width(), window.height()) == size     # the window could not be 929 px
        assert not _unreachable(qapp, window)
    finally:
        window.close()


def test_a_tab_scrolls_to_its_end_after_its_content_grows(qapp, studio_dialogs, tmp_path,
                                                          monkeypatch):
    from PySide6.QtWidgets import QLabel

    from auralite.studio.qt.common import Collapsible
    window = _window(tmp_path, monkeypatch)
    long_text = "A report that wraps over many lines of the window, word after word. " * 12
    try:
        window.show()
        window.resize(1280, 700)
        _settle(qapp)
        for key in ("train", "finetune"):
            window.goto(key)
            _settle(qapp)
            area = window.tabs.currentWidget()
            closed, _ = _reach(area)
            for section in window.pages[key].findChildren(Collapsible):
                section.set_expanded(True)
            _settle(qapp)
            opened, reachable = _reach(area)
            assert opened > closed and opened <= reachable + 1, key
            for label in window.pages[key].findChildren(QLabel):
                if label.wordWrap() and label.isVisible():
                    label.setText(long_text)
            _settle(qapp)
            grown, reachable = _reach(area)
            assert grown > opened and grown <= reachable + 1, key
        window.resize(1280, 560)                 # and with the window made shorter
        _settle(qapp)
        assert not _unreachable(qapp, window)
    finally:
        window.close()


class _Screen:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)

    def availableGeometry(self):
        from PySide6.QtCore import QRect
        return QRect(0, 0, *self.size)


def test_the_window_opens_within_a_small_screen(qapp, studio_dialogs, tmp_path, monkeypatch):
    from auralite.studio.qt.window import MainWindow
    monkeypatch.setattr(MainWindow, "screen", lambda self: _Screen(1280, 760))
    window = _window(tmp_path, monkeypatch)
    try:
        assert window._initial_size() == (1080, 700)          # 800 px did not fit
        assert window.height() <= 700
    finally:
        window.close()
    monkeypatch.setattr(MainWindow, "screen", lambda self: _Screen(1920, 1040))
    window = _window(tmp_path, monkeypatch)
    try:
        assert window._initial_size() == (1080, 800)          # room: as before
    finally:
        window.close()
    monkeypatch.setattr(MainWindow, "screen", lambda self: _Screen(800, 500))
    window = _window(tmp_path, monkeypatch)
    try:
        assert window._initial_size() == (760, 480)           # never below 640 x 480
    finally:
        window.close()


def test_windows_speaks_its_display_language_whatever_git_bash_says(monkeypatch):
    import ctypes
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LANG", "en_US.UTF-8")                 # what Git Bash sets
    monkeypatch.delenv("LANGUAGE", raising=False)
    monkeypatch.delenv("LC_ALL", raising=False)

    class Kernel32:
        langid = 0x0419                                       # Russian

        def GetUserDefaultUILanguage(self):  # noqa: N802 - the Windows name
            return self.langid

    kernel = Kernel32()
    monkeypatch.setattr(ctypes, "windll", type("WinDLL", (), {"kernel32": kernel})(),
                        raising=False)
    assert i18n.system_language() == "ru"
    kernel.langid = 0x0409
    monkeypatch.setenv("LANG", "ru_RU.UTF-8")
    assert i18n.system_language() == "en"                     # the display language wins
    monkeypatch.setattr(ctypes, "windll", None, raising=False)  # no Windows API: $LANG
    assert i18n.system_language() == "ru"


# ---- the classic interface ---------------------------------------------------------------
class _Canvas:
    """The calls `_sync_scroll` makes on a tab's canvas."""

    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.item: dict = {}
        self.region = None
        self.configured = 0

    def winfo_width(self) -> int:
        return self.size[0]

    def winfo_height(self) -> int:
        return self.size[1]

    def itemconfigure(self, item, **options) -> None:
        assert item == 7
        self.item.update(options)
        self.configured += 1

    def configure(self, scrollregion=None) -> None:
        self.region = scrollregion

    def bbox(self, _tag):                     # what the old code asked for
        return (0, 0, self.size[0], self.size[1])


class _Inner:
    def __init__(self, height: int) -> None:
        self.height = height

    def winfo_reqheight(self) -> int:
        return self.height


def _classic_app():
    from auralite.gui.app import AIApp
    app = AIApp.__new__(AIApp)                # the method alone, without a window
    app._scroll_sizes = {}
    return app


def test_a_classic_tab_follows_content_that_grows_after_it_filled_the_window():
    app = _classic_app()
    canvas, inner = _Canvas(1260, 609), _Inner(390)
    app._sync_scroll(canvas, inner, 7)        # short content: the tab fills the window
    assert canvas.item == {"width": 1260, "height": 609}
    assert canvas.region == (0, 0, 1260, 609)
    inner.height = 1290                       # a result shown: 900 px more
    app._sync_scroll(canvas, inner, 7)
    assert canvas.item["height"] == 1290 and canvas.region == (0, 0, 1260, 1290)
    inner.height = 390                        # and gone again: no blank page to scroll
    app._sync_scroll(canvas, inner, 7)
    assert canvas.item["height"] == 609 and canvas.region == (0, 0, 1260, 609)
    calls = canvas.configured
    app._sync_scroll(canvas, inner, 7)        # nothing changed: nothing configured
    assert canvas.configured == calls
    canvas.size = (1000, 469)                 # the window made smaller
    app._sync_scroll(canvas, inner, 7)
    assert canvas.item == {"width": 1000, "height": 469}
    assert canvas.region == (0, 0, 1000, 469)


def test_the_classic_tab_shown_is_measured_again_every_moment(monkeypatch):
    import tkinter as tk
    app = _classic_app()
    seen = []
    app._sync_visible_tab = lambda: seen.append("sync")

    class Root(tk.Misc):
        def __init__(self) -> None:        # no Tcl: only `after` is used
            self.timers = []

        def after(self, ms, fn=None, *args):
            self.timers.append((ms, fn))
            return f"after#{len(self.timers)}"

    app.root = Root()
    cancelled = []
    app.root.after_cancel = cancelled.append
    app._watch_tab_content()
    assert seen == ["sync"] and app.root.timers[0][0] == 400
    app.root.timers[0][1]()                   # the timer fires: measured again, re-armed
    assert seen == ["sync", "sync"] and len(app.root.timers) == 2
    app._stop_scroll_ticks()                  # closing the window stops the loop
    assert cancelled == ["after#2"] and app._scroll_timer is None
    app.root = object()                       # a test double for the window: no timer loop
    seen.clear()
    app._watch_tab_content()
    assert seen == []


def test_the_classic_tab_shown_is_the_one_measured():
    app = _classic_app()
    seen = []
    app._scroll_syncs = [lambda: seen.append("start"), lambda: seen.append("train")]

    class Notebook:
        def index(self, what):
            assert what == "current"
            return 1

    app.notebook = Notebook()
    app._sync_visible_tab()
    assert seen == ["train"]
