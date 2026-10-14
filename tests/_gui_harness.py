"""Shared tkinter doubles for the GUI test suites (no display required).

The GUI cannot be started in CI, so every test that needs `AIApp` replaces the
whole tkinter surface with these recording doubles. They live here — instead of
inside one test module — because three suites need them now
(`test_gui_tabs_smoke.py`, `test_gui_tab_flows.py`, `test_gui_modules.py`), and a
second copy of a widget double is exactly how a "smoke" suite stops testing the
real thing.

Nothing here asserts anything: `build_gui_app()` is the entry point, the fixtures
do the assertions.
"""
from __future__ import annotations

import contextlib
import gc
import weakref
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

class MockVar:
    def __init__(self, value=None, *a, **k):
        self._val = value

    def get(self):
        return self._val

    def set(self, val):
        self._val = val

    def trace_add(self, *a, **k):
        return "trace-1"

    def trace(self, *a, **k):
        return "trace-1"


class MockText:
    """Bare-bones `tk.Text`: enough state for insert/delete/get round-trips."""

    def __init__(self, *a, **k):
        self.content = ""
        self.config_calls = 0
        self.tags: dict[str, dict] = {}
        self._index_reply = "1.0"      # overridden per test when needed

    # widget protocol
    def pack(self, *a, **k): pass
    def grid(self, *a, **k): pass
    def place(self, *a, **k): pass
    def bind(self, *a, **k): pass
    def see(self, *a, **k): pass
    def yview(self, *a, **k): pass
    def xview(self, *a, **k): pass

    def insert(self, index, text, *tags):
        self.content += text

    def index(self, spec):
        """`tk.Text.index` — the sandbox log uses it to trim old lines."""
        return self._index_reply

    def mark_set(self, *a, **k): pass
    def tag_remove(self, *a, **k): pass

    def delete(self, *a, **k):
        self.content = ""

    def get(self, *a, **k):
        return self.content

    def config(self, **k):
        self.config_calls += 1

    configure = config

    def tag_configure(self, tag, **k):
        self.tags[tag] = k

    def tag_add(self, *a, **k): pass


class DummyTk:
    """Minimal root: `after()` runs the callback immediately (documented)."""

    def __init__(self, *a, **k):
        self.destroyed = False
        self.clipboard = ""
        self._geometry = "820x620+10+10"
        self.protocols: dict[str, object] = {}

    def title(self, *a, **k): pass

    # dialog helpers (messagebox/simpledialog ask for these on a real root)
    def withdraw(self, *a, **k): pass
    def deiconify(self, *a, **k): pass
    def iconify(self, *a, **k): pass
    def attributes(self, *a, **k): return ""

    def geometry(self, value=None):
        if value is not None:
            self._geometry = value
        return self._geometry

    def minsize(self, *a, **k): pass
    def configure(self, *a, **k): pass
    def resizable(self, *a, **k): pass

    def protocol(self, name, fn):
        self.protocols[name] = fn

    def after(self, ms, fn=None, *a):
        if fn is not None:
            fn(*a)
        return "after-1"

    def after_idle(self, fn, *a):
        fn(*a)

    def winfo_screenwidth(self):
        return 1280

    def winfo_screenheight(self):
        return 800

    def winfo_children(self):
        return []

    def update(self, *a, **k): pass
    def update_idletasks(self, *a, **k): pass
    def mainloop(self, *a, **k): pass
    def quit(self, *a, **k): pass
    def destroy(self): self.destroyed = True
    def clipboard_clear(self): self.clipboard = ""
    def clipboard_append(self, text): self.clipboard += text
    def focus_set(self, *a, **k): pass
    def grab_release(self, *a, **k): pass
    def state(self, *a, **k): pass


def make_mock(*a, **k):
    return MagicMock()


_TK_PATCHES = {
    "tkinter.Variable": MockVar, "tkinter.StringVar": MockVar,
    "tkinter.BooleanVar": MockVar, "tkinter.DoubleVar": MockVar,
    "tkinter.IntVar": MockVar, "tkinter.Text": MockText, "tkinter.Tk": DummyTk,
    "tkinter.Menu": make_mock, "tkinter.Canvas": make_mock,
    "tkinter.ttk.Notebook": make_mock, "tkinter.ttk.Frame": make_mock,
    "tkinter.ttk.LabelFrame": make_mock, "tkinter.ttk.Button": make_mock,
    "tkinter.ttk.Label": make_mock, "tkinter.ttk.Entry": make_mock,
    "tkinter.ttk.Checkbutton": make_mock, "tkinter.ttk.Radiobutton": make_mock,
    "tkinter.ttk.Combobox": make_mock, "tkinter.ttk.Progressbar": make_mock,
    "tkinter.ttk.Scrollbar": make_mock, "tkinter.ttk.Scale": make_mock,
    "tkinter.ttk.Style": make_mock, "tkinter.ttk.OptionMenu": make_mock,
    "tkinter.ttk.Separator": make_mock, "tkinter.ttk.Spinbox": make_mock,
    "tkinter.ttk.Treeview": make_mock, "tkinter.ttk.PanedWindow": make_mock,
}

#: Dialog boxes must not pop up (and must not touch a real Tk root) in tests.
_DIALOG_PATCHES = {
    "messagebox.showinfo": MagicMock(), "messagebox.showwarning": MagicMock(),
    "messagebox.showerror": MagicMock(), "messagebox.askyesno": MagicMock(return_value=False),
    "filedialog.askopenfilename": MagicMock(return_value=""),
    "filedialog.asksaveasfilename": MagicMock(return_value=""),
    "filedialog.askdirectory": MagicMock(return_value=""),
}


class FakeAxis:
    def __init__(self):
        self.kinds: list[str] = []
        self.title = None

    def clear(self): self.kinds.append("clear")
    def bar(self, *a, **k): self.kinds.append("bar")
    def imshow(self, *a, **k): self.kinds.append("imshow")
    def plot(self, *a, **k): self.kinds.append("plot")
    def legend(self, *a, **k): self.kinds.append("legend")
    def set_xticks(self, *a, **k): pass
    def set_yticks(self, *a, **k): pass
    def set_xlabel(self, *a, **k): pass
    def set_ylabel(self, *a, **k): pass
    def set_title(self, text, **kwargs): self.title = text
    def axis(self, *a, **k): self.kinds.append("axis")
    def grid(self, *a, **k): pass
    def set_ylim(self, *a, **k): pass
    def axhline(self, *a, **k): self.kinds.append("axhline")
    def text(self, *a, **k): self.kinds.append("text")
    def annotate(self, *a, **k): self.kinds.append("annotate")


class FakeFigure:
    """Minimal `matplotlib.figure.Figure`: records axes and tolerates drawing."""

    def __init__(self, *a, **k):
        self.figsize = k.get("figsize")
        self.dpi = k.get("dpi")
        self.axes: list[FakeAxis] = []
        self.saved: list[str] = []

    def add_subplot(self, *a, **k):
        axis = FakeAxis()
        self.axes.append(axis)
        return axis

    def subplots(self, *a, **k):
        axis = self.add_subplot(111)
        return self, axis

    def tight_layout(self, *a, **k): pass
    def subplots_adjust(self, *a, **k): pass
    def suptitle(self, *a, **k): pass
    def colorbar(self, *a, **k): return MagicMock()
    def cla(self): self.axes.clear()
    def clear(self): self.axes.clear()
    def savefig(self, path, *a, **k): self.saved.append(str(path))


@contextlib.contextmanager
def build_gui_app(tmp_path, monkeypatch, prepare=None, config=None,
                  session=None, language=None, restore_session=True,
                  matplotlib=False):
    """Build a real `AIApp` on top of the doubles, with state redirected.

    Args:
        tmp_path/monkeypatch: pytest fixtures (files and env go to the temp dir).
        prepare: optional callable run *after* import but *before* `AIApp` is
            built — used to pre-load an engine or patch module globals.
        config/session/language: overrides for `GuiConfig` fields.
        restore_session: pass `False` to start from a clean slate.
        matplotlib: keep the optional matplotlib path enabled (the caller is then
            responsible for patching `Figure`/`FigureCanvasTkAgg` in the tab
            modules it exercises — the defaults are text fallbacks).
    """
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
        stack.enter_context(patch("auralite.gui.common.HAS_MATPLOTLIB", bool(matplotlib)))
        for target, value in _DIALOG_PATCHES.items():
            module_name, _, attr = target.rpartition(".")
            module = {"messagebox": sys.modules["tkinter.messagebox"],
                      "filedialog": sys.modules["tkinter.filedialog"]}[module_name]
            stack.enter_context(patch.object(module, attr, value))

        from auralite.gui.app import AIApp
        from auralite.gui import config as gui_config_module
        from auralite.gui import session as gui_session

        # `SESSION_PATH`/`DEFAULT_CONFIG_PATH` are computed at import time, so
        # point the modules at the temp dir explicitly.
        stack.enter_context(patch.object(gui_session, "SESSION_PATH",
                                         tmp_path / "gui_session.json"))
        stack.enter_context(patch.object(gui_config_module, "DEFAULT_CONFIG_PATH",
                                         tmp_path / "gui_config.json"))

        if config is not None or not restore_session or language:
            from auralite.gui.config import GuiConfig

            base = config if config is not None else GuiConfig()
            base.restore_session = bool(restore_session and base.restore_session)
            if language:
                base.language = language
            stack.enter_context(patch.object(gui_config_module, "load_config",
                                             lambda *a, **k: base))
        if prepare is not None:
            prepare()
        # `AIApp.__init__` replaces `sys.stdout`/`sys.stderr` with a console
        # redirector that references the Tk root. Restore the real streams on
        # exit so the redirectors of finished tests do not pile up.
        saved_streams = (sys.stdout, sys.stderr)

        def _restore_streams():
            sys.stdout, sys.stderr = saved_streams
            # Every GUI test builds a real app (engine, tensors, figures), so
            # collect its cycles eagerly instead of leaving them to a later
            # generation-2 pass.
            gc.collect()

        stack.callback(_restore_streams)
        instance = AIApp(DummyTk())
        instance._tmp_path = tmp_path

        def _shutdown_bridge():
            """Stop the app's background bridge before the app is dropped.

            `TkAsyncBridge` owns a daemon thread running an asyncio loop plus a
            `ThreadPoolExecutor`. Those live threads keep the bridge alive; the
            bridge holds the Tk root and the root holds the app (through
            `root.protocol("WM_DELETE_WINDOW", app._on_close)`), so an app whose
            bridge is never stopped can never be collected. That leaked the whole
            app graph — engine, tensors and figures — once per test: the GUI
            suite grew to ~1.3 GB peak RSS and a full run swap-thrashed on a 2 GB
            box. Shutting the bridge down keeps the suite flat (~0.5 GB).
            """
            bridge = getattr(instance, "bridge", None)
            if bridge is not None:
                with contextlib.suppress(Exception):
                    bridge.shutdown(timeout=2.0)
                instance.bridge = None

        stack.callback(_shutdown_bridge)
        yield instance


def dialog_double(name: str, value):
    """Patch one stdlib dialog (e.g. `filedialog.askopenfilename`) for one test."""
    module_name, _, attr = name.rpartition(".")
    module = sys.modules[{"messagebox": "tkinter.messagebox",
                          "filedialog": "tkinter.filedialog",
                          "simpledialog": "tkinter.simpledialog"}[module_name]]
    return patch.object(module, attr,
                        MagicMock(return_value=value) if value is not None else MagicMock(),
                        create=True)


def wait_for(predicate, timeout: float = 60.0, interval: float = 0.02) -> bool:
    """Poll `predicate` until it is true (used for the GUI's worker threads)."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class InlineThread:
    """`threading.Thread` replacement that runs the target synchronously.

    `created` holds *weak* references on purpose: a strong list would keep every
    worker closure — and through it the whole `AIApp`, its engine and its tensors —
    alive for the rest of the session.  That leak pushed the GUI suite's peak RSS
    past 1.3 GB and made the full test run swap-thrash on low-memory machines.
    """

    created: list = []

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        InlineThread.created.append(weakref.ref(self))

    def start(self):
        if self.target is not None:
            self.target(*self.args, **self.kwargs)

    def join(self, timeout=None):  # pragma: no cover - inline threads are finished
        return None

    def is_alive(self):  # pragma: no cover
        return False
