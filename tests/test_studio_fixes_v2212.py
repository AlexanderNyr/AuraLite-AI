"""2.21.2: what the first runs of the Windows app on a PC from 2011 showed.

* a run with a time limit (the Studio sets one): its progress - the chart's x, the bar - in
  one measure for every event of an epoch, so it never goes back;
* "RAM: 0.0 GB" on Windows: the process' memory with GetProcessMemoryInfo's types declared,
  psutil first;
* the console: ``.\\auralite.exe`` in every hint (PowerShell runs nothing from the current
  folder by its name alone), another ``auralite`` on PATH, ``auralite doctor --out``;
* the Train tab lists a model size by its parameters and shape.
"""
from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from auralite import bundle

REPO = Path(__file__).resolve().parents[1]
INFO = {"app": "AuraLite Studio", "version": "2.21.2", "platform": "windows-x64",
        "python": "3.13.9", "torch": "2.15.0+cpu", "qt": "6.9.3", "built": "2026-11-07",
        "commit": "abc1234"}
CORPUS = ("the quick brown fox jumps over the lazy dog. "
          "a journey of a thousand miles begins with a single step. ") * 30
BASE = {"d_model": 32, "d_ff": 64, "n_heads": 2, "n_layers": 1, "seq_length": 16, "seed": 0,
        "tokenizer": "char", "batch_size": 8, "lr": 3e-3, "progress_every_seconds": 1e9,
        "progress_every_batches": 1}


def _load(name: str):
    """A script of packaging/app as a module (the folder is not a package)."""
    spec = importlib.util.spec_from_file_location(f"_auralite_app_v2212_{name}",
                                                  REPO / "packaging" / "app" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_app(tmp_path, monkeypatch):
    """This process as the app: frozen, its programs in tmp_path/app, with its note."""
    folder = tmp_path / "app"
    folder.mkdir()
    (folder / bundle.APP_INFO).write_text(json.dumps(INFO), encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(folder / bundle.program_name("studio")))
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    bundle._read_info.cache_clear()
    yield folder.resolve()
    bundle._read_info.cache_clear()


# ----------------------------------------------------------------------
#  A timed run's progress
# ----------------------------------------------------------------------
def _events(params: dict) -> tuple[list[tuple[str, float]], dict]:
    from auralite.engine import AuraLiteEngine
    seen: list[tuple[str, float]] = []

    def on_progress(epoch, total, loss, val_loss, info=None):
        info = info or {}
        if info.get("phase") in ("train", "val", "epoch_end", "stopped"):
            seen.append((info["phase"], float(info["percent"])))

    summary = AuraLiteEngine().train(CORPUS, params, progress_callback=on_progress)
    return seen, summary


def _in_order(seen: list[tuple[str, float]]) -> bool:
    percents = [p for _phase, p in seen]
    return all(b >= a for a, b in zip(percents, percents[1:], strict=False))


def test_a_timed_runs_progress_never_goes_back_when_the_epochs_end_it():
    """A time limit far beyond the run, as the Studio's 60 minutes for a small text: the
    epochs end the run, and every event takes their share. 2.21.1's steps took the time's
    share (about 0 %), validation and the epochs' ends 33, 67 and 100 %."""
    pytest.importorskip("torch")
    seen, summary = _events(dict(BASE, epochs=3, val_split=0.1, time_budget_s=3600.0))
    assert {"train", "val", "epoch_end"} <= {phase for phase, _ in seen}
    assert _in_order(seen), seen
    ends = [p for phase, p in seen if phase == "epoch_end"]
    assert ends == pytest.approx([100 / 3, 200 / 3, 100.0], abs=0.01)
    assert max(p for phase, p in seen if phase != "epoch_end") <= 99.99
    assert summary["stop_reason"] != "time_budget"


def test_a_timed_runs_progress_never_goes_back_when_the_time_ends_it():
    """A time limit that ends the run long before its epochs: the time's share is further
    on, and validation and the epochs' ends take it too (2.21.1: the epochs' share, a
    fraction of a percent behind the steps)."""
    pytest.importorskip("torch")
    seen, summary = _events(dict(BASE, epochs=10_000, val_split=0.1, time_budget_s=1.5,
                                 early_stopping_patience=0))     # only the time ends it
    assert summary["stop_reason"] == "time_budget"
    assert {"train", "epoch_end"} <= {phase for phase, _ in seen}
    assert _in_order(seen), seen
    ends = [p for phase, p in seen if phase == "epoch_end"]
    assert ends[-1] == 100.0                    # the run ends here (not 0.0x % of 10,000 epochs)
    assert max(p for phase, p in seen if phase != "epoch_end") <= 99.99
    # the last steps come close to the time limit: its share, not the epochs' (0.0x %)
    assert max(p for phase, p in seen if phase == "train") > 50.0


def test_a_stop_in_a_timed_run_reports_where_it_stopped():
    """A stop in the middle of an epoch: its event in the measure of the steps before it
    (2.21.1: the share of the epochs alone - next to nothing of 10,000)."""
    pytest.importorskip("torch")
    import threading

    from auralite.engine import AuraLiteEngine
    stop = threading.Event()
    seen: list[tuple[str, float]] = []

    def on_progress(epoch, total, loss, val_loss, info=None):
        info = info or {}
        if info.get("phase") in ("train", "val", "epoch_end", "stopped"):
            seen.append((info["phase"], float(info["percent"])))
            if info["phase"] == "train" and float(info["percent"]) >= 1.0 \
                    and 2 <= int(info["batch"]) <= int(info["batches"]) - 3:
                stop.set()                                    # mid-epoch

    params = dict(BASE, epochs=10_000, val_split=0.1, time_budget_s=30.0,
                  early_stopping_patience=0)
    summary = AuraLiteEngine().train(CORPUS, params, progress_callback=on_progress,
                                     stop_event=stop)
    assert summary["stop_reason"] == "stopped"
    assert seen[-1][0] == "stopped" and seen[-1][1] >= 1.0
    assert _in_order(seen), seen[-5:]


def test_without_a_time_limit_the_epochs_share_as_before():
    pytest.importorskip("torch")
    seen, _summary = _events(dict(BASE, epochs=2, val_split=0.1))
    assert _in_order(seen), seen
    assert [p for phase, p in seen if phase == "epoch_end"] == pytest.approx([50.0, 100.0])
    # validation reports its batches too: the same percent per epoch
    assert sorted({p for phase, p in seen if phase == "val"}) == pytest.approx([50.0, 99.99])


def test_the_studios_progress_never_goes_back():
    from auralite.studio.progress import ProgressTracker
    tracker = ProgressTracker()
    tracker.update(1, 100, 5.0, None, {"phase": "train", "batch": 5, "batches": 5,
                                       "percent": 6.2})
    view = tracker.update(1, 100, 4.9, 4.95, {"phase": "epoch_end", "batch": 5, "batches": 5,
                                              "percent": 6.0})
    assert view.percent == pytest.approx(6.2)
    tracker.update(2, 100, 4.8, None, {"phase": "train", "batch": 1, "batches": 5,
                                       "percent": 6.4})
    xs = [x for x, _loss in tracker.losses]
    assert xs == sorted(xs) and xs[-1] == pytest.approx(6.4)
    assert tracker.val_losses == [(pytest.approx(6.2), 4.95)]


# ----------------------------------------------------------------------
#  The Studio's memory on Windows
# ----------------------------------------------------------------------
def test_process_memory_asks_psutil_first(monkeypatch):
    from auralite.studio import hardware

    class _Process:
        def memory_info(self):
            return types.SimpleNamespace(rss=123_456_789)

    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(Process=_Process))
    assert hardware.process_memory() == 123_456_789


def test_without_psutil_windows_asks_the_system(monkeypatch):
    from auralite.studio import hardware
    monkeypatch.setitem(sys.modules, "psutil", None)          # `import psutil` fails
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware, "_windows_working_set", lambda: 777 << 20)
    assert hardware.process_memory() == 777 << 20

    def broken() -> int:
        raise OSError("no psapi")

    monkeypatch.setattr(hardware, "_windows_working_set", broken)
    assert hardware.process_memory() == 0


class _Function:
    """A DLL's function: what ctypes sets on it, and a Python body."""

    def __init__(self, body):
        self.body, self.restype, self.argtypes = body, None, None

    def __call__(self, *args):
        return self.body(self, *args)


def test_the_windows_call_declares_its_handle_types():
    """GetCurrentProcess() is -1, a pointer-sized handle: returned as ctypes' default int it
    reaches GetProcessMemoryInfo as 0xFFFFFFFF on 64-bit Windows, an invalid handle - the
    call fails and 2.21.1's status bar said "RAM: 0.0 GB"."""
    from auralite.studio import hardware
    seen: dict = {}

    def current_process(fn):
        return -1 if fn.restype is ctypes.c_void_p else 0xFFFFFFFF

    def memory_info(fn, handle, counters, size):
        seen.update(handle=handle, size=size)
        if fn.argtypes is None or fn.argtypes[0] is not ctypes.c_void_p or handle != -1:
            return 0                                          # ERROR_INVALID_HANDLE
        counters._obj.WorkingSetSize = 512 << 20
        return 1

    dlls = {"kernel32": types.SimpleNamespace(GetCurrentProcess=_Function(current_process)),
            "psapi": types.SimpleNamespace(GetProcessMemoryInfo=_Function(memory_info))}
    assert hardware._windows_working_set(dlls.__getitem__) == 512 << 20
    assert seen == {"handle": -1, "size": ctypes.sizeof(hardware._MemoryCounters)}
    assert dlls["psapi"].GetProcessMemoryInfo.restype is ctypes.c_int
    assert dlls["psapi"].GetProcessMemoryInfo.argtypes[1]._type_ is hardware._MemoryCounters


# ----------------------------------------------------------------------
#  The console
# ----------------------------------------------------------------------
def test_outside_the_app_the_console_line_is_the_pip_command():
    assert bundle.console_command("doctor", "--studio") == "auralite doctor --studio"
    assert bundle.console_hint("runs") == "auralite runs"
    assert bundle.other_cli_on_path() is None


def test_in_the_app_the_console_line_has_its_folder_in_front(fake_app):
    assert bundle.console_command("doctor", "--studio", system="win32") == \
        ".\\auralite.exe doctor --studio"
    assert bundle.console_command("train", "--data", "my notes.txt", system="win32") == \
        '.\\auralite.exe train --data "my notes.txt"'
    assert bundle.console_command("runs", system="linux") == "./auralite runs"
    assert bundle.console_hint("studio") == \
        f"{bundle.console_command('studio')} (in a console opened in {fake_app})"


def test_the_start_up_hints_say_the_console_line(fake_app, tmp_path):
    from auralite.studio import app
    full_path = bundle.quoted(bundle.cli_command("studio"))      # 2.21.1's hint
    text = app.failure_text(RuntimeError("boom"))
    assert bundle.console_hint("studio") in text
    assert bundle.console_hint("doctor", "--studio") in text
    assert full_path not in text
    output, faults = tmp_path / "out.log", tmp_path / "faults.log"
    output.write_text("ImportError: DLL load failed while importing QtCore\n", encoding="utf-8")
    crash = app.crash_text(0xC0000409, 1.0, output, faults)
    assert bundle.console_hint("studio") in crash and full_path not in crash


def test_another_auralite_on_path_is_found(fake_app, tmp_path):
    name = bundle.program_name("cli")
    pip = tmp_path / "Python313" / "Scripts"
    pip.mkdir(parents=True)
    (pip / name).write_text("", encoding="utf-8")
    (fake_app / name).write_text("", encoding="utf-8")
    assert bundle.other_cli_on_path(os.pathsep.join([str(pip), str(fake_app)])) == pip / name
    assert bundle.other_cli_on_path(os.pathsep.join([str(fake_app), str(pip)])) is None
    missing = tmp_path / "nothing here"
    assert bundle.other_cli_on_path(os.pathsep.join(["", f'"{missing}"'])) is None
    assert bundle.other_cli_on_path(os.pathsep.join([f'"{pip}"'])) == pip / name


def test_doctor_in_the_app_names_another_auralite(fake_app, tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from auralite import cli
    name = bundle.program_name("cli")
    pip = tmp_path / "Scripts"
    pip.mkdir()
    (pip / name).write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", os.pathsep.join([str(pip), os.environ.get("PATH", "")]))
    assert cli.main(["doctor", "--no-speed", "--goal", "pretrain"]) == 0
    out = capsys.readouterr().out
    assert f"In PowerShell `auralite` alone runs {pip / name}, another AuraLite" in out
    assert f"For this app's type {bundle.console_command()} in its folder" in out
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert cli.main(["doctor", "--no-speed", "--goal", "pretrain"]) == 0
    assert "alone runs" not in capsys.readouterr().out


def test_doctor_writes_its_report_to_a_file(tmp_path, capsys):
    pytest.importorskip("torch")
    from auralite import cli
    out = tmp_path / "report.txt"
    assert cli.main(["doctor", "--no-speed", "--goal", "pretrain", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    report = out.read_text(encoding="utf-8")
    assert report.startswith("AuraLite ") and "Processor:" in report
    assert report in printed and f"The report is in {out.resolve()}" in printed
    assert "The report is in" not in report


def test_the_readme_says_the_console_line():
    build = _load("build_app")
    windows = build.readme_text(dict(INFO), "win32")
    assert ".\\auralite.exe doctor --studio" in windows
    assert ".\\auralite.exe doctor --studio --out r.txt" in windows
    assert "powershell in Explorer's address bar" in windows
    assert "    auralite.exe" not in windows               # never the name alone
    linux = build.readme_text(dict(INFO, platform="linux-x64"), "linux")
    assert "./auralite doctor --studio" in linux and "powershell" not in linux


def test_the_guide_and_the_faq_say_the_console_line():
    guide = (REPO / "docs" / "windows-app.md").read_text(encoding="utf-8")
    assert ".\\auralite.exe doctor --studio --out report.txt" in guide
    assert "\nauralite.exe " not in guide                  # no example with the name alone
    assert '"AuraLite Studio.exe" --self-test' not in guide
    faq = (REPO / "docs" / "faq.md").read_text(encoding="utf-8")
    assert "## 24. `auralite.exe doctor` says `invalid choice: 'doctor'`" in faq


# ----------------------------------------------------------------------
#  The Train tab's model sizes
# ----------------------------------------------------------------------
def test_a_size_is_its_parameters_and_shape():
    from auralite.studio.advisor import PresetFit, preset_label
    assert preset_label(PresetFit("25m", 6_600_000, 700 << 20, True, 857.0, 0.02)) == \
        "7M parameters - 8 layers, width 256"
    assert preset_label(PresetFit("125m", 85_000_000, 0, True, None, None)) == \
        "85M parameters - 12 layers, width 768"
    assert preset_label(PresetFit("custom", 800_000, 0, True, None, None)) == "0.8M parameters"


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from tests._integration import require
    QtWidgets = require("PySide6.QtWidgets")
    QtCore = require("PySide6.QtCore")
    from auralite.engine import AuraLiteEngine
    from auralite.studio.hardware import Hardware
    from auralite.studio.qt import common
    from auralite.studio.qt.window import MainWindow
    from auralite.studio.settings import StudioSettings
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    common.DIALOGS.update(error=lambda friendly: None, inform=lambda title, text: None,
                          confirm=lambda title, text: True, save=lambda title: "",
                          open=lambda title: "", dir=lambda title: "")
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    window = MainWindow(engine=AuraLiteEngine(), settings=StudioSettings(first_run_done=True),
                        detect=False, capture_streams=False, save_on_close=False,
                        auto_speed_test=False)
    gb = 1 << 30
    window.hardware = Hardware(system="Windows", cpu="Intel(R) Core(TM) i5-2400", cores=4,
                               threads=4, simd="AVX", x86_64_v2=True, memory_total=16 * gb,
                               memory_available=12 * gb, train_threads=4, gflops=60.0)
    window.hardware_changed.emit(window.hardware)
    app.processEvents()
    yield window
    window.model_dirty = False
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()
    common.DIALOGS.clear()


def test_the_train_tab_lists_sizes_by_parameters(studio):
    from PySide6.QtCore import Qt
    tab = studio.train_tab
    tab._update_estimates()
    texts = [tab.size.itemText(i) for i in range(tab.size.count())]
    assert texts and all(" parameters - " in text and " layers, width " in text
                         for text in texts)
    assert not any(text.startswith(f"{tab.size.itemData(i)}:") for i, text in enumerate(texts))
    assert sum("(recommended)" in text for text in texts) == 1
    first = tab.size.itemData(0)
    assert tab.size.itemData(0, Qt.ItemDataRole.ToolTipRole) == \
        f"the {first} preset: auralite train --preset {first}"
