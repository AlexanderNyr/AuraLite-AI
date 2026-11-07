"""AuraLite Studio's main window: menu, tabs, status bar, and the shared state (2.19)."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont, QIcon, QKeySequence,
                           QLinearGradient, QPainter, QPixmap)
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QMenu, QScrollArea,
                               QTabWidget, QWidget)

from ..hardware import Hardware, auralite_home, detect_hardware, format_gb, measure_gflops, \
    process_memory
from ..library import remember_model
from ..logs import install_excepthooks, log_dir, setup_logging
from ..settings import StudioSettings, load_settings, save_settings
from .common import Job, ask_open, ask_save, confirm, inform, show_error
from .tab_chat import ChatTab
from .tab_export import ExportTab
from .tab_finetune import FineTuneTab
from .tab_library import LibraryTab
from .tab_log import LogTab
from .tab_start import StartTab
from .tab_train import TrainTab
from .theme import ACCENT, apply_theme

GUIDE_URL = "https://github.com/AlexanderNyr/AuraLite-AI/blob/main/docs/studio.md"
TAB_TITLES = (("start", "Start"), ("train", "Train"), ("finetune", "Fine-tune"),
              ("chat", "Chat"), ("export", "Export"), ("library", "Library"), ("log", "Log"))


def studio_icon() -> QIcon:
    """The app's icon, drawn (no image file to ship)."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    gradient = QLinearGradient(QPointF(0, 0), QPointF(64, 64))
    gradient.setColorAt(0, QColor(ACCENT))
    gradient.setColorAt(1, QColor("#7b4fd6"))
    painter.setBrush(gradient)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QRectF(2, 2, 60, 60), 14, 14)
    font = QFont()
    font.setPixelSize(40)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor("white"))
    painter.drawText(QRectF(0, 0, 64, 62), Qt.AlignmentFlag.AlignCenter, "A")
    painter.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    """The window; tabs reach the engine, the settings and each other through it."""

    hardware_changed = Signal(object)
    model_changed = Signal()
    busy_changed = Signal(str)
    error_raised = Signal(object)
    runs_changed = Signal()                  # 2.20: a run of the library began or ended

    def __init__(self, engine: Any = None, settings: StudioSettings | None = None, *,
                 detect: bool = True, capture_streams: bool = True,
                 save_on_close: bool = True, auto_speed_test: bool = True):
        super().__init__()
        from auralite import __version__
        if engine is None:
            from auralite.engine import AuraLiteEngine
            engine = AuraLiteEngine()
        self.engine = engine
        self.settings = settings if settings is not None else load_settings()
        self.hardware: Hardware | None = None
        self.busy = ""
        self.model_dirty = False
        self.model_path = ""
        self.model_origin = ""              # where the model in memory came from
        self.save_on_close = save_on_close
        self.auto_speed_test = auto_speed_test
        self._jobs: list[Job] = []
        self.version = __version__
        self.setWindowTitle(f"AuraLite Studio {__version__}")
        self.setWindowIcon(studio_icon())
        self.resize(1080, 800)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.start_tab = StartTab(self)
        self.train_tab = TrainTab(self)
        self.finetune_tab = FineTuneTab(self)
        self.chat_tab = ChatTab(self)
        self.export_tab = ExportTab(self)
        self.library_tab = LibraryTab(self)
        self.log_tab = LogTab(self)
        self.pages: dict[str, QWidget] = {"start": self.start_tab, "train": self.train_tab,
                                          "finetune": self.finetune_tab, "chat": self.chat_tab,
                                          "export": self.export_tab, "library": self.library_tab,
                                          "log": self.log_tab}
        for key, title in TAB_TITLES:
            page = self.pages[key]
            if key in ("chat", "library", "log"):
                self.tabs.addTab(page, title)
                continue
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, title)
        self._build_menu()
        self.device_label = QLabel("")
        self.model_status = QLabel("")
        self.memory_label = QLabel("")
        for label in (self.device_label, self.model_status, self.memory_label):
            label.setContentsMargins(8, 0, 8, 0)
            self.statusBar().addPermanentWidget(label)
        self.memory_timer = QTimer(self)
        self.memory_timer.setInterval(2000)
        self.memory_timer.timeout.connect(self._update_memory)
        self.memory_timer.start()
        self.model_changed.connect(self._update_status)
        self.hardware_changed.connect(self._hardware_status)
        self.error_raised.connect(self._uncaught)
        # 2.20: the library follows the runs; the Start tab offers an interrupted one
        self.runs_changed.connect(self.library_tab.on_runs_changed)
        self.runs_changed.connect(self.start_tab.refresh_interrupted)
        self.busy_changed.connect(lambda _name: self.library_tab._update_run_buttons())
        self.tabs.currentChanged.connect(self._tab_changed)
        self.log_tab.attach(capture_streams)
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, self.settings.dark)
        if self.settings.geometry:
            try:
                self.restoreGeometry(QByteArray.fromHex(self.settings.geometry.encode()))
            except Exception:
                pass
        self._update_status()
        self._update_memory()
        if detect:
            QTimer.singleShot(0, self.detect)

    # ---- menu ----------------------------------------------------------------
    def _action(self, menu: QMenu, text: str, slot: Any, shortcut: str = "") -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self._action(file_menu, "Open a model…", self.open_model_dialog, "Ctrl+O")
        self.save_action = self._action(file_menu, "Save the model as…", self.save_model_dialog,
                                        "Ctrl+S")
        self.recent_menu = file_menu.addMenu("Recent models")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        file_menu.addSeparator()
        self._action(file_menu, "Quit", self.close, "Ctrl+Q")
        view = bar.addMenu("&View")
        self.dark_action = self._action(view, "Dark theme", self.toggle_dark)
        self.dark_action.setCheckable(True)
        self.dark_action.setChecked(self.settings.dark)
        tools = bar.addMenu("&Tools")
        self._action(tools, "Setup wizard…", lambda: self.open_wizard("finetune"), "Ctrl+N")
        self._action(tools, "Speed test", self.run_speed_test)
        self._action(tools, "Library: runs, models, datasets", lambda: self.goto("library"),
                     "Ctrl+L")
        tools.addSeparator()
        self._action(tools, "Classic interface (tkinter)", self.open_classic)
        self._action(tools, "Open the AuraLite folder", lambda: self._open(str(auralite_home())))
        self._action(tools, "Open the log folder", lambda: self._open(str(log_dir())))
        help_menu = bar.addMenu("&Help")
        self._action(help_menu, "Guide", lambda: QDesktopServices.openUrl(QUrl(GUIDE_URL)), "F1")
        self._action(help_menu, "Check for updates…", self.check_updates)
        self._action(help_menu, "About AuraLite Studio", self.about)

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        if not self.settings.recent_models:
            action = self.recent_menu.addAction("(none yet)")
            action.setEnabled(False)
        for path in self.settings.recent_models:
            action = self.recent_menu.addAction(path)
            action.triggered.connect(lambda _c=False, p=path: self.open_model(p))

    def _open(self, path: str) -> None:
        from .common import open_path
        open_path(path)

    # ---- hardware --------------------------------------------------------------
    def detect(self) -> None:
        settings = self.settings

        def work(_job: Job) -> Hardware:
            hw = detect_hardware()
            if settings.gflops and settings.gflops_threads == hw.train_threads:
                hw.gflops = settings.gflops
            elif self.auto_speed_test:
                # one second, once: every time estimate after it is measured
                hw.gflops = measure_gflops(hw.accelerator if hw.accelerator != "cpu" else "cpu")
            return hw

        self._run_background("detect", work, self._detected)

    def _detected(self, hw: Hardware) -> None:
        self.hardware = hw
        device = getattr(self.engine, "device", None)
        if hw.accelerator == "cpu" and getattr(device, "type", "") == "cuda":
            # a GPU this PyTorch has no kernels for (hardware.py names it): train on the CPU
            import torch
            self.engine.device = torch.device("cpu")
        if hw.gflops and hw.gflops != self.settings.gflops:
            self.settings.gflops = float(hw.gflops)
            self.settings.gflops_threads = hw.train_threads
            save_settings(self.settings)
        self._update_status()
        self.hardware_changed.emit(hw)
        if not self.settings.first_run_done and self.isVisible():
            QTimer.singleShot(200, lambda: self.open_wizard("finetune"))

    def run_speed_test(self) -> None:
        if self.hardware is None:
            return
        device = self.hardware.accelerator if self.hardware.accelerator != "cpu" else "cpu"
        self.statusBar().showMessage("Speed test…")

        def work(_job: Job) -> float:
            return measure_gflops(device)

        self._run_background("speed-test", work, self._measured)

    def _measured(self, gflops: float) -> None:
        if self.hardware is None:
            return
        self.hardware.gflops = float(gflops)
        self.settings.gflops = float(gflops)
        self.settings.gflops_threads = self.hardware.train_threads
        save_settings(self.settings)
        self.statusBar().showMessage(f"Speed test: {gflops:,.0f} GFLOP/s", 6000)
        self.hardware_changed.emit(self.hardware)

    def _run_background(self, name: str, fn: Any, done: Any) -> Job:
        job = Job(name, fn, self)
        job.finished.connect(done)
        job.failed.connect(lambda exc: self.report_error(exc, name.replace("-", " ")))
        job.finished.connect(lambda _r, j=job: self._forget(j))
        job.failed.connect(lambda _e, j=job: self._forget(j))
        self._jobs.append(job)
        return job.start()

    def _forget(self, job: Job) -> None:
        if job in self._jobs:
            self._jobs.remove(job)

    # ---- shared state ------------------------------------------------------------
    def goto(self, name: str) -> None:
        index = [key for key, _ in TAB_TITLES].index(name)
        self.tabs.setCurrentIndex(index)

    def _tab_changed(self, index: int) -> None:
        if TAB_TITLES[index][0] == "library":
            self.library_tab.refresh()

    def begin(self, name: str, quiet: bool = False) -> bool:
        """Claim the engine for `name` (training, an export, a reply); False when busy."""
        if self.busy:
            if not quiet:
                inform(self, "AuraLite is busy", f"AuraLite is busy with {self.busy}: wait "
                                                 f"until it ends, or stop it first.")
            return False
        self.busy = name
        self.save_action.setEnabled(False)
        self.busy_changed.emit(name)
        return True

    def end(self, model_dirty: bool | None = None) -> None:
        self.busy = ""
        self.save_action.setEnabled(True)
        if model_dirty is not None:
            self.model_dirty = bool(model_dirty)
        self.busy_changed.emit("")
        self.model_changed.emit()

    def confirm_replace_model(self, text: str) -> bool:
        if self.engine.model is None or not self.model_dirty:
            return True
        return confirm(self, "Replace the model in memory?",
                       f"{text}\n\nThe model in memory has changes that are not saved. "
                       f"Continue without saving them?")

    def model_description(self) -> str:
        engine = self.engine
        model = engine.model
        if model is None:
            return "no model"
        if engine.is_gguf_model():
            return f"GGUF model {os.path.basename(engine.gguf_path or '')} (chat only)"
        if engine.is_hf_model():
            return f"transformers model {engine.hf_path or ''} (chat only here)"
        native = getattr(model, "module", model)
        parts = []
        used = getattr(engine, "params_used", {}) or {}
        source = self.model_origin or str(used.get("imported_from") or "").removeprefix("hf:")
        parts.append(source or (os.path.basename(self.model_path) if self.model_path
                                else "trained here"))
        try:
            parts.append(f"{_count(native.count_parameters())} parameters")
        except Exception:
            pass
        try:
            from auralite.engine.nf4 import nf4_layers
            if nf4_layers(native):
                parts.append("4-bit base")
        except Exception:
            pass
        try:
            if engine.has_lora():
                parts.append(f"LoRA adapters (rank {native.lora_rank})"
                             if getattr(native, "lora_rank", 0) else "LoRA adapters")
        except Exception:
            pass
        if self.model_dirty:
            parts.append("not saved")
        return " · ".join(parts)

    def _update_status(self) -> None:
        hw = self.hardware
        if hw is not None:
            self.device_label.setText(f"GPU: {hw.gpu}" if hw.accelerator != "cpu"
                                      else f"CPU · {hw.train_threads} threads")
        self.model_status.setText(f"Model: {self.model_description()}")

    def _hardware_status(self, _hw: Any) -> None:
        self._update_status()
        self._update_memory()

    def _update_memory(self) -> None:
        used = process_memory()
        total = self.hardware.memory_total if self.hardware is not None else 0
        self.memory_label.setText(f"RAM: {format_gb(used)}" + (f" of {format_gb(total)}"
                                                               if total else ""))

    # ---- models --------------------------------------------------------------------
    def open_model_dialog(self) -> None:
        path = ask_open(self, "Open a model", "AuraLite models (*.pt);;GGUF (*.gguf);;"
                                              "All files (*)", self.settings.last_model_dir)
        if path:
            self.open_model(path)

    def open_model(self, path: str, origin: str = "", then: str = "") -> None:
        """Load a model file; `origin` names it in the status bar (2.20: a run of
        the library), `then` is the tab to open once it is loaded."""
        if not os.path.exists(path):
            show_error(self, FileNotFoundError(f"not found: {path}"))
            return
        if not self.confirm_replace_model("Opening a model replaces the one in memory."):
            return
        if not self.begin("opening a model"):
            return
        engine = self.engine
        self.statusBar().showMessage(f"Opening {os.path.basename(path)}…")
        def work(_job: Job) -> Any:
            engine.unload_model()             # 2.20: one model in memory, not two
            return engine.load_model(path)

        job = Job("open-model", work, self)
        job.finished.connect(lambda _r: self._opened(path, origin, then))
        job.failed.connect(self._open_failed)
        self._jobs.append(job)
        job.start()

    def _opened(self, path: str, origin: str = "", then: str = "") -> None:
        self.model_loaded(path)
        if origin:
            self.model_origin = origin
        self.end(model_dirty=False)
        self.statusBar().showMessage(f"Opened {origin or path}", 6000)
        if then:
            self.goto(then)

    def import_model(self, spec: str, then: str = "") -> None:
        """Import a Hugging Face model into memory (2.20: a download of the library,
        which needs no network the second time)."""
        if not self.confirm_replace_model("Importing a model replaces the one in memory."):
            return
        if not self.begin("importing a model"):
            return
        engine = self.engine
        self.statusBar().showMessage(f"Importing {spec}…")
        def work(_job: Job) -> Any:
            engine.unload_model()
            return engine.import_hf_model(spec)

        job = Job("import-model", work, self)
        job.finished.connect(lambda _r: self._opened(spec, "", then))
        job.failed.connect(self._open_failed)
        self._jobs.append(job)
        job.start()

    def _open_failed(self, exc: BaseException) -> None:
        self.end(model_dirty=None)
        self.report_error(exc, "opening the model")

    def model_loaded(self, spec: str) -> None:
        text = str(spec)
        self.model_origin = (text[3:] if text.lower().startswith("hf:")
                             else os.path.basename(os.path.normpath(text)))
        if os.path.isfile(spec) and spec.endswith((".pt", ".gguf")):
            self.model_path = os.path.abspath(spec)
            self.settings.add_recent(spec)
            self.settings.last_model_dir = os.path.dirname(self.model_path)
            save_settings(self.settings)
            self.start_tab.refresh_recent()
        else:
            self.model_path = ""
        self.model_changed.emit()

    def model_unloaded(self) -> None:
        """The engine let the model in memory go (a new one is loading)."""
        self.model_origin = ""
        self.model_path = ""
        self.model_changed.emit()

    def model_trained(self, how: str) -> None:
        """After a run: a new model, or the base fine-tuned."""
        if how == "new":
            self.model_origin = "new model"
        elif self.model_origin and not self.model_origin.endswith("(fine-tuned)"):
            self.model_origin += " (fine-tuned)"
        self.model_path = ""

    def model_saved(self, path: str) -> None:
        self.model_path = os.path.abspath(path)
        self.model_origin = os.path.basename(path)
        self.settings.add_recent(path)
        self.settings.last_model_dir = os.path.dirname(self.model_path)
        save_settings(self.settings)
        self.start_tab.refresh_recent()
        remember_model(self.model_path, "saved")

    # ---- the run library (2.20) ------------------------------------------------------
    def run_hardware(self) -> dict[str, Any]:
        """This computer, as a run's record keeps it."""
        from auralite.runs import machine_summary
        info = machine_summary(self.engine)
        hw = self.hardware
        if hw is not None:
            info.update(cpu=hw.cpu or info.get("cpu", ""), cores=hw.cores, threads=hw.threads,
                        memory_gb=round(hw.memory_total / (1 << 30), 1) if hw.memory_total
                        else None, accelerator=hw.accelerator, system=hw.system)
        return info

    def run_saved(self, run: Any) -> bool:
        """After a recorded run: the model in memory is the run's result, kept in
        the library (nothing is lost on closing). True when it was saved."""
        path = run.result_path
        if path is None or run.record.get("save_error"):
            return False
        self.model_path = str(path)
        self.model_origin = run.name
        return True

    def save_model_dialog(self) -> None:
        if self.engine.model is None:
            return
        start = self.model_path or os.path.join(self.settings.last_model_dir or
                                                os.path.expanduser("~"), "my-model.pt")
        path = ask_save(self, "Save the model as", "AuraLite models (*.pt)", start)
        if not path:
            return
        if not path.endswith(".pt"):
            path += ".pt"
        if not self.begin("saving the model"):
            return
        engine = self.engine
        job = Job("save-model", lambda _job: engine.save_model(path), self)
        job.finished.connect(lambda _r: self._saved(path))
        job.failed.connect(self._save_failed)
        self._jobs.append(job)
        job.start()

    def _saved(self, path: str) -> None:
        self.model_saved(path)
        self.end(model_dirty=False)
        self.statusBar().showMessage(f"Saved {path}", 6000)

    def _save_failed(self, exc: BaseException) -> None:
        self.end(model_dirty=None)
        self.report_error(exc, "saving the model")

    # ---- wizard, errors, help ------------------------------------------------------
    def open_wizard(self, goal: str = "finetune") -> None:
        from .wizard import SetupWizard
        if self.busy:
            inform(self, "AuraLite is busy", f"AuraLite is busy with {self.busy}.")
            return
        wizard = SetupWizard(self, goal)
        try:
            accepted = wizard.exec()
            self.settings.first_run_done = True
            save_settings(self.settings)
            if accepted:
                self.apply_plan(wizard.plan, wizard.data_page.report())
        finally:
            wizard.deleteLater()

    def apply_plan(self, plan: dict[str, Any], report: Any = None) -> None:
        """Open the tab of the wizard's plan, filled in, and start it if asked."""
        goal = plan.get("goal", "finetune")
        tab: Any = self.train_tab if goal == "pretrain" else self.finetune_tab
        if goal == "pretrain":
            self.train_tab.apply_plan(plan.get("settings") or {})
        else:
            self.finetune_tab.apply_plan({k: v for k, v in plan.items() if k != "data"})
        if report is not None:
            tab.data.set_report(report)
        elif plan.get("data"):
            tab.data.set_path(str(plan["data"]))
        self.goto("train" if goal == "pretrain" else "finetune")
        if plan.get("start"):
            QTimer.singleShot(100, tab.start)

    def report_error(self, exc: BaseException, action: str = "") -> None:
        show_error(self, exc, action)

    def _uncaught(self, exc: BaseException) -> None:
        show_error(self, exc)

    def toggle_dark(self) -> None:
        self.settings.dark = self.dark_action.isChecked()
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, self.settings.dark)
        save_settings(self.settings)
        self.chat_tab.render()

    def open_classic(self) -> None:
        try:
            subprocess.Popen([sys.executable, "-m", "auralite.gui"],
                             close_fds=True, start_new_session=sys.platform != "win32")
        except OSError as err:
            self.report_error(err, "starting the classic interface")

    def check_updates(self) -> None:
        from ..updates import check_for_update
        self.statusBar().showMessage("Checking for updates…")
        self._run_background("update-check", lambda _job: check_for_update(self.version),
                             lambda info: inform(self, "Updates", info.message()))

    def about(self) -> None:
        import torch
        try:
            import PySide6
            qt = PySide6.__version__
        except Exception:
            qt = "?"
        inform(self, "About AuraLite Studio",
               f"AuraLite Studio {self.version}\n\n"
               f"Train, fine-tune and run language models on your own computer.\n\n"
               f"Python {platform.python_version()} · PyTorch {torch.__version__} · "
               f"Qt (PySide6) {qt}\nLicense: Apache-2.0\n"
               f"https://github.com/AlexanderNyr/AuraLite-AI\n\n"
               f"Models: {auralite_home()}\nLogs: {log_dir()}")

    # ---- closing ---------------------------------------------------------------------
    def running_jobs(self) -> list[Job]:
        jobs = [tab.job for tab in (self.train_tab, self.finetune_tab, self.chat_tab,
                                    self.export_tab) if getattr(tab, "job", None) is not None]
        if self.finetune_tab.distill_job is not None:
            jobs.append(self.finetune_tab.distill_job)
        return [j for j in jobs if j is not None and j.running()]

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        running = self.running_jobs()
        if running and not confirm(self, "Quit AuraLite Studio?",
                                   f"AuraLite is busy with {self.busy or 'work'}. Stop it and "
                                   f"quit? (A model being trained is not saved.)"):
            event.ignore()
            return
        if not running and self.engine.model is not None and self.model_dirty and \
                not confirm(self, "Quit AuraLite Studio?",
                            "The model in memory is not saved. Quit anyway?"):
            event.ignore()
            return
        for job in running:
            job.stop_event.set()
        for job in running:
            if job.thread is not None:
                job.thread.join(timeout=10)
        if self.save_on_close:
            self.settings.geometry = bytes(self.saveGeometry().toHex().data()).decode()
            save_settings(self.settings)
        self.memory_timer.stop()
        self.log_tab.detach()
        event.accept()


def _count(n: int) -> str:
    """``"1.5B"``, ``"494M"``, ``"30K"``."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(n)


def run(argv: list[str] | None = None) -> int:
    """Start the Studio (the window and Qt's event loop); returns the exit code."""
    import multiprocessing
    multiprocessing.freeze_support()          # tokenizer workers (spawn) in a packaged app
    if sys.platform == "win32":
        try:                                  # its own taskbar button and icon on Windows
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(  # type: ignore[attr-defined]
                "AuraLite.Studio")
        except Exception:
            pass
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("AuraLite Studio")
    app.setOrganizationName("AuraLite")
    setup_logging("studio")
    window = MainWindow()
    install_excepthooks(window.error_raised.emit)
    window.show()
    return int(app.exec())
