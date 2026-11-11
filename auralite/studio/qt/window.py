"""AuraLite Studio's main window: menu, tabs, status bar, and the shared state (2.19)."""
from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from typing import Any

from PySide6.QtCore import QByteArray, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QIcon, QKeySequence
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QMainWindow, QMenu,
                               QScrollArea, QTabWidget, QWidget)

from ... import bundle
from ..hardware import Hardware, auralite_home, detect_hardware, format_gb, measure_gflops, \
    process_memory
from ..library import remember_model
from ..logs import install_excepthooks, log_dir, setup_logging
from ..settings import StudioSettings, load_settings, save_settings
from .common import Job, ScrollPage, ask_open, ask_save, confirm, inform, show_error
from .tab_chat import ChatTab
from .tab_evaluate import EvaluateTab
from .tab_export import ExportTab
from .tab_finetune import FineTuneTab
from .tab_generate import GenerateTab
from .tab_library import LibraryTab
from .tab_log import LogTab
from .tab_images import ImagesTab
from .tab_model import ModelTab
from .tab_quantize import QuantizeTab
from .tab_start import StartTab
from .tab_train import TrainTab
from .theme import apply_theme, icon_pixmap
from ..i18n import LANGUAGES, N_, language, resolve_language, set_language, tr

log = logging.getLogger("auralite.studio")

GUIDE_URL = "https://github.com/AlexanderNyr/AuraLite-AI/blob/main/docs/studio.md"
TAB_TITLES = (("start", N_("Start")), ("train", N_("Train")), ("finetune", N_("Fine-tune")),
              ("chat", N_("Chat")), ("generate", N_("Generate")), ("images", N_("Images")),
              ("model", N_("Model")), ("quantize", N_("Quantize")), ("evaluate", N_("Evaluate")),
              ("export", N_("Export")), ("library", N_("Library")), ("log", N_("Log")))


def studio_icon() -> QIcon:
    """The app's icon, from 16 to 256 pixels."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(icon_pixmap(size))
    return icon


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
        self.bench_job: Job | None = None                   # 2.22: Tools -> Benchmark
        self.bench_choice: Any = None                       # 2.23: its last settings
        self.version = __version__
        self.setWindowTitle(f"AuraLite Studio {__version__}")
        self.setWindowIcon(studio_icon())
        self.resize(*self._initial_size())             # 2.28.1: within the screen

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.start_tab = StartTab(self)
        self.train_tab = TrainTab(self)
        self.finetune_tab = FineTuneTab(self)
        self.chat_tab = ChatTab(self)
        self.generate_tab = GenerateTab(self)          # 2.24: the classic Generate tab
        self.images_tab = ImagesTab(self)              # 2.26: the classic Image tab
        self.model_tab = ModelTab(self)                # 2.25: the classic Model, Quant and
        self.quantize_tab = QuantizeTab(self)          # Eval tabs
        self.evaluate_tab = EvaluateTab(self)
        self.export_tab = ExportTab(self)
        self.library_tab = LibraryTab(self)
        self.log_tab = LogTab(self)
        self.pages: dict[str, QWidget] = {"start": self.start_tab, "train": self.train_tab,
                                          "finetune": self.finetune_tab, "chat": self.chat_tab,
                                          "generate": self.generate_tab,
                                          "images": self.images_tab,
                                          "model": self.model_tab,
                                          "quantize": self.quantize_tab,
                                          "evaluate": self.evaluate_tab,
                                          "export": self.export_tab, "library": self.library_tab,
                                          "log": self.log_tab}
        # 2.28.1: every tab scrolls. Chat, Generate, Library and Log did not: Generate's
        # 849 px made the window at least 929 px tall, taller than a 768 or 800 px screen, so
        # the bottom of every tab - scrolled ones too - lay below the screen, out of reach.
        for key, title in TAB_TITLES:
            self.tabs.addTab(ScrollPage(self.pages[key]), tr(title))
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
        self._fit_to_tabs()                              # 2.28: a saved size included
        if self.settings.last_tab in self.pages:         # 2.24: where the last session was
            self.goto(self.settings.last_tab)
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
        file_menu = bar.addMenu(tr("&File"))
        self._action(file_menu, tr("Open a model…"), self.open_model_dialog, "Ctrl+O")
        self.save_action = self._action(file_menu, tr("Save the model as…"), self.save_model_dialog,
                                        "Ctrl+S")
        self.recent_menu = file_menu.addMenu(tr("Recent models"))
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        file_menu.addSeparator()
        self._action(file_menu, tr("Quit"), self.close, "Ctrl+Q")
        view = bar.addMenu(tr("&View"))
        self.dark_action = self._action(view, tr("Dark theme"), self.toggle_dark)
        self.dark_action.setCheckable(True)
        self.dark_action.setChecked(self.settings.dark)
        # 2.28: the interface's language - saved, and spoken from the next start
        self.language_menu = view.addMenu(tr("Language"))
        self.language_group = QActionGroup(self)
        self.language_group.setExclusive(True)
        self.language_actions: dict[str, QAction] = {}
        for code, name in (("", tr("Automatic (the computer's language)")), *LANGUAGES.items()):
            action = QAction(name, self)
            action.setCheckable(True)
            action.setChecked(code == (self.settings.language or ""))
            action.triggered.connect(lambda _checked=False, c=code: self.choose_language(c))
            self.language_group.addAction(action)
            self.language_menu.addAction(action)
            self.language_actions[code] = action
        tools = bar.addMenu(tr("&Tools"))
        self._action(tools, tr("Setup wizard…"), lambda: self.open_wizard("finetune"), "Ctrl+N")
        self._action(tools, tr("Speed test"), self.run_speed_test)
        # 2.22: auralite bench in the window
        self.bench_action = self._action(tools, tr("Benchmark…"), self.run_benchmark)
        self._action(tools, tr("Library: runs, models, datasets"), lambda: self.goto("library"),
                     "Ctrl+L")
        tools.addSeparator()
        if not bundle.packaged():             # 2.21: the app leaves the classic GUI out
            self._action(tools, tr("Classic interface (tkinter)"), self.open_classic)
        self._action(tools, tr("Open the AuraLite folder"), lambda: self._open(str(auralite_home())))
        self._action(tools, tr("Open the log folder"), lambda: self._open(str(log_dir())))
        help_menu = bar.addMenu(tr("&Help"))
        self._action(help_menu, tr("Guide"), lambda: QDesktopServices.openUrl(QUrl(GUIDE_URL)), "F1")
        self._action(help_menu, tr("Check for updates…"), self.check_updates)
        self._action(help_menu, tr("About AuraLite Studio"), self.about)

    def choose_language(self, code: str) -> None:
        """View -> Language (2.28): the choice is saved and the Studio speaks it from the next
        start - now, after a restart, if the user wants."""
        code = code if code in LANGUAGES else ""
        if code == (self.settings.language or ""):
            return
        self.settings.language = code
        save_settings(self.settings)
        action = self.language_actions.get(code)
        if action is not None:
            action.setChecked(True)
        if resolve_language(code or None) == language():
            return
        if confirm(self, tr("Language"),
                   tr("AuraLite Studio speaks the new language after a restart.") + "\n\n"
                   + tr("Restart now?")):
            self.restart()

    def restart(self) -> bool:
        """Close the window - asking what closing asks - and start the Studio again."""
        from ..app import restart_command
        if not self.close():
            return False
        options: dict[str, Any] = ({"creationflags": getattr(subprocess, "DETACHED_PROCESS", 0)
                                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
                                   if sys.platform == "win32" else {"start_new_session": True})
        try:
            subprocess.Popen(restart_command(), stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             close_fds=True, **options)
        except OSError as exc:
            logging.getLogger("auralite.studio").warning("could not start the Studio again: %s",
                                                         exc)
            return False
        return True

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        if not self.settings.recent_models:
            action = self.recent_menu.addAction(tr("(none yet)"))
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
        self.statusBar().showMessage(tr("Speed test…"))

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
        self.statusBar().showMessage(tr("Speed test: {gflops:,.0f} GFLOP/s", gflops=gflops), 6000)
        self.hardware_changed.emit(self.hardware)

    def run_benchmark(self) -> None:
        """2.22: *Tools -> Benchmark*: ``auralite bench`` - how fast this computer trains and
        writes three sizes of a new model; the report goes to ~/.auralite/bench. While it
        runs the menu item stops it. 2.23: a dialog first - the seconds per size and, if
        asked, a fine-tune of a catalog model or of one already downloaded (a folder or a
        checkpoint)."""
        from auralite import bench

        from .bench_dialog import choose_benchmark, describe
        if self.bench_job is not None:
            self.bench_job.stop_event.set()
            self.statusBar().showMessage(tr("Stopping the benchmark…"))
            return
        sizes = list(bench.SIZES)
        choice = choose_benchmark(self, self.bench_choice)
        if choice is None:
            return
        self.bench_choice = choice
        if not self.begin(N_("benchmark")):
            return
        log.info("[AuraLite] bench: %s", describe(choice))

        def work(job: Job) -> Any:
            report = bench.run_bench(sizes=sizes, seconds=choice.seconds,
                                     finetune=choice.finetune, method=choice.method,
                                     progress=job.progress.emit, stop=job.stop_event)
            return report, bench.save_report(report)

        job = Job("benchmark", work, self)
        job.progress.connect(self._bench_progress)
        job.finished.connect(self._benchmarked)
        job.failed.connect(self._bench_failed)
        self.bench_job = job
        self.bench_action.setText(tr("Stop the benchmark"))
        job.start()

    def _bench_progress(self, line: Any) -> None:
        text = str(line).strip()
        self.statusBar().showMessage(tr("Benchmark: {line}", line=text))
        log.info("[AuraLite] bench: %s", text)

    def _bench_over(self) -> None:
        self.bench_job = None
        self.bench_action.setText(tr("Benchmark…"))
        self.statusBar().clearMessage()
        self.end()

    def _benchmarked(self, result: Any) -> None:
        from auralite import bench
        report, (md, _js) = result
        self._bench_over()
        inform(self, tr("Benchmark"), "\n".join(bench.summary_lines(report)) + "\n\n"
               + tr("The report, a table for an article or a README: {path}", path=md))

    def _bench_failed(self, exc: BaseException) -> None:
        self._bench_over()
        if isinstance(exc, InterruptedError):
            self.statusBar().showMessage(tr("The benchmark was stopped"), 6000)
            return
        self.report_error(exc, tr("benchmark"))

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
    def _initial_size(self) -> tuple[int, int]:
        """2.28.1: 1080 x 800, smaller on a screen that has less room - 800 px and the title
        bar did not fit a 768 or 800 px screen."""
        width, height = 1080, 800
        try:
            screen = self.screen()
            area = screen.availableGeometry() if screen is not None else None
        except Exception:                            # pragma: no cover - an exotic platform
            area = None
        if area is not None and area.width() > 0 and area.height() > 0:
            width = min(width, area.width() - 40)
            height = min(height, area.height() - 60)    # the title bar and a margin
        return max(640, width), max(480, height)

    def current_page(self) -> QWidget:
        """2.28.1: the page of the tab shown (the tabs hold scroll areas)."""
        area = self.tabs.currentWidget()
        return area.widget() if isinstance(area, QScrollArea) else area

    def _fit_to_tabs(self) -> None:
        """2.28: a window at least as wide as its row of tabs when the screen has room - the
        Russian titles are longer than the English ones 1080 was chosen for, and a tab behind
        the scroll arrows is easy to miss. A maximized window stays as it is."""
        if self.isMaximized() or self.isFullScreen():
            return
        try:
            bar = self.tabs.tabBar()
            bar.ensurePolished()                     # the theme's padding before it is shown
            need = int(bar.sizeHint().width()) + 16
            screen = self.screen()
            area = screen.availableGeometry() if screen is not None else None
        except Exception:                            # pragma: no cover - an exotic platform
            return
        if area is None:
            return
        wanted = min(need, area.width() - 40)
        if wanted <= self.width():
            return
        self.resize(wanted, self.height())
        if self.x() + wanted > area.right():         # a place saved further right
            self.move(max(area.left(), area.right() - wanted), self.y())

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
                inform(self, tr("AuraLite is busy"),
                       tr("AuraLite is busy with {busy}: wait until it ends, or stop it first.",
                          busy=tr(self.busy)))
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
        return confirm(self, tr("Replace the model in memory?"),
                       text + "\n\n" + tr("The model in memory has changes that are not saved. "
                                           "Continue without saving them?"))

    def model_description(self) -> str:
        engine = self.engine
        model = engine.model
        if model is None:
            return tr("no model")
        if engine.is_gguf_model():
            return tr("GGUF model {name} (chat only)",
                      name=os.path.basename(engine.gguf_path or ""))
        if engine.is_hf_model():
            if getattr(engine.hf_proxy, "is_peft", False):
                return tr("transformers model {path} with LoRA adapters", path=engine.hf_path or "")
            return tr("transformers model {path}", path=engine.hf_path or "")
        native = getattr(model, "module", model)
        parts = []
        used = getattr(engine, "params_used", {}) or {}
        source = self.model_origin or str(used.get("imported_from") or "").removeprefix("hf:")
        parts.append(origin_text(source) if source else (
            os.path.basename(self.model_path) if self.model_path else tr("trained here")))
        try:
            parts.append(tr("{count} parameters", count=_count(native.count_parameters())))
        except Exception:
            pass
        try:
            from auralite.engine.nf4 import nf4_layers
            if nf4_layers(native):
                parts.append(tr("4-bit base"))
        except Exception:
            pass
        try:
            if engine.has_lora():
                parts.append(tr("LoRA adapters (rank {rank})", rank=native.lora_rank)
                             if getattr(native, "lora_rank", 0) else tr("LoRA adapters"))
        except Exception:
            pass
        if self.model_dirty:
            parts.append(tr("not saved"))
        return " · ".join(parts)

    def _update_status(self) -> None:
        hw = self.hardware
        if hw is not None:
            self.device_label.setText(f"GPU: {hw.gpu}" if hw.accelerator != "cpu"
                                      else tr("CPU · {threads} threads", threads=hw.train_threads))
        self.model_status.setText(tr("Model: {model}", model=self.model_description()))

    def _hardware_status(self, _hw: Any) -> None:
        self._update_status()
        self._update_memory()

    def _update_memory(self) -> None:
        used = process_memory()
        total = self.hardware.memory_total if self.hardware is not None else 0
        self.memory_label.setText(tr("RAM: {used} of {total}", used=format_gb(used),
                                     total=format_gb(total)) if total else
                                  tr("RAM: {used}", used=format_gb(used)))

    # ---- models --------------------------------------------------------------------
    def open_model_dialog(self) -> None:
        path = ask_open(self, tr("Open a model"), tr("AuraLite models (*.pt);;GGUF (*.gguf);;"
                                              "All files (*)"), self.settings.last_model_dir)
        if path:
            self.open_model(path)

    def open_model(self, path: str, origin: str = "", then: str = "") -> None:
        """Load a model file; `origin` names it in the status bar (2.20: a run of
        the library), `then` is the tab to open once it is loaded."""
        if not os.path.exists(path):
            show_error(self, FileNotFoundError(tr("not found: {path}", path=path)))
            return
        if not self.confirm_replace_model(tr("Opening a model replaces the one in memory.")):
            return
        if not self.begin(N_("opening a model")):
            return
        engine = self.engine
        self.statusBar().showMessage(tr("Opening {name}…", name=os.path.basename(path)))
        gguf = path.lower().endswith(".gguf")
        options = self.model_tab.gguf_options() if gguf else {}   # 2.25: llama.cpp's settings

        def work(_job: Job) -> Any:
            engine.unload_model()             # 2.20: one model in memory, not two
            if gguf:
                return engine.load_gguf_model(path, **options)
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
        self.statusBar().showMessage(tr("Opened {name}", name=origin or path), 6000)
        if then:
            self.goto(then)

    def import_model(self, spec: str, then: str = "") -> None:
        """Import a Hugging Face model into memory (2.20: a download of the library,
        which needs no network the second time)."""
        if not self.confirm_replace_model(tr("Importing a model replaces the one in memory.")):
            return
        if not self.begin(N_("importing a model")):
            return
        engine = self.engine
        self.statusBar().showMessage(tr("Importing {spec}…", spec=spec))
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
        self.report_error(exc, tr("opening the model"))

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
        """After a run: a new model, the model trained further (2.24), or the base
        fine-tuned."""
        if how == "new":
            self.model_origin = N_("new model")
        elif how == "more":
            if self.model_origin and not self.model_origin.endswith("(trained further)"):
                self.model_origin += " " + N_("(trained further)")
        elif self.model_origin and not self.model_origin.endswith("(fine-tuned)"):
            self.model_origin += " " + N_("(fine-tuned)")
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
        from ..runner import machine_record
        return machine_record(self.engine, self.hardware)

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
        path = ask_save(self, tr("Save the model as"), tr("AuraLite models (*.pt)"), start)
        if not path:
            return
        if not path.endswith(".pt"):
            path += ".pt"
        if not self.begin(N_("saving the model")):
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
        self.statusBar().showMessage(tr("Saved {path}", path=path), 6000)

    def _save_failed(self, exc: BaseException) -> None:
        self.end(model_dirty=None)
        self.report_error(exc, tr("saving the model"))

    # ---- wizard, errors, help ------------------------------------------------------
    def open_wizard(self, goal: str = "finetune") -> None:
        from .wizard import SetupWizard
        if self.busy:
            inform(self, tr("AuraLite is busy"),
                   tr("AuraLite is busy with {busy}.", busy=tr(self.busy)))
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
            self.report_error(err, tr("starting the classic interface"))

    def check_updates(self) -> None:
        from ..updates import check_for_update
        self.statusBar().showMessage(tr("Checking for updates…"))
        self._run_background("update-check", lambda _job: check_for_update(self.version),
                             lambda info: inform(self, tr("Updates"), info.message()))

    def about(self) -> None:
        import torch
        try:
            import PySide6
            qt = PySide6.__version__
        except Exception:
            qt = "?"
        app = ("\n" + tr("This is {app}", app=bundle.describe()) + "\n") if bundle.packaged() \
            else ""
        inform(self, tr("About AuraLite Studio"),
               f"AuraLite Studio {self.version}\n\n"
               + tr("Train, fine-tune and run language models on your own computer.") + "\n\n"
               + f"Python {platform.python_version()} · PyTorch {torch.__version__} · "
               f"Qt (PySide6) {qt}\n" + tr("License: {license}", license="Apache-2.0") + "\n"
               "https://github.com/AlexanderNyr/AuraLite-AI\n" + app + "\n"
               + tr("Models: {path}", path=auralite_home()) + "\n"
               + tr("Logs: {path}", path=log_dir()))

    # ---- closing ---------------------------------------------------------------------
    def running_jobs(self) -> list[Job]:
        jobs = [tab.job for tab in (self.train_tab, self.finetune_tab, self.chat_tab,
                                    self.generate_tab, self.images_tab, self.model_tab,
                                    self.quantize_tab, self.evaluate_tab, self.export_tab)
                if getattr(tab, "job", None) is not None]
        for extra in (self.finetune_tab.distill_job, self.train_tab.worker_job):
            if extra is not None:
                jobs.append(extra)
        return [j for j in jobs if j is not None and j.running()]

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        running = self.running_jobs()
        if running and not confirm(self, tr("Quit AuraLite Studio?"),
                                   tr("AuraLite is busy with {busy}. Stop it and quit? (A model "
                                      "being trained is not saved.)",
                                      busy=tr(self.busy or N_("work")))):
            event.ignore()
            return
        if not running and self.engine.model is not None and self.model_dirty and \
                not confirm(self, tr("Quit AuraLite Studio?"),
                            tr("The model in memory is not saved. Quit anyway?")):
            event.ignore()
            return
        if not running and self.images_tab.unsaved and self.images_tab.has_model() and \
                not confirm(self, tr("Quit AuraLite Studio?"),
                            tr("The image generator in memory is not saved. Quit anyway?")):
            event.ignore()
            return
        for job in running:
            job.stop_event.set()
        for job in running:
            if job.thread is not None:
                job.thread.join(timeout=10)
        if self.save_on_close:
            self.settings.geometry = bytes(self.saveGeometry().toHex().data()).decode()
            self.settings.last_tab = TAB_TITLES[self.tabs.currentIndex()][0] \
                if 0 <= self.tabs.currentIndex() < len(TAB_TITLES) else ""
            save_settings(self.settings)
        self.memory_timer.stop()
        self.log_tab.detach()
        event.accept()


def origin_text(origin: str) -> str:
    """2.28: where the model came from, in the interface's language. It is kept in English:
    run records and the library note it."""
    text = tr(origin) if origin == "new model" else origin
    for suffix in ("(trained further)", "(fine-tuned)"):
        text = text.replace(suffix, tr(suffix))
    return text


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
    settings = load_settings()
    set_language(settings.language or None)        # 2.28: before a word is shown
    install_qt_translations(app)
    window = MainWindow(settings=settings)
    install_excepthooks(window.error_raised.emit)
    window.show()
    log = logging.getLogger("auralite.studio")
    log.info("AuraLite Studio %s: the window is open (%s)", window.version,
             bundle.describe() if bundle.packaged() else sys.executable)
    seconds = _smoke_seconds()
    if seconds:
        # 2.21: CI starts the app's program the way a user does and needs it to close
        log.info("%s=%s: closing the window in %s s", SMOKE_ENV, seconds, seconds)
        QTimer.singleShot(int(seconds * 1000), lambda: _smoke_close(window))
    return int(app.exec())


def install_qt_translations(app: QApplication) -> Any:
    """2.28: Qt's own words - the buttons of its dialogs (Yes, No, Cancel) - in the
    interface's language, when Qt brings them. Returns the translator (None: English)."""
    code = language()
    if code == "en":
        return None
    from PySide6.QtCore import QLibraryInfo, QTranslator
    folders = [QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)]
    try:
        import PySide6
        base = os.path.dirname(PySide6.__file__ or "")
        folders += [os.path.join(base, "Qt", "translations"), os.path.join(base, "translations")]
    except Exception:                         # pragma: no cover - PySide6 is there
        pass
    translator = QTranslator(app)
    for folder in folders:
        if folder and translator.load(f"qtbase_{code}", folder):
            app.installTranslator(translator)
            return translator
    return None


#: 2.21: close the window this many seconds after it opened (a smoke test of a build).
SMOKE_ENV = "AURALITE_STUDIO_SMOKE_SECONDS"


def _smoke_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get(SMOKE_ENV, "") or 0))
    except ValueError:
        return 0.0


def _smoke_close(window: MainWindow) -> None:
    """Close what is open - the setup wizard of a first start is a modal dialog - then the
    window; quit for sure a little later."""
    logging.getLogger("auralite.studio").info("%s: closing the window now", SMOKE_ENV)
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QDialog) and widget.isVisible():
            widget.reject()
    window.close()
    QTimer.singleShot(5000, window, QApplication.quit)     # gone with the window
