"""gui/menu.py - the classic interface's menu bar (2.25: from AuraLite Studio).

File (open, save, recent models, quit), View (the dark theme, remembered), Tools (the
benchmark, the speed test, the AuraLite folder, AuraLite Studio) and Help (the guide,
updates, about). The footer shows the memory the app uses and the last speed test.
The benchmark's choice is the Studio's (:mod:`auralite.studio.benchchoice`).
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
import webbrowser
from typing import Any

from .common import messagebox, tk, ttk

GUIDE_URL = "https://github.com/AlexanderNyr/AuraLite-AI/blob/main/docs/classic.md"
#: How often the footer's memory figure is refreshed (milliseconds).
MEMORY_EVERY_MS = 5000


def open_folder(path: str) -> bool:
    """Show a folder in the system's file manager (made when missing)."""
    if not path:
        return False
    if os.path.isfile(path):
        path = os.path.dirname(path) or path
    os.makedirs(path, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(path)                                # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except (OSError, AttributeError):
        return False
    return True


class MenuMixin:
    """The menu bar of `AIApp` (needs `root`, `engine`, `gui_config`, `dark_mode`,
    `load_model`, `save_model`, `_toggle_dark_mode`, `_on_close`)."""

    root: Any
    engine: Any
    gui_config: Any
    dark_mode: Any

    # ---- the bar ---------------------------------------------------------------------
    def _build_menu(self) -> None:
        bar = tk.Menu(self.root)
        self.menubar = bar
        file_menu = tk.Menu(bar, tearoff=0)
        file_menu.add_command(label="Open a model…", accelerator="Ctrl+O",
                              command=lambda: self.load_model())         # type: ignore[attr-defined]
        file_menu.add_command(label="Save the model as…", accelerator="Ctrl+S",
                              command=lambda: self.save_model())         # type: ignore[attr-defined]
        self.recent_menu = tk.Menu(file_menu, tearoff=0,
                                   postcommand=self._fill_recent_menu)
        file_menu.add_cascade(label="Recent models", menu=self.recent_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", accelerator="Ctrl+Q",
                              command=lambda: self._on_close())          # type: ignore[attr-defined]
        bar.add_cascade(label="File", menu=file_menu)
        view = tk.Menu(bar, tearoff=0)
        view.add_checkbutton(label="Dark theme", variable=self.dark_mode,
                             command=lambda: self._toggle_dark_mode())   # type: ignore[attr-defined]
        bar.add_cascade(label="View", menu=view)
        tools = tk.Menu(bar, tearoff=0)
        self.tools_menu = tools
        tools.add_command(label="Benchmark…", command=self._benchmark_clicked)
        tools.add_command(label="Speed test", command=self._run_speed_test)
        tools.add_separator()
        tools.add_command(label="Open the AuraLite folder", command=self._open_home)
        tools.add_command(label="AuraLite Studio (Qt)", command=self._open_studio)
        bar.add_cascade(label="Tools", menu=tools)
        help_menu = tk.Menu(bar, tearoff=0)
        help_menu.add_command(label="Guide", accelerator="F1",
                              command=lambda: webbrowser.open(GUIDE_URL))
        help_menu.add_command(label="Check for updates…", command=self._check_updates)
        help_menu.add_command(label="About AuraLite", command=self._show_about)
        bar.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=bar)
        for key, action in (("<Control-o>", lambda _e: self.load_model()),  # type: ignore[attr-defined]
                            ("<Control-s>", lambda _e: self.save_model()),  # type: ignore[attr-defined]
                            ("<Control-q>", lambda _e: self._on_close()),   # type: ignore[attr-defined]
                            ("<F1>", lambda _e: webbrowser.open(GUIDE_URL))):
            try:
                self.root.bind_all(key, action)
            except (AttributeError, tk.TclError):              # a test double of the root
                pass

    def _build_footer_tools(self, info_row: Any) -> None:
        """The footer's memory figure and the last speed test."""
        self.memory_label = ttk.Label(info_row, text="RAM: —", style="Sub.TLabel")
        self.memory_label.pack(side=tk.LEFT, padx=8)
        self.speed_label = ttk.Label(info_row, text=self._speed_text(), style="Sub.TLabel")
        self.speed_label.pack(side=tk.LEFT, padx=8)
        self._memory_ticking = False
        self._memory_after: Any = None
        self._tick_memory()

    def _speed_text(self) -> str:
        """The last speed test, for the footer ("" before the first)."""
        speed = float(getattr(self.gui_config, "gflops", 0.0) or 0.0)
        return f"Speed: {speed:,.0f} GFLOP/s" if speed else ""

    # ---- recent models -------------------------------------------------------------------
    def _remember_model(self, path: str) -> None:
        """`path` first in File -> Recent models (saved with the settings)."""
        try:
            self.gui_config = self.gui_config.add_recent(path)
            self.gui_config.save()
        except Exception:                          # a read-only home: remembered this run
            pass

    def _fill_recent_menu(self) -> None:
        menu = self.recent_menu
        menu.delete(0, tk.END)
        recent = [p for p in getattr(self.gui_config, "recent_models", []) if isinstance(p, str)]
        if not recent:
            menu.add_command(label="(none yet)", state=tk.DISABLED)
        for path in recent:
            menu.add_command(label=path, command=lambda p=path: self._open_recent(p))

    def _open_recent(self, path: str) -> None:
        if not os.path.exists(path):
            messagebox.showwarning("Recent models", f"The file is gone:\n{path}")
            self.gui_config = self.gui_config.update(
                recent_models=[p for p in self.gui_config.recent_models if p != path])
            return
        self.load_model(path)                                    # type: ignore[attr-defined]

    # ---- the footer ------------------------------------------------------------------------
    def _update_memory_label(self) -> None:
        from auralite.studio.hardware import format_gb, process_memory
        used = process_memory()
        label = getattr(self, "memory_label", None)
        if label is not None:
            label.config(text=f"RAM: {format_gb(used)}" if used else "RAM: —")

    def _tick_memory(self) -> None:
        # A test's root runs after() at once: the flag makes that one update, not a loop.
        if self._memory_ticking:
            return
        self._memory_ticking = True
        try:
            self._update_memory_label()
            self._memory_after = self.root.after(MEMORY_EVERY_MS, self._tick_memory)
        except Exception:                          # the window is closing
            pass
        finally:
            self._memory_ticking = False

    def _stop_memory_ticks(self) -> None:
        after = getattr(self, "_memory_after", None)
        if after is not None:
            try:
                self.root.after_cancel(after)
            except Exception:
                pass
            self._memory_after = None

    # ---- tools ------------------------------------------------------------------------------
    def _open_home(self) -> None:
        from auralite.studio.hardware import auralite_home
        open_folder(str(auralite_home()))

    def _open_studio(self) -> None:
        import importlib.util
        if importlib.util.find_spec("PySide6") is None:
            messagebox.showinfo("AuraLite Studio",
                                "AuraLite Studio needs Qt 6:\n"
                                "pip install \"auralite-ai[studio]\"\nthen start it again.")
            return
        try:
            subprocess.Popen([sys.executable, "-m", "auralite.studio"], close_fds=True,
                             start_new_session=sys.platform != "win32")
        except OSError as err:
            messagebox.showerror("AuraLite Studio", f"Could not start it:\n{err}")

    def _run_speed_test(self) -> None:
        """Tools -> Speed test: the GFLOP/s that every time estimate builds on."""
        device = "cuda" if getattr(getattr(self.engine, "device", None), "type", "") == "cuda" \
            else "cpu"
        self.speed_label.config(text="Speed test…")

        def run() -> None:
            from auralite.studio.hardware import measure_gflops
            try:
                gflops = float(measure_gflops(device))
            except Exception as err:
                message = str(err)
                self.root.after(0, lambda: self._speed_failed(message))
                return
            self.root.after(0, lambda: self._speed_measured(gflops))

        threading.Thread(target=run, daemon=True).start()

    def _speed_measured(self, gflops: float) -> None:
        threads = int(getattr(self.engine, "num_threads", 0) or 0)
        self.gui_config = self.gui_config.update(gflops=float(gflops), gflops_threads=threads)
        try:
            self.gui_config.save()
        except Exception:
            pass
        self.speed_label.config(text=self._speed_text())
        print(f"[AuraLite] Speed test: {gflops:,.0f} GFLOP/s")

    def _speed_failed(self, message: str) -> None:
        self.speed_label.config(text=self._speed_text())
        messagebox.showerror("Speed test", message)

    # ---- the benchmark -------------------------------------------------------------------
    def _benchmark_clicked(self) -> None:
        """Tools -> Benchmark… opens the dialog; while it runs, it stops it."""
        stop = getattr(self, "_bench_stop", None)
        if stop is not None:
            stop.set()
            self.speed_label.config(text="Stopping the benchmark…")
            return
        choice = self._ask_benchmark()
        if choice is not None:
            self._run_benchmark(choice)

    def _ask_benchmark(self, start: Any = None) -> Any:
        """The benchmark's settings from a dialog (a BenchChoice), None for Cancel."""
        from auralite import bench
        from auralite.studio.benchchoice import METHODS, BenchChoice, check_choice, \
            default_choice
        from auralite.studio.catalog import CATALOG, find_model
        start = start or getattr(self, "bench_choice", None) or default_choice()
        dlg = tk.Toplevel(self.root)
        dlg.title("Benchmark")
        dlg.transient(self.root)
        dlg.resizable(False, False)
        outer = ttk.Frame(dlg, padding="12")
        outer.pack(fill=tk.BOTH, expand=True)
        sizes = list(bench.SIZES)
        ttk.Label(outer, wraplength=520, justify=tk.LEFT,
                  text=f"Measures how fast this computer trains and writes: three sizes of a "
                       f"new model ({', '.join(sizes)}), offline. Close other programs: they "
                       f"slow it down. The report (Markdown and JSON) goes to "
                       f"{bench.default_folder()}.").pack(anchor=tk.W)
        row = ttk.Frame(outer)
        row.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(row, text="Training for each size (seconds):").pack(side=tk.LEFT)
        seconds_var = tk.StringVar(value=str(int(round(start.seconds))))
        ttk.Spinbox(row, from_=15, to=600, increment=15, width=6,
                    textvariable=seconds_var).pack(side=tk.LEFT, padx=6)
        finetune_var = tk.BooleanVar(value=bool(start.finetune))
        ttk.Checkbutton(outer, text="Also measure fine-tuning a model (LoRA / QLoRA)",
                        variable=finetune_var).pack(anchor=tk.W, pady=(6, 2))
        group = ttk.LabelFrame(outer, text="The model to fine-tune", padding="8")
        group.pack(fill=tk.X)
        labels = [f"{m.display_name} - {m.size_label}, {m.download_gb:.1f} GB to download"
                  for m in CATALOG]
        specs = [m.spec for m in CATALOG]
        spec = start.finetune or ""
        model = find_model(spec) if spec.lower().startswith("hf:") else None
        shown = model or find_model(bench.FINETUNE_MODEL)
        where_var = tk.StringVar(value="disk" if start.finetune and model is None
                                 else "catalog")
        catalog_var = tk.StringVar(value=labels[specs.index(shown.spec)]
                                   if shown is not None and shown.spec in specs
                                   else (labels[0] if labels else ""))
        path_var = tk.StringVar(value=spec if start.finetune and model is None else "")
        ttk.Radiobutton(group, text="From the catalog (downloaded once into ~/.auralite/hf):",
                        variable=where_var, value="catalog").pack(anchor=tk.W)
        ttk.Combobox(group, textvariable=catalog_var, values=labels, state="readonly",
                     width=60).pack(anchor=tk.W, padx=18)
        ttk.Radiobutton(group, text="Already downloaded - a Hugging Face model folder or an "
                                    "AuraLite checkpoint:",
                        variable=where_var, value="disk").pack(anchor=tk.W, pady=(6, 0))
        line = ttk.Frame(group)
        line.pack(fill=tk.X, padx=18)
        ttk.Entry(line, textvariable=path_var, width=44).pack(side=tk.LEFT, fill=tk.X,
                                                             expand=True)

        def pick_folder() -> None:
            from .common import filedialog
            path = filedialog.askdirectory(title="The downloaded model's folder "
                                                 "(with config.json)", parent=dlg)
            if path:
                path_var.set(path)
                where_var.set("disk")

        def pick_file() -> None:
            from .common import filedialog
            path = filedialog.askopenfilename(
                title="An AuraLite checkpoint to fine-tune", parent=dlg,
                filetypes=[("AuraLite checkpoints", "*.pt *.safetensors"), ("All files", "*.*")])
            if path:
                path_var.set(path)
                where_var.set("disk")

        ttk.Button(line, text="Folder…", command=pick_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(line, text="File…", command=pick_file).pack(side=tk.LEFT)
        method_row = ttk.Frame(group)
        method_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(method_row, text="Method:").pack(side=tk.LEFT)
        method_labels = [label for _key, label in METHODS]
        method_var = tk.StringVar(value=next((label for key, label in METHODS
                                              if key == start.method), method_labels[0]))
        ttk.Combobox(method_row, textvariable=method_var, values=method_labels,
                     state="readonly", width=48).pack(side=tk.LEFT, padx=6)
        problem = ttk.Label(outer, text="", foreground="#d93025", wraplength=520)
        problem.pack(anchor=tk.W, pady=(6, 0))
        result: dict[str, Any] = {}

        def read() -> Any:
            try:
                seconds = max(15.0, min(600.0, float(seconds_var.get())))
            except (TypeError, ValueError):
                seconds = bench.DEFAULT_SECONDS
            finetune = None
            if finetune_var.get():
                if where_var.get() == "catalog":
                    label = catalog_var.get()
                    finetune = specs[labels.index(label)] if label in labels else ""
                else:
                    finetune = str(path_var.get() or "").strip()
            method = next((key for key, label in METHODS if label == method_var.get()), "")
            return BenchChoice(seconds=seconds, finetune=finetune, method=method)

        def start_clicked() -> None:
            choice = read()
            reason = check_choice(choice)
            if reason:
                problem.config(text=reason)
                return
            result["choice"] = choice
            dlg.destroy()

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(buttons, text="Start", command=start_clicked).pack(side=tk.RIGHT, padx=4)
        self._bench_dialog = {"read": read, "start": start_clicked, "problem": problem,
                              "result": result,
                              "vars": {"seconds": seconds_var, "finetune": finetune_var,
                                       "where": where_var, "catalog": catalog_var,
                                       "path": path_var, "method": method_var}}
        try:
            dlg.grab_set()
            self.root.wait_window(dlg)
        except (AttributeError, tk.TclError):                 # a test double of the root
            pass
        return result.get("choice")

    def _run_benchmark(self, choice: Any) -> None:
        """``auralite bench`` in the background: progress in the footer and the console,
        the summary and the report's path at the end."""
        from auralite import bench
        from auralite.studio.benchchoice import describe
        self.bench_choice = choice
        stop = threading.Event()
        self._bench_stop = stop
        self._set_bench_label("Stop the benchmark")
        print(f"[AuraLite] bench: {describe(choice)}")

        def progress(line: str) -> None:
            text = str(line).strip()
            print(f"[AuraLite] bench: {text}")
            self.root.after(0, lambda: self.speed_label.config(text=f"Benchmark: {text}"[:90]))

        def run() -> None:
            try:
                report = bench.run_bench(sizes=list(bench.SIZES), seconds=choice.seconds,
                                         finetune=choice.finetune, method=choice.method,
                                         progress=progress, stop=stop)
                md, _js = bench.save_report(report)
            except InterruptedError:
                self.root.after(0, lambda: self._bench_over("The benchmark was stopped."))
                return
            except Exception as err:
                message = str(err)
                self.root.after(0, lambda: self._bench_over("", error=message))
                return
            summary = "\n".join(bench.summary_lines(report))
            self.root.after(0, lambda: self._bench_over(
                f"{summary}\n\nThe report, a table for an article or a README: {md}"))

        threading.Thread(target=run, daemon=True).start()

    def _set_bench_label(self, label: str) -> None:
        try:
            self.tools_menu.entryconfig(0, label=label)
        except (AttributeError, tk.TclError):
            pass

    def _bench_over(self, text: str, error: str = "") -> None:
        self._bench_stop = None
        self._set_bench_label("Benchmark…")
        self.speed_label.config(text=self._speed_text())
        if error:
            messagebox.showerror("Benchmark", error)
        elif text:
            messagebox.showinfo("Benchmark", text)
        self.last_bench_text = text or error

    # ---- help ---------------------------------------------------------------------------------
    def _check_updates(self) -> None:
        from auralite import __version__
        from auralite.studio.updates import check_for_update

        def run() -> None:
            info = check_for_update(__version__)
            self.root.after(0, lambda: messagebox.showinfo("Updates", info.message()))

        threading.Thread(target=run, daemon=True).start()

    def about_text(self) -> str:
        import torch

        from auralite import __version__
        from auralite.studio.hardware import auralite_home
        return (f"AuraLite {__version__} - the classic interface (tkinter {tk.TkVersion})\n\n"
                f"Train, fine-tune and run language models on your own computer.\n\n"
                f"Python {platform.python_version()} · PyTorch {torch.__version__}\n"
                f"License: Apache-2.0\nhttps://github.com/AlexanderNyr/AuraLite-AI\n\n"
                f"Models and settings: {auralite_home()}")

    def _show_about(self) -> None:
        messagebox.showinfo("About AuraLite", self.about_text())
