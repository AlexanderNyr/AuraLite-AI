"""gui/tabs/start.py — the Start tab of the classic interface (2.26, as the Studio's).

This computer and what fits it (the advisor), the ways in - fine-tune a ready model,
train a new one, open one -, the recent models, and a run that was interrupted (a crash, a
power cut), which continues from its checkpoint exactly where it stood: a new model's run
on the Train tab, a fine-tune on the Fine-tune tab. The texts are the Studio's
(`studio/startinfo.py`).
"""
from __future__ import annotations

import os
import time
from typing import Any

from auralite import runs as runlib
from auralite.studio import runner
from auralite.studio.progress import summary_text
from auralite.studio.startinfo import fit_lines, interrupted_text

from .. import common as gui_common
from ..common import messagebox, threading, tk, ttk


class StartTabMixin:
    """The Start tab (needs `tab_start`, `engine`, `gui_config`, `notebook`, `root`)."""

    root: Any
    engine: Any
    gui_config: Any

    # ---- the tab -----------------------------------------------------------------
    def _build_start_tab(self) -> None:
        tab = self.tab_start                                   # type: ignore[attr-defined]
        self.hardware = None
        self.start_interrupted_run = None
        ttk.Label(tab, text="Welcome to AuraLite", style="Header.TLabel").pack(anchor=tk.W)
        self.start_intro = ttk.Label(
            tab, wraplength=760, justify=tk.LEFT,
            text="Train language models on your own computer: make a ready model answer your "
                 "way (fine-tuning), or grow a new one from your texts. Nothing leaves this "
                 "computer.")
        self.start_intro.pack(anchor=tk.W, pady=(2, 8))

        # a run that was interrupted continues from here (shown when there is one)
        self.start_interrupted_frame = ttk.LabelFrame(tab, text="  ⏸  An interrupted run  ",
                                                      padding="8")
        self.start_interrupted_label = ttk.Label(self.start_interrupted_frame, text="",
                                                 wraplength=740, justify=tk.LEFT)
        self.start_interrupted_label.pack(anchor=tk.W)
        row = ttk.Frame(self.start_interrupted_frame)
        row.pack(anchor=tk.W, pady=(6, 0))
        self.start_continue_btn = ttk.Button(row, text="▶ Continue it",
                                             command=self._start_continue_interrupted)
        self.start_continue_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.start_dismiss_btn = ttk.Button(row, text="Dismiss",
                                            command=self._start_dismiss_interrupted)
        self.start_dismiss_btn.pack(side=tk.LEFT)
        self._start_interrupted_shown = False

        actions = ttk.LabelFrame(tab, text="  What do you want to do?  ", padding="8")
        actions.pack(fill=tk.X, pady=(0, 8))
        self.start_actions_frame = actions
        for title, text, command in (
                ("🎯 Fine-tune a ready model  (recommended)",
                 "Qwen, Llama, Mistral or SmolLM2 learns from your dialogs or texts",
                 lambda: self._goto_tab("finetune")),
                ("🧠 Train a new model from scratch",
                 "A small model of your own, from your texts: architecture and tokenizer "
                 "included", lambda: self._goto_tab("train")),
                ("📂 Open a model…",
                 "An AuraLite checkpoint (.pt) or a GGUF file to chat with, train further or "
                 "export", lambda: self.load_model())):              # type: ignore[attr-defined]
            line = ttk.Frame(actions)
            line.pack(fill=tk.X, pady=2)
            ttk.Button(line, text=title, width=42, command=command).pack(side=tk.LEFT)
            ttk.Label(line, text=text, style="Sub.TLabel").pack(side=tk.LEFT, padx=8)

        computer = ttk.LabelFrame(tab, text="  💻  This computer  ", padding="8")
        computer.pack(fill=tk.X, pady=(0, 8))
        self.start_hardware_label = ttk.Label(computer, text="Looking at this computer…",
                                              justify=tk.LEFT, wraplength=740)
        self.start_hardware_label.pack(anchor=tk.W)
        self.start_speed_btn = ttk.Button(computer, text="Run the speed test (1 s)",
                                          command=self._run_speed_test)  # type: ignore[attr-defined]
        self.start_speed_btn.pack(anchor=tk.W, pady=4)
        ttk.Label(computer, text="What fits", style="TLabelframe.Label").pack(anchor=tk.W,
                                                                            pady=(4, 0))
        self.start_fit_label = ttk.Label(computer, text="…", justify=tk.LEFT, wraplength=740)
        self.start_fit_label.pack(anchor=tk.W)

        recent = ttk.LabelFrame(tab, text="  🕘  Recent models  ", padding="8")
        recent.pack(fill=tk.BOTH, expand=True)
        self.start_recent = ttk.Treeview(recent, show="tree", height=5, selectmode="browse")
        self.start_recent.pack(fill=tk.X)
        self.start_recent.bind("<Double-1>", self._start_recent_clicked)
        self.start_recent_paths: list[str] = []
        ttk.Label(recent, text="Double-click to open.", style="Sub.TLabel").pack(anchor=tk.W)

        self._start_refresh_recent()
        self._start_refresh_interrupted()
        # once the window shows (the engine is imported by then: it takes milliseconds)
        self.root.after(100, self._start_detect_hardware)

    def _goto_tab(self, name: str) -> None:
        """Bring a tab of the notebook to the front."""
        widget = getattr(self, "_tab_widgets", {}).get(name)
        if widget is not None:
            try:
                self.notebook.select(widget)                   # type: ignore[attr-defined]
            except Exception:                  # a test double of the notebook
                pass
            self._active_tab_name = name

    # ---- this computer -------------------------------------------------------------
    def _start_detect_hardware(self) -> None:
        """Look at this computer - fast, a few milliseconds: no benchmark (the speed test
        is a button). The last speed test counts when it ran with the threads AuraLite
        trains with."""
        from auralite.studio.hardware import detect_hardware
        try:
            hw = detect_hardware()
        except Exception as exc:                       # an exotic system: say so, go on
            gui_common.logger.warning("could not look at this computer: %s", exc)
            hw = None
        if hw is not None:
            speed = float(getattr(self.gui_config, "gflops", 0.0) or 0.0)
            threads = int(getattr(self.gui_config, "gflops_threads", 0) or 0)
            if speed and threads in (hw.train_threads,
                                     int(getattr(self.engine, "num_threads", 0) or 0)):
                hw.gflops = speed
        self._start_show_hardware(hw)

    def _start_show_hardware(self, hw: Any) -> None:
        self.hardware = hw
        if hw is None:
            self.start_hardware_label.config(text="Could not look at this computer.")
            self.start_fit_label.config(text="")
        else:
            self.start_hardware_label.config(text="\n".join(hw.summary_lines()))
            self.start_speed_btn.config(text="Run the speed test again" if hw.gflops
                                        else "Run the speed test (1 s)")
            self.start_fit_label.config(text="\n".join(fit_lines(hw)))
        update = getattr(self, "_ft_update_estimates", None)
        if callable(update):
            update()

    def _hardware_measured(self, gflops: float) -> None:
        """After the speed test (Tools or the button): the estimates are measured now."""
        hw = getattr(self, "hardware", None)
        if hw is not None:
            hw.gflops = float(gflops)
            self._start_show_hardware(hw)

    # ---- recent models ---------------------------------------------------------------
    def _start_refresh_recent(self) -> None:
        tree = self.start_recent
        try:
            for item in tree.get_children():
                tree.delete(item)
        except Exception:                      # a test double
            pass
        self.start_recent_paths = [p for p in getattr(self.gui_config, "recent_models", [])
                                   if isinstance(p, str)]
        if not self.start_recent_paths:
            tree.insert("", tk.END, iid="none",
                        text="No models yet - the ones you train, open or save appear here.")
        for i, path in enumerate(self.start_recent_paths):
            gone = "" if os.path.exists(path) else "   (gone)"
            tree.insert("", tk.END, iid=str(i),
                        text=f"{os.path.basename(path)}   —   {path}{gone}")

    def _start_recent_clicked(self, _event: Any = None) -> None:
        try:
            iid = str(self.start_recent.focus())
        except Exception:
            return
        if iid.isdigit() and int(iid) < len(self.start_recent_paths):
            self._open_recent(self.start_recent_paths[int(iid)])  # type: ignore[attr-defined]
            self._start_refresh_recent()

    # ---- an interrupted run ---------------------------------------------------------
    def _start_refresh_interrupted(self) -> None:
        """Offer the newest interrupted run of this computer that can continue."""
        try:
            interrupted = runlib.interrupted_runs()
        except Exception as exc:                       # an unreadable runs folder
            gui_common.logger.warning("could not read the runs: %s", exc)
            interrupted = []
        run = self.start_interrupted_run = interrupted[0] if interrupted else None
        frame = self.start_interrupted_frame
        if run is None:
            if self._start_interrupted_shown:
                frame.pack_forget()
                self._start_interrupted_shown = False
            return
        headline, rest = interrupted_text(run, len(interrupted) - 1)
        self.start_interrupted_label.config(text=headline + rest)
        if not self._start_interrupted_shown:
            frame.pack(fill=tk.X, pady=(0, 8), before=self.start_actions_frame)
            self._start_interrupted_shown = True

    def _start_continue_interrupted(self) -> None:
        if self.start_interrupted_run is not None:
            self.continue_recorded_run(self.start_interrupted_run)

    def _start_dismiss_interrupted(self) -> None:
        if self.start_interrupted_run is not None:
            runlib.update_run(self.start_interrupted_run, dismissed=True)
        self._start_refresh_interrupted()

    # ---- what runs now -------------------------------------------------------------------
    def _busy_with(self) -> str:
        """What trains now ("" when nothing does): one run at a time."""
        thread = getattr(self, "_ft_thread", None)
        if thread is not None and thread.is_alive():
            return "a fine-tune"
        if getattr(self, "_train_phase", ""):
            return "a training run"
        return ""

    def _model_in_memory_changed(self, label: str = "") -> None:
        """The model in memory changed (loaded, trained, gone): the tabs follow it."""
        engine = self.engine
        model = getattr(engine, "model", None)
        has = model is not None
        self.is_trained = has
        try:
            gguf = bool(engine.is_gguf_model())
        except Exception:
            gguf = False
        self.gen_btn.config(state=tk.NORMAL if has else tk.DISABLED)   # type: ignore[attr-defined]
        self.save_btn.config(state=tk.NORMAL if has and not gguf      # type: ignore[attr-defined]
                             else tk.DISABLED)
        try:
            count = int(model.count_parameters()) if has else 0
        except Exception:
            count = 0
        self.param_label.config(text=f"Parameters: {count:,}" if count  # type: ignore[attr-defined]
                                else "Parameters: —")
        if label:
            self.model_file_label.config(text=label, foreground="black")  # type: ignore[attr-defined]
        if gui_common.HAS_QUANTIZATION and hasattr(self, "_update_quant_buttons"):
            try:
                self._update_quant_buttons()                   # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            self._refresh_model_info()                         # type: ignore[attr-defined]
        except Exception as exc:                               # cosmetic
            gui_common.logger.debug("model info not refreshed: %s", exc)
        follow = getattr(self, "_ft_model_changed", None)
        if callable(follow):
            follow()

    # ---- continuing a recorded run --------------------------------------------------------
    def continue_recorded_run(self, run: Any) -> bool:
        """Continue a stopped, failed or interrupted run of the library from its checkpoint,
        exactly where it stood: a new model's run on the Train tab, a fine-tune on the
        Fine-tune tab (`studio/runner.continue_run`, as the Studio). True when it started."""
        ok, why = run.continue_state()
        if not ok:
            messagebox.showinfo("This run cannot continue", f"{run.name} cannot continue: {why}.")
            return False
        busy = self._busy_with()
        if busy:
            messagebox.showinfo("One run at a time", f"Wait until {busy} ends, or stop it.")
            return False
        if self.engine.model is not None and not messagebox.askyesno(
                "Continue the run", "Continuing the run loads it in place of the model in "
                                    "memory.\n\nContinue?"):
            return False
        kind = "train" if run.kind == "train" else "finetune"
        if kind == "train":
            self._goto_tab("train")
            self._begin_train_ui()                             # type: ignore[attr-defined]
            progress, stop, finish = (self.update_progress,    # type: ignore[attr-defined]
                                      self.stop_event, self.finish_event)  # type: ignore[attr-defined]
        else:
            self._goto_tab("finetune")
            self._ft_begin(f"Continuing {run.name}…")          # type: ignore[attr-defined]
            progress, stop, finish = (self._ft_progress,       # type: ignore[attr-defined]
                                      self.ft_stop_event, self.ft_finish_event)  # type: ignore[attr-defined]
        holder: dict[str, Any] = {}
        started = time.time()

        def on_event(event: str, *payload: Any) -> None:
            if event == "recorder":
                holder["recorder"] = payload[0]
            else:
                self.root.after(0, self._run_event, kind, event, payload)

        def work() -> None:
            try:
                summary = runner.continue_run(self.engine, run, progress, stop_event=stop,
                                              finish_event=finish, on_event=on_event)
            except BaseException as exc:                       # shown on the window's thread
                self.root.after(0, self._run_failed, kind, exc)
                return
            self.root.after(0, self._run_done, kind, summary, holder.get("recorder"),
                            time.time() - started, stop.is_set())

        thread = threading.Thread(target=work, daemon=True)
        if kind == "finetune":
            self._ft_thread = thread
        thread.start()
        return True

    def _run_event(self, kind: str, event: str, payload: tuple) -> None:
        """A step of a run before or after training, on the window's thread."""
        if event == "stage":
            text = str(payload[0])
            if kind == "train":
                self.status_label.config(text=f"Status: {text}")  # type: ignore[attr-defined]
            else:
                self._ft_show_stage(text, payload[1] if len(payload) > 1 else None)  # type: ignore[attr-defined]
        elif event == "import" and kind == "finetune":
            self._ft_show_import(payload[0])                   # type: ignore[attr-defined]
        elif event == "unloaded":
            self._model_in_memory_changed("No model loaded")
        elif event in ("loaded", "continued"):
            spec = str(payload[0])
            label = spec[3:] if spec.lower().startswith("hf:") else os.path.basename(spec) or spec
            self._model_in_memory_changed(f"Loaded: {label}")
            if os.path.isfile(spec) and spec.endswith(".pt"):
                remember = getattr(self, "_remember_model", None)
                if callable(remember):
                    remember(spec)

    def _run_done(self, kind: str, summary: Any, recorder: Any, seconds: float,
                  stopped: bool) -> None:
        text = summary_text(summary, seconds, stopped)
        run = recorder.run if recorder is not None else None
        if run is not None and run.result_path is not None and not run.record.get("save_error"):
            text += (". Saved in the library: it can continue from there."
                     if run.continue_state()[0] else ". Saved in the library.")
            label = f"Run: {run.name}"
        else:
            label = ""
        print(f"[AuraLite] {text}")
        if kind == "train":
            self._reset_train_buttons()                        # type: ignore[attr-defined]
            self.status_label.config(text=f"Status: {text}")   # type: ignore[attr-defined]
        else:
            self._ft_end(text, summary, stopped)               # type: ignore[attr-defined]
        self._model_in_memory_changed(label)
        self._start_refresh_interrupted()

    def _run_failed(self, kind: str, exc: BaseException) -> None:
        if kind == "train":
            self._reset_train_buttons()                        # type: ignore[attr-defined]
            self.status_label.config(text="Status: The run failed - see the message.")  # type: ignore[attr-defined]
        else:
            self._ft_end("The fine-tuning failed - see the message.")  # type: ignore[attr-defined]
        self._model_in_memory_changed()
        self._start_refresh_interrupted()
        messagebox.showerror("The run failed", f"{type(exc).__name__}: {exc}")
